# 文件说明：该文件属于评测运行器，集中实现 generation runner 相关逻辑。
from __future__ import annotations

import json
import logging
from pathlib import Path
from statistics import mean
from typing import Any, Callable

import numpy as np
from PIL import Image

from mmsec_eval.config.schema import AppConfig
from mmsec_eval.defenses.sanitize_v1 import SanitizeDefenseV1
from mmsec_eval.io.jsonl_io import read_jsonl
from mmsec_eval.metrics.generation import answer_matches, normalize_answer, object_jaccard, object_present, text_similarity, yes_no_value
from mmsec_eval.plugins.registry import create
from mmsec_eval.model_adapters.local_vlm_lifecycle import (
    empty_cuda_cache,
    ensure_local_vlm_adapters_ready,
    local_vlm_adapters,
    stop_local_vlm_servers,
)
from mmsec_eval.risk.scoring import compute_risk_score, normalize_inverse
from mmsec_eval.runner.artifacts import make_run_dir, new_run_id, write_env_snapshot, write_json_snapshot, write_results, write_summary
from mmsec_eval.runner.report import write_report
from mmsec_eval.sample_store.serializer import save_image_png, write_json, write_jsonl
from mmsec_eval.types import AttackContext, DefenseContext, ModelOutput, RunArtifacts, Sample
from mmsec_eval.utils.seed import set_seed

LOG = logging.getLogger(__name__)


# 发送 `进度` 回调或事件，让调用方及时感知运行状态。
def _emit_progress(progress: Callable[[str, str, float | None, str], None] | None, stage_key: str, state: str, progress_percent: float | None, message: str) -> None:
    if progress is not None:
        progress(stage_key, state, progress_percent, message)


# 安全计算 `均值`，在空值或异常输入下返回可控结果。
def _safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


# 解析 `路径` 的真实位置或配置值，减少调用方重复分支。
def _resolve_path(value: object, *, base_dir: Path) -> Path:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError("missing image path")
    path = Path(raw)
    if path.is_absolute():
        return path
    candidate = base_dir / path
    if candidate.exists():
        return candidate
    return Path.cwd() / path


# 加载 `图像`，把外部文件、配置或运行产物转换为内存结构。
def _load_image(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float32) / 255.0


# 把 `list` 输入规整为列表，过滤空文本后交给后续流程使用。
def _list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    return [text] if text else []


# 整理 `id` 字段，统一生成式案例在 runner 内的读取口径。
def _case_id(row: dict[str, Any], idx: int) -> str:
    return str(row.get("id") or row.get("sample_id") or row.get("case_id") or f"gen-{idx:04d}")


# 定位 `案例 图像 路径`，把配置值或请求上下文转换成实际文件系统路径。
def _case_image_path(row: dict[str, Any], *, cases_dir: Path) -> Path:
    return _resolve_path(
        row.get("image")
        or row.get("image_path")
        or row.get("clean_image")
        or row.get("clean_image_path"),
        base_dir=cases_dir,
    )


# 从 `案例` 构造评测样本，保留原始行数据作为元信息。
def _sample_from_case(row: dict[str, Any], idx: int, *, image: np.ndarray, image_path: Path, stage: str) -> Sample:
    metadata = dict(row)
    metadata["source"] = str(image_path)
    metadata["image_path"] = str(image_path)
    metadata["generation_stage"] = stage
    references = _list(row.get("reference_captions") or row.get("caption"))
    text = str(
        row.get("question")
        or row.get("prompt")
        or row.get("clean_caption")
        or (references[0] if references else "")
        or row.get("target_object")
        or ""
    )
    target = str(row.get("target_object") or row.get("added_object") or "")
    return Sample(sample_id=_case_id(row, idx), image=np.asarray(image, dtype=np.float32), text=text, target_text=target, metadata=metadata)


# 标记 `样本` 阶段，区分 clean、attacked 和 defended 样本。
def _stage_sample(sample: Sample, stage: str) -> Sample:
    metadata = dict(sample.metadata)
    metadata["generation_stage"] = stage
    return Sample(sample_id=sample.sample_id, image=np.asarray(sample.image, dtype=np.float32), text=sample.text, target_text=sample.target_text, metadata=metadata)


# 组装 `调试轨迹 调试 载荷`，把分散字段整理成后端任务或风险评分使用的载荷。
def _trace_debug_payload(row: dict[str, Any], attack_name: str, trace: list[Any]) -> dict[str, Any]:
    return {
        "sample_id": str(row.get("sample_id") or row.get("id") or ""),
        "attack": attack_name,
        "trace_steps": len(trace),
        "trace_tail": [
            {
                "step": int(item.step),
                "loss_total": float(item.loss_total),
                "loss_parts": dict(item.loss_parts),
                "metadata": dict(item.metadata),
            }
            for item in trace[-5:]
        ],
    }


# 生成 `视觉问答`，补齐前端展示或后续评测需要的样本资产。
def _generate_vqa(model: Any, sample: Sample, row: dict[str, Any], cfg: AppConfig) -> ModelOutput:
    question = str(row.get("question") or sample.text or "").strip()
    if not question:
        raise ValueError(f"VQA case {sample.sample_id} missing question")
    return model.generate_answer(sample, question, prompt=str(cfg.task.vqa_prompt), max_tokens=64)


# 生成 `图像描述`，补齐前端展示或后续评测需要的样本资产。
def _generate_caption(model: Any, sample: Sample, cfg: AppConfig) -> ModelOutput:
    return model.generate_caption(sample, prompt=str(cfg.task.caption_prompt), max_tokens=96)


# 探测 `存在性` 状态，优先调用模型探针并在失败时回退到文本匹配。
def _probe_present(model: Any, sample: Sample, object_name: str, aliases: list[str], caption_text: str, cfg: AppConfig) -> bool:
    if bool(cfg.task.object_probe_enabled) and object_name:
        try:
            out = model.object_probe(sample, object_name, prompt=str(cfg.task.object_probe_prompt), max_tokens=8)
            parsed = yes_no_value(out.text)
            if parsed is not None:
                return bool(parsed)
        except (AttributeError, NotImplementedError, RuntimeError, ValueError, TypeError):
            return object_present(caption_text, object_name, aliases)
    return object_present(caption_text, object_name, aliases)


# 计算 `视觉问答 指标`，把原始模型输出汇总成页面和报告使用的指标字段。
def _vqa_metrics(row: dict[str, Any], clean: ModelOutput, attacked: ModelOutput, defended: ModelOutput | None) -> dict[str, Any]:
    aliases = _list(row.get("answer_aliases") or row.get("answers") or row.get("acceptable_answers"))
    answer = str(row.get("answer") or row.get("ground_truth") or row.get("label") or (aliases[0] if aliases else "")).strip()
    clean_correct = answer_matches(clean.text, answer, aliases)
    attacked_correct = answer_matches(attacked.text, answer, aliases)
    defended_correct = answer_matches(defended.text, answer, aliases) if defended is not None else False
    attack_success = bool(clean_correct and not attacked_correct)
    return {
        "answer": answer,
        "answer_aliases": aliases,
        "clean_correct": clean_correct,
        "attacked_correct": attacked_correct,
        "defended_correct": defended_correct,
        "answer_changed": normalize_answer(clean.text) != normalize_answer(attacked.text),
        "attack_success": attack_success,
        "defense_recovered": bool(attack_success and defended_correct),
    }


# 计算 `图像描述 指标`，把原始模型输出汇总成页面和报告使用的指标字段。
def _caption_metrics(row: dict[str, Any], model: Any, clean_sample: Sample, attacked_sample: Sample, defended_sample: Sample | None, clean: ModelOutput, attacked: ModelOutput, defended: ModelOutput | None, cfg: AppConfig) -> dict[str, Any]:
    target = str(row.get("target_object") or row.get("added_object") or clean_sample.target_text or "").strip()
    aliases = _list(row.get("target_aliases"))
    non_targets = _list(row.get("non_target_objects"))
    goal = str(row.get("attack_goal") or row.get("goal") or "remove_object").strip().lower()

    clean_present = _probe_present(model, clean_sample, target, aliases, clean.text, cfg) if target else False
    attacked_present = _probe_present(model, attacked_sample, target, aliases, attacked.text, cfg) if target else False
    defended_present = (
        _probe_present(model, defended_sample, target, aliases, defended.text, cfg)
        if target and defended_sample is not None and defended is not None
        else False
    )

    if goal in {"add", "add_object", "insert_object"}:
        attack_success = bool(not clean_present and attacked_present)
        defense_recovered = bool(attack_success and defended is not None and defended_present == clean_present)
    else:
        attack_success = bool(clean_present and not attacked_present)
        defense_recovered = bool(attack_success and defended is not None and defended_present == clean_present)

    clean_non_target = []
    attacked_non_target = []
    defended_non_target = []
    for obj in non_targets:
        if _probe_present(model, clean_sample, obj, [], clean.text, cfg):
            clean_non_target.append(obj)
        if _probe_present(model, attacked_sample, obj, [], attacked.text, cfg):
            attacked_non_target.append(obj)
        if defended_sample is not None and defended is not None and _probe_present(model, defended_sample, obj, [], defended.text, cfg):
            defended_non_target.append(obj)

    clean_non_target_set = set(clean_non_target)
    spr = 1.0 if not clean_non_target_set else float(len(clean_non_target_set & set(attacked_non_target)) / len(clean_non_target_set))
    return {
        "target_object": target,
        "target_aliases": aliases,
        "attack_goal": goal,
        "target_present_clean": clean_present,
        "target_present_attacked": attacked_present,
        "target_present_defended": defended_present,
        "non_target_objects": non_targets,
        "clean_non_target_present": clean_non_target,
        "attacked_non_target_present": attacked_non_target,
        "defended_non_target_present": defended_non_target,
        "semantic_preservation_rate": spr,
        "object_jaccard": object_jaccard(clean_non_target + ([target] if clean_present else []), attacked_non_target + ([target] if attacked_present else [])),
        "caption_text_similarity": text_similarity(clean.text, attacked.text),
        "attack_success": attack_success,
        "defense_recovered": defense_recovered,
    }


# 标记 `output scores` 阶段，区分 clean、attacked 和 defended 样本。
def _stage_output_scores(task_kind: str, stage_metrics: dict[str, Any]) -> dict[str, float]:
    if task_kind == "vqa":
        return {
            "clean": 1.0 if bool(stage_metrics.get("clean_correct", False)) else 0.0,
            "adv": 1.0 if bool(stage_metrics.get("attacked_correct", False)) else 0.0,
            "defended": 1.0 if bool(stage_metrics.get("defended_correct", False)) else 0.0,
        }

    clean_present = stage_metrics.get("target_present_clean")
    attacked_present = stage_metrics.get("target_present_attacked")
    defended_present = stage_metrics.get("target_present_defended")
    if clean_present is None:
        return {
            "clean": 1.0,
            "adv": float(stage_metrics.get("caption_text_similarity", 0.0) or 0.0),
            "defended": 1.0 if bool(stage_metrics.get("defense_recovered", False)) else 0.0,
        }
    return {"clean": 1.0, "adv": 1.0 if attacked_present == clean_present else 0.0, "defended": 1.0 if defended_present == clean_present else 0.0}


# 整理 `证据包` 字段，统一生成式案例在 runner 内的读取口径。
def _case_bundle(
    *,
    cfg: AppConfig,
    row: dict[str, Any],
    clean_sample: Sample,
    attacked_sample: Sample,
    defended_sample: Sample | None,
    clean_output: ModelOutput,
    attacked_output: ModelOutput,
    defended_output: ModelOutput | None,
    stage_metrics: dict[str, Any],
    refs: dict[str, str],
    perturbation: dict[str, float],
) -> dict[str, Any]:
    task_kind = str(cfg.task.kind)
    input_label = "问题" if task_kind == "vqa" else "描述指令"
    clean_input = str(row.get("question") or cfg.task.caption_prompt)
    output_scores = _stage_output_scores(task_kind, stage_metrics)
    defended_text = defended_output.text if defended_output is not None else ""
    bundle = {
        "task_kind": task_kind,
        "sample": {
            "sample_id": clean_sample.sample_id,
            "text": clean_input,
            "target_text": str(row.get("target_object") or row.get("added_object") or ""),
            "metadata": dict(clean_sample.metadata),
        },
        "adversarial": {
            "sample_id": attacked_sample.sample_id,
            "text": clean_input,
            "perturbation_l0": int(perturbation.get("l0", 0.0)),
            "perturbation_l2": float(perturbation.get("l2", 0.0)),
            "perturbation_linf": float(perturbation.get("linf", 0.0)),
            "metadata": dict(attacked_sample.metadata),
        },
        "defended": {
            "sample_id": defended_sample.sample_id if defended_sample is not None else "",
            "text": clean_input,
            "metadata": dict(defended_sample.metadata) if defended_sample is not None else {},
        },
        "inputs": {
            "clean": {"text": clean_input, "label": input_label},
            "adv": {"text": clean_input, "label": input_label},
            "defended": {"text": clean_input, "label": input_label},
        },
        "dataset_tag": str(cfg.dataset.benchmark_tag or cfg.dataset.kind),
        "model_tag": str(cfg.plugins.model_adapter),
        "outputs": {
            "clean": {"text": clean_output.text, "score": output_scores["clean"]},
            "adv": {"text": attacked_output.text, "score": output_scores["adv"]},
            "defended": {"text": defended_text, "score": output_scores["defended"]},
        },
        "metrics": {
            **stage_metrics,
            "perturbation_l2": float(perturbation.get("l2", 0.0)),
            "perturbation_linf": float(perturbation.get("linf", 0.0)),
        },
        "judge": {
            "success": bool(stage_metrics.get("attack_success", False)),
            "reason": "generation_output_changed_successfully" if stage_metrics.get("attack_success", False) else "generation_output_not_successful",
        },
        "diagnostics": {
            "text_diff_score": 1.0 - text_similarity(clean_output.text, attacked_output.text),
            "embedding_shift": float(perturbation.get("l2", 0.0)),
            "generation_task": task_kind,
        },
        "artifact_refs": refs,
        "visual_labels": {
            "clean": "原始图片",
            "adv": "攻击后图片",
            "defended": "防御后图片",
        },
    }
    return bundle


# 计算 `样本 delta 指标`，把原始模型输出汇总成页面和报告使用的指标字段。
def _sample_delta_metrics(clean: Sample, attacked: Sample) -> dict[str, float]:
    delta = np.asarray(attacked.image, dtype=np.float32) - np.asarray(clean.image, dtype=np.float32)
    flat = delta.reshape(-1)
    return {
        "l0": float(np.count_nonzero(np.abs(flat) > 1e-8)),
        "l2": float(np.linalg.norm(flat, ord=2)),
        "linf": float(np.max(np.abs(flat))) if flat.size else 0.0,
    }


# 组装 `风险 载荷`，把分散字段整理成后端任务或风险评分使用的载荷。
def _risk_payload(cfg: AppConfig, *, asr: float, semantic: float, avg_l2: float, avg_linf: float, stability: float) -> dict[str, Any]:
    if not bool(cfg.risk.enabled):
        return {"risk_score": 0.0, "risk_level": "disabled", "risk_scenario": str(cfg.task.kind), "risk_breakdown": {}, "risk_weights": {}, "risk_recommendations": []}
    return compute_risk_score(
        scenario="qa" if str(cfg.task.kind) == "vqa" else "caption",
        components={
            "effectiveness": float(asr),
            "semantic": float(semantic),
            "cost": 0.5 * (normalize_inverse(avg_l2, float(cfg.risk.l2_reference)) + normalize_inverse(avg_linf, float(cfg.risk.linf_reference))),
            "transfer": 0.0,
            "stability": float(stability),
        },
        weights=dict(cfg.risk.weights or {}),
    )


# 加载 `生成式评测 rows`，把外部文件、配置或运行产物转换为内存结构。
def _load_generation_rows(cfg: AppConfig, progress: Callable[[str, str, float | None, str], None] | None) -> tuple[Path, Path, list[dict[str, Any]]]:
    cases_path = Path(str(cfg.task.cases_jsonl))
    cases_dir = cases_path.parent if cases_path.parent.exists() else Path.cwd()
    _emit_progress(progress, "dataset_loading", "running", 38, f"正在读取生成式评测样本：{cases_path}")
    rows = read_jsonl(str(cases_path))
    max_samples = int(cfg.runner.max_samples or cfg.dataset.max_items or 0)
    return cases_path, cases_dir, rows[:max_samples] if max_samples > 0 else rows


# 执行 `生成式评测 components` 辅助逻辑，保持评测运行器中的输入处理和结果输出一致。
def _generation_components(cfg: AppConfig) -> tuple[Any, Any, Any, Any, str]:
    gen_model = create("model_adapter", cfg.plugins.model_adapter)
    surrogate_name = str(cfg.runner.surrogate_model_adapter or "clip_hf")
    surrogate = create("model_adapter", surrogate_name)
    attack = create("attack", cfg.plugins.attack)
    defense = create("defense", cfg.plugins.defense) if bool(cfg.defense.enabled) else None
    return gen_model, surrogate, attack, defense, surrogate_name


# 推断 `攻击 and defend`，从样本、配置或运行记录中提取统一名称。
def _attack_and_defend(
    cfg: AppConfig,
    clean_sample: Sample,
    surrogate: Any,
    attack: Any,
    defense: Any,
    *,
    run_dir: str,
    sample_debug_dir: Path,
) -> tuple[Any, Sample, Sample | None]:
    attacked = attack.attack(
        clean_sample,
        AttackContext(config=cfg, model_adapter=surrogate, surrogate_model_adapter=surrogate, run_dir=run_dir, sample_debug_dir=str(sample_debug_dir)),
    )
    attacked_sample = _stage_sample(attacked.sample, "attacked")
    if defense is None:
        return attacked, attacked_sample, None
    defended = defense.defend(
        attacked_sample,
        DefenseContext(config=cfg, model_adapter=surrogate, stage="attacked", run_dir=run_dir, sample_debug_dir=str(sample_debug_dir / "defense")),
    )
    return attacked, attacked_sample, _stage_sample(defended.sample, "defended")


# 执行 `staged lifecycle enabled` 辅助逻辑，保持评测运行器中的输入处理和结果输出一致。
def _staged_lifecycle_enabled(cfg: AppConfig) -> bool:
    return bool(getattr(cfg.runner, "staged_model_lifecycle", True))


# 推断 `release 本地 视觉语言模型 所属 攻击`，从样本、配置或运行记录中提取统一名称。
def _release_local_vlm_for_attack(cfg: AppConfig, progress: Callable[[str, str, float | None, str], None] | None) -> None:
    if not _staged_lifecycle_enabled(cfg) or not bool(getattr(cfg.runner, "stop_local_vlm_before_attack", True)):
        return
    _emit_progress(progress, "model_preflight", "running", 42, "正在停止本地 VLM，释放显存给攻击生成阶段。")
    stop_local_vlm_servers()
    empty_cuda_cache()


# 准备 `生成式评测 模型 所属 evaluation` 数据，补齐后续运行、报告或测试需要的字段。
def _prepare_generation_model_for_evaluation(cfg: AppConfig, adapters: list[str], progress: Callable[[str, str, float | None, str], None] | None) -> None:
    if not _staged_lifecycle_enabled(cfg):
        return
    empty_cuda_cache()
    if not adapters or not bool(getattr(cfg.runner, "restart_local_vlm_for_evaluation", True)):
        return
    _emit_progress(progress, "model_preflight", "running", 55, "攻击图已生成，正在启动本地 VLM 进行评测。")
    ensure_local_vlm_adapters_ready(adapters)
    _emit_progress(progress, "model_preflight", "success", 58, "本地 VLM 已就绪，开始评测攻击图。")


# 生成 `outputs`，补齐前端展示或后续评测需要的样本资产。
def _generate_outputs(cfg: AppConfig, row: dict[str, Any], gen_model: Any, clean_sample: Sample, attacked_sample: Sample, defended_sample: Sample | None) -> tuple[ModelOutput, ModelOutput, ModelOutput | None, dict[str, Any], float]:
    if str(cfg.task.kind) == "vqa":
        clean_out = _generate_vqa(gen_model, clean_sample, row, cfg)
        attacked_out = _generate_vqa(gen_model, attacked_sample, row, cfg)
        defended_out = _generate_vqa(gen_model, defended_sample, row, cfg) if defended_sample is not None else None
        metrics = _vqa_metrics(row, clean_out, attacked_out, defended_out)
        return clean_out, attacked_out, defended_out, metrics, 1.0 - (1.0 if metrics["answer_changed"] else 0.0)
    clean_out = _generate_caption(gen_model, clean_sample, cfg)
    attacked_out = _generate_caption(gen_model, attacked_sample, cfg)
    defended_out = _generate_caption(gen_model, defended_sample, cfg) if defended_sample is not None else None
    metrics = _caption_metrics(row, gen_model, clean_sample, attacked_sample, defended_sample, clean_out, attacked_out, defended_out, cfg)
    return clean_out, attacked_out, defended_out, metrics, float(metrics.get("semantic_preservation_rate", 0.0))


# 整理 `产物 refs` 路径信息，把本地文件或产物引用转换成统一表示。
def _artifact_refs(case_dir: Path, sample_debug_dir: Path, cfg: AppConfig, clean_sample: Sample, attacked_sample: Sample, defended_sample: Sample | None, attacked: Any) -> dict[str, str]:
    refs = {"clean_image": save_image_png(str(case_dir / "clean.png"), clean_sample.image), "adv_image": save_image_png(str(case_dir / "adv.png"), attacked_sample.image)}
    if defended_sample is not None:
        refs["defended_image"] = save_image_png(str(case_dir / "defended.png"), defended_sample.image)
    sample_debug_dir.mkdir(parents=True, exist_ok=True)
    refs["attack_debug"] = write_json(str(sample_debug_dir / "debug.json"), _trace_debug_payload({}, cfg.plugins.attack, attacked.attack_trace))
    refs.update(dict(attacked.artifact_refs or {}))
    for src_key, ref_key in (("attention_debug_path", "attention_map"), ("mask_debug_path", "mask_map"), ("patch_preview", "patch_preview"), ("joint_debug_path", "joint_debug")):
        value = str(dict(attacked.metadata or {}).get(src_key) or "").strip()
        if value and not str(refs.get(ref_key) or "").strip():
            refs[ref_key] = value
    return refs


# 整理 `result rows` 字段，统一生成式案例在 runner 内的读取口径。
def _case_result_rows(sid: str, cfg: AppConfig, row: dict[str, Any], case_dir: Path, outputs: tuple[ModelOutput, ModelOutput, ModelOutput | None], metrics: dict[str, Any], refs: dict[str, str], perturbation: dict[str, float]) -> tuple[dict[str, Any], dict[str, Any]]:
    clean_out, attacked_out, defended_out = outputs
    result_row = {
        "sample_id": sid,
        "task_kind": str(cfg.task.kind),
        "question": str(row.get("question") or ""),
        "target_object": str(row.get("target_object") or row.get("added_object") or ""),
        "clean_output": clean_out.text,
        "attacked_output": attacked_out.text,
        "defended_output": defended_out.text if defended_out is not None else "",
        "attack_success": bool(metrics.get("attack_success", False)),
        "defense_recovered": bool(metrics.get("defense_recovered", False)),
        "perturbation_l2": float(perturbation["l2"]),
        "perturbation_linf": float(perturbation["linf"]),
        "metrics": metrics,
        "artifact_refs": refs,
    }
    index_row = {
        "sample_id": sid,
        "case_dir": str(case_dir),
        "judge_success": bool(metrics.get("attack_success", False)),
        "perturbation_l2": float(perturbation["l2"]),
        "perturbation_linf": float(perturbation["linf"]),
        "defense_gain_sample": 1.0 if bool(metrics.get("defense_recovered", False)) else 0.0,
    }
    return result_row, index_row


# 执行 `one 生成式评测 案例` 流程，按配置驱动评测运行器完成一次任务。
def _run_one_generation_case(ctx: dict[str, Any], idx: int, row: dict[str, Any]) -> dict[str, Any]:
    cfg: AppConfig = ctx["cfg"]
    sid = _case_id(row, idx)
    image_path = _case_image_path(row, cases_dir=ctx["cases_dir"])
    clean_sample = _sample_from_case(row, idx, image=_load_image(image_path), image_path=image_path, stage="clean")
    case_dir = ctx["cases_root"] / sid
    case_dir.mkdir(parents=True, exist_ok=True)
    _release_local_vlm_for_attack(cfg, ctx.get("progress"))
    attacked, attacked_sample, defended_sample = _attack_and_defend(cfg, clean_sample, ctx["surrogate"], ctx["attack"], ctx["defense"], run_dir=ctx["run_dir"], sample_debug_dir=ctx["debug_root"] / sid)
    _prepare_generation_model_for_evaluation(cfg, ctx.get("local_generation_adapters", []), ctx.get("progress"))
    clean_out, attacked_out, defended_out, metrics, semantic = _generate_outputs(cfg, row, ctx["gen_model"], clean_sample, attacked_sample, defended_sample)
    perturbation = _sample_delta_metrics(clean_sample, attacked_sample)
    refs = _artifact_refs(case_dir, ctx["debug_root"] / sid, cfg, clean_sample, attacked_sample, defended_sample, attacked)
    bundle = _case_bundle(cfg=cfg, row=row, clean_sample=clean_sample, attacked_sample=attacked_sample, defended_sample=defended_sample, clean_output=clean_out, attacked_output=attacked_out, defended_output=defended_out, stage_metrics=metrics, refs=refs, perturbation=perturbation)
    refs["case_bundle"] = write_json(str(case_dir / "case_bundle.json"), bundle)
    result_row, index_row = _case_result_rows(sid, cfg, row, case_dir, (clean_out, attacked_out, defended_out), metrics, refs, perturbation)
    return {"result": result_row, "index": index_row, "l2": perturbation["l2"], "linf": perturbation["linf"], "semantic": semantic, "success": metrics.get("attack_success", False), "recovered": metrics.get("defense_recovered", False)}


# 计算 `生成式评测 指标`，把原始模型输出汇总成页面和报告使用的指标字段。
def _generation_metrics(task_kind: str, results: list[dict[str, Any]], asr_attack: float, semantic_score: float, recovered_values: list[float]) -> dict[str, float]:
    if task_kind == "vqa":
        return {
            "clean_accuracy": _safe_mean([1.0 if row.get("metrics", {}).get("clean_correct", False) else 0.0 for row in results]),
            "attacked_accuracy": _safe_mean([1.0 if row.get("metrics", {}).get("attacked_correct", False) else 0.0 for row in results]),
            "defended_accuracy": _safe_mean([1.0 if row.get("metrics", {}).get("defended_correct", False) else 0.0 for row in results]),
            "answer_change_rate": _safe_mean([1.0 if row.get("metrics", {}).get("answer_changed", False) else 0.0 for row in results]),
            "target_flip_rate": asr_attack,
            "semantic_preservation_rate": semantic_score,
        }
    return {
        "target_flip_rate": asr_attack,
        "semantic_preservation_rate": semantic_score,
        "caption_text_similarity": _safe_mean([float(row.get("metrics", {}).get("caption_text_similarity", 0.0) or 0.0) for row in results]),
        "object_jaccard": _safe_mean([float(row.get("metrics", {}).get("object_jaccard", 0.0) or 0.0) for row in results]),
        "recovery_rate": _safe_mean(recovered_values),
    }


# 组装 `摘要 载荷`，把分散字段整理成后端任务或风险评分使用的载荷。
def _summary_payload(cfg: AppConfig, surrogate_name: str, defense: Any, results: list[dict[str, Any]], values: dict[str, list[float]]) -> tuple[dict[str, Any], dict[str, Any]]:
    asr_attack = _safe_mean(values["success"])
    asr_defended = _safe_mean(values["defended_success"])
    semantic_score = _safe_mean(values["semantic"])
    risk_payload = _risk_payload(cfg, asr=asr_attack, semantic=semantic_score, avg_l2=_safe_mean(values["l2"]), avg_linf=_safe_mean(values["linf"]), stability=asr_attack)
    gen_metrics = _generation_metrics(str(cfg.task.kind), results, asr_attack, semantic_score, values["recovered"])
    summary = {
        "task_kind": str(cfg.task.kind),
        "dataset_name": str(cfg.dataset.kind),
        "benchmark_tag": str(cfg.dataset.benchmark_tag or cfg.dataset.kind),
        "model_adapter": str(cfg.plugins.model_adapter),
        "surrogate_model_adapter": surrogate_name,
        "victim_model_adapters": [str(cfg.plugins.model_adapter)],
        "attack": str(cfg.plugins.attack),
        "eval_scope": str(cfg.task.eval_scope or "image"),
        "attack_mode": str(cfg.attack.mode),
        "defense": str(cfg.plugins.defense) if defense is not None else "",
        "defense_enabled": bool(defense is not None),
        "num_samples": len(results),
        "num_effective": len(results),
        "asr": round(float(asr_attack), 6),
        "asr_attack": round(float(asr_attack), 6),
        "asr_defended": round(float(asr_defended), 6),
        "defense_gain": round(float(asr_attack - asr_defended), 6),
        "recovery_rate": round(_safe_mean(values["recovered"]), 6),
        "avg_l2": round(_safe_mean(values["l2"]), 6),
        "avg_linf": round(_safe_mean(values["linf"]), 6),
        "semantic_preservation_score": round(float(semantic_score), 6),
        "generation_metrics": gen_metrics,
        "metric_series": {"l2": values["l2"], "linf": values["linf"]},
        "staged_model_lifecycle": bool(getattr(cfg.runner, "staged_model_lifecycle", True)),
        **risk_payload,
    }
    return summary, risk_payload


# 组装 `报告 载荷`，把分散字段整理成后端任务或风险评分使用的载荷。
def _report_payload(cfg: AppConfig, cases_path: Path, summary: dict[str, Any], risk_payload: dict[str, Any], results: list[dict[str, Any]], values: dict[str, list[float]]) -> dict[str, Any]:
    gen_metrics = dict(summary.get("generation_metrics", {}))
    return {
        "summary": summary,
        "rows_preview": results[:20],
        "metric_series": summary["metric_series"],
        "generation": {"task_kind": str(cfg.task.kind), "metrics": gen_metrics, "cases_jsonl": str(cases_path)},
        "stage_metrics": {
            "clean": {"accuracy": gen_metrics.get("clean_accuracy", 0.0)},
            "attacked": {"asr": _safe_mean(values["success"]), "accuracy": gen_metrics.get("attacked_accuracy", 0.0)},
            "defended": {"asr": _safe_mean(values["defended_success"]), "recovery_rate": summary["recovery_rate"]},
        },
        "risk": {**risk_payload, "components_raw": {"asr_attack": _safe_mean(values["success"]), "semantic_preservation": _safe_mean(values["semantic"]), "avg_l2": _safe_mean(values["l2"]), "avg_linf": _safe_mean(values["linf"])}},
    }


# 作为 `generation_runner.py` 的执行入口，串联参数读取、核心处理和退出状态。
def run(cfg: AppConfig, progress: Callable[[str, str, float | None, str], None] | None = None) -> RunArtifacts:
    if str(cfg.task.kind) not in {"vqa", "caption"}:
        raise ValueError("generation runner requires task.kind='vqa' or 'caption'")
    set_seed(cfg.seed)
    run_id = new_run_id()
    run_dir = make_run_dir(cfg.artifacts_dir, run_id)
    write_env_snapshot(run_dir)
    cases_path, cases_dir, rows = _load_generation_rows(cfg, progress)
    gen_model, surrogate, attack, defense, surrogate_name = _generation_components(cfg)
    cases_root = Path(run_dir) / "cases"
    debug_root = Path(run_dir) / "attack_debug"
    cases_root.mkdir(parents=True, exist_ok=True)
    debug_root.mkdir(parents=True, exist_ok=True)
    ctx = {
        "cfg": cfg,
        "run_dir": run_dir,
        "cases_dir": cases_dir,
        "cases_root": cases_root,
        "debug_root": debug_root,
        "gen_model": gen_model,
        "surrogate": surrogate,
        "attack": attack,
        "defense": defense,
        "progress": progress,
        "local_generation_adapters": local_vlm_adapters([str(cfg.plugins.model_adapter)]),
    }
    _emit_progress(progress, "attack_execution", "running", 48, f"正在执行 {len(rows)} 条生成式样本的攻击与生成。")
    results: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []
    values = {"l2": [], "linf": [], "semantic": [], "success": [], "defended_success": [], "recovered": []}
    for idx, row in enumerate(rows):
        item = _run_one_generation_case(ctx, idx, row)
        results.append(item["result"])
        index_rows.append(item["index"])
        values["l2"].append(float(item["l2"]))
        values["linf"].append(float(item["linf"]))
        values["semantic"].append(float(item["semantic"]))
        values["success"].append(1.0 if bool(item["success"]) else 0.0)
        values["recovered"].append(1.0 if bool(item["recovered"]) else 0.0)
        values["defended_success"].append(0.0 if bool(item["recovered"]) else (1.0 if bool(item["success"]) else 0.0))
        _emit_progress(progress, "attack_execution", "running", 48 + 35 * ((idx + 1) / max(1, len(rows))), f"已完成 {idx + 1} / {len(rows)} 条生成式样本")
    summary, risk_payload = _summary_payload(cfg, surrogate_name, defense, results, values)
    summary["run_id"] = run_id
    report_data = _report_payload(cfg, cases_path, summary, risk_payload, results, values)
    _emit_progress(progress, "result_aggregation", "running", 90, "正在汇总生成式评测结果。")
    results_path = write_results(run_dir, results)
    run_index_path = write_jsonl(str(Path(run_dir) / "cases_index.jsonl"), index_rows)
    summary_path = write_summary(run_dir, summary)
    write_json_snapshot(run_dir, "report_data.json", report_data)
    _emit_progress(progress, "report_writing", "running", 97, "正在写入报告。")
    report_path = write_report(run_dir, summary, results)
    _emit_progress(progress, "completed", "success", 100, "生成式评测完成。")
    if bool(getattr(cfg.runner, "stop_local_vlm_after_run", False)):
        stop_local_vlm_servers(adapters=local_vlm_adapters([str(cfg.plugins.model_adapter)]))
        empty_cuda_cache()
    return RunArtifacts(run_id=run_id, run_dir=run_dir, results_path=results_path, summary_path=summary_path, report_path=report_path, run_index_path=run_index_path)
