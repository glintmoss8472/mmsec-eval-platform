# 文件说明：该文件属于后端业务服务，集中实现 asset evaluator 相关逻辑。
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Callable

import numpy as np
from PIL import Image

from mmsec_api.store.sqlite import SQLiteStore
from mmsec_api.utils import utc_now_iso
from mmsec_eval.metrics.generation import answer_matches, normalize_answer, object_jaccard, object_present, text_similarity
from mmsec_eval.plugins.builtin import register_builtin_plugins
from mmsec_eval.plugins.registry import create
from mmsec_eval.risk.scoring import compute_risk_score, normalize_inverse
from mmsec_eval.types import ModelOutput, Sample


# 规范化 `文本` 字段，把空值和非字符串输入转换为稳定文本。
def _text(value: object) -> str:
    return str(value or "").strip()


# 把 `数值` 输入转换为数值，无法解析时返回调用方指定的默认值。
def _num(value: object, default: float = 0.0) -> float:
    try:
        return float(value if value is not None else default)
    except (TypeError, ValueError):
        return float(default)


# 确认 `record` 是字典记录，避免后续字段读取直接接触异常类型。
def _record(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


# 把 `list` 输入规整为列表，过滤空文本后交给后续流程使用。
def _list(value: object) -> list[str]:
    if isinstance(value, list):
        return [_text(item) for item in value if _text(item)]
    text = _text(value)
    return [text] if text else []


# 安全计算 `案例 id`，在空值或异常输入下返回可控结果。
def _safe_case_id(raw: str, index: int, used: set[str]) -> str:
    token = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in raw.strip())[:100] or f"asset-{index:03d}"
    candidate = token
    suffix = 2
    while candidate in used:
        candidate = f"{token}-{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


# 解析 `引用路径` 的真实位置或配置值，减少调用方重复分支。
def _resolve_ref(project_root: Path, ref: str) -> Path | None:
    text = _text(ref)
    if not text:
        return None
    path = Path(text)
    if path.exists():
        return path
    if path.is_absolute():
        return path if path.exists() else None
    candidate = project_root / path
    return candidate if candidate.exists() else None


# 复制 `引用路径` 对应的文件引用，并返回可写入结果记录的路径。
def _copy_ref(project_root: Path, ref: str, dest: Path) -> str:
    src = _resolve_ref(project_root, ref)
    if src is None or not src.is_file():
        return ""
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return dest.as_posix()


# 加载 `图像`，把外部文件、配置或运行产物转换为内存结构。
def _load_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


# 读取 `证据包` 来源数据，缺失或格式异常时返回空结构。
def _source_bundle(project_root: Path, asset: dict[str, Any]) -> dict[str, Any]:
    path = project_root / "artifacts" / "runs" / _text(asset.get("run_id")) / "cases" / _text(asset.get("sample_id")) / "case_bundle.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


# 归类 `风险 level`，把连续分数或多条记录整理成稳定分组。
def _risk_level(score: float) -> str:
    if score >= 0.8:
        return "critical"
    if score >= 0.6:
        return "high"
    if score >= 0.4:
        return "medium"
    if score >= 0.2:
        return "low"
    return "very_low"


# 写出 `JSON`，保证后续报告、页面或复现实验能读取。
def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# 写出 `JSONL`，保证后续报告、页面或复现实验能读取。
def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


# 计算目标项在降序分数列表中的排名，使用稳定排序保证指标结果可复现。
def _rank_desc(values: np.ndarray, target: int) -> int:
    order = np.argsort(-np.asarray(values, dtype=np.float32), kind="mergesort")
    hit = np.where(order == int(target))[0]
    return int(hit[0]) + 1 if hit.size else int(len(order) + 1)


# 计算 `recall at` 指标，统计目标样本落入 Top-K 的比例。
def _recall_at(ranks: list[int], k: int) -> float:
    return float(sum(1 for rank in ranks if int(rank) <= int(k)) / len(ranks)) if ranks else 0.0


# 计算平均排名，空输入时返回可控的默认结果。
def _mean_rank(ranks: list[int]) -> float:
    return float(mean([float(rank) for rank in ranks])) if ranks else 0.0


# 计算 `矩阵`，为指标、风险或调度决策提供数值依据。
def _score_matrix(adapter: Any, images: list[np.ndarray], texts: list[str]) -> np.ndarray:
    pairs = [(images[i], texts[j]) for i in range(len(images)) for j in range(len(texts))]
    scores = adapter.score_pairs(pairs, batch_size=8)
    return np.asarray(scores, dtype=np.float32).reshape(len(images), len(texts))


# 计算 `图文检索 stage 指标`，把原始模型输出汇总成页面和报告使用的指标字段。
def _vlr_stage_metrics(ir_ranks: list[int], tr_ranks: list[int]) -> dict[str, float]:
    return {
        "ir_r@1": _recall_at(ir_ranks, 1),
        "ir_r@5": _recall_at(ir_ranks, 5),
        "tr_r@1": _recall_at(tr_ranks, 1),
        "tr_r@5": _recall_at(tr_ranks, 5),
        "mean_rank_ir": _mean_rank(ir_ranks),
        "mean_rank_tr": _mean_rank(tr_ranks),
    }


# 计算 `ASR` 条件指标，只在满足前置条件的样本上统计。
def _conditional_asr(clean_ranks: list[int], attacked_ranks: list[int]) -> float:
    eligible = [idx for idx, rank in enumerate(clean_ranks) if int(rank) <= 1]
    if not eligible:
        return 0.0
    return float(sum(1 for idx in eligible if int(attacked_ranks[idx]) > 1) / len(eligible))


# 推断 `asset 任务`，从样本、配置或运行记录中提取统一名称。
def _asset_task(entries: list[dict[str, Any]]) -> str:
    tasks = sorted({_text(entry.get("task_kind")) for entry in entries if _text(entry.get("task_kind"))})
    return tasks[0] if len(tasks) == 1 else "asset_mixed"


# 推断 `数据集 名称`，从样本、配置或运行记录中提取统一名称。
def _dataset_name(entries: list[dict[str, Any]]) -> str:
    datasets = sorted({_text(entry.get("benchmark_tag") or entry.get("dataset_name")) for entry in entries if _text(entry.get("benchmark_tag") or entry.get("dataset_name"))})
    return datasets[0] if len(datasets) == 1 else "sample_asset_library"


# 推断 `攻击 名称`，从样本、配置或运行记录中提取统一名称。
def _attack_name(entries: list[dict[str, Any]]) -> str:
    attacks = sorted({_text(entry.get("attack")) for entry in entries if _text(entry.get("attack"))})
    return attacks[0] if len(attacks) == 1 else "asset_mixed"


# 读取 `行记录` 来源数据，缺失或格式异常时返回空结构。
def _source_row(entry: dict[str, Any]) -> dict[str, Any]:
    bundle = _record(entry.get("source_bundle"))
    sample = _record(bundle.get("sample"))
    sample_meta = _record(sample.get("metadata"))
    metrics = _record(bundle.get("metrics"))
    row = dict(sample_meta)
    row.setdefault("question", sample_meta.get("question") or sample.get("text") or entry.get("text"))
    row.setdefault("answer", sample_meta.get("answer") or sample.get("target_text") or metrics.get("answer") or entry.get("target_text"))
    row.setdefault("answer_aliases", sample_meta.get("answer_aliases") or metrics.get("answer_aliases") or [])
    row.setdefault("target_object", sample_meta.get("target_object") or sample.get("target_text") or metrics.get("target_object") or entry.get("target_text"))
    row.setdefault("target_aliases", sample_meta.get("target_aliases") or metrics.get("target_aliases") or [])
    row.setdefault("non_target_objects", sample_meta.get("non_target_objects") or metrics.get("non_target_objects") or [])
    row.setdefault("attack_goal", sample_meta.get("attack_goal") or metrics.get("attack_goal") or "remove_object")
    return row


# 构建 `样本` 数据，集中整理后端业务服务需要的输出结构。
def _make_sample(entry: dict[str, Any], *, image: np.ndarray, stage: str) -> Sample:
    bundle = _record(entry.get("source_bundle"))
    sample = _record(bundle.get("sample"))
    metadata = dict(_record(sample.get("metadata")))
    metadata.update(
        {
            "source_asset_id": entry.get("asset_id"),
            "source_run_id": entry.get("source_run_id"),
            "source_case_id": entry.get("source_case_id"),
            "asset_evaluation_stage": stage,
        }
    )
    return Sample(
        sample_id=_text(entry.get("case_id")),
        image=np.asarray(image, dtype=np.float32),
        text=_text(entry.get("text") or sample.get("text")),
        target_text=_text(entry.get("target_text") or sample.get("target_text")),
        metadata=metadata,
    )


# 计算 `视觉问答 指标`，把原始模型输出汇总成页面和报告使用的指标字段。
def _vqa_metrics(row: dict[str, Any], clean: ModelOutput, attacked: ModelOutput) -> dict[str, Any]:
    aliases = _list(row.get("answer_aliases") or row.get("answers") or row.get("acceptable_answers"))
    answer = _text(row.get("answer") or row.get("ground_truth") or row.get("label") or (aliases[0] if aliases else ""))
    clean_correct = answer_matches(clean.text, answer, aliases)
    attacked_correct = answer_matches(attacked.text, answer, aliases)
    attack_success = bool(clean_correct and not attacked_correct)
    return {
        "answer": answer,
        "answer_aliases": aliases,
        "clean_correct": clean_correct,
        "attacked_correct": attacked_correct,
        "defended_correct": False,
        "answer_changed": normalize_answer(clean.text) != normalize_answer(attacked.text),
        "attack_success": attack_success,
        "defense_recovered": False,
    }


# 计算 `图像描述 指标`，把原始模型输出汇总成页面和报告使用的指标字段。
def _caption_metrics(row: dict[str, Any], clean_sample: Sample, attacked_sample: Sample, clean: ModelOutput, attacked: ModelOutput) -> dict[str, Any]:
    target = _text(row.get("target_object") or row.get("added_object") or clean_sample.target_text)
    aliases = _list(row.get("target_aliases"))
    non_targets = _list(row.get("non_target_objects"))
    goal = _text(row.get("attack_goal") or row.get("goal") or "remove_object").lower()
    clean_present = object_present(clean.text, target, aliases) if target else False
    attacked_present = object_present(attacked.text, target, aliases) if target else False
    if goal in {"add", "add_object", "insert_object"}:
        attack_success = bool(not clean_present and attacked_present)
    else:
        attack_success = bool(clean_present and not attacked_present)
    clean_non_target = [obj for obj in non_targets if object_present(clean.text, obj, [])]
    attacked_non_target = [obj for obj in non_targets if object_present(attacked.text, obj, [])]
    clean_semantic = clean_non_target + ([target] if clean_present else [])
    attacked_semantic = attacked_non_target + ([target] if attacked_present else [])
    return {
        "target_object": target,
        "target_aliases": aliases,
        "attack_goal": goal,
        "target_present_clean": clean_present,
        "target_present_attacked": attacked_present,
        "target_present_defended": False,
        "non_target_objects": non_targets,
        "clean_non_target_present": clean_non_target,
        "attacked_non_target_present": attacked_non_target,
        "defended_non_target_present": [],
        "semantic_preservation_rate": text_similarity(clean.text, attacked.text),
        "object_jaccard": object_jaccard(clean_semantic, attacked_semantic),
        "caption_text_similarity": text_similarity(clean.text, attacked.text),
        "attack_success": attack_success,
        "defense_recovered": False,
    }


# 组装 `风险 载荷 所属 任务`，把分散字段整理成后端任务或风险评分使用的载荷。
def _risk_payload_for_task(task_kind: str, *, asr: float, semantic: float, avg_l2: float, avg_linf: float, transfer: float = 0.0) -> dict[str, Any]:
    scenario = "retrieval" if task_kind == "vlr" else "qa" if task_kind == "vqa" else "caption"
    return compute_risk_score(
        scenario=scenario,
        components={
            "effectiveness": float(asr),
            "semantic": float(semantic),
            "cost": 0.5 * (normalize_inverse(avg_l2, 25.0) + normalize_inverse(avg_linf, 0.2)),
            "transfer": float(transfer),
            "stability": float(asr),
        },
        weights={},
    )


# 评估 `图文检索` 结果，汇总攻击前后指标和风险证据。
def _evaluate_vlr(entries: list[dict[str, Any]], victim_adapters: list[str], progress: Callable[[str, str, float | None, str], None]) -> dict[str, Any]:
    texts = [_text(entry.get("text")) for entry in entries]
    clean_images = [np.asarray(entry["clean_image"], dtype=np.float32) for entry in entries]
    adv_images = [np.asarray(entry["adv_image"], dtype=np.float32) for entry in entries]
    victim_metrics: dict[str, Any] = {}
    victim_compare: list[dict[str, Any]] = []
    per_case: dict[str, dict[str, Any]] = {str(entry["case_id"]): {} for entry in entries}
    victim_asrs: list[float] = []

    for victim_name in victim_adapters:
        progress("victim_evaluation", "running", 58, f"正在用受测模型 {victim_name} 重新评分样本集。")
        adapter = create("model_adapter", victim_name)
        clean_matrix = _score_matrix(adapter, clean_images, texts)
        adv_matrix = _score_matrix(adapter, adv_images, texts)
        n = len(entries)
        clean_ir = [_rank_desc(clean_matrix[:, idx], idx) for idx in range(n)]
        adv_ir = [_rank_desc(adv_matrix[:, idx], idx) for idx in range(n)]
        clean_tr = [_rank_desc(clean_matrix[idx, :], idx) for idx in range(n)]
        adv_tr = [_rank_desc(adv_matrix[idx, :], idx) for idx in range(n)]
        clean_metrics = _vlr_stage_metrics(clean_ir, clean_tr)
        adv_metrics = _vlr_stage_metrics(adv_ir, adv_tr)
        cond = {
            "ir_cond_asr@1": _conditional_asr(clean_ir, adv_ir),
            "tr_cond_asr@1": _conditional_asr(clean_tr, adv_tr),
        }
        victim_asr = float(mean([cond["ir_cond_asr@1"], cond["tr_cond_asr@1"]]))
        victim_asrs.append(victim_asr)
        victim_metrics[victim_name] = {"clean": clean_metrics, "attacked": adv_metrics, "conditional": cond, "status": {"state": "success", "note": "asset images retested"}}
        victim_compare.append(
            {
                "victim": victim_name,
                "status": victim_metrics[victim_name]["status"],
                "clean": clean_metrics,
                "attacked": adv_metrics,
                "conditional": cond,
                "delta_mean_rank_ir": float(adv_metrics["mean_rank_ir"] - clean_metrics["mean_rank_ir"]),
                "delta_mean_rank_tr": float(adv_metrics["mean_rank_tr"] - clean_metrics["mean_rank_tr"]),
            }
        )
        for idx, entry in enumerate(entries):
            case = per_case[str(entry["case_id"])]
            case.setdefault("victim_scores", {})[victim_name] = {
                "clean_score": float(clean_matrix[idx, idx]),
                "adv_score": float(adv_matrix[idx, idx]),
                "clean_ir_rank": clean_ir[idx],
                "adv_ir_rank": adv_ir[idx],
                "clean_tr_rank": clean_tr[idx],
                "adv_tr_rank": adv_tr[idx],
            }
            if "primary_victim" not in case:
                ir_success = clean_ir[idx] <= 1 and adv_ir[idx] > 1
                tr_success = clean_tr[idx] <= 1 and adv_tr[idx] > 1
                case.update(
                    {
                        "primary_victim": victim_name,
                        "judge_success": bool(ir_success or tr_success),
                        "judge_reason": "asset_retest_top1_drop" if bool(ir_success or tr_success) else "asset_retest_top1_not_dropped",
                        "clean_score": float(clean_matrix[idx, idx]),
                        "adv_score": float(adv_matrix[idx, idx]),
                        "clean_ir_rank": clean_ir[idx],
                        "adv_ir_rank": adv_ir[idx],
                        "clean_tr_rank": clean_tr[idx],
                        "adv_tr_rank": adv_tr[idx],
                    }
                )

    asr_attack = float(mean(victim_asrs)) if victim_asrs else 0.0
    return {"asr_attack": asr_attack, "victims": victim_metrics, "victim_compare": victim_compare, "per_case": per_case}


# 评估 `生成式评测` 结果，汇总攻击前后指标和风险证据。
def _evaluate_generation(entries: list[dict[str, Any]], victim_adapters: list[str], task_kind: str, progress: Callable[[str, str, float | None, str], None]) -> dict[str, Any]:
    victim_name = victim_adapters[0]
    progress("victim_evaluation", "running", 58, f"正在用受测模型 {victim_name} 重新生成 clean/adv 输出。")
    model = create("model_adapter", victim_name)
    per_case: dict[str, dict[str, Any]] = {}
    success_values: list[float] = []
    semantic_values: list[float] = []
    clean_correct_values: list[float] = []
    attacked_correct_values: list[float] = []
    answer_changed_values: list[float] = []
    caption_similarity_values: list[float] = []
    object_jaccard_values: list[float] = []

    for entry in entries:
        row = _source_row(entry)
        clean_sample = _make_sample(entry, image=entry["clean_image"], stage="clean")
        adv_sample = _make_sample(entry, image=entry["adv_image"], stage="attacked")
        if task_kind == "vqa":
            question = _text(row.get("question") or clean_sample.text)
            clean_out = model.generate_answer(clean_sample, question, max_tokens=64)
            adv_out = model.generate_answer(adv_sample, question, max_tokens=64)
            metrics = _vqa_metrics(row, clean_out, adv_out)
            semantic = 1.0 - (1.0 if metrics["answer_changed"] else 0.0)
            clean_correct_values.append(1.0 if metrics["clean_correct"] else 0.0)
            attacked_correct_values.append(1.0 if metrics["attacked_correct"] else 0.0)
            answer_changed_values.append(1.0 if metrics["answer_changed"] else 0.0)
        else:
            clean_out = model.generate_caption(clean_sample, max_tokens=96)
            adv_out = model.generate_caption(adv_sample, max_tokens=96)
            metrics = _caption_metrics(row, clean_sample, adv_sample, clean_out, adv_out)
            semantic = float(metrics.get("semantic_preservation_rate", 0.0) or 0.0)
            caption_similarity_values.append(float(metrics.get("caption_text_similarity", 0.0) or 0.0))
            object_jaccard_values.append(float(metrics.get("object_jaccard", 0.0) or 0.0))
        success_values.append(1.0 if metrics.get("attack_success") else 0.0)
        semantic_values.append(float(semantic))
        per_case[str(entry["case_id"])] = {
            "primary_victim": victim_name,
            "judge_success": bool(metrics.get("attack_success")),
            "judge_reason": "asset_retest_generation_success" if metrics.get("attack_success") else "asset_retest_generation_not_successful",
            "outputs": {"clean": clean_out, "adv": adv_out},
            "metrics": metrics,
        }

    asr_attack = float(mean(success_values)) if success_values else 0.0
    generation_metrics: dict[str, float] = {}
    if task_kind == "vqa":
        generation_metrics = {
            "clean_accuracy": float(mean(clean_correct_values)) if clean_correct_values else 0.0,
            "attacked_accuracy": float(mean(attacked_correct_values)) if attacked_correct_values else 0.0,
            "answer_change_rate": float(mean(answer_changed_values)) if answer_changed_values else 0.0,
            "target_flip_rate": asr_attack,
            "semantic_preservation_rate": float(mean(semantic_values)) if semantic_values else 0.0,
        }
    else:
        generation_metrics = {
            "target_flip_rate": asr_attack,
            "semantic_preservation_rate": float(mean(semantic_values)) if semantic_values else 0.0,
            "caption_text_similarity": float(mean(caption_similarity_values)) if caption_similarity_values else 0.0,
            "object_jaccard": float(mean(object_jaccard_values)) if object_jaccard_values else 0.0,
        }
    return {"asr_attack": asr_attack, "per_case": per_case, "generation_metrics": generation_metrics, "semantic": float(mean(semantic_values)) if semantic_values else 0.0}


# 准备 `entries` 数据，补齐后续运行、报告或测试需要的字段。
def _prepare_entries(project_root: Path, cases_root: Path, assets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    used_case_ids: set[str] = set()
    for idx, asset in enumerate(assets, start=1):
        case_id = _safe_case_id(str(asset.get("sample_id") or f"asset-{idx}"), idx, used_case_ids)
        case_dir = cases_root / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        clean_src = _resolve_ref(project_root, _text(asset.get("clean_image_ref")))
        adv_src = _resolve_ref(project_root, _text(asset.get("adv_image_ref")))
        if clean_src is None or adv_src is None:
            raise ValueError(f"样本资产 {asset.get('asset_id')} 缺少原始图像或对抗图像，无法进行真实复测。")
        clean_ref = _copy_ref(project_root, _text(asset.get("clean_image_ref")), case_dir / "clean.png")
        adv_ref = _copy_ref(project_root, _text(asset.get("adv_image_ref")), case_dir / "adv.png")
        source_bundle = _source_bundle(project_root, asset)
        sample = _record(source_bundle.get("sample"))
        clean_input = _record(_record(source_bundle.get("inputs")).get("clean"))
        entries.append(
            {
                "asset": asset,
                "asset_id": asset.get("asset_id"),
                "case_id": case_id,
                "case_dir": case_dir,
                "source_bundle": source_bundle,
                "source_run_id": asset.get("run_id"),
                "source_case_id": asset.get("sample_id"),
                "task_kind": asset.get("task_kind") or source_bundle.get("task_kind") or "vlr",
                "dataset_name": asset.get("dataset_name"),
                "benchmark_tag": asset.get("benchmark_tag") or asset.get("dataset_name"),
                "attack": asset.get("attack"),
                "attack_scope": asset.get("attack_scope"),
                "text": asset.get("source_text") or sample.get("text") or clean_input.get("text"),
                "target_text": asset.get("target_text") or sample.get("target_text"),
                "clean_ref": clean_ref,
                "adv_ref": adv_ref,
                "clean_image": _load_image(clean_src),
                "adv_image": _load_image(adv_src),
                "l2": _num(asset.get("perturbation_l2")),
                "linf": _num(asset.get("perturbation_linf")),
                "source_risk": _num(asset.get("risk_score")),
            }
        )
    return entries


# 执行 `asset evaluation` 流程，按配置驱动后端业务服务完成一次任务。
def run_asset_evaluation(
    *,
    store: SQLiteStore,
    artifacts_dir: str,
    override: dict[str, Any],
    job_id: str,
    log: Callable[[str, str], None],
    progress: Callable[[str, str, float | None, str], None],
) -> dict[str, Any]:
    extra = _record(override.get("extra"))
    runner = _record(override.get("runner"))
    plugins = _record(override.get("plugins"))
    asset_ids = [str(item).strip() for item in extra.get("sample_asset_ids", []) if str(item).strip()] if isinstance(extra.get("sample_asset_ids"), list) else []
    asset_batch_id = _text(extra.get("sample_asset_batch_id"))
    if not asset_ids:
        raise ValueError("样本集复测需要至少调用 1 个证据完整样本。")
    limit = int(_num(runner.get("max_samples"), len(asset_ids)))
    selected_asset_ids = asset_ids if limit <= 0 else asset_ids[: max(1, limit)]
    assets = store.get_sample_assets(selected_asset_ids)
    if not assets:
        raise ValueError("样本集复测没有读取到可用的证据完整样本。")

    register_builtin_plugins()
    project_root = Path(__file__).resolve().parents[3]
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_root = project_root / artifacts_dir / "runs" / run_id
    cases_root = run_root / "cases"
    run_root.mkdir(parents=True, exist_ok=True)
    cases_root.mkdir(parents=True, exist_ok=True)

    progress("config_validation", "running", 18, "正在校验样本集真实复测配置。")
    victim_adapters = [str(item) for item in runner.get("victim_model_adapters", []) if str(item).strip()] if isinstance(runner.get("victim_model_adapters"), list) else []
    if not victim_adapters:
        victim_adapters = [_text(plugins.get("model_adapter")) or _text(assets[0].get("model_adapter"))]
    victim_adapters = [item for item in victim_adapters if item]
    if not victim_adapters:
        raise ValueError("样本集复测缺少受测模型配置。")
    progress("config_validation", "success", 26, "样本集真实复测配置校验完成。")
    progress("dataset_loading", "running", 38, f"正在装载 {len(assets)} 个已生成对抗样本。")
    entries = _prepare_entries(project_root, cases_root, assets)

    task_kind = _asset_task(entries)
    if task_kind == "asset_mixed":
        raise ValueError("一次样本集复测只能包含同一任务类型，请在对抗样本库按任务筛选后再调用。")

    if task_kind == "vlr":
        eval_result = _evaluate_vlr(entries, victim_adapters, progress)
        semantic_score = 1.0
        generation_metrics: dict[str, Any] = {}
        victims = eval_result["victims"]
        victim_compare = eval_result["victim_compare"]
    elif task_kind in {"vqa", "caption"}:
        eval_result = _evaluate_generation(entries, victim_adapters, task_kind, progress)
        semantic_score = float(eval_result.get("semantic", 0.0) or 0.0)
        generation_metrics = dict(eval_result.get("generation_metrics", {}))
        victims = {}
        victim_compare = []
    else:
        raise ValueError(f"当前样本集任务类型 {task_kind} 暂不支持真实复测。")

    progress("result_aggregation", "running", 88, "正在按真实复测输出汇总样本级指标。")
    per_case = _record(eval_result.get("per_case"))
    now_iso = utc_now_iso()
    case_rows: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    promoted_assets: list[dict[str, Any]] = []
    l2_values = [float(entry["l2"]) for entry in entries]
    linf_values = [float(entry["linf"]) for entry in entries]

    for entry in entries:
        case_eval = _record(per_case.get(str(entry["case_id"])))
        case_dir = Path(entry["case_dir"])
        source_bundle = _record(entry.get("source_bundle"))
        sample = dict(_record(source_bundle.get("sample")))
        adversarial = dict(_record(source_bundle.get("adversarial")))
        sample.setdefault("sample_id", entry["case_id"])
        sample.setdefault("text", entry.get("text", ""))
        sample_meta = dict(_record(sample.get("metadata")))
        sample_meta.update({"source_asset_id": entry.get("asset_id"), "source_run_id": entry.get("source_run_id"), "source_case_id": entry.get("source_case_id"), "asset_evaluation_run_id": run_id})
        sample["metadata"] = sample_meta
        adversarial.setdefault("sample_id", entry["case_id"])
        adversarial.setdefault("text", entry.get("text", ""))
        adv_meta = dict(_record(adversarial.get("metadata")))
        adv_meta.update({"source_asset_id": entry.get("asset_id"), "source_run_id": entry.get("source_run_id"), "source_case_id": entry.get("source_case_id"), "attack": entry.get("attack", ""), "attack_name": entry.get("attack", ""), "attack_scope": entry.get("attack_scope", ""), "workflow_type": "asset_evaluation_retest"})
        adversarial["metadata"] = adv_meta

        if task_kind == "vlr":
            clean_score = _num(case_eval.get("clean_score"))
            adv_score = _num(case_eval.get("adv_score"))
            outputs = {
                "clean": {"text": f"受测模型重新评分={clean_score:.4f}", "score": clean_score},
                "adv": {"text": f"受测模型重新评分={adv_score:.4f}", "score": adv_score},
            }
            metrics = {
                "asset_retest": True,
                "score_drop": float(clean_score - adv_score),
                "clean_ir_rank": int(case_eval.get("clean_ir_rank", 0) or 0),
                "adv_ir_rank": int(case_eval.get("adv_ir_rank", 0) or 0),
                "clean_tr_rank": int(case_eval.get("clean_tr_rank", 0) or 0),
                "adv_tr_rank": int(case_eval.get("adv_tr_rank", 0) or 0),
                "perturbation_l2": float(entry["l2"]),
                "perturbation_linf": float(entry["linf"]),
            }
        else:
            out = _record(case_eval.get("outputs"))
            clean_out = out.get("clean")
            adv_out = out.get("adv")
            outputs = {
                "clean": {"text": getattr(clean_out, "text", ""), "score": getattr(clean_out, "score", None)},
                "adv": {"text": getattr(adv_out, "text", ""), "score": getattr(adv_out, "score", None)},
            }
            metrics = {**_record(case_eval.get("metrics")), "asset_retest": True, "perturbation_l2": float(entry["l2"]), "perturbation_linf": float(entry["linf"])}

        judge_success = bool(case_eval.get("judge_success"))
        bundle = {
            "task_kind": task_kind,
            "sample": sample,
            "adversarial": adversarial,
            "defended": {"sample_id": "", "text": "", "metadata": {}},
            "inputs": _record(source_bundle.get("inputs")) or {"clean": {"text": entry.get("text", "")}, "adv": {"text": entry.get("text", "")}, "defended": {"text": ""}},
            "dataset_tag": entry.get("benchmark_tag") or entry.get("dataset_name") or "",
            "model_tag": case_eval.get("primary_victim") or victim_adapters[0],
            "outputs": outputs,
            "metrics": metrics,
            "judge": {"success": judge_success, "reason": case_eval.get("judge_reason") or "asset_retest"},
            "diagnostics": {"source_asset_id": entry.get("asset_id"), "source_run_id": entry.get("source_run_id"), "source_case_id": entry.get("source_case_id"), "asset_reuse_evaluation": True, "asset_retest": True},
            "artifact_refs": {"clean_image": entry["clean_ref"], "adv_image": entry["adv_ref"]},
            "artifact_capability": {
                "clean_image": {"status": "available", "reason": "来自样本集，并已用于本次复测"},
                "adv_image": {"status": "available", "reason": "来自样本集，并已用于本次复测"},
            },
            "visual_labels": {"clean": "原始图像", "adv": "对抗图像"},
            "asset_lineage": {"asset_id": entry.get("asset_id"), "source_run_id": entry.get("source_run_id"), "source_case_id": entry.get("source_case_id"), "source_report_url": f"/reports/{entry.get('source_run_id')}", "source_case_url": f"/reports/{entry.get('source_run_id')}/cases/{entry.get('source_case_id')}"},
        }
        _write_json(case_dir / "case_bundle.json", bundle)
        row = {
            "sample_id": entry["case_id"],
            "source_sample_id": entry.get("source_case_id"),
            "source_asset_id": entry.get("asset_id"),
            "source_run_id": entry.get("source_run_id"),
            "task_kind": task_kind,
            "dataset_name": entry.get("dataset_name"),
            "benchmark_tag": entry.get("benchmark_tag"),
            "model_adapter": case_eval.get("primary_victim") or victim_adapters[0],
            "attack": entry.get("attack"),
            "artifact_status": "complete",
            "judge_success": judge_success,
            "risk_level": _risk_level(1.0 if judge_success else 0.0),
            "risk_score": 1.0 if judge_success else 0.0,
            "perturbation_l2": float(entry["l2"]),
            "perturbation_linf": float(entry["linf"]),
            "created_at": now_iso,
        }
        case_rows.append(row)
        rows.append({**row, "scope": "asset_retest", "asset_reuse": True, "asset_retest": True})
        source_asset = _record(entry.get("asset"))
        if _text(source_asset.get("reusable_status")) == "pending_evaluation":
            source_meta = _record(source_asset.get("metadata"))
            source_meta.update({"promoted_by_evaluation_run_id": run_id, "promoted_model_adapter": row["model_adapter"]})
            promoted_assets.append(
                {
                    **source_asset,
                    "asset_id": source_asset.get("asset_id") or entry.get("asset_id"),
                    "source_run_id": source_asset.get("source_run_id") or entry.get("source_run_id"),
                    "source_case_id": source_asset.get("source_case_id") or entry.get("source_case_id"),
                    "model_adapter": row["model_adapter"],
                    "artifact_status": "complete",
                    "reusable_status": "ready",
                    "reusable_note": "已完成受测模型测评，可纳入正式调用。",
                    "judge_success": judge_success,
                    "risk_level": row["risk_level"],
                    "risk_score": row["risk_score"],
                    "perturbation_l2": row["perturbation_l2"],
                    "perturbation_linf": row["perturbation_linf"],
                    "metadata": source_meta,
                }
            )

    asr = float(eval_result.get("asr_attack", 0.0) or 0.0)
    avg_l2 = float(mean(l2_values)) if l2_values else 0.0
    avg_linf = float(mean(linf_values)) if linf_values else 0.0
    risk_payload = _risk_payload_for_task(task_kind, asr=asr, semantic=semantic_score, avg_l2=avg_l2, avg_linf=avg_linf, transfer=0.0)
    summary = {
        "run_id": run_id,
        "task_kind": task_kind,
        "dataset_name": _dataset_name(entries),
        "benchmark_tag": _dataset_name(entries),
        "eval_scope": "asset_retest",
        "num_images": len(entries),
        "num_texts": len(entries),
        "num_effective": len(entries),
        "num_samples": len(entries),
        "sample_pair_count": len(entries),
        "attack": _attack_name(entries),
        "attack_mode": "asset_fixed_retest",
        "defense": "",
        "defense_enabled": False,
        "model_adapter": _text(plugins.get("model_adapter")) or victim_adapters[0],
        "surrogate_model_adapter": _text(runner.get("surrogate_model_adapter")) or "asset_library",
        "victim_model_adapters": victim_adapters,
        "asr": asr,
        "asr_attack": asr,
        "conditional_asr_attack": asr,
        "asr_definition": "asset_retest_clean_top1_drop" if task_kind == "vlr" else "asset_retest_output_success",
        "asr_defended": asr,
        "defense_gain": 0.0,
        "avg_l2": avg_l2,
        "avg_linf": avg_linf,
        "generation_metrics": generation_metrics,
        "victims": victims,
        "victim_compare": victim_compare,
        **risk_payload,
        "risk": risk_payload,
        "risk_scenario": "asset_retest",
        "asset_evaluation_mode": True,
        "asset_retest_mode": True,
        "asset_workflow": {
            "type": "asset_evaluation",
            "execution": "retest_existing_adversarial_images",
            "batch_id": asset_batch_id,
            "asset_count": len(entries),
            "source_asset_ids": [entry.get("asset_id") for entry in entries],
            "source_run_ids": sorted({_text(entry.get("source_run_id")) for entry in entries if _text(entry.get("source_run_id"))}),
            "note": "本次调用对抗样本库中的原始图像与对抗图像，并使用当前受测模型重新评测 clean/adv 输入。",
        },
        "risk_recommendations": list(risk_payload.get("risk_recommendations", [])),
    }
    report_data = {
        "summary": summary,
        "rows_preview": rows,
        "metric_series": {"l2": l2_values, "linf": linf_values},
        "mode_stats": {f"{summary['attack']}:asset_fixed_retest": {"count": len(entries), "asr": asr}},
        "stage_metrics": {name: payload for name, payload in victims.items()},
        "vlr": {"victim_compare": victim_compare, "failure_cases": rows} if task_kind == "vlr" else {},
        "generation_metrics": generation_metrics,
        "asset_lineage": [
            {"asset_id": entry.get("asset_id"), "source_run_id": entry.get("source_run_id"), "source_case_id": entry.get("source_case_id"), "attack": entry.get("attack"), "dataset": entry.get("benchmark_tag") or entry.get("dataset_name")}
            for entry in entries
        ],
    }
    progress("report_writing", "running", 97, "正在写入样本集真实复测报告。")
    _write_json(run_root / "summary.json", summary)
    _write_json(run_root / "report_data.json", report_data)
    _write_jsonl(run_root / "cases_index.jsonl", case_rows)
    _write_jsonl(run_root / "results.jsonl", rows)
    html = "<html><body><h2>对抗样本集真实复测</h2><pre>" + json.dumps(summary, ensure_ascii=False, indent=2) + "</pre></body></html>"
    (run_root / "report.html").write_text(html, encoding="utf-8")
    store.upsert_run_cache(summary, str(run_root))
    if promoted_assets:
        store.upsert_sample_assets(promoted_assets)
    store.record_sample_asset_usage(asset_ids=[str(entry.get("asset_id")) for entry in entries], evaluation_run_id=run_id, job_id=job_id, note="asset_retest")
    log("info", f"asset retest success: run_id={run_id} assets={len(entries)} asr={asr:.4f}")
    return {"run_id": run_id, "summary_path": str(run_root / "summary.json"), "results_path": str(run_root / "results.jsonl"), "report_path": str(run_root / "report.html")}
