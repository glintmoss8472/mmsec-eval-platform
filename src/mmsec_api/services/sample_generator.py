from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any, Callable

import numpy as np

from mmsec_api.services.model_runtime import ensure_models_ready
from mmsec_api.store.sqlite import SQLiteStore
from mmsec_api.utils import utc_now_iso
from mmsec_eval.attacks.catalog import attack_surrogate_error
from mmsec_eval.config.loader import load_config
from mmsec_eval.config.sweep import apply_override
from mmsec_eval.config.validate import validate_config
from mmsec_eval.datasets.registry import load_dataset
from mmsec_eval.logging import setup_logging
from mmsec_eval.plugins.builtin import register_builtin_plugins
from mmsec_eval.plugins.registry import create
from mmsec_eval.runtime import apply_config_env
from mmsec_eval.runner.artifacts import make_run_dir, new_run_id, write_env_snapshot, write_json_snapshot
from mmsec_eval.runner.generation_runner import _case_image_path, _case_id, _load_generation_rows, _load_image, _sample_from_case, _stage_sample, _trace_debug_payload
from mmsec_eval.sample_store.serializer import save_image_png, write_json, write_jsonl
from mmsec_eval.types import AttackContext, Sample
from mmsec_eval.utils.seed import set_seed


PENDING_REUSABLE_STATUS = "pending_evaluation"


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    return value


def _safe_case_id(value: object, idx: int) -> str:
    text = str(value or "").strip() or f"sample-{idx:04d}"
    text = re.sub(r"[^0-9A-Za-z_.:-]+", "_", text).strip("._-")
    return (text or f"sample-{idx:04d}")[:120]


def _sample_with_id(sample: Sample, sample_id: str) -> Sample:
    metadata = dict(sample.metadata or {})
    metadata["source_sample_id"] = str(sample.sample_id)
    return Sample(sample_id=sample_id, image=np.asarray(sample.image, dtype=np.float32), text=sample.text, target_text=sample.target_text, metadata=metadata)


def _dataset_label(cfg: Any) -> str:
    return str(getattr(cfg.dataset, "benchmark_tag", "") or getattr(cfg.dataset, "kind", "") or "sample_generation")


def _asset_scope(task_kind: str, attack: str, eval_scope: str) -> str:
    attack_key = str(attack or "").lower()
    if str(eval_scope or "").lower() == "joint" or attack_key in {"tmm", "advedm_plus"}:
        return "图文联合扰动"
    if str(task_kind) in {"vqa", "caption"}:
        return "图像扰动"
    if "text" in attack_key:
        return "文本扰动"
    return "图像扰动"


def _trace_rows(trace: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in trace or []:
        if hasattr(item, "__dataclass_fields__"):
            rows.append(_json_ready(asdict(item)))
        elif isinstance(item, dict):
            rows.append(_json_ready(item))
    return rows


def _load_clean_samples(cfg: Any, progress: Callable[[str, str, float | None, str], None] | None) -> list[tuple[dict[str, Any], Sample]]:
    task_kind = str(getattr(cfg.task, "kind", "") or "").strip()
    if task_kind in {"vqa", "caption"}:
        _, cases_dir, rows = _load_generation_rows(cfg, progress)
        out: list[tuple[dict[str, Any], Sample]] = []
        for idx, row in enumerate(rows, start=1):
            image_path = _case_image_path(row, cases_dir=cases_dir)
            sample = _sample_from_case(row, idx, image=_load_image(image_path), image_path=image_path, stage="clean")
            out.append((dict(row), sample))
        return out

    if progress is not None:
        progress("dataset_loading", "running", 38, "正在读取来源数据集并准备样本生成。")
    dataset = load_dataset(cfg)
    max_samples = int(getattr(cfg.runner, "max_samples", 0) or getattr(cfg.dataset, "max_items", 0) or 0)
    if max_samples > 0:
        dataset = dataset[:max_samples]
    if progress is not None:
        progress("dataset_loading", "success", 42, f"来源数据集读取完成，共纳入 {len(dataset)} 条样本。")
    return [({}, sample) for sample in dataset]


def _artifact_refs(case_dir: Path, debug_dir: Path, cfg: Any, clean_sample: Sample, attacked: Any, attacked_sample: Sample) -> dict[str, str]:
    refs = {
        "clean_image": save_image_png(str(case_dir / "clean.png"), clean_sample.image),
        "adv_image": save_image_png(str(case_dir / "adv.png"), attacked_sample.image),
    }
    trace = _trace_rows(list(getattr(attacked, "attack_trace", []) or []))
    if trace:
        refs["attack_trace"] = write_jsonl(str(case_dir / "attack_trace.jsonl"), trace)
    debug_dir.mkdir(parents=True, exist_ok=True)
    refs["attack_debug"] = write_json(str(debug_dir / "debug.json"), _trace_debug_payload({}, str(cfg.plugins.attack), list(getattr(attacked, "attack_trace", []) or [])))
    refs.update({str(k): str(v) for k, v in dict(getattr(attacked, "artifact_refs", {}) or {}).items() if str(v)})
    metadata = dict(getattr(attacked, "metadata", {}) or {})
    for src_key, ref_key in (("attention_debug_path", "attention_map"), ("mask_debug_path", "mask_map"), ("patch_preview", "patch_preview"), ("joint_debug_path", "joint_debug")):
        value = str(metadata.get(src_key) or "").strip()
        if value and not str(refs.get(ref_key) or "").strip():
            refs[ref_key] = value
    return refs


def _case_bundle(
    *,
    cfg: Any,
    row: dict[str, Any],
    clean_sample: Sample,
    attacked: Any,
    attacked_sample: Sample,
    refs: dict[str, str],
    dataset_tag: str,
) -> dict[str, Any]:
    task_kind = str(cfg.task.kind)
    attack = str(cfg.plugins.attack)
    perturbation_l2 = float(getattr(attacked, "perturbation_l2", 0.0) or 0.0)
    perturbation_linf = float(getattr(attacked, "perturbation_linf", 0.0) or 0.0)
    attack_meta = dict(getattr(attacked, "metadata", {}) or {})
    attack_meta.update({"attack": attack, "attack_name": attack, "workflow_type": "sample_generation_only", "requires_evaluation": True})
    sample_meta = dict(clean_sample.metadata or {})
    sample_meta.update({"source_row": row, "workflow_type": "sample_generation_only", "requires_evaluation": True})
    return {
        "task_kind": task_kind,
        "sample": {
            "sample_id": clean_sample.sample_id,
            "text": clean_sample.text,
            "target_text": clean_sample.target_text,
            "metadata": _json_ready(sample_meta),
        },
        "adversarial": {
            "sample_id": clean_sample.sample_id,
            "text": attacked_sample.text,
            "perturbation_l0": int(getattr(attacked, "perturbation_l0", 0) or 0),
            "perturbation_l2": perturbation_l2,
            "perturbation_linf": perturbation_linf,
            "metadata": _json_ready(attack_meta),
        },
        "defended": {"sample_id": "", "text": "", "metadata": {}},
        "inputs": {
            "clean": {"text": clean_sample.text},
            "adv": {"text": attacked_sample.text},
            "defended": {"text": ""},
        },
        "dataset_tag": dataset_tag,
        "model_tag": "",
        "outputs": {
            "clean": {"text": "", "score": 0.0, "note": "尚未选择受测模型。"},
            "adv": {"text": "", "score": 0.0, "note": "尚未选择受测模型。"},
        },
        "metrics": {
            "sample_generation_only": True,
            "requires_evaluation": True,
            "perturbation_l2": perturbation_l2,
            "perturbation_linf": perturbation_linf,
        },
        "judge": {"success": False, "reason": "尚未选择受测模型，等待测评。"},
        "diagnostics": {"sample_generation_only": True, "requires_evaluation": True},
        "artifact_refs": refs,
        "artifact_capability": {
            "clean_image": {"status": "available", "reason": "原始图像已保存。"},
            "adv_image": {"status": "available", "reason": "对抗图像已生成并保存。"},
            "model_outputs": {"status": "pending", "reason": "需要选择受测模型后生成。"},
        },
        "visual_labels": {"clean": "原始图像", "adv": "对抗图像"},
    }


def _prepare_generation_config(config_path: str, override: dict[str, Any], artifacts_dir: str) -> tuple[Any, str]:
    register_builtin_plugins()
    cfg = load_config(config_path)
    if override:
        cfg = apply_override(cfg, override)
    cfg.artifacts_dir = artifacts_dir
    cfg.defense.enabled = False
    surrogate_name = str(getattr(cfg.runner, "surrogate_model_adapter", "") or getattr(cfg.plugins, "model_adapter", "") or "clip_hf").strip()
    cfg.plugins.model_adapter = surrogate_name
    cfg.runner.surrogate_model_adapter = surrogate_name
    cfg.runner.victim_model_adapters = []
    apply_config_env(cfg)
    return cfg, surrogate_name


def _validate_generation_config(
    cfg: Any,
    surrogate_name: str,
    log: Callable[[str, str], None],
    progress: Callable[[str, str, float | None, str], None],
) -> None:
    progress("config_validation", "running", 18, "正在校验样本生成配置。")
    error = attack_surrogate_error(str(cfg.plugins.attack), surrogate_name)
    if error:
        raise ValueError(error)
    validate_config(cfg)
    progress("config_validation", "success", 26, "样本生成配置校验完成。")

    progress("model_preflight", "running", 12, "正在检查攻击生成所需代理模型。")
    ensure_models_ready([surrogate_name], project_root=Path(__file__).resolve().parents[3], log=log)
    progress("model_preflight", "success", 16, "代理模型检查完成。")


def _start_generation_run(cfg: Any) -> tuple[str, Path]:
    setup_logging(cfg.artifacts_dir)
    set_seed(int(getattr(cfg, "seed", 0) or 0))
    run_id = new_run_id()
    run_dir = Path(make_run_dir(cfg.artifacts_dir, run_id))
    write_json_snapshot(str(run_dir), "config_snapshot.json", asdict(cfg))
    write_env_snapshot(str(run_dir))
    return run_id, run_dir


def _generated_row_payload(
    *,
    run_id: str,
    sample_id: str,
    task_kind: str,
    dataset_kind: str,
    dataset_tag: str,
    attack: str,
    attack_scope: str,
    now_iso: str,
    clean_sample: Sample,
    perturbation_l2: float,
    perturbation_linf: float,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "sample_id": sample_id,
        "task_kind": task_kind,
        "dataset_name": dataset_kind,
        "benchmark_tag": dataset_tag,
        "model_adapter": "",
        "attack": attack,
        "attack_scope": attack_scope,
        "artifact_status": "generated_only",
        "judge_success": False,
        "risk_level": "",
        "risk_score": 0.0,
        "perturbation_l2": perturbation_l2,
        "perturbation_linf": perturbation_linf,
        "created_at": now_iso,
        "text": clean_sample.text,
        "target_text": clean_sample.target_text,
    }


def _pending_asset_payload(
    *,
    row_payload: dict[str, Any],
    run_id: str,
    sample_id: str,
    variant_id: str,
    refs: dict[str, str],
    source_text: str,
    target_text: str,
    surrogate_name: str,
) -> dict[str, Any]:
    return {
        "asset_id": f"{run_id}::{sample_id}",
        "variant_id": variant_id,
        "source_run_id": run_id,
        "source_case_id": sample_id,
        "task_kind": row_payload["task_kind"],
        "dataset_name": row_payload["dataset_name"],
        "benchmark_tag": row_payload["benchmark_tag"],
        "model_adapter": "",
        "attack": row_payload["attack"],
        "attack_scope": row_payload["attack_scope"],
        "source_text": source_text,
        "target_text": target_text,
        "clean_image_ref": refs.get("clean_image", ""),
        "adv_image_ref": refs.get("adv_image", ""),
        "artifact_status": "generated_only",
        "reusable_status": PENDING_REUSABLE_STATUS,
        "reusable_note": "已生成原始图像和对抗图像，等待选择受测模型完成测评。",
        "judge_success": False,
        "risk_level": "",
        "risk_score": 0.0,
        "perturbation_l2": row_payload["perturbation_l2"],
        "perturbation_linf": row_payload["perturbation_linf"],
        "semantic_score": 0.0,
        "created_at": row_payload["created_at"],
        "metadata": {
            "source_run_id": run_id,
            "source_case_id": sample_id,
            "source_case_url": "",
            "source_report_url": "",
            "requires_evaluation": True,
            "sample_generation_only": True,
            "surrogate_model_adapter": surrogate_name,
        },
    }


def _generate_pending_asset_case(
    *,
    cfg: Any,
    attack: Any,
    surrogate: Any,
    run_id: str,
    run_dir: Path,
    cases_root: Path,
    debug_root: Path,
    dataset_tag: str,
    task_kind: str,
    attack_scope: str,
    now_iso: str,
    row: dict[str, Any],
    original_sample: Sample,
    idx: int,
    surrogate_name: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], float, float]:
    sid = _safe_case_id(_case_id(row, idx) if row else original_sample.sample_id, idx)
    clean_sample = _sample_with_id(original_sample, sid)
    case_dir = cases_root / sid
    case_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = debug_root / sid
    ctx = AttackContext(config=cfg, model_adapter=surrogate, surrogate_model_adapter=surrogate, run_dir=str(run_dir), sample_debug_dir=str(debug_dir))
    attacked = attack.attack(clean_sample, ctx)
    attacked_sample = _stage_sample(attacked.sample, "attacked")
    refs = _artifact_refs(case_dir, debug_dir, cfg, clean_sample, attacked, attacked_sample)
    bundle = _case_bundle(cfg=cfg, row=row, clean_sample=clean_sample, attacked=attacked, attacked_sample=attacked_sample, refs=refs, dataset_tag=dataset_tag)
    refs["case_bundle"] = write_json(str(case_dir / "case_bundle.json"), bundle)
    l2 = float(getattr(attacked, "perturbation_l2", 0.0) or 0.0)
    linf = float(getattr(attacked, "perturbation_linf", 0.0) or 0.0)
    row_payload = _generated_row_payload(
        run_id=run_id,
        sample_id=sid,
        task_kind=task_kind,
        dataset_kind=str(cfg.dataset.kind),
        dataset_tag=dataset_tag,
        attack=str(cfg.plugins.attack),
        attack_scope=attack_scope,
        now_iso=now_iso,
        clean_sample=clean_sample,
        perturbation_l2=l2,
        perturbation_linf=linf,
    )
    asset = _pending_asset_payload(
        row_payload=row_payload,
        run_id=run_id,
        sample_id=sid,
        variant_id=f"{run_id}::{sid}::{cfg.plugins.attack}",
        refs=refs,
        source_text=clean_sample.text,
        target_text=clean_sample.target_text,
        surrogate_name=surrogate_name,
    )
    return row_payload, {**row_payload, "case_dir": str(case_dir)}, asset, l2, linf


def _generation_summary(
    *,
    cfg: Any,
    run_id: str,
    now_iso: str,
    task_kind: str,
    dataset_tag: str,
    attack_scope: str,
    surrogate_name: str,
    total: int,
    l2_values: list[float],
    linf_values: list[float],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "created_at": now_iso,
        "task_kind": task_kind,
        "dataset_name": str(cfg.dataset.kind),
        "benchmark_tag": dataset_tag,
        "eval_scope": str(cfg.task.eval_scope or "image"),
        "num_samples": total,
        "num_effective": total,
        "attack": str(cfg.plugins.attack),
        "attack_scope": attack_scope,
        "model_adapter": "",
        "surrogate_model_adapter": surrogate_name,
        "victim_model_adapters": [],
        "asr": 0.0,
        "asr_attack": 0.0,
        "risk_score": 0.0,
        "risk_level": "",
        "avg_l2": float(mean(l2_values)) if l2_values else 0.0,
        "avg_linf": float(mean(linf_values)) if linf_values else 0.0,
        "sample_generation_only": True,
        "requires_evaluation": True,
        "result_type": "generated_only",
        "result_type_note": "该批次只完成样本生成，尚未进行受测模型测评。",
        "risk_recommendations": ["选择受测模型完成测评后再判定风险。"],
    }


def _write_generation_outputs(
    *,
    run_dir: Path,
    summary: dict[str, Any],
    result_rows: list[dict[str, Any]],
    index_rows: list[dict[str, Any]],
    assets: list[dict[str, Any]],
    l2_values: list[float],
    linf_values: list[float],
) -> tuple[str, str, Path]:
    report_data = {
        "summary": summary,
        "rows_preview": result_rows[:50],
        "metric_series": {"l2": l2_values, "linf": linf_values},
        "generation_only": True,
        "asset_lineage": [{"asset_id": item["asset_id"], "source_run_id": summary["run_id"], "source_case_id": item["source_case_id"]} for item in assets],
    }
    summary_path = write_json(str(run_dir / "summary.json"), summary)
    write_json(str(run_dir / "report_data.json"), report_data)
    results_path = write_jsonl(str(run_dir / "results.jsonl"), result_rows)
    write_jsonl(str(run_dir / "cases_index.jsonl"), index_rows)
    html_path = run_dir / "report.html"
    html_path.write_text("<html><body><h2>对抗样本生成记录</h2><pre>" + json.dumps(summary, ensure_ascii=False, indent=2) + "</pre></body></html>", encoding="utf-8")
    return summary_path, results_path, html_path


def _generate_pending_assets(
    *,
    cfg: Any,
    attack: Any,
    surrogate: Any,
    run_id: str,
    run_dir: Path,
    clean_rows: list[tuple[dict[str, Any], Sample]],
    dataset_tag: str,
    task_kind: str,
    attack_scope: str,
    now_iso: str,
    surrogate_name: str,
    progress: Callable[[str, str, float | None, str], None],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[float], list[float]]:
    cases_root = run_dir / "cases"
    debug_root = run_dir / "attack_debug"
    assets: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []
    l2_values: list[float] = []
    linf_values: list[float] = []

    total = len(clean_rows)
    progress("attack_execution", "running", 45, f"正在生成对抗样本 0/{total}。")
    for idx, (row, original_sample) in enumerate(clean_rows, start=1):
        row_payload, index_row, asset, l2, linf = _generate_pending_asset_case(
            cfg=cfg,
            attack=attack,
            surrogate=surrogate,
            run_id=run_id,
            run_dir=run_dir,
            cases_root=cases_root,
            debug_root=debug_root,
            dataset_tag=dataset_tag,
            task_kind=task_kind,
            attack_scope=attack_scope,
            now_iso=now_iso,
            row=row,
            original_sample=original_sample,
            idx=idx,
            surrogate_name=surrogate_name,
        )
        l2_values.append(l2)
        linf_values.append(linf)
        result_rows.append({**row_payload, "requires_evaluation": True})
        index_rows.append(index_row)
        assets.append(asset)
        if idx == total or idx % 5 == 0:
            progress("attack_execution", "running", 45 + (idx / total) * 35, f"正在生成对抗样本 {idx}/{total}。")
    return assets, result_rows, index_rows, l2_values, linf_values


def run_sample_generation_only(
    *,
    config_path: str,
    override: dict[str, Any],
    artifacts_dir: str,
    store: SQLiteStore,
    log: Callable[[str, str], None],
    progress: Callable[[str, str, float | None, str], None],
) -> dict[str, Any]:
    cfg, surrogate_name = _prepare_generation_config(config_path, override, artifacts_dir)
    _validate_generation_config(cfg, surrogate_name, log, progress)
    run_id, run_dir = _start_generation_run(cfg)
    clean_rows = _load_clean_samples(cfg, progress)
    if not clean_rows:
        raise ValueError("来源数据集没有可生成的样本。")

    attack = create("attack", str(cfg.plugins.attack))
    surrogate = create("model_adapter", surrogate_name)
    dataset_tag = _dataset_label(cfg)
    task_kind = str(cfg.task.kind)
    attack_scope = _asset_scope(task_kind, str(cfg.plugins.attack), str(cfg.task.eval_scope or ""))
    now_iso = utc_now_iso()

    total = len(clean_rows)
    assets, result_rows, index_rows, l2_values, linf_values = _generate_pending_assets(
        cfg=cfg,
        attack=attack,
        surrogate=surrogate,
        run_id=run_id,
        run_dir=run_dir,
        clean_rows=clean_rows,
        dataset_tag=dataset_tag,
        task_kind=task_kind,
        attack_scope=attack_scope,
        now_iso=now_iso,
        surrogate_name=surrogate_name,
        progress=progress,
    )

    summary = _generation_summary(
        cfg=cfg,
        run_id=run_id,
        now_iso=now_iso,
        task_kind=task_kind,
        dataset_tag=dataset_tag,
        attack_scope=attack_scope,
        surrogate_name=surrogate_name,
        total=total,
        l2_values=l2_values,
        linf_values=linf_values,
    )
    progress("result_aggregation", "running", 88, "正在写入待测评样本资产。")
    summary_path, results_path, html_path = _write_generation_outputs(
        run_dir=run_dir,
        summary=summary,
        result_rows=result_rows,
        index_rows=index_rows,
        assets=assets,
        l2_values=l2_values,
        linf_values=linf_values,
    )
    written = store.upsert_sample_assets(assets)
    log("info", f"sample generation success: run_id={run_id} assets={written}")
    progress("completed", "success", 100, f"样本生成完成，已写入 {written} 条待测评资产。")
    return {"run_id": run_id, "summary_path": summary_path, "results_path": results_path, "report_path": str(html_path)}
