from __future__ import annotations

import logging
import json
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any, Callable

import numpy as np

from mmsec_eval.config.schema import AppConfig
from mmsec_eval.datasets.registry import load_dataset
from mmsec_eval.plugins.registry import create
from mmsec_eval.runner.artifacts import (
    make_run_dir,
    new_run_id,
    write_env_snapshot,
    write_json_snapshot,
    write_results,
    write_summary,
)
from mmsec_eval.runner.report import write_report
from mmsec_eval.risk.scoring import compute_risk_score, normalize_inverse
from mmsec_eval.sample_store.manager import SampleStoreManager
from mmsec_eval.types import AttackContext, AttackedSample, DefenseContext, EvalRecord, RunArtifacts
from mmsec_eval.utils.cot_trace import parse_cot_trace
from mmsec_eval.utils.seed import set_seed
from mmsec_eval.viz.plots import plot_asr_bar, plot_attack_comparison, plot_metric_curve, plot_stage_compare_bar

LOG = logging.getLogger(__name__)


def _emit_progress(progress: Callable[[str, str, float | None, str], None] | None, stage_key: str, state: str, progress_percent: float | None, message: str) -> None:
    if progress is not None:
        progress(stage_key, state, progress_percent, message)


def _safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def _text_diff_score(clean: str, adv: str) -> float:
    a = set((clean or "").lower().split())
    b = set((adv or "").lower().split())
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    similarity = inter / max(1, union)
    return float(1.0 - similarity)


def _embedding_shift(clean: np.ndarray | None, adv: np.ndarray | None) -> float:
    if clean is None or adv is None:
        return 0.0
    c = np.asarray(clean, dtype=np.float32).reshape(-1)
    a = np.asarray(adv, dtype=np.float32).reshape(-1)
    n = min(c.shape[0], a.shape[0])
    if n == 0:
        return 0.0
    return float(np.linalg.norm(c[:n] - a[:n], ord=2))


def _cot_shift(clean_trace: dict[str, Any], adv_trace: dict[str, Any]) -> float:
    c = str(clean_trace.get("final_action", "")).strip().lower()
    a = str(adv_trace.get("final_action", "")).strip().lower()
    if not c and not a:
        return 0.0
    return 0.0 if c == a else 1.0


def _mode_key(row: dict[str, Any]) -> str:
    name = str(row.get("attack_name") or row.get("attack") or "unknown")
    mode = str(row.get("attack_mode") or "A")
    return f"{name}:{mode}"


def _recovery_success(
    *,
    adv_text_diff: float,
    adv_emb_shift: float,
    def_text_diff: float,
    def_emb_shift: float,
) -> float:
    text_th = max(0.02, float(adv_text_diff) * 0.85)
    emb_th = max(0.02, float(adv_emb_shift) * 0.85)
    ok = (float(def_text_diff) <= text_th) and (float(def_emb_shift) <= emb_th)
    return 1.0 if ok else 0.0


def _build_mode_stats(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        key = _mode_key(row)
        grouped.setdefault(key, []).append(1.0 if bool(row.get("judge_success", False)) else 0.0)
    out: dict[str, dict[str, float]] = {}
    for key, vals in grouped.items():
        out[key] = {"count": float(len(vals)), "asr": float(_safe_mean(vals))}
    return out


def _model_tag(cfg: AppConfig) -> str:
    if cfg.plugins.model_adapter == "clip_hf":
        return f"clip_hf:{cfg.model.clip_model_name}"
    if cfg.plugins.model_adapter == "http":
        return f"http:{cfg.model.http_endpoint}"
    return cfg.plugins.model_adapter


def _reproduction_fidelity() -> list[dict[str, str]]:
    return [
        {
            "paper": "AdvEDM",
            "status": "approx",
            "source": "src/mmsec_eval/attacks/advedm/*",
        },
        {
            "paper": "AdvCLIP",
            "status": "approx",
            "source": "src/mmsec_eval/attacks/advclip/*",
        },
        {
            "paper": "TMM",
            "status": "approx",
            "source": "src/mmsec_eval/attacks/tmm/*",
        },
    ]


def _write_attack_debug(sample_debug_dir: Path, record: EvalRecord, cfg: AppConfig) -> str:
    sample_debug_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "sample_id": record.sample.sample_id,
        "attack": cfg.plugins.attack,
        "attack_mode": record.attacked.sample.metadata.get("attack_mode", cfg.attack.mode),
        "attack_metadata": record.attacked.metadata,
        "trace_steps": len(record.attacked.attack_trace),
        "trace_tail": [
            {
                "step": t.step,
                "loss_total": t.loss_total,
                "loss_parts": t.loss_parts,
                "metadata": t.metadata,
            }
            for t in record.attacked.attack_trace[-3:]
        ],
        "clean_text": record.pred_clean.text,
        "adv_text": record.pred_adv.text,
        "diagnostics": record.diagnostics,
        "reproduction_fidelity": _reproduction_fidelity(),
    }
    out = sample_debug_dir / "debug.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(out)


def _benchmark_summary(cfg: AppConfig, summary: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "benchmark": True,
        "benchmark_tag": cfg.dataset.benchmark_tag or cfg.dataset.kind,
        "dataset_name": cfg.dataset.kind,
        "model_adapter": cfg.plugins.model_adapter,
        "attack": cfg.plugins.attack,
        "attack_mode": cfg.attack.mode,
        "num_samples": summary.get("num_samples", len(rows)),
        "asr": summary.get("asr", 0.0),
        "avg_l2": summary.get("avg_l2", 0.0),
        "avg_linf": summary.get("avg_linf", 0.0),
        "transfer_success_rate": summary.get("transfer_success_rate", 0.0),
        "risk_score": summary.get("risk_score", 0.0),
        "risk_level": summary.get("risk_level", ""),
        "risk_scenario": summary.get("risk_scenario", ""),
        "mode_stats": _build_mode_stats(rows),
    }


def _create_sample_store(cfg: AppConfig, run_dir: str) -> SampleStoreManager | None:
    if not cfg.sample_store.enabled:
        return None
    return SampleStoreManager(
        run_dir=run_dir,
        save_images=cfg.sample_store.save_images,
        save_traces=cfg.sample_store.save_traces,
        dataset_tag=(cfg.dataset.benchmark_tag or cfg.dataset.kind),
        model_tag=_model_tag(cfg),
    )


def _build_attack_record(sample: Any, *, cfg: AppConfig, model: Any, attack: Any, metric: Any, judge: Any, run_dir: str, sample_debug_dir: Path) -> tuple[EvalRecord, dict[str, Any]]:
    pred_clean = model.predict(sample)
    attacked = attack.attack(
        sample,
        AttackContext(
            config=cfg,
            model_adapter=model,
            run_dir=run_dir,
            sample_debug_dir=str(sample_debug_dir),
        ),
    )
    pred_adv = model.predict(attacked.sample)
    record = EvalRecord(sample=sample, attacked=attacked, pred_clean=pred_clean, pred_adv=pred_adv)
    cot_clean = parse_cot_trace(pred_clean.text)
    cot_adv = parse_cot_trace(pred_adv.text)
    record.diagnostics = {
        "text_diff_score": _text_diff_score(pred_clean.text, pred_adv.text),
        "embedding_shift": _embedding_shift(pred_clean.embedding, pred_adv.embedding),
        "clean_score": float(pred_clean.score),
        "adv_score": float(pred_adv.score),
        "cot_clean": cot_clean,
        "cot_adv": cot_adv,
        "cot_shift_score": _cot_shift(cot_clean, cot_adv),
    }
    record.judge = judge.judge(record)
    record.metrics = metric.compute(record)
    return record, {"cot_clean": cot_clean}


def _apply_attacked_defense(
    record: EvalRecord,
    *,
    cfg: AppConfig,
    model: Any,
    defense: Any | None,
    metric: Any,
    judge: Any,
    run_dir: str,
    sample_debug_dir: Path,
    cot_clean: dict[str, Any],
) -> dict[str, Any]:
    success_attack = 1.0 if bool(record.judge and record.judge.success) else 0.0
    info: dict[str, Any] = {
        "pred_defended": None,
        "defended_sample": None,
        "defense_refs": {},
        "j_defended_success": success_attack,
        "recovery_success": 0.0,
    }
    if defense is None or not bool(cfg.defense.apply_on_attacked):
        return info

    defended = defense.defend(
        record.attacked.sample,
        DefenseContext(
            config=cfg,
            model_adapter=model,
            stage="attacked",
            run_dir=run_dir,
            sample_debug_dir=str(sample_debug_dir),
        ),
    )
    pred_defended = model.predict(defended.sample)
    record_def = _defense_eval_record(record=record, defended_sample=defended.sample, pred_defended=pred_defended)
    cot_defended = parse_cot_trace(pred_defended.text)
    record_def.diagnostics.update(
        {
            "cot_clean": cot_clean,
            "cot_defended": cot_defended,
            "cot_shift_score": _cot_shift(cot_clean, cot_defended),
        }
    )
    j_def = judge.judge(record_def)
    j_defended_success = 1.0 if j_def.success else 0.0
    def_metrics = metric.compute(record_def)
    _merge_defense_diagnostics(record, record_def, pred_defended, cot_defended, def_metrics)
    recovery_success = _recovery_success(
        adv_text_diff=float(record.diagnostics["text_diff_score"]),
        adv_emb_shift=float(record.diagnostics["embedding_shift"]),
        def_text_diff=float(record_def.diagnostics["text_diff_score"]),
        def_emb_shift=float(record_def.diagnostics["embedding_shift"]),
    )
    info.update(
        {
            "pred_defended": pred_defended,
            "defended_sample": defended.sample,
            "defense_refs": dict(defended.artifact_refs),
            "j_defended_success": j_defended_success,
            "recovery_success": recovery_success,
        }
    )
    return info


def _defense_eval_record(record: EvalRecord, *, defended_sample: Any, pred_defended: Any) -> EvalRecord:
    clean_img = np.asarray(record.sample.image, dtype=np.float32)
    def_img = np.asarray(defended_sample.image, dtype=np.float32)
    delta = def_img - clean_img
    attacked_def = AttackedSample(
        sample=defended_sample,
        perturbation_l0=int(np.count_nonzero(np.abs(delta) > 1e-8)),
        perturbation_l2=float(np.linalg.norm(delta.reshape(-1), ord=2)),
        perturbation_linf=float(np.max(np.abs(delta))) if delta.size else 0.0,
        attack_trace=[],
        metadata={"defended_from": "attacked"},
    )
    record_def = EvalRecord(
        sample=record.sample,
        attacked=attacked_def,
        pred_clean=record.pred_clean,
        pred_adv=pred_defended,
    )
    record_def.diagnostics = {
        "text_diff_score": _text_diff_score(record.pred_clean.text, pred_defended.text),
        "embedding_shift": _embedding_shift(record.pred_clean.embedding, pred_defended.embedding),
        "clean_score": float(record.pred_clean.score),
        "defended_score": float(pred_defended.score),
    }
    return record_def


def _merge_defense_diagnostics(record: EvalRecord, record_def: EvalRecord, pred_defended: Any, cot_defended: dict[str, Any], def_metrics: dict[str, float]) -> None:
    record.diagnostics["defended_text_diff_score"] = float(record_def.diagnostics["text_diff_score"])
    record.diagnostics["defended_embedding_shift"] = float(record_def.diagnostics["embedding_shift"])
    record.diagnostics["defended_score"] = float(pred_defended.score)
    record.diagnostics["cot_defended"] = cot_defended
    record.diagnostics["cot_shift_score_defended"] = float(record_def.diagnostics["cot_shift_score"])
    record.metrics.update({f"defended_{k}": float(v) for k, v in def_metrics.items() if isinstance(v, (int, float))})


def _apply_clean_defense(
    record: EvalRecord,
    *,
    cfg: AppConfig,
    model: Any,
    defense: Any | None,
    run_dir: str,
    sample_debug_dir: Path,
) -> tuple[dict[str, str], float | None, float | None]:
    if defense is None or not bool(cfg.defense.apply_on_clean):
        return {}, None, None
    defended_clean = defense.defend(
        record.sample,
        DefenseContext(
            config=cfg,
            model_adapter=model,
            stage="clean",
            run_dir=run_dir,
            sample_debug_dir=str(sample_debug_dir / "clean"),
        ),
    )
    pred_clean_defended = model.predict(defended_clean.sample)
    clean_utility_text = _text_diff_score(record.pred_clean.text, pred_clean_defended.text)
    clean_utility_emb = _embedding_shift(record.pred_clean.embedding, pred_clean_defended.embedding)
    record.diagnostics["clean_utility_text_diff"] = float(clean_utility_text)
    record.diagnostics["clean_utility_embedding_shift"] = float(clean_utility_emb)
    return dict(defended_clean.artifact_refs), float(clean_utility_text), float(clean_utility_emb)


def _pairwise_stats(record: EvalRecord, defense_info: dict[str, Any], clean_utility: tuple[float | None, float | None], defense: Any | None) -> dict[str, Any]:
    success_attack = 1.0 if bool(record.judge and record.judge.success) else 0.0
    success_defended = None
    if defense is not None:
        success_defended = float(defense_info["j_defended_success"])
    return {
        "success_attack": success_attack,
        "success_defended": success_defended,
        "recovery_success": float(defense_info["recovery_success"]),
        "l0": float(record.metrics.get("perturbation_l0", record.attacked.perturbation_l0)),
        "l2": float(record.metrics.get("perturbation_l2", record.attacked.perturbation_l2)),
        "linf": float(record.metrics.get("perturbation_linf", record.attacked.perturbation_linf)),
        "transfer_success": float(record.metrics.get("transfer_success", success_attack)),
        "semantic_similarity": float(record.metrics.get("semantic_similarity", 0.0)),
        "ssim": float(record.metrics.get("ssim", 0.0)),
        "clean_utility_text_diff": clean_utility[0],
        "clean_utility_embedding_shift": clean_utility[1],
    }


def _append_pairwise_stats(stats: dict[str, Any], lists: dict[str, list[float]]) -> None:
    lists["success_attack"].append(float(stats["success_attack"]))
    if stats["success_defended"] is not None:
        lists["success_defended"].append(float(stats["success_defended"]))
    lists["recovery"].append(float(stats["recovery_success"]))
    lists["l0"].append(float(stats["l0"]))
    lists["l2"].append(float(stats["l2"]))
    lists["linf"].append(float(stats["linf"]))
    lists["transfer"].append(float(stats["transfer_success"]))
    lists["semantic"].append(float(stats["semantic_similarity"]))
    lists["ssim"].append(float(stats["ssim"]))
    if stats["clean_utility_text_diff"] is not None:
        lists["clean_utility_text_diff"].append(float(stats["clean_utility_text_diff"]))
    if stats["clean_utility_embedding_shift"] is not None:
        lists["clean_utility_embedding_shift"].append(float(stats["clean_utility_embedding_shift"]))


def _persist_pairwise_artifacts(record: EvalRecord, *, cfg: AppConfig, sample_debug_dir: Path, store: SampleStoreManager | None, defense_info: dict[str, Any]) -> dict[str, str]:
    refs: dict[str, str] = {}
    pred_defended = defense_info["pred_defended"]
    defended_sample = defense_info["defended_sample"]
    defense_gain_sample = float(_pairwise_attack_success(record) - float(defense_info["j_defended_success"]))
    defense_refs = dict(defense_info["defense_refs"])
    if store is not None:
        refs = store.persist_record(
            record,
            defended_sample=defended_sample,
            pred_defended=pred_defended,
            defense_refs=defense_refs,
            defense_gain_sample=defense_gain_sample,
        )
        record.attacked.artifact_refs.update(refs)
    refs["attack_debug"] = _write_attack_debug(sample_debug_dir, record, cfg)
    return refs


def _pairwise_attack_success(record: EvalRecord) -> float:
    return 1.0 if bool(record.judge and record.judge.success) else 0.0


def _pairwise_result_row(record: EvalRecord, *, cfg: AppConfig, defense: Any | None, defense_info: dict[str, Any], refs: dict[str, str]) -> dict[str, Any]:
    pred_defended = defense_info["pred_defended"]
    j_defended_success = float(defense_info["j_defended_success"])
    defense_gain_sample = float(_pairwise_attack_success(record) - j_defended_success)
    return {
        "sample_id": record.sample.sample_id,
        "clean_text": record.pred_clean.text,
        "adv_text": record.pred_adv.text,
        "defended_text": pred_defended.text if pred_defended is not None else "",
        "clean_score": float(record.pred_clean.score),
        "adv_score": float(record.pred_adv.score),
        "defended_score": float(pred_defended.score) if pred_defended is not None else float(record.pred_adv.score),
        "judge_success": bool(record.judge and record.judge.success),
        "judge_reason": record.judge.reason if record.judge is not None else "",
        "judge_success_attack": bool(record.judge and record.judge.success),
        "judge_success_defended": bool(j_defended_success),
        "error_code": record.pred_adv.error_code or record.pred_clean.error_code,
        "perturbation_l0": record.attacked.perturbation_l0,
        "perturbation_l2": record.attacked.perturbation_l2,
        "perturbation_linf": record.attacked.perturbation_linf,
        "attack_name": record.attacked.sample.metadata.get("attack_name", cfg.plugins.attack),
        "attack_mode": record.attacked.sample.metadata.get("attack_mode", cfg.attack.mode),
        "defense_name": cfg.plugins.defense if defense is not None else "",
        "defense_enabled": bool(defense is not None),
        "defense_gain_sample": defense_gain_sample,
        "recovery_success": float(defense_info["recovery_success"]),
        "dataset_name": cfg.dataset.kind,
        "benchmark_tag": cfg.dataset.benchmark_tag or cfg.dataset.kind,
        "diagnostics": dict(record.diagnostics),
        "artifact_refs": refs,
        **record.metrics,
    }


def _run_pairwise_sample(
    sample: Any,
    *,
    cfg: AppConfig,
    model: Any,
    attack: Any,
    defense: Any | None,
    metric: Any,
    judge: Any,
    run_dir: str,
    attack_debug_root: Path,
    store: SampleStoreManager | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    sample_debug_dir = attack_debug_root / sample.sample_id
    try:
        record, trace_info = _build_attack_record(
            sample,
            cfg=cfg,
            model=model,
            attack=attack,
            metric=metric,
            judge=judge,
            run_dir=run_dir,
            sample_debug_dir=sample_debug_dir,
        )
        defense_info = _apply_attacked_defense(
            record,
            cfg=cfg,
            model=model,
            defense=defense,
            metric=metric,
            judge=judge,
            run_dir=run_dir,
            sample_debug_dir=sample_debug_dir,
            cot_clean=trace_info["cot_clean"],
        )
        clean_refs, clean_text_diff, clean_emb_shift = _apply_clean_defense(
            record,
            cfg=cfg,
            model=model,
            defense=defense,
            run_dir=run_dir,
            sample_debug_dir=sample_debug_dir,
        )
        defense_info["defense_refs"].update(clean_refs)
        stats = _pairwise_stats(record, defense_info, (clean_text_diff, clean_emb_shift), defense)
        refs = _persist_pairwise_artifacts(
            record,
            cfg=cfg,
            sample_debug_dir=sample_debug_dir,
            store=store,
            defense_info=defense_info,
        )
        return _pairwise_result_row(record, cfg=cfg, defense=defense, defense_info=defense_info, refs=refs), stats
    except (AttributeError, RuntimeError, TypeError, ValueError, OSError) as exc:
        LOG.exception("sample failed: %s", sample.sample_id)
        sample_debug_dir.mkdir(parents=True, exist_ok=True)
        (sample_debug_dir / "error.txt").write_text(str(exc), encoding="utf-8")
        raise RuntimeError(f"sample failed: {sample.sample_id}: {exc}") from exc


def _pairwise_aggregates(cfg: AppConfig, stat_lists: dict[str, list[float]]) -> dict[str, Any]:
    success_attack_list = stat_lists["success_attack"]
    success_defended_list = stat_lists["success_defended"]
    asr_attack = _safe_mean(success_attack_list)
    asr_defended = _safe_mean(success_defended_list) if success_defended_list else asr_attack
    l2_values = stat_lists["l2"]
    linf_values = stat_lists["linf"]
    transfer_values = stat_lists["transfer"]
    semantic_values = stat_lists["semantic"]
    ssim_values = stat_lists["ssim"]
    cost_score = 0.5 * (
        normalize_inverse(_safe_mean(l2_values), float(cfg.risk.l2_reference))
        + normalize_inverse(_safe_mean(linf_values), float(cfg.risk.linf_reference))
    )
    semantic_score = 0.5 * (_safe_mean(semantic_values) + _safe_mean(ssim_values))
    stability_score = 1.0 if any(float(x) > 0.5 for x in success_attack_list) else 0.0
    risk_payload = _pairwise_risk_payload(
        cfg,
        asr_attack=asr_attack,
        semantic_score=semantic_score,
        cost_score=cost_score,
        transfer_rate=_safe_mean(transfer_values),
        stability_score=stability_score,
    )
    return {
        "asr_attack": asr_attack,
        "asr_defended": asr_defended,
        "defense_gain": float(asr_attack - asr_defended),
        "recovery_rate": _safe_mean(stat_lists["recovery"]),
        "risk_payload": risk_payload,
    }


def _pairwise_risk_payload(cfg: AppConfig, *, asr_attack: float, semantic_score: float, cost_score: float, transfer_rate: float, stability_score: float) -> dict[str, Any]:
    if bool(cfg.risk.enabled):
        return compute_risk_score(
            scenario=str(cfg.risk.scenario or "general"),
            components={
                "effectiveness": float(asr_attack),
                "semantic": float(semantic_score),
                "cost": float(cost_score),
                "transfer": float(transfer_rate),
                "stability": float(stability_score),
            },
            weights=dict(cfg.risk.weights or {}),
        )
    return {
        "risk_score": 0.0,
        "risk_level": "disabled",
        "risk_scenario": str(cfg.risk.scenario or "general"),
        "risk_breakdown": {},
        "risk_weights": {},
        "risk_recommendations": [],
    }


def _pairwise_summary_payload(cfg: AppConfig, run_id: str, rows: list[dict[str, Any]], stat_lists: dict[str, list[float]], aggregates: dict[str, Any], defense: Any | None) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "task_kind": "pairwise",
        "num_samples": len(rows),
        "num_effective": len(stat_lists["success_attack"]),
        "asr": round(float(aggregates["asr_attack"]), 6),
        "asr_attack": round(float(aggregates["asr_attack"]), 6),
        "asr_defended": round(float(aggregates["asr_defended"]), 6),
        "defense_gain": round(float(aggregates["defense_gain"]), 6),
        "recovery_rate": round(float(aggregates["recovery_rate"]), 6),
        "avg_l0": round(_safe_mean(stat_lists["l0"]), 6),
        "avg_l2": round(_safe_mean(stat_lists["l2"]), 6),
        "avg_linf": round(_safe_mean(stat_lists["linf"]), 6),
        "transfer_success_rate": round(_safe_mean(stat_lists["transfer"]), 6),
        "attack": cfg.plugins.attack,
        "attack_mode": cfg.attack.mode,
        "defense": cfg.plugins.defense if defense is not None else "",
        "defense_enabled": bool(defense is not None),
        "experiment_id": str(cfg.runner.experiment_id or ""),
        "clean_utility_text_diff": round(_safe_mean(stat_lists["clean_utility_text_diff"]), 6),
        "clean_utility_embedding_shift": round(_safe_mean(stat_lists["clean_utility_embedding_shift"]), 6),
        "model_adapter": cfg.plugins.model_adapter,
        "metric": cfg.plugins.metric,
        "judge": cfg.plugins.judge,
        "dataset_name": cfg.dataset.kind,
        "benchmark_tag": cfg.dataset.benchmark_tag or cfg.dataset.kind,
        "reproduction_fidelity": {x["paper"].lower(): x["status"] for x in _reproduction_fidelity()},
        **aggregates["risk_payload"],
    }


def _pairwise_report_payload(summary: dict[str, Any], rows: list[dict[str, Any]], stat_lists: dict[str, list[float]], aggregates: dict[str, Any]) -> dict[str, Any]:
    risk_payload = aggregates["risk_payload"]
    return {
        "summary": summary,
        "mode_stats": _build_mode_stats(rows),
        "stage_metrics": {
            "clean": {"asr": 0.0},
            "attacked": {"asr": float(aggregates["asr_attack"])},
            "defended": {
                "asr": float(aggregates["asr_defended"]),
                "recovery_rate": float(aggregates["recovery_rate"]),
            },
        },
        "defense_compare": {
            "asr_attack": float(aggregates["asr_attack"]),
            "asr_defended": float(aggregates["asr_defended"]),
            "defense_gain": float(aggregates["defense_gain"]),
            "recovery_rate": float(aggregates["recovery_rate"]),
            "clean_utility_text_diff": float(_safe_mean(stat_lists["clean_utility_text_diff"])),
            "clean_utility_embedding_shift": float(_safe_mean(stat_lists["clean_utility_embedding_shift"])),
        },
        "risk": {
            **risk_payload,
            "components_raw": {
                "avg_semantic_similarity": float(_safe_mean(stat_lists["semantic"])),
                "avg_ssim": float(_safe_mean(stat_lists["ssim"])),
                "avg_l2": float(_safe_mean(stat_lists["l2"])),
                "avg_linf": float(_safe_mean(stat_lists["linf"])),
                "transfer_success_rate": float(_safe_mean(stat_lists["transfer"])),
            },
        },
        "rows_preview": rows[: min(20, len(rows))],
        "metric_series": {
            "l2": stat_lists["l2"],
            "linf": stat_lists["linf"],
            "l0": stat_lists["l0"],
        },
        "reproduction_fidelity": _reproduction_fidelity(),
    }


def _write_pairwise_plots(cfg: AppConfig, run_dir: str, rows: list[dict[str, Any]], stat_lists: dict[str, list[float]], aggregates: dict[str, Any], defense: Any | None) -> None:
    if not (cfg.runner.save_plots and rows):
        return
    if stat_lists["l2"]:
        plot_metric_curve(stat_lists["l2"], "perturbation_l2", f"{run_dir}/metric_curve_l2.png")
    if stat_lists["linf"]:
        plot_metric_curve(stat_lists["linf"], "perturbation_linf", f"{run_dir}/metric_curve_linf.png")
    plot_asr_bar({"asr": aggregates["asr_attack"], "transfer": _safe_mean(stat_lists["transfer"])}, f"{run_dir}/asr_bar.png")
    plot_attack_comparison(_build_mode_stats(rows), f"{run_dir}/attack_compare_bar.png")
    if defense is not None:
        plot_stage_compare_bar(
            out_path=f"{run_dir}/stage_compare_asr.png",
            asr_attack=float(aggregates["asr_attack"]),
            asr_defended=float(aggregates["asr_defended"]),
            title="Pairwise Stage Compare",
        )


def _finish_pairwise_run(
    *,
    cfg: AppConfig,
    run_id: str,
    run_dir: str,
    rows: list[dict[str, Any]],
    stat_lists: dict[str, list[float]],
    store: SampleStoreManager | None,
    defense: Any | None,
    benchmark_mode: bool,
    progress: Callable[[str, str, float | None, str], None] | None,
) -> RunArtifacts:
    _emit_progress(progress, "result_aggregation", "running", 90, "正在汇总攻击指标、防御收益和风险分数。")
    aggregates = _pairwise_aggregates(cfg, stat_lists)
    summary = _pairwise_summary_payload(cfg, run_id, rows, stat_lists, aggregates, defense)
    results_path = write_results(run_dir, rows)
    summary_path = write_summary(run_dir, summary)
    report_data = _pairwise_report_payload(summary, rows, stat_lists, aggregates)
    write_json_snapshot(run_dir, "report_data.json", report_data)
    _write_pairwise_plots(cfg, run_dir, rows, stat_lists, aggregates, defense)

    _emit_progress(progress, "report_writing", "running", 97, "正在写入摘要、报告和图表文件。")
    report_path = write_report(run_dir, summary=summary, rows=rows)
    _emit_progress(progress, "report_writing", "success", 99, "运行报告写入完成。")

    run_index_path = store.flush() if store is not None else ""
    benchmark_summary_path = ""
    if benchmark_mode or cfg.dataset.kind in {"flickr30k", "coco_subset"}:
        bench = _benchmark_summary(cfg, summary, rows)
        benchmark_summary_path = write_json_snapshot(run_dir, "benchmark_summary.json", bench)

    return RunArtifacts(
        run_id=run_id,
        run_dir=run_dir,
        results_path=results_path,
        summary_path=summary_path,
        report_path=report_path,
        run_index_path=run_index_path,
        benchmark_summary_path=benchmark_summary_path,
    )


def run(cfg: AppConfig, benchmark_mode: bool = False, progress: Callable[[str, str, float | None, str], None] | None = None) -> RunArtifacts:
    set_seed(cfg.seed)
    run_id = new_run_id()
    run_dir = make_run_dir(cfg.artifacts_dir, run_id)
    attack_debug_root = Path(run_dir) / "attack_debug"
    attack_debug_root.mkdir(parents=True, exist_ok=True)

    write_json_snapshot(run_dir, "config_snapshot.json", asdict(cfg))
    write_env_snapshot(run_dir)

    model = create("model_adapter", cfg.plugins.model_adapter)
    attack = create("attack", cfg.plugins.attack)
    defense = create("defense", cfg.plugins.defense) if bool(cfg.defense.enabled) else None
    metric = create("metric", cfg.plugins.metric)
    judge = create("judge", cfg.plugins.judge)

    _emit_progress(progress, "dataset_loading", "running", 32, "正在装载评测数据集。")
    dataset = load_dataset(cfg)
    if cfg.runner.max_samples > 0:
        dataset = dataset[: cfg.runner.max_samples]
    LOG.info("Loaded dataset: %d samples", len(dataset))
    _emit_progress(progress, "dataset_loading", "success", 38, f"数据集装载完成，共纳入 {len(dataset)} 条样本。")

    store = _create_sample_store(cfg, run_dir)

    rows: list[dict[str, Any]] = []
    stat_lists: dict[str, list[float]] = {
        "success_attack": [],
        "success_defended": [],
        "recovery": [],
        "l0": [],
        "l2": [],
        "linf": [],
        "transfer": [],
        "semantic": [],
        "ssim": [],
        "clean_utility_text_diff": [],
        "clean_utility_embedding_shift": [],
    }

    _emit_progress(progress, "attack_execution", "running", 58, "正在执行对抗攻击并逐样本评测。")
    for sample in dataset:
        row, stats = _run_pairwise_sample(
            sample,
            cfg=cfg,
            model=model,
            attack=attack,
            defense=defense,
            metric=metric,
            judge=judge,
            run_dir=run_dir,
            attack_debug_root=attack_debug_root,
            store=store,
        )
        rows.append(row)
        _append_pairwise_stats(stats, stat_lists)

    return _finish_pairwise_run(
        cfg=cfg,
        run_id=run_id,
        run_dir=run_dir,
        rows=rows,
        stat_lists=stat_lists,
        store=store,
        defense=defense,
        benchmark_mode=benchmark_mode,
        progress=progress,
    )
