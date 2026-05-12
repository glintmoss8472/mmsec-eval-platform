# 文件说明：该文件属于运维与实验脚本，集中实现 reconfigure seven vlm results 相关逻辑。
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mmsec_eval.risk.scoring import compute_risk_score, normalize_direct  # noqa: E402


# 转换 `as float` 输入，在无法解析时返回 None 或调用方默认值。
def _as_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


# 计算 `avg 指标` 均值，空输入时返回可控的默认结果。
def _avg_metric(payload: dict[str, Any], left: str, right: str) -> float:
    return 0.5 * (_as_float(payload.get(left)) + _as_float(payload.get(right)))


# 读取 `JSON`，并对缺失或异常输入做边界处理。
def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# 整理 `iter success rows` 行记录，把原始结果转换成列表接口和报告可消费的结构。
def _iter_success_rows(results_index: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in results_index.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rc = row.get("rc", 1)
        if int(0 if rc is None else rc) == 0 and row.get("summary"):
            rows.append(row)
    return rows


# 执行 `风险 来源 conditional` 辅助逻辑，保持运维与实验脚本中的输入处理和结果输出一致。
def _risk_from_conditional(summary: dict[str, Any], conditional_asr: float, conditional: dict[str, Any]) -> dict[str, Any]:
    risk_node = summary.get("risk", {}) if isinstance(summary.get("risk"), dict) else {}
    old_breakdown = summary.get("risk_breakdown", {}) if isinstance(summary.get("risk_breakdown"), dict) else {}
    raw = risk_node.get("components_raw", {}) if isinstance(risk_node.get("components_raw"), dict) else {}

    avg_rank_delta = 0.5 * (
        _as_float(conditional.get("ir_rank_delta_mean"))
        + _as_float(conditional.get("tr_rank_delta_mean"))
    )
    if avg_rank_delta == 0.0:
        avg_rank_delta = _as_float(raw.get("avg_rank_delta"))
    stability = max(normalize_direct(avg_rank_delta, 100.0), float(conditional_asr))

    return compute_risk_score(
        scenario=str(summary.get("risk_scenario") or summary.get("risk", {}).get("risk_scenario") or "retrieval"),
        components={
            "effectiveness": float(conditional_asr),
            "semantic": _as_float(old_breakdown.get("semantic")),
            "cost": _as_float(old_breakdown.get("cost")),
            "transfer": 0.0 if len(summary.get("victim_model_adapters", []) or []) <= 1 else _as_float(old_breakdown.get("transfer")),
            "stability": float(stability),
        },
        weights=summary.get("risk_weights", {}) if isinstance(summary.get("risk_weights"), dict) else {},
    )


# 整理 `行记录 来源 索引` 行记录，把原始结果转换成列表接口和报告可消费的结构。
def _row_from_index(index_row: dict[str, Any], project_root: Path) -> dict[str, Any]:
    summary_path = project_root / str(index_row["summary"])
    summary = _read_json(summary_path)
    victim_names = list(summary.get("victim_model_adapters", []) or [])
    victim_name = victim_names[0] if victim_names else str(index_row.get("adapter", ""))
    victim = summary.get("victims", {}).get(victim_name, {}) if isinstance(summary.get("victims"), dict) else {}
    clean = victim.get("clean", {}) if isinstance(victim.get("clean"), dict) else {}
    attacked = victim.get("attacked", {}) if isinstance(victim.get("attacked"), dict) else {}
    conditional = victim.get("conditional", {}) if isinstance(victim.get("conditional"), dict) else {}

    conditional_asr = _as_float(summary.get("conditional_asr_attack"))
    if conditional_asr == 0.0 and conditional:
        conditional_asr = _avg_metric(conditional, "ir_cond_asr@1", "tr_cond_asr@1")
    attacked_error = _as_float(
        summary.get("attacked_error_rate@1")
        or summary.get("unconditional_asr_attack")
        or summary.get("asr_attack")
    )
    reconfigured_risk = _risk_from_conditional(summary, conditional_asr, conditional)
    semantic = summary.get("semantic_preservation", {})
    semantic_value = (
        _as_float(semantic.get("combined_semantic_preservation"))
        if isinstance(semantic, dict)
        else 0.0
    )

    return {
        "model": str(index_row.get("model", "")),
        "attack": str(index_row.get("attack", "")),
        "run_id": str(summary.get("run_id", summary_path.parent.name)),
        "duration_seconds": int(index_row.get("duration_seconds", 0) or 0),
        "asr_attack": round(float(conditional_asr), 6),
        "conditional_asr_attack": round(float(conditional_asr), 6),
        "attacked_error_rate@1": round(float(attacked_error), 6),
        "risk_score": round(float(reconfigured_risk.get("risk_score", 0.0)), 6),
        "risk_level": str(reconfigured_risk.get("risk_level", "")),
        "semantic_preservation": semantic_value,
        "clean_ir_r1": _as_float(clean.get("ir_r@1")),
        "attacked_ir_r1": _as_float(attacked.get("ir_r@1")),
        "clean_tr_r1": _as_float(clean.get("tr_r@1")),
        "attacked_tr_r1": _as_float(attacked.get("tr_r@1")),
        "clean_r1_avg": _avg_metric(clean, "ir_r@1", "tr_r@1"),
        "attacked_r1_avg": _avg_metric(attacked, "ir_r@1", "tr_r@1"),
        "cond_support_ir@1": _as_float(conditional.get("ir_cond_support@1")),
        "cond_support_tr@1": _as_float(conditional.get("tr_cond_support@1")),
        "rank_delta_ir": _as_float(conditional.get("ir_rank_delta_mean")),
        "rank_delta_tr": _as_float(conditional.get("tr_rank_delta_mean")),
        "num_images": int(summary.get("num_images", 0) or 0),
        "num_texts": int(summary.get("num_texts", 0) or 0),
        "summary": str(index_row.get("summary", "")),
        "log": str(index_row.get("log", "")),
    }


# 写出 `outputs`，保证后续报告、页面或复现实验能读取。
def _write_outputs(out_base: Path, rows: list[dict[str, Any]]) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "asr_definition": "conditional_clean_top1_drop",
        "attacked_error_rate_definition": "1 - attacked first-rank recall; diagnostic only",
        "rows": rows,
    }
    out_base.with_suffix(".json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = list(rows[0].keys()) if rows else []
    with out_base.with_suffix(".csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "| model | attack | conditional attack success rate | attacked first-rank error | risk | semantic | clean first-rank recall | attacked first-rank recall | support | duration(s) | run_id |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        support = 0.5 * (_as_float(row.get("cond_support_ir@1")) + _as_float(row.get("cond_support_tr@1")))
        lines.append(
            f"| {row['model']} | {row['attack']} | {float(row['conditional_asr_attack']):.4f} | "
            f"{float(row['attacked_error_rate@1']):.4f} | {float(row['risk_score']):.4f} | "
            f"{float(row['semantic_preservation']):.4f} | {float(row['clean_r1_avg']):.4f} | "
            f"{float(row['attacked_r1_avg']):.4f} | {support:.1f} | {int(row['duration_seconds'])} | {row['run_id']} |"
        )
    out_base.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# 执行 `backup 已有` 辅助逻辑，保持运维与实验脚本中的输入处理和结果输出一致。
def _backup_existing(run_dir: Path) -> None:
    tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    for suffix in (".csv", ".json", ".md"):
        src = run_dir / f"effective_results_28{suffix}"
        if src.exists():
            shutil.copy2(src, run_dir / f"effective_results_28.legacy_stage_error_{tag}{suffix}")


# 作为 `reconfigure_seven_vlm_results.py` 的执行入口，串联参数读取、核心处理和退出状态。
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    results_index = run_dir / "results_index.jsonl"
    if not results_index.exists():
        raise FileNotFoundError(results_index)

    rows = [_row_from_index(row, PROJECT_ROOT) for row in _iter_success_rows(results_index)]
    out_base = run_dir / ("effective_results_28" if args.overwrite else "effective_results_28_reconfigured")
    if args.overwrite:
        _backup_existing(run_dir)
    _write_outputs(out_base, rows)
    print(json.dumps({"rows": len(rows), "out_base": str(out_base)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
