# 文件说明：该文件属于后端业务服务，集中实现 run reader 相关逻辑。
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from mmsec_api.services.risk_compat import derive_compatible_risk


# 读取 `JSON`，并对缺失或异常输入做边界处理。
def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


# 读取 `JSONL`，并对缺失或异常输入做边界处理。
def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            if isinstance(data, dict):
                rows.append(data)
        except json.JSONDecodeError:
            continue
    return rows


# 计算分页窗口，把完整列表裁剪成接口本次响应需要的记录范围。
def paginate(items: list[Any], page: int, page_size: int) -> tuple[int, list[Any]]:
    total = len(items)
    start = max(0, (page - 1) * page_size)
    end = start + page_size
    return total, items[start:end]


# 执行 `created at 来源 运行记录 id` 辅助逻辑，保持后端业务服务中的输入处理和结果输出一致。
def _created_at_from_run_id(run_id: str) -> str:
    parts = str(run_id or "").split("_")
    if len(parts) < 2:
        return str(run_id or "")
    try:
        captured = datetime.strptime(f"{parts[0]}{parts[1]}", "%Y%m%d%H%M%S")
    except ValueError:
        return str(run_id or "")
    server_tz = datetime.now().astimezone().tzinfo
    if server_tz is None:
        return captured.isoformat()
    return captured.replace(tzinfo=server_tz).isoformat()


# 转换 `as float` 输入，在无法解析时返回 None 或调用方默认值。
def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


# 计算 `均值` 均值，空输入时返回可控的默认结果。
def _mean(values: list[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


# 组装 `victim 载荷`，把分散字段整理成后端任务或风险评分使用的载荷。
def _victim_payloads(summary: dict[str, Any]) -> list[dict[str, Any]]:
    victims = summary.get("victims")
    if isinstance(victims, dict):
        return [payload for payload in victims.values() if isinstance(payload, dict)]
    return []


# 标记 `指标` 阶段，区分 clean、attacked 和 defended 样本。
def _stage_metric(payloads: list[dict[str, Any]], stage: str, keys: list[str]) -> float | None:
    values: list[float | None] = []
    for payload in payloads:
        node = payload.get(stage)
        if not isinstance(node, dict):
            continue
        for key in keys:
            if key in node:
                values.append(_as_float(node.get(key)))
                break
    return _mean(values)


# 计算 `指标` 条件指标，只在满足前置条件的样本上统计。
def _conditional_metric(payloads: list[dict[str, Any]], keys: list[str]) -> float | None:
    values: list[float | None] = []
    for payload in payloads:
        node = payload.get("conditional")
        if not isinstance(node, dict):
            continue
        for key in keys:
            if key in node:
                values.append(_as_float(node.get(key)))
                break
    return _mean(values)


# 计算 `摘要 dashboard 指标`，把原始模型输出汇总成页面和报告使用的指标字段。
def _summary_dashboard_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    payloads = _victim_payloads(summary)
    clean_ir = _stage_metric(payloads, "clean", ["ir_r@1"])
    clean_tr = _stage_metric(payloads, "clean", ["tr_r@1"])
    attacked_ir = _stage_metric(payloads, "attacked", ["ir_r@1"])
    attacked_tr = _stage_metric(payloads, "attacked", ["tr_r@1"])
    clean_rank = _mean(
        [
            _stage_metric(payloads, "clean", ["mean_rank_ir"]),
            _stage_metric(payloads, "clean", ["mean_rank_tr"]),
        ]
    )
    attacked_rank = _mean(
        [
            _stage_metric(payloads, "attacked", ["mean_rank_ir"]),
            _stage_metric(payloads, "attacked", ["mean_rank_tr"]),
        ]
    )
    rank_delta = _mean(
        [
            _conditional_metric(payloads, ["ir_rank_delta_mean"]),
            _conditional_metric(payloads, ["tr_rank_delta_mean"]),
        ]
    )
    if rank_delta is None and clean_rank is not None and attacked_rank is not None:
        rank_delta = attacked_rank - clean_rank

    clean_r1 = _mean([clean_ir, clean_tr])
    attacked_r1 = _mean([attacked_ir, attacked_tr])
    attack_drop = clean_r1 - attacked_r1 if clean_r1 is not None and attacked_r1 is not None else None
    num_images = int(summary.get("num_images", 0) or 0)
    num_texts = int(summary.get("num_texts", 0) or 0)
    sample_pair_count = summary.get("sample_pair_count") or summary.get("num_pairs")
    if not sample_pair_count and num_images > 0 and num_texts > 0:
        sample_pair_count = num_images * num_texts
    generation_metrics = summary.get("generation_metrics") if isinstance(summary.get("generation_metrics"), dict) else {}
    return {
        "eval_scope": str(summary.get("eval_scope", "")),
        "surrogate_model_adapter": str(summary.get("surrogate_model_adapter", "")),
        "victim_model_adapters": summary.get("victim_model_adapters", []),
        "sample_pair_count": int(sample_pair_count or 0),
        "clean_r1_mean": clean_r1,
        "attacked_r1_mean": attacked_r1,
        "attack_drop_r1_mean": attack_drop,
        "clean_mean_rank": clean_rank,
        "attacked_mean_rank": attacked_rank,
        "rank_delta_mean": rank_delta,
        "clean_accuracy": _as_float(generation_metrics.get("clean_accuracy")),
        "attacked_accuracy": _as_float(generation_metrics.get("attacked_accuracy")),
        "answer_change_rate": _as_float(generation_metrics.get("answer_change_rate")),
        "target_flip_rate": _as_float(generation_metrics.get("target_flip_rate")),
        "semantic_preservation_rate": _as_float(generation_metrics.get("semantic_preservation_rate")),
        "caption_text_similarity": _as_float(generation_metrics.get("caption_text_similarity")),
        "object_jaccard": _as_float(generation_metrics.get("object_jaccard")),
        "avg_linf": _as_float(summary.get("avg_linf")),
        "semantic_preservation_score": _as_float(
            (summary.get("semantic_preservation") or {}).get("combined_semantic_preservation")
            if isinstance(summary.get("semantic_preservation"), dict)
            else None
        ),
    }


# 发现 `运行记录 来源 产物`，从文件系统产物中恢复可展示的运行记录。
def discover_runs_from_artifacts(artifacts_dir: str = "artifacts") -> list[dict[str, Any]]:
    runs_root = Path(artifacts_dir) / "runs"
    if not runs_root.exists():
        return []
    items: list[dict[str, Any]] = []
    for run_dir in sorted([p for p in runs_root.iterdir() if p.is_dir()], reverse=True):
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue
        summary = read_json(summary_path, {})
        run_id = str(summary.get("run_id") or run_dir.name)
        created_at = str(summary.get("created_at") or _created_at_from_run_id(run_id))
        dashboard_metrics = _summary_dashboard_metrics(summary if isinstance(summary, dict) else {})
        risk_payload = derive_compatible_risk(summary if isinstance(summary, dict) else {}, dashboard_metrics)
        items.append(
            {
                "run_id": run_id,
                "created_at": created_at,
                "task_kind": str(summary.get("task_kind", "")),
                "dataset_name": str(summary.get("dataset_name", "")),
                "benchmark_tag": str(summary.get("benchmark_tag", "")),
                "attack": str(summary.get("attack", "")),
                "mode": str(summary.get("attack_mode", "")),
                "experiment_id": str(summary.get("experiment_id", "")),
                "model_adapter": str(summary.get("model_adapter", "")),
                "asr": float(summary.get("asr", 0.0) or 0.0),
                "asr_attack": float(summary.get("asr_attack", summary.get("asr", 0.0)) or 0.0),
                "risk_score": float(risk_payload.get("risk_score", 0.0) or 0.0),
                "risk_level": str(risk_payload.get("risk_level", "")),
                "risk_scenario": str(risk_payload.get("risk_scenario", summary.get("risk_scenario", ""))),
                "avg_l2": float(summary.get("avg_l2", 0.0) or 0.0),
                "path": str(run_dir),
                **dashboard_metrics,
                "semantic_preservation_score": float(summary.get("semantic_preservation_score", 0.0) or 0.0),
            }
        )
    return items
