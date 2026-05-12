# 文件说明：该文件属于后端业务服务，集中实现 risk compat 相关逻辑。
from __future__ import annotations

from typing import Any, Mapping


COMPAT_RISK_WEIGHTS: dict[str, float] = {
    "task_damage": 0.30,
    "output_instability": 0.25,
    "semantic_disguise": 0.15,
    "low_perturbation": 0.15,
    "tail_case": 0.15,
}

COMPAT_RISK_COMPONENTS: tuple[dict[str, str], ...] = (
    {
        "key": "task_damage",
        "label_zh": "任务破坏风险",
        "label_en": "task damage risk",
        "description": "衡量攻击是否破坏图文匹配、视觉问答正确性或图像描述目标。",
    },
    {
        "key": "output_instability",
        "label_zh": "输出失稳风险",
        "label_en": "output instability risk",
        "description": "衡量攻击后召回、排名、答案或描述文本是否出现明显变化。",
    },
    {
        "key": "semantic_disguise",
        "label_zh": "语义伪装风险",
        "label_en": "semantic disguise risk",
        "description": "衡量样本仍保留原始语义或低扰动外观时攻击仍然有效的程度。",
    },
    {
        "key": "low_perturbation",
        "label_zh": "低扰动可达风险",
        "label_en": "low perturbation reachability risk",
        "description": "衡量攻击在较低扰动代价下达到任务破坏效果的程度。",
    },
    {
        "key": "tail_case",
        "label_zh": "尾部案例风险",
        "label_en": "tail case risk",
        "description": "衡量最坏样本或历史稳定性信号中仍然存在的高风险案例。",
    },
)


# 中文注释：封装 _record 的内部步骤，让后端业务服务主流程保持清晰并隔离边界细节。
def _record(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


# 中文注释：封装 _clamp01 的内部步骤，让后端业务服务主流程保持清晰并隔离边界细节。
def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


# 中文注释：封装 _float_or_none 的内部步骤，让后端业务服务主流程保持清晰并隔离边界细节。
def _float_or_none(value: object) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


# 中文注释：封装 _first_float 的内部步骤，让后端业务服务主流程保持清晰并隔离边界细节。
def _first_float(*values: object) -> float | None:
    for value in values:
        parsed = _float_or_none(value)
        if parsed is not None:
            return parsed
    return None


# 中文注释：封装 _mean 的内部步骤，让后端业务服务主流程保持清晰并隔离边界细节。
def _mean(values: list[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


# 中文注释：封装 _victim_payloads 的内部步骤，让后端业务服务主流程保持清晰并隔离边界细节。
def _victim_payloads(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    victims = summary.get("victims")
    if isinstance(victims, dict):
        return [payload for payload in victims.values() if isinstance(payload, dict)]
    return []


# 中文注释：封装 _stage_metric 的内部步骤，让后端业务服务主流程保持清晰并隔离边界细节。
def _stage_metric(payloads: list[dict[str, Any]], stage: str, keys: tuple[str, ...]) -> float | None:
    values: list[float | None] = []
    for payload in payloads:
        node = payload.get(stage)
        if not isinstance(node, dict):
            continue
        for key in keys:
            if key in node:
                values.append(_float_or_none(node.get(key)))
                break
    return _mean(values)


# 中文注释：封装 _conditional_metric 的内部步骤，让后端业务服务主流程保持清晰并隔离边界细节。
def _conditional_metric(payloads: list[dict[str, Any]], keys: tuple[str, ...]) -> float | None:
    values: list[float | None] = []
    for payload in payloads:
        node = payload.get("conditional")
        if not isinstance(node, dict):
            continue
        for key in keys:
            if key in node:
                values.append(_float_or_none(node.get(key)))
                break
    return _mean(values)


# 中文注释：封装 _source_breakdown 的内部步骤，让后端业务服务主流程保持清晰并隔离边界细节。
def _source_breakdown(summary: Mapping[str, Any]) -> dict[str, float]:
    raw = _record(summary.get("risk_breakdown"))
    if not raw:
        raw = _record(_record(summary.get("risk")).get("risk_breakdown"))
    out: dict[str, float] = {}
    for key, value in raw.items():
        parsed = _float_or_none(value)
        if parsed is not None:
            out[str(key)] = _clamp01(parsed)
    return out


# 中文注释：封装 _level_from_score 的内部步骤，让后端业务服务主流程保持清晰并隔离边界细节。
def _level_from_score(score: float) -> str:
    value = _clamp01(score)
    if value >= 0.80:
        return "critical"
    if value >= 0.60:
        return "high"
    if value >= 0.40:
        return "medium"
    if value >= 0.20:
        return "low"
    return "minimal"


# 中文注释：封装 _weighted_score 的内部步骤，让后端业务服务主流程保持清晰并隔离边界细节。
def _weighted_score(values: Mapping[str, float]) -> float:
    return _clamp01(sum(float(values.get(key, 0.0)) * weight for key, weight in COMPAT_RISK_WEIGHTS.items()))


# 中文注释：封装 _audit_rows 的内部步骤，让后端业务服务主流程保持清晰并隔离边界细节。
def _audit_rows(values: Mapping[str, float]) -> list[dict[str, object]]:
    specs = {item["key"]: item for item in COMPAT_RISK_COMPONENTS}
    rows: list[dict[str, object]] = []
    for key in COMPAT_RISK_WEIGHTS:
        spec = specs[key]
        value = _clamp01(float(values.get(key, 0.0)))
        weight = float(COMPAT_RISK_WEIGHTS[key])
        rows.append(
            {
                "key": key,
                "label_zh": spec["label_zh"],
                "label_en": spec["label_en"],
                "value": round(value, 6),
                "weight": round(weight, 6),
                "contribution": round(value * weight, 6),
                "direction": "higher_is_riskier",
                "description": spec["description"],
            }
        )
    return rows


# 中文注释：封装 _risk_observations 的内部步骤，让后端业务服务主流程保持清晰并隔离边界细节。
def _risk_observations(level: str, values: Mapping[str, float]) -> list[str]:
    observations: list[str] = []
    if float(values.get("task_damage", 0.0)) >= 0.6:
        observations.append("任务破坏风险高：优先复核高分运行和对应样本证据。")
    if float(values.get("output_instability", 0.0)) >= 0.6:
        observations.append("输出失稳明显：重点对比原始输出和攻击后输出差异。")
    if float(values.get("semantic_disguise", 0.0)) >= 0.6 or float(values.get("low_perturbation", 0.0)) >= 0.6:
        observations.append("攻击扰动较隐蔽：结合扰动图、掩码和文本差异解释结论边界。")
    if float(values.get("tail_case", 0.0)) >= 0.6:
        observations.append("尾部案例风险突出：讲解时应展示最坏样本和证据链。")
    if observations:
        return observations
    if level in {"minimal", "low"}:
        return ["当前风险较低：保留样本规模和证据置信度提示，避免过度解释。"]
    return ["当前风险需要复核：结合分任务指标和样本复盘确认风险来源。"]


# 中文注释：封装 _low_cost_from_summary 的内部步骤，让后端业务服务主流程保持清晰并隔离边界细节。
def _low_cost_from_summary(summary: Mapping[str, Any], row: Mapping[str, Any], source_breakdown: Mapping[str, float]) -> float:
    source_cost = _float_or_none(source_breakdown.get("cost"))
    if source_cost is not None:
        return _clamp01(source_cost)
    avg_l2 = _first_float(summary.get("avg_l2"), row.get("avg_l2"))
    avg_linf = _first_float(summary.get("avg_linf"), summary.get("avg_l_inf"), row.get("avg_linf"))
    l2_cost = _clamp01(1.0 - ((avg_l2 or 0.0) / 25.0)) if avg_l2 is not None else 0.0
    linf_cost = _clamp01(1.0 - ((avg_linf or 0.0) / 0.2)) if avg_linf is not None else 0.0
    if avg_l2 is None and avg_linf is None:
        return 0.0
    if avg_l2 is None:
        return linf_cost
    if avg_linf is None:
        return l2_cost
    return _clamp01((l2_cost + linf_cost) / 2.0)


# 中文注释：封装 _semantic_base 的内部步骤，让后端业务服务主流程保持清晰并隔离边界细节。
def _semantic_base(summary: Mapping[str, Any], generation: Mapping[str, Any], source_breakdown: Mapping[str, float]) -> float:
    nested = _record(summary.get("semantic_preservation")).get("combined_semantic_preservation")
    value = _first_float(
        summary.get("semantic_preservation_score"),
        nested,
        generation.get("semantic_preservation_rate"),
        source_breakdown.get("semantic"),
    )
    return _clamp01(value if value is not None else 0.0)


# 中文注释：封装 _vlr_metrics 的内部步骤，让后端业务服务主流程保持清晰并隔离边界细节。
def _vlr_metrics(summary: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, float]:
    payloads = _victim_payloads(summary)
    clean_r1 = _first_float(
        row.get("clean_r1_mean"),
        summary.get("clean_r1_mean"),
        _mean(
            [
                _stage_metric(payloads, "clean", ("ir_r@1",)),
                _stage_metric(payloads, "clean", ("tr_r@1",)),
            ]
        ),
    )
    attacked_r1 = _first_float(
        row.get("attacked_r1_mean"),
        summary.get("attacked_r1_mean"),
        _mean(
            [
                _stage_metric(payloads, "attacked", ("ir_r@1",)),
                _stage_metric(payloads, "attacked", ("tr_r@1",)),
            ]
        ),
    )
    clean_rank = _first_float(
        row.get("clean_mean_rank"),
        summary.get("clean_mean_rank"),
        _mean(
            [
                _stage_metric(payloads, "clean", ("mean_rank_ir",)),
                _stage_metric(payloads, "clean", ("mean_rank_tr",)),
            ]
        ),
    )
    attacked_rank = _first_float(
        row.get("attacked_mean_rank"),
        summary.get("attacked_mean_rank"),
        _mean(
            [
                _stage_metric(payloads, "attacked", ("mean_rank_ir",)),
                _stage_metric(payloads, "attacked", ("mean_rank_tr",)),
            ]
        ),
    )
    rank_delta = _first_float(
        row.get("rank_delta_mean"),
        summary.get("rank_delta_mean"),
        _mean(
            [
                _conditional_metric(payloads, ("ir_rank_delta_mean",)),
                _conditional_metric(payloads, ("tr_rank_delta_mean",)),
            ]
        ),
    )
    if rank_delta is None and clean_rank is not None and attacked_rank is not None:
        rank_delta = attacked_rank - clean_rank
    attack_drop = _first_float(row.get("attack_drop_r1_mean"), summary.get("attack_drop_r1_mean"))
    if attack_drop is None and clean_r1 is not None and attacked_r1 is not None:
        attack_drop = clean_r1 - attacked_r1
    return {
        "recall_drop": _clamp01(attack_drop if attack_drop is not None else 0.0),
        "rank_shift": _clamp01(max(rank_delta if rank_delta is not None else 0.0, 0.0) / 100.0),
    }


# 中文注释：实现 derive_compatible_risk 的核心流程，支撑后端业务服务中的业务语义和异常边界。
def derive_compatible_risk(summary: Mapping[str, Any] | None, row: Mapping[str, Any] | None = None) -> dict[str, object]:
    source = _record(summary)
    row_data = _record(row)
    generation = _record(source.get("generation_metrics"))
    source_breakdown = _source_breakdown(source)
    task_kind = str(source.get("task_kind") or row_data.get("task_kind") or "").strip().lower()
    asr = _clamp01(_first_float(source.get("asr_attack"), row_data.get("asr_attack"), source.get("asr"), row_data.get("asr")) or 0.0)
    low_cost = _low_cost_from_summary(source, row_data, source_breakdown)

    if task_kind in {"vlr", "retrieval"}:
        metrics = _vlr_metrics(source, row_data)
        recall_drop = metrics["recall_drop"]
        rank_shift = metrics["rank_shift"]
        task_damage = _clamp01(0.7 * asr + 0.3 * recall_drop)
        output_instability = _clamp01(0.6 * recall_drop + 0.4 * rank_shift)
        semantic_base = _semantic_base(source, generation, source_breakdown)
        tail_case = _clamp01(_first_float(source_breakdown.get("stability"), max(task_damage, output_instability)) or 0.0)
    elif task_kind == "vqa":
        clean_acc = _clamp01(_first_float(generation.get("clean_accuracy"), row_data.get("clean_accuracy")) or 0.0)
        attacked_acc = _clamp01(_first_float(generation.get("attacked_accuracy"), row_data.get("attacked_accuracy")) or 0.0)
        answer_change = _clamp01(_first_float(generation.get("answer_change_rate"), row_data.get("answer_change_rate"), asr) or 0.0)
        cond_asr = _clamp01(asr / clean_acc) if clean_acc > 0 else asr
        acc_drop = _clamp01((clean_acc - attacked_acc) / clean_acc) if clean_acc > 0 else asr
        task_damage = _clamp01(0.6 * cond_asr + 0.4 * acc_drop)
        output_instability = answer_change
        semantic_base = _clamp01(_first_float(source_breakdown.get("cost"), low_cost) or 0.0)
        tail_case = _clamp01(_first_float(source_breakdown.get("stability"), task_damage) or 0.0)
    elif task_kind == "caption":
        text_similarity = _clamp01(_first_float(generation.get("caption_text_similarity"), row_data.get("caption_text_similarity"), 1.0 - asr) or 0.0)
        object_jaccard = _clamp01(_first_float(generation.get("object_jaccard"), row_data.get("object_jaccard"), 1.0 - asr) or 0.0)
        text_shift = _clamp01(1.0 - text_similarity)
        object_shift = _clamp01(1.0 - object_jaccard)
        task_damage = _clamp01(0.5 * asr + 0.3 * object_shift + 0.2 * text_shift)
        output_instability = _clamp01(0.7 * text_shift + 0.3 * object_shift)
        semantic_base = _semantic_base(source, generation, source_breakdown)
        tail_case = _clamp01(_first_float(source_breakdown.get("stability"), task_damage) or 0.0)
    else:
        task_damage = _clamp01(_first_float(source_breakdown.get("effectiveness"), asr) or 0.0)
        output_instability = _clamp01(_first_float(source_breakdown.get("effectiveness"), asr) or 0.0)
        semantic_base = _clamp01(_first_float(source_breakdown.get("semantic"), 0.0) or 0.0)
        tail_case = _clamp01(_first_float(source_breakdown.get("stability"), max(task_damage, output_instability)) or 0.0)

    values = {
        "task_damage": _clamp01(task_damage),
        "output_instability": _clamp01(output_instability),
        "semantic_disguise": _clamp01(task_damage * semantic_base),
        "low_perturbation": _clamp01(task_damage * low_cost),
        "tail_case": _clamp01(tail_case),
    }
    score = round(_weighted_score(values), 6)
    level = _level_from_score(score)
    return {
        "risk_score": score,
        "risk_level": level,
        "risk_scenario": task_kind or str(source.get("risk_scenario") or row_data.get("risk_scenario") or "general"),
        "risk_breakdown": {key: round(value, 6) for key, value in values.items()},
        "risk_weights": {key: round(value, 6) for key, value in COMPAT_RISK_WEIGHTS.items()},
        "risk_component_audit": _audit_rows(values),
        "risk_recommendations": _risk_observations(level, values),
    }


# 中文注释：实现 apply_compatible_risk 的核心流程，支撑后端业务服务中的业务语义和异常边界。
def apply_compatible_risk(summary: Mapping[str, Any] | None, row: Mapping[str, Any] | None = None) -> dict[str, Any]:
    out = dict(_record(summary))
    out.update(derive_compatible_risk(out, row))
    return out


# 中文注释：实现 apply_compatible_report_data 的核心流程，支撑后端业务服务中的业务语义和异常边界。
def apply_compatible_report_data(report_data: Mapping[str, Any] | None) -> dict[str, Any]:
    out = dict(_record(report_data))
    summary = _record(out.get("summary")) or out
    risk_payload = derive_compatible_risk(summary)
    if "summary" in out and isinstance(out.get("summary"), dict):
        out["summary"] = {**dict(_record(out.get("summary"))), **risk_payload}
    risk_node = dict(_record(out.get("risk")))
    risk_node.update(risk_payload)
    risk_node["components_raw"] = dict(risk_payload["risk_breakdown"])
    out["risk"] = risk_node
    return out
