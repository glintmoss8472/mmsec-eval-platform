# 文件说明：该文件属于项目工程，集中实现 paper evidence 相关逻辑。
from __future__ import annotations

import json
from typing import Any


JOINT_TEXT_ATTACKS = {"tmm", "advedm_plus"}
IMAGE_ATTACKS = {
    "advclip",
    "advedm",
    "tmm",
    "advedm_plus",
    "vqa_visual_corruption",
    "xtransfer_uap",
    "foa_attack",
    "anyattack",
    "mpc_attack",
    "m_attack",
}
INFERRED_JOINT_NOTE = "原始 summary.attack_debug 未保留在当前归档包内，当前联合执行标记依据冻结实验定义中的 eval_scope 和 no_text 消融标识补写。"
OBSERVED_JOINT_NOTE = "联合执行标记直接来自冻结的 summary.attack_debug 字段。"


# 中文注释：实现 parse_optional_bool 的核心流程，支撑项目工程中的业务语义和异常边界。
def parse_optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    text = str(value or "").strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


# 中文注释：实现 parse_optional_float 的核心流程，支撑项目工程中的业务语义和异常边界。
def parse_optional_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


# 中文注释：实现 contains_no_text_marker 的核心流程，支撑项目工程中的业务语义和异常边界。
def contains_no_text_marker(*parts: Any) -> bool:
    for part in parts:
        text = str(part or "").strip().lower()
        if not text:
            continue
        if "no_text" in text or "disable_text_branch" in text or "去文本" in text:
            return True
    return False


# 中文注释：实现 marker_json 的核心流程，支撑项目工程中的业务语义和异常边界。
def marker_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False) if isinstance(value, dict) else ""


# 中文注释：封装 _finalize_joint_payload 的内部步骤，让项目工程主流程保持清晰并隔离边界细节。
def _finalize_joint_payload(
    *,
    image_branch_enabled: bool | None,
    text_branch_enabled: bool | None,
    text_edit_applied: bool | None,
    text_changed_ratio: float | None,
    evidence_source: str,
    evidence_basis: str,
    note: str,
) -> dict[str, Any]:
    joint_execution_declared = bool(image_branch_enabled and text_branch_enabled and text_edit_applied)
    return {
        "image_branch_enabled": bool(image_branch_enabled),
        "text_branch_enabled": bool(text_branch_enabled),
        "text_edit_applied": bool(text_edit_applied),
        "text_changed_ratio": text_changed_ratio,
        "joint_execution_declared": joint_execution_declared,
        "joint_execution_confirmed": bool(joint_execution_declared and evidence_basis == "observed"),
        "joint_execution_evidence_source": evidence_source,
        "joint_execution_basis": evidence_basis,
        "joint_execution_note": note,
    }


# 中文注释：实现 build_row_joint_execution 的核心流程，支撑项目工程中的业务语义和异常边界。
def build_row_joint_execution(row: dict[str, Any]) -> dict[str, Any]:
    attack = str(row.get("attack", "") or "").strip()
    eval_scope = str(row.get("eval_scope", "") or "").strip().lower()
    attack_debug = row.get("attack_debug", {})
    attack_debug = attack_debug if isinstance(attack_debug, dict) else {}
    explicit_no_text = contains_no_text_marker(
        row.get("id", ""),
        row.get("row_id", ""),
        row.get("benchmark_tag", ""),
        row.get("experiment_id", ""),
        marker_json(row.get("extra", {})),
    )

    image_branch_enabled = parse_optional_bool(row.get("image_branch_enabled"))
    if image_branch_enabled is None:
        image_branch_enabled = parse_optional_bool(attack_debug.get("need_image"))
    if image_branch_enabled is None:
        image_branch_enabled = eval_scope in {"image", "joint"} or attack in IMAGE_ATTACKS

    text_branch_enabled = parse_optional_bool(row.get("text_branch_enabled"))
    if text_branch_enabled is None:
        text_branch_enabled = parse_optional_bool(attack_debug.get("need_text"))
    if text_branch_enabled is None:
        text_branch_enabled = attack in JOINT_TEXT_ATTACKS and eval_scope in {"text", "joint"} and not explicit_no_text

    text_changed_ratio = parse_optional_float(row.get("text_changed_ratio"))
    if text_changed_ratio is None:
        text_changed_ratio = parse_optional_float(attack_debug.get("text_changed_ratio"))

    text_edit_applied = parse_optional_bool(row.get("text_edit_applied"))
    evidence_source = str(row.get("joint_execution_evidence_source", "") or "").strip()
    evidence_basis = str(row.get("joint_execution_basis", "") or "").strip()
    if not evidence_source:
        evidence_source = "summary_attack_debug" if attack_debug else "frozen_experiment_definition"
    if not evidence_basis:
        evidence_basis = "observed" if attack_debug else "inferred"

    if text_edit_applied is None:
        if text_changed_ratio is not None:
            text_edit_applied = text_changed_ratio > 0.0
        elif explicit_no_text:
            text_edit_applied = False
        else:
            text_edit_applied = bool(text_branch_enabled)

    note = str(row.get("joint_execution_note", "") or "").strip()
    if not note:
        note = OBSERVED_JOINT_NOTE if attack_debug else INFERRED_JOINT_NOTE

    return _finalize_joint_payload(
        image_branch_enabled=image_branch_enabled,
        text_branch_enabled=text_branch_enabled,
        text_edit_applied=text_edit_applied,
        text_changed_ratio=text_changed_ratio,
        evidence_source=evidence_source,
        evidence_basis=evidence_basis,
        note=note,
    )


# 中文注释：实现 build_summary_joint_execution 的核心流程，支撑项目工程中的业务语义和异常边界。
def build_summary_joint_execution(
    row: dict[str, Any],
    summary: dict[str, Any],
    *,
    observed_note: str = "联合执行标记直接来自 summary.attack_debug 中冻结的 need_text / text_changed_ratio 字段。",
) -> dict[str, Any]:
    attack = str(summary.get("attack", row.get("attack", "")) or "").strip()
    eval_scope = str(summary.get("eval_scope", row.get("eval_scope", "")) or "").strip().lower()
    attack_debug = summary.get("attack_debug", {})
    attack_debug = attack_debug if isinstance(attack_debug, dict) else {}
    explicit_no_text = contains_no_text_marker(
        row.get("id", ""),
        row.get("benchmark_tag", ""),
        summary.get("experiment_id", ""),
        marker_json(summary.get("extra", {})),
        marker_json(row.get("extra", {})),
    )

    image_branch_enabled = parse_optional_bool(attack_debug.get("need_image"))
    if image_branch_enabled is None:
        image_branch_enabled = eval_scope in {"image", "joint"} or attack in IMAGE_ATTACKS

    text_branch_enabled = parse_optional_bool(attack_debug.get("need_text"))
    if text_branch_enabled is None:
        text_branch_enabled = attack in JOINT_TEXT_ATTACKS and eval_scope in {"text", "joint"} and not explicit_no_text

    text_changed_ratio = parse_optional_float(attack_debug.get("text_changed_ratio"))
    text_edit_applied = parse_optional_bool(row.get("text_edit_applied"))
    evidence_source = "summary_attack_debug" if attack_debug else "frozen_experiment_definition"
    evidence_basis = "observed" if attack_debug else "inferred"

    if text_edit_applied is None:
        if text_changed_ratio is not None:
            text_edit_applied = text_changed_ratio > 0.0
        elif explicit_no_text:
            text_edit_applied = False
        else:
            text_edit_applied = bool(text_branch_enabled)

    return _finalize_joint_payload(
        image_branch_enabled=image_branch_enabled,
        text_branch_enabled=text_branch_enabled,
        text_edit_applied=text_edit_applied,
        text_changed_ratio=text_changed_ratio,
        evidence_source=evidence_source,
        evidence_basis=evidence_basis,
        note=observed_note if attack_debug else INFERRED_JOINT_NOTE,
    )


# 中文注释：封装 _joint_payload_from_summary 的内部步骤，让项目工程主流程保持清晰并隔离边界细节。
def _joint_payload_from_summary(summary_data: dict[str, Any] | None) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if isinstance(summary_data, dict):
        candidate = summary_data.get("joint_execution", {})
        if isinstance(candidate, dict):
            payload = dict(candidate)
        if not payload:
            attack_debug = summary_data.get("attack_debug", {})
            nested_summary = summary_data.get("summary", {})
            if not attack_debug and isinstance(nested_summary, dict):
                attack_debug = dict(nested_summary.get("attack_debug", {}))
            if isinstance(attack_debug, dict) and attack_debug:
                payload = {
                    "image_branch_enabled": parse_optional_bool(attack_debug.get("need_image")),
                    "text_branch_enabled": parse_optional_bool(attack_debug.get("need_text")),
                    "text_changed_ratio": parse_optional_float(attack_debug.get("text_changed_ratio")),
                    "joint_execution_evidence_source": "summary_attack_debug",
                    "joint_execution_basis": "observed",
                    "joint_execution_note": OBSERVED_JOINT_NOTE,
                }
    return payload


# 中文注释：封装 _formal_branch_flags 的内部步骤，让项目工程主流程保持清晰并隔离边界细节。
def _formal_branch_flags(
    row: dict[str, Any],
    payload: dict[str, Any],
    *,
    attack: str,
    eval_scope: str,
    explicit_no_text: bool,
) -> tuple[bool | None, bool | None, float | None, bool | None]:
    image_branch_enabled = parse_optional_bool(payload.get("image_branch_enabled"))
    if image_branch_enabled is None:
        image_branch_enabled = parse_optional_bool(row.get("image_branch_enabled"))
    if image_branch_enabled is None:
        image_branch_enabled = eval_scope in {"image", "joint"} or attack in IMAGE_ATTACKS

    text_branch_enabled = parse_optional_bool(payload.get("text_branch_enabled"))
    if text_branch_enabled is None:
        text_branch_enabled = parse_optional_bool(row.get("text_branch_enabled"))
    if text_branch_enabled is None:
        text_branch_enabled = attack in JOINT_TEXT_ATTACKS and eval_scope in {"text", "joint"} and not explicit_no_text

    text_changed_ratio = parse_optional_float(payload.get("text_changed_ratio"))
    if text_changed_ratio is None:
        text_changed_ratio = parse_optional_float(row.get("text_changed_ratio"))

    text_edit_applied = parse_optional_bool(payload.get("text_edit_applied"))
    if text_edit_applied is None:
        text_edit_applied = parse_optional_bool(row.get("text_edit_applied"))
    if text_edit_applied is None:
        text_edit_applied = (text_changed_ratio > 0.0) if text_changed_ratio is not None else bool(text_branch_enabled and not explicit_no_text)
    return image_branch_enabled, text_branch_enabled, text_changed_ratio, text_edit_applied


# 中文注释：封装 _formal_evidence_text 的内部步骤，让项目工程主流程保持清晰并隔离边界细节。
def _formal_evidence_text(row: dict[str, Any], payload: dict[str, Any]) -> tuple[str, str, str]:
    evidence_source = str(payload.get("joint_execution_evidence_source", row.get("joint_execution_evidence_source", "")) or "").strip()
    if not evidence_source:
        evidence_source = "frozen_experiment_definition"
    evidence_basis = str(payload.get("joint_execution_basis", row.get("joint_execution_basis", "")) or "").strip()
    if not evidence_basis:
        evidence_basis = "inferred" if evidence_source == "frozen_experiment_definition" else "observed"
    note = str(payload.get("joint_execution_note", row.get("joint_execution_note", "")) or "").strip()
    return evidence_source, evidence_basis, note or INFERRED_JOINT_NOTE


# 中文注释：实现 build_formal_joint_execution 的核心流程，支撑项目工程中的业务语义和异常边界。
def build_formal_joint_execution(row: dict[str, Any], summary_data: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = _joint_payload_from_summary(summary_data)
    attack = str(row.get("attack", "") or "").strip()
    eval_scope = str(row.get("eval_scope", "") or "").strip().lower()
    explicit_no_text = contains_no_text_marker(
        row.get("evidence_row_id", ""),
        row.get("experiment_id", ""),
        row.get("benchmark_tag", ""),
        row.get("summary_path", ""),
        row.get("report_path", ""),
        row.get("joint_execution_note", ""),
    )
    image_branch_enabled, text_branch_enabled, text_changed_ratio, text_edit_applied = _formal_branch_flags(
        row, payload, attack=attack, eval_scope=eval_scope, explicit_no_text=explicit_no_text
    )
    evidence_source, evidence_basis, note = _formal_evidence_text(row, payload)
    return _finalize_joint_payload(
        image_branch_enabled=image_branch_enabled,
        text_branch_enabled=text_branch_enabled,
        text_edit_applied=text_edit_applied,
        text_changed_ratio=text_changed_ratio,
        evidence_source=evidence_source,
        evidence_basis=evidence_basis,
        note=note,
    )
