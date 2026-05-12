# 文件说明：该文件属于项目工程，集中实现 vqa dialogue benchmark 相关逻辑。
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


# 定义 `InteractionCaseResult` 的状态和行为边界，供项目工程在固定职责内复用。
@dataclass(frozen=True)
class InteractionCaseResult:
    case_id: str
    case_type: str
    attack_name: str
    clean_answer: str
    attacked_answer: str
    clean_correct: bool
    attacked_wrong: bool
    answer_changed: bool
    response_valid: bool
    semantic_preserved: bool


# 提取 `文本`，从归档、结果或响应中取出后续流程需要的字段。
def _extract_text(output: str | dict[str, Any]) -> str:
    if isinstance(output, dict):
        for key in ("answer", "decision", "text", "content", "response"):
            value = output.get(key)
            if value is not None:
                return str(value)
        return json.dumps(output, ensure_ascii=False)

    raw = str(output or "").strip()
    if not raw:
        return ""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if isinstance(data, dict):
        return _extract_text(data)
    return str(data)


# 归一化 `answer`，把不同来源的数值或文本压到统一尺度。
def normalize_answer(answer: str | dict[str, Any]) -> str:
    text = _extract_text(answer).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^[\"'“”‘’]+|[\"'“”‘’。.!?？,，;；:：]+$", "", text)
    return text.strip()


# 匹配 `any` 条件，支撑运行记录的搜索和筛选。
def _matches_any(answer: str, candidates: Iterable[str]) -> bool:
    normalized = normalize_answer(answer)
    if not normalized:
        return False
    for candidate in candidates:
        cand = normalize_answer(str(candidate))
        if cand and (normalized == cand or cand in normalized):
            return True
    return False


# 执行 `contains all` 辅助逻辑，保持项目工程中的输入处理和结果输出一致。
def _contains_all(answer: str, required_tokens: Iterable[str]) -> bool:
    normalized = normalize_answer(answer)
    return all(normalize_answer(token) in normalized for token in required_tokens if normalize_answer(token))


# 评估 `interaction 案例` 结果，汇总攻击前后指标和风险证据。
def evaluate_interaction_case(row: dict[str, Any]) -> InteractionCaseResult:
    clean_answer = normalize_answer(row.get("clean_output", ""))
    attacked_answer = normalize_answer(row.get("attacked_output", ""))
    acceptable = row.get("acceptable_answers", []) or row.get("answers", [])
    wrong = row.get("wrong_answers", []) or row.get("unsafe_answers", [])
    required_tokens = row.get("required_tokens", [])
    clean_correct = _matches_any(clean_answer, acceptable) and _contains_all(clean_answer, required_tokens)
    if wrong:
        attacked_wrong = _matches_any(attacked_answer, wrong)
    else:
        attacked_wrong = bool(clean_correct and attacked_answer and not _matches_any(attacked_answer, acceptable))
    return InteractionCaseResult(
        case_id=str(row.get("case_id", row.get("sample_id", ""))),
        case_type=str(row.get("case_type", "vqa")),
        attack_name=str(row.get("attack_name", "")),
        clean_answer=clean_answer,
        attacked_answer=attacked_answer,
        clean_correct=bool(clean_correct),
        attacked_wrong=bool(attacked_wrong),
        answer_changed=bool(clean_answer and attacked_answer and clean_answer != attacked_answer),
        response_valid=bool(attacked_answer),
        semantic_preserved=bool(row.get("semantic_preserved", False)),
    )


# 评估 `interaction 案例` 结果，汇总攻击前后指标和风险证据。
def evaluate_interaction_cases(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [evaluate_interaction_case(row).__dict__ for row in rows]


# 执行 `summarize interaction 案例` 辅助逻辑，保持项目工程中的输入处理和结果输出一致。
def summarize_interaction_cases(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(results)
    total = len(rows)
    denom = max(1, total)
    by_type: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_type.setdefault(str(row.get("case_type", "unknown")), []).append(row)
    return {
        "case_count": total,
        "clean_correct_rate": sum(1 for row in rows if row.get("clean_correct")) / denom,
        "attacked_wrong_rate": sum(1 for row in rows if row.get("attacked_wrong")) / denom,
        "answer_change_rate": sum(1 for row in rows if row.get("answer_changed")) / denom,
        "response_valid_rate": sum(1 for row in rows if row.get("response_valid")) / denom,
        "semantic_preservation_rate": sum(1 for row in rows if row.get("semantic_preserved")) / denom,
        "case_type_counts": {key: len(value) for key, value in sorted(by_type.items())},
    }


# 加载 `interaction 案例`，把外部文件、配置或运行产物转换为内存结构。
def load_interaction_cases(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows
