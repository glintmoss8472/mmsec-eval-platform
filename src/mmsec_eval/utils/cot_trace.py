# 文件说明：该文件属于项目工程，集中实现 cot trace 相关逻辑。
from __future__ import annotations

import re
from typing import Any


_CALL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(([^()]*)\)")
_ACTION_TOKEN_RE = re.compile(
    r"\b("
    r"brake|accelerate|turn|stop|move|pickup|drop|open|close|wait|park|"
    r"刹车|加速|转向|停止|前进|后退|拾取|放下|打开|关闭|等待|停车"
    r")\b",
    flags=re.IGNORECASE,
)


# 解析 `cot 调试轨迹`，把文本或载荷转换成可校验的字段。
def parse_cot_trace(text: str) -> dict[str, Any]:
    s = str(text or "")
    lines = [x.strip() for x in s.splitlines() if x.strip()]
    if not lines and s.strip():
        lines = [s.strip()]

    reasoning: list[str] = []
    dialogue: list[str] = []
    actions: list[str] = []

    for line in lines:
        low = line.lower()
        is_reason = any(k in low for k in ["reason", "analysis", "think", "because", "推理", "分析", "因为"])
        is_dialogue = any(k in low for k in ["dialog", "response", "reply", "对话", "回复", "回答"])
        is_action = any(k in low for k in ["action", "command", "execute", "动作", "指令", "执行"])

        calls = _CALL_RE.findall(line)
        call_text = [f"{fn}({args})" for fn, args in calls]
        if call_text:
            actions.extend(call_text)
            is_action = True

        if _ACTION_TOKEN_RE.search(line):
            is_action = True

        if is_reason:
            reasoning.append(line)
        if is_dialogue:
            dialogue.append(line)
        if is_action and line not in actions:
            actions.append(line)

    # Fallback: if no explicit sections, treat first line as reasoning and last line as action candidate.
    if not reasoning and lines:
        reasoning = [lines[0]]
    if not actions and lines:
        actions = [lines[-1]]

    final_action = actions[-1] if actions else ""
    return {
        "reasoning": reasoning[:6],
        "dialogue": dialogue[:6],
        "actions": actions[:6],
        "final_action": final_action,
        "num_lines": len(lines),
    }

