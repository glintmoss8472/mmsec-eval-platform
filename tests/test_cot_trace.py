# 文件说明：该文件属于自动化测试，集中实现 test cot trace 相关逻辑。
from __future__ import annotations

from mmsec_eval.utils.cot_trace import parse_cot_trace


# 验证 `parse cot 调试轨迹 extracts sections` 场景，防止相关行为在后续修改中退化。
def test_parse_cot_trace_extracts_sections():
    text = """
Reasoning: The road is blocked so we should slow down.
Dialogue: I will keep a safe distance.
Action: Brake()
"""
    out = parse_cot_trace(text)
    assert out["reasoning"]
    assert out["dialogue"]
    assert out["actions"]
    assert "brake" in str(out["final_action"]).lower()


# 验证 `parse cot 调试轨迹 fallback` 场景，防止相关行为在后续修改中退化。
def test_parse_cot_trace_fallback():
    out = parse_cot_trace("just one line")
    assert out["reasoning"]
    assert out["actions"]
