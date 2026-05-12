# 文件说明：该文件属于自动化测试，集中实现 test cot trace 相关逻辑。
from __future__ import annotations

from mmsec_eval.utils.cot_trace import parse_cot_trace


# 中文注释：验证 test_parse_cot_trace_extracts_sections 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
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


# 中文注释：验证 test_parse_cot_trace_fallback 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_parse_cot_trace_fallback():
    out = parse_cot_trace("just one line")
    assert out["reasoning"]
    assert out["actions"]
