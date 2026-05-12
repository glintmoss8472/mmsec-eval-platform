from __future__ import annotations

from mmsec_eval.utils.cot_trace import parse_cot_trace


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


def test_parse_cot_trace_fallback():
    out = parse_cot_trace("just one line")
    assert out["reasoning"]
    assert out["actions"]
