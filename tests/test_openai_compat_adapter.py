from __future__ import annotations

import numpy as np

from mmsec_eval.model_adapters.openai_compat_adapter import OpenAICompatAdapter


def test_fallback_score_from_plain_decimal():
    assert OpenAICompatAdapter._fallback_score_from_text("0.83") == 0.83


def test_fallback_score_from_ratio():
    assert OpenAICompatAdapter._fallback_score_from_text("I would rate it 8/10.") == 0.8


def test_fallback_score_from_yes_no_terms():
    assert OpenAICompatAdapter._fallback_score_from_text("yes, the image matches") == 0.8
    assert OpenAICompatAdapter._fallback_score_from_text("no, it does not match") == 0.1


def test_prompt_order_defaults_to_image_first(monkeypatch):
    monkeypatch.delenv("MMSEC_OPENAI_QWEN2_VL_PROMPT_ORDER", raising=False)
    adapter = OpenAICompatAdapter(variant="QWEN2_VL")
    content = adapter._payload(np.zeros((2, 2, 3), dtype=np.float32), "a test caption")["messages"][0]["content"]

    assert adapter.prompt_order == "image_first"
    assert content[0]["type"] == "image_url"
    assert content[1]["type"] == "text"


def test_prompt_order_can_be_text_first(monkeypatch):
    monkeypatch.setenv("MMSEC_OPENAI_QWEN2_VL_PROMPT_ORDER", "text_first")
    adapter = OpenAICompatAdapter(variant="QWEN2_VL")
    content = adapter._payload(np.zeros((2, 2, 3), dtype=np.float32), "a test caption")["messages"][0]["content"]

    assert adapter.prompt_order == "text_first"
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"
