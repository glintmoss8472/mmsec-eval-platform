from __future__ import annotations

import json

from mmsec_api.services.job_executor import JobExecutor


def test_parse_override_accepts_legacy_singular_victim_adapter():
    parsed = JobExecutor._parse_override(
        json.dumps({"runner": {"victim_model_adapter": "openai_qwen35_9b"}})
    )

    assert "victim_model_adapter" not in parsed["runner"]
    assert parsed["runner"]["victim_model_adapters"] == ["openai_qwen35_9b"]


def test_parse_override_keeps_explicit_victim_adapter_list():
    parsed = JobExecutor._parse_override(
        json.dumps({"runner": {"victim_model_adapter": "openai_qwen35_9b", "victim_model_adapters": ["openai_qwen3_vl"]}})
    )

    assert "victim_model_adapter" not in parsed["runner"]
    assert parsed["runner"]["victim_model_adapters"] == ["openai_qwen3_vl"]
