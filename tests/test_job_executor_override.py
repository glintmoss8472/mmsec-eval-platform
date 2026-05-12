# 文件说明：该文件属于自动化测试，集中实现 test job executor override 相关逻辑。
from __future__ import annotations

import json

from mmsec_api.services.job_executor import JobExecutor


# 中文注释：验证 test_parse_override_accepts_legacy_singular_victim_adapter 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_parse_override_accepts_legacy_singular_victim_adapter():
    parsed = JobExecutor._parse_override(
        json.dumps({"runner": {"victim_model_adapter": "openai_qwen35_9b"}})
    )

    assert "victim_model_adapter" not in parsed["runner"]
    assert parsed["runner"]["victim_model_adapters"] == ["openai_qwen35_9b"]


# 中文注释：验证 test_parse_override_keeps_explicit_victim_adapter_list 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_parse_override_keeps_explicit_victim_adapter_list():
    parsed = JobExecutor._parse_override(
        json.dumps({"runner": {"victim_model_adapter": "openai_qwen35_9b", "victim_model_adapters": ["openai_qwen3_vl"]}})
    )

    assert "victim_model_adapter" not in parsed["runner"]
    assert parsed["runner"]["victim_model_adapters"] == ["openai_qwen3_vl"]
