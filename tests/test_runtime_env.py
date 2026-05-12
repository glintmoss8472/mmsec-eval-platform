# 文件说明：该文件属于自动化测试，集中实现 test runtime env 相关逻辑。
from __future__ import annotations

import os

from mmsec_eval.config.loader import load_config
from mmsec_eval.model_adapters.local_vlm_catalog import LOCAL_OPENAI_COMPAT_MODEL_SPECS
from mmsec_eval.runtime import apply_config_env, apply_local_vlm_env_defaults


# 中文注释：验证 test_apply_config_env_sets_common_model_and_adapter_settings 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_apply_config_env_sets_common_model_and_adapter_settings(monkeypatch):
    cfg = load_config("configs/mvp.yaml")
    monkeypatch.setattr(cfg.model, "http_endpoint", "http://127.0.0.1:9999/score")
    monkeypatch.setattr(cfg.model, "http_retries", 3)
    monkeypatch.setattr(cfg.model, "http_timeout", 12)
    monkeypatch.setattr(cfg.judge, "llm_enabled", True)
    monkeypatch.setattr(cfg.judge, "llm_provider", "local")
    monkeypatch.setattr(cfg.judge, "llm_endpoint", "http://127.0.0.1:9998")

    apply_config_env(cfg)

    assert os.environ["MMSEC_CLIP_MODEL_NAME"] == cfg.model.clip_model_name
    assert os.environ["MMSEC_OPENAI_COMPAT_MODEL_NAME"] == cfg.model.openai_model_name
    assert os.environ["MMSEC_GEMINI_MODEL_NAME"] == cfg.model.gemini_model_name
    assert os.environ["MMSEC_HTTP_ADAPTER_ENDPOINT"] == "http://127.0.0.1:9999/score"
    assert os.environ["MMSEC_HTTP_ADAPTER_RETRIES"] == "3"
    assert os.environ["MMSEC_HTTP_ADAPTER_TIMEOUT"] == "12"
    assert os.environ["MMSEC_LLM_JUDGE_ENABLED"] == "1"
    assert os.environ["MMSEC_LLM_PROVIDER"] == "local"
    assert os.environ["MMSEC_LLM_ENDPOINT"] == "http://127.0.0.1:9998"


# 中文注释：验证 test_apply_local_vlm_env_defaults_preserves_overrides_and_can_set_api_fields 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_apply_local_vlm_env_defaults_preserves_overrides_and_can_set_api_fields(monkeypatch):
    first = LOCAL_OPENAI_COMPAT_MODEL_SPECS[0]
    monkeypatch.setenv(first.endpoint_env, "http://127.0.0.1:19999/v1")
    monkeypatch.delenv(first.api_key_env, raising=False)
    monkeypatch.delenv(first.timeout_env, raising=False)

    apply_local_vlm_env_defaults(include_api_key_env=True, include_timeout=True)

    assert os.environ[first.endpoint_env] == "http://127.0.0.1:19999/v1"
    assert os.environ[first.api_key_env] == first.api_key_env_default
    assert os.environ[first.timeout_env] == first.timeout_default
