from __future__ import annotations

import os

from mmsec_eval.cli import _apply_runtime_env
from mmsec_eval.config.loader import load_config


def test_apply_runtime_env_sets_local_openai_compat_defaults(monkeypatch):
    for key in [
        "MMSEC_OPENAI_QWEN35_9B_BASE_URL",
        "MMSEC_OPENAI_QWEN3_VL_BASE_URL",
        "MMSEC_OPENAI_QWEN25_VL_BASE_URL",
        "MMSEC_OPENAI_INTERNVL35_BASE_URL",
        "MMSEC_OPENAI_MINICPM_V_BASE_URL",
        "MMSEC_OPENAI_OVIS25_BASE_URL",
        "MMSEC_OPENAI_GEMMA3_12B_BASE_URL",
    ]:
        monkeypatch.delenv(key, raising=False)

    cfg = load_config("configs/mvp.yaml")
    _apply_runtime_env(cfg)

    assert os.environ["MMSEC_OPENAI_QWEN35_9B_BASE_URL"] == "http://127.0.0.1:8011/v1"
    assert os.environ["MMSEC_OPENAI_QWEN3_VL_BASE_URL"] == "http://127.0.0.1:8012/v1"
    assert os.environ["MMSEC_OPENAI_QWEN25_VL_BASE_URL"] == "http://127.0.0.1:8013/v1"
    assert os.environ["MMSEC_OPENAI_INTERNVL35_BASE_URL"] == "http://127.0.0.1:8014/v1"
    assert os.environ["MMSEC_OPENAI_MINICPM_V_BASE_URL"] == "http://127.0.0.1:8015/v1"
    assert os.environ["MMSEC_OPENAI_OVIS25_BASE_URL"] == "http://127.0.0.1:8016/v1"
    assert os.environ["MMSEC_OPENAI_GEMMA3_12B_BASE_URL"] == "http://127.0.0.1:8017/v1"
