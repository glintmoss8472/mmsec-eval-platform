# 文件说明：该文件属于自动化测试，集中实现 test config extra passthrough 相关逻辑。
from __future__ import annotations

from mmsec_eval.config.loader import _to_config


# 中文注释：验证 test_to_config_preserves_extra_without_double_nesting 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_to_config_preserves_extra_without_double_nesting() -> None:
    cfg = _to_config(
        {
            "plugins": {"attack": "advedm_plus"},
            "extra": {"advedm_plus_ablation": {"disable_text_branch": True}},
        }
    )

    assert "extra" not in cfg.extra
    assert cfg.extra["advedm_plus_ablation"]["disable_text_branch"] is True


# 中文注释：验证 test_to_config_keeps_unknown_top_level_keys_in_extra 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_to_config_keeps_unknown_top_level_keys_in_extra() -> None:
    cfg = _to_config(
        {
            "plugins": {"attack": "advedm_plus"},
            "custom_note": {"value": 7},
        }
    )

    assert cfg.extra["custom_note"]["value"] == 7
