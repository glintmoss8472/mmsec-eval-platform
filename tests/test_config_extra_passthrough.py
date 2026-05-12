# 文件说明：该文件属于自动化测试，集中实现 test config extra passthrough 相关逻辑。
from __future__ import annotations

from mmsec_eval.config.loader import _to_config


# 验证 `to 配置 preserves extra 移除 double nesting` 场景，防止相关行为在后续修改中退化。
def test_to_config_preserves_extra_without_double_nesting() -> None:
    cfg = _to_config(
        {
            "plugins": {"attack": "advedm_plus"},
            "extra": {"advedm_plus_ablation": {"disable_text_branch": True}},
        }
    )

    assert "extra" not in cfg.extra
    assert cfg.extra["advedm_plus_ablation"]["disable_text_branch"] is True


# 验证 `to 配置 keeps unknown top level keys in extra` 场景，防止相关行为在后续修改中退化。
def test_to_config_keeps_unknown_top_level_keys_in_extra() -> None:
    cfg = _to_config(
        {
            "plugins": {"attack": "advedm_plus"},
            "custom_note": {"value": 7},
        }
    )

    assert cfg.extra["custom_note"]["value"] == 7
