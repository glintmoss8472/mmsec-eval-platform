from __future__ import annotations

from mmsec_eval.config.loader import _to_config


def test_to_config_preserves_extra_without_double_nesting() -> None:
    cfg = _to_config(
        {
            "plugins": {"attack": "advedm_plus"},
            "extra": {"advedm_plus_ablation": {"disable_text_branch": True}},
        }
    )

    assert "extra" not in cfg.extra
    assert cfg.extra["advedm_plus_ablation"]["disable_text_branch"] is True


def test_to_config_keeps_unknown_top_level_keys_in_extra() -> None:
    cfg = _to_config(
        {
            "plugins": {"attack": "advedm_plus"},
            "custom_note": {"value": 7},
        }
    )

    assert cfg.extra["custom_note"]["value"] == 7
