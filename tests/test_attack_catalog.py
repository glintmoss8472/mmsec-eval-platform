from __future__ import annotations

from mmsec_eval.attacks.catalog import (
    CLASSIC_GRADIENT_ATTACKS,
    JOINT_TEXT_ATTACKS,
    EXTERNAL_ATTACKS,
    EXTERNAL_ATTACK_METADATA,
    PAPER_ATTACKS,
    attack_surrogate_error,
)
from mmsec_eval.plugins.builtin import register_builtin_plugins
from mmsec_eval.plugins.registry import list_plugins


def test_attack_surrogate_policy_rejects_incompatible_surrogates() -> None:
    assert attack_surrogate_error("fgsm", "clip_hf") is None
    assert attack_surrogate_error("fgsm", "openai_qwen25_vl")
    assert attack_surrogate_error("tmm", "vilt_itm") is None
    assert attack_surrogate_error("tmm", "openai_qwen3_vl")
    assert attack_surrogate_error("advedm_plus", "clip_hf") is None
    assert attack_surrogate_error("advedm_plus", "blip_itm")
    assert attack_surrogate_error("advclip", "openai_qwen3_vl") is None


def test_attack_catalog_matches_builtin_registration() -> None:
    register_builtin_plugins()
    registered_attacks = set(list_plugins("attack"))
    assert CLASSIC_GRADIENT_ATTACKS.issubset(registered_attacks)
    assert PAPER_ATTACKS.issubset(registered_attacks)
    assert EXTERNAL_ATTACKS.issubset(registered_attacks)
    assert JOINT_TEXT_ATTACKS == {"tmm", "advedm_plus"}


def test_external_attack_metadata_is_synchronized() -> None:
    register_builtin_plugins()
    registered_attacks = set(list_plugins("attack"))
    for attack_id in {"vqa_visual_corruption", "xtransfer_uap", "foa_attack", "anyattack", "mpc_attack", "m_attack"}:
        assert attack_id in EXTERNAL_ATTACK_METADATA
        assert attack_id in registered_attacks
        meta = EXTERNAL_ATTACK_METADATA[attack_id]
        assert meta.attack_scope == "image"
        assert meta.recommended_tasks
    assert EXTERNAL_ATTACK_METADATA["foa_attack"].requires_external_repo is True
    assert EXTERNAL_ATTACK_METADATA["vqa_visual_corruption"].requires_external_repo is True
    assert EXTERNAL_ATTACK_METADATA["vqa_visual_corruption"].requires_checkpoint is False
    assert EXTERNAL_ATTACK_METADATA["anyattack"].requires_checkpoint is True
    assert EXTERNAL_ATTACK_METADATA["mpc_attack"].requires_checkpoint is False
    assert EXTERNAL_ATTACK_METADATA["mpc_attack"].requires_target_image is True
