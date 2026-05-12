import pytest

from mmsec_eval.exceptions import PluginNotFoundError
from mmsec_eval.plugins.builtin import register_builtin_plugins
from mmsec_eval.plugins.registry import create, list_plugins


def test_registry_builtin_create():
    register_builtin_plugins()
    adapters = set(list_plugins("model_adapter"))
    assert "clip_hf" in adapters
    assert "blip_itm" in adapters
    assert "vilt_itm" in adapters
    assert "openai_compat" in adapters
    assert "openai_gpt4o" in adapters
    assert "openai_qwen35_9b" in adapters
    assert "openai_qwen3_vl" in adapters
    assert "openai_qwen25_vl" in adapters
    assert "openai_internvl35" in adapters
    assert "openai_minicpm_v" in adapters
    assert "openai_ovis25" in adapters
    assert "openai_gemma3_12b" in adapters
    assert "openai_qwen2_vl" not in adapters
    assert "gemini_vision" in adapters
    assert "dummy" not in adapters
    assert "toy_torch" not in adapters
    obj = create("attack", "advedm")
    assert obj is not None
    obj_plus = create("attack", "advedm_plus")
    assert obj_plus is not None
    assert create("attack", "fgsm") is not None
    assert create("attack", "pgd") is not None
    assert create("attack", "cw") is not None
    for attack_id in ("vqa_visual_corruption", "xtransfer_uap", "foa_attack", "anyattack", "mpc_attack", "m_attack"):
        assert create("attack", attack_id) is not None


def test_registry_missing():
    with pytest.raises(PluginNotFoundError):
        create("attack", "missing_attack")
