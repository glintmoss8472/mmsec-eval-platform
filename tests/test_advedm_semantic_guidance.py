from __future__ import annotations

from pathlib import Path

import numpy as np

from mmsec_eval.attacks.advedm.attack import ADVEDMAttack
from mmsec_eval.config.loader import load_config
from mmsec_eval.plugins.builtin import register_builtin_plugins
from mmsec_eval.plugins.registry import create
from mmsec_eval.types import AttackContext, Sample


def test_advedm_writes_debug_and_variant(tmp_path: Path):
    register_builtin_plugins()
    img = np.zeros((64, 64, 3), dtype=np.float32)
    img[16:48, 16:48] = 0.8
    s = Sample(sample_id="s1", image=img, text="a scene", target_text="dog")

    cfg = load_config("configs/mvp.yaml")
    cfg.artifacts_dir = str(tmp_path / "artifacts")
    cfg.attack.mode = "A"
    cfg.attack.steps = 2
    cfg.attack.patch_size = 8
    cfg.attack.topk = 4

    dbg = tmp_path / "dbg"
    adapter = create("model_adapter", "clip_hf")
    ctx = AttackContext(config=cfg, model_adapter=adapter, surrogate_model_adapter=adapter, sample_debug_dir=str(dbg))

    out = ADVEDMAttack().attack(s, ctx)
    assert out.sample.metadata.get("attack_variant") == "ADVEDM-R"
    assert out.metadata.get("score_provider") == "surrogate_semantic_score"

    assert (dbg / "advedm_mask.png").exists()
    assert (dbg / "advedm_attention.png").exists()
    assert (dbg / "advedm_regions.json").exists()
