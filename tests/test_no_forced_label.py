# 文件说明：该文件属于自动化测试，集中实现 test no forced label 相关逻辑。
from __future__ import annotations

import numpy as np

from mmsec_eval.attacks.advedm.attack import ADVEDMAttack
from mmsec_eval.attacks.advclip.attack import AdvCLIPPatchAttack
from mmsec_eval.attacks.advclip.train import train_advclip_patch
from mmsec_eval.attacks.tmm.attack import TMMAttack
from mmsec_eval.config.loader import load_config
from mmsec_eval.plugins.builtin import register_builtin_plugins
from mmsec_eval.plugins.registry import create
from mmsec_eval.types import AttackContext, Sample


# 中文注释：封装 _sample 的内部步骤，让自动化测试主流程保持清晰并隔离边界细节。
def _sample() -> Sample:
    img = np.zeros((32, 32, 3), dtype=np.float32)
    img[8:24, 8:24] = 0.9
    return Sample(sample_id="s1", image=img, text="a blue square", target_text="circle")


# 中文注释：验证 test_attacks_do_not_inject_forced_label 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_attacks_do_not_inject_forced_label(tmp_path):
    register_builtin_plugins()
    cfg = load_config("configs/mvp.yaml")
    cfg.artifacts_dir = str(tmp_path / "artifacts")
    cfg.attack.steps = 2
    cfg.attack.patch_size = 8
    cfg.attack.eps_t = 0

    # Ensure AdvCLIP patch exists (no random-init fallback).
    cfg.attack.patch_train_steps = 5
    train_advclip_patch(cfg)

    adapter = create("model_adapter", "clip_hf")
    ctx = AttackContext(config=cfg, model_adapter=adapter, surrogate_model_adapter=adapter)

    for plugin in (ADVEDMAttack(), AdvCLIPPatchAttack(), TMMAttack()):
        attacked = plugin.attack(_sample(), ctx)
        assert "forced_label" not in attacked.sample.metadata
