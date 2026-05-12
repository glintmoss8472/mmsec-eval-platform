# 文件说明：该文件属于自动化测试，集中实现 test advclip patch registry 相关逻辑。
from __future__ import annotations

import numpy as np
from pathlib import Path

from mmsec_eval.attacks.advclip.attack import AdvCLIPPatchAttack
from mmsec_eval.attacks.advclip.registry import make_key, read_registry, resolve_patch
from mmsec_eval.attacks.advclip.train import train_advclip_patch
from mmsec_eval.config.loader import load_config
from mmsec_eval.config.validate import validate_config
from mmsec_eval.plugins.builtin import register_builtin_plugins
from mmsec_eval.plugins.registry import create
from mmsec_eval.types import AttackContext, Sample


# 中文注释：验证 test_advclip_patch_registry_roundtrip 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_advclip_patch_registry_roundtrip(tmp_path: Path) -> None:
    register_builtin_plugins()

    cfg = load_config("configs/mvp.yaml")
    cfg.artifacts_dir = str(tmp_path / "artifacts")
    cfg.plugins.model_adapter = "clip_hf"
    cfg.attack.mode = "A"
    cfg.attack.patch_size = 32
    cfg.attack.use_gan = False
    cfg.attack.gan_steps = 1
    cfg.attack.patch_train_steps = 5
    cfg.dataset.kind = "toy_shapes"
    cfg.dataset.num_samples = 2
    cfg.runner.max_samples = 2
    cfg.runner.continue_on_error = False
    validate_config(cfg)

    out = train_advclip_patch(cfg)
    train_patch = Path(out.run_dir) / "attack_debug" / "advclip_patch_A_32.npy"
    assert train_patch.exists()

    reg_key = make_key(
        clip_model_name=str(cfg.model.clip_model_name),
        mode=str(cfg.attack.mode),
        patch_size=int(cfg.attack.patch_size),
    )
    reg = read_registry(str(cfg.artifacts_dir))
    assert reg_key in reg.get("entries", {})
    resolved = resolve_patch(str(cfg.artifacts_dir), reg_key)
    assert resolved

    # New run directory: should not already have patch.
    eval_run_dir = tmp_path / "artifacts" / "runs" / "eval_run"
    eval_patch = eval_run_dir / "attack_debug" / "advclip_patch_A_32.npy"
    assert not eval_patch.exists()

    attack = AdvCLIPPatchAttack()
    sample = Sample(
        sample_id="s1",
        image=np.zeros((64, 64, 3), dtype=np.float32),
        text="a cat sitting on a chair",
    )
    ctx = AttackContext(config=cfg, model_adapter=create("model_adapter", "clip_hf"), run_dir=str(eval_run_dir))
    attacked = attack.attack(sample, ctx)

    assert attacked.metadata.get("patch_source") == "registry"
    assert attacked.metadata.get("registry_key") == reg_key
    assert eval_patch.exists()

    # The copied patch should match the registry-resolved patch.
    a = np.load(resolved)
    b = np.load(eval_patch)
    assert a.shape == b.shape
    assert np.allclose(a, b)
