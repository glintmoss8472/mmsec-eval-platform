# 文件说明：该文件属于自动化测试，集中实现 test attack modes 相关逻辑。
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mmsec_eval.attacks.advedm.attack import ADVEDMAttack
from mmsec_eval.attacks.advedm_plus.attack import ADVEDMPlusAttack
from mmsec_eval.attacks.advclip.attack import AdvCLIPPatchAttack
from mmsec_eval.attacks.advclip.train import train_advclip_patch
from mmsec_eval.attacks.tmm.attack import TMMAttack
from mmsec_eval.config.loader import load_config
from mmsec_eval.plugins.builtin import register_builtin_plugins
from mmsec_eval.plugins.registry import create
from mmsec_eval.types import AttackContext, Sample


# 中文注释：封装 _cuda_available 的内部步骤，让自动化测试主流程保持清晰并隔离边界细节。
def _cuda_available() -> bool:
    try:
        import torch

        return bool(getattr(torch.version, "cuda", None) is not None and torch.cuda.is_available())
    except (ImportError, OSError, RuntimeError):
        return False


pytestmark = pytest.mark.skipif(not _cuda_available(), reason="attack mode integration tests require CUDA")


# 中文注释：封装 _sample 的内部步骤，让自动化测试主流程保持清晰并隔离边界细节。
def _sample() -> Sample:
    img = np.zeros((64, 64, 3), dtype=np.float32)
    img[16:48, 16:48] = 0.8
    return Sample(sample_id="s1", image=img, text="a blue square", target_text="circle")


# 中文注释：封装 _run_attack 的内部步骤，让自动化测试主流程保持清晰并隔离边界细节。
def _run_attack(plugin, mode: str, *, artifacts_dir: str):
    register_builtin_plugins()
    cfg = load_config("configs/mvp.yaml")
    cfg.artifacts_dir = str(artifacts_dir)
    cfg.attack.mode = mode
    cfg.attack.steps = 3
    cfg.attack.patch_size = 8
    # Keep tests offline-fast: do not run BERT-based token replacement.
    cfg.attack.eps_t = 0

    # Ensure AdvCLIP has a trained patch (no random-init fallback).
    if isinstance(plugin, AdvCLIPPatchAttack):
        cfg.attack.patch_train_steps = 5
        train_advclip_patch(cfg)

    adapter = create("model_adapter", "clip_hf")
    ctx = AttackContext(config=cfg, model_adapter=adapter, surrogate_model_adapter=adapter)
    return plugin.attack(_sample(), ctx)


# 中文注释：验证 test_advedm_a_b_modes 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_advedm_a_b_modes(tmp_path: Path):
    p = ADVEDMAttack()
    a = _run_attack(p, "A", artifacts_dir=str(tmp_path / "artifacts"))
    b = _run_attack(p, "B", artifacts_dir=str(tmp_path / "artifacts"))
    assert a.sample.metadata["attack_mode"] == "A"
    assert b.sample.metadata["attack_mode"] == "B"
    assert len(a.attack_trace) > 0
    assert len(b.attack_trace) > 0


# 中文注释：验证 test_advedm_plus_a_b_modes 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_advedm_plus_a_b_modes(tmp_path: Path):
    p = ADVEDMPlusAttack()
    a = _run_attack(p, "A", artifacts_dir=str(tmp_path / "artifacts"))
    b = _run_attack(p, "B", artifacts_dir=str(tmp_path / "artifacts"))
    assert a.sample.metadata["attack_mode"] == "A"
    assert b.sample.metadata["attack_mode"] == "B"
    assert len(a.attack_trace) > 0
    assert len(b.attack_trace) > 0


# 中文注释：验证 test_advclip_a_b_modes 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_advclip_a_b_modes(tmp_path: Path):
    p = AdvCLIPPatchAttack()
    a = _run_attack(p, "A", artifacts_dir=str(tmp_path / "artifacts"))
    b = _run_attack(p, "B", artifacts_dir=str(tmp_path / "artifacts"))
    assert a.sample.metadata["attack_mode"] == "A"
    assert b.sample.metadata["attack_mode"] == "B"
    assert len(a.attack_trace) > 0
    assert len(b.attack_trace) > 0


# 中文注释：验证 test_tmm_a_b_modes 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_tmm_a_b_modes(tmp_path: Path):
    p = TMMAttack()
    a = _run_attack(p, "A", artifacts_dir=str(tmp_path / "artifacts"))
    b = _run_attack(p, "B", artifacts_dir=str(tmp_path / "artifacts"))
    assert a.sample.metadata["attack_mode"] == "A"
    assert b.sample.metadata["attack_mode"] == "B"
    assert len(a.attack_trace) > 0
    assert len(b.attack_trace) > 0
