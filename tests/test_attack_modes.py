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


# 执行 `CUDA available` 辅助逻辑，保持自动化测试中的输入处理和结果输出一致。
def _cuda_available() -> bool:
    try:
        import torch

        return bool(getattr(torch.version, "cuda", None) is not None and torch.cuda.is_available())
    except (ImportError, OSError, RuntimeError):
        return False


pytestmark = pytest.mark.skipif(not _cuda_available(), reason="attack mode integration tests require CUDA")


# 执行 `样本` 辅助逻辑，保持自动化测试中的输入处理和结果输出一致。
def _sample() -> Sample:
    img = np.zeros((64, 64, 3), dtype=np.float32)
    img[16:48, 16:48] = 0.8
    return Sample(sample_id="s1", image=img, text="a blue square", target_text="circle")


# 执行 `攻击` 流程，按配置驱动自动化测试完成一次任务。
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


# 验证 `advedm a b modes` 场景，防止相关行为在后续修改中退化。
def test_advedm_a_b_modes(tmp_path: Path):
    p = ADVEDMAttack()
    a = _run_attack(p, "A", artifacts_dir=str(tmp_path / "artifacts"))
    b = _run_attack(p, "B", artifacts_dir=str(tmp_path / "artifacts"))
    assert a.sample.metadata["attack_mode"] == "A"
    assert b.sample.metadata["attack_mode"] == "B"
    assert len(a.attack_trace) > 0
    assert len(b.attack_trace) > 0


# 验证 `advedm plus a b modes` 场景，防止相关行为在后续修改中退化。
def test_advedm_plus_a_b_modes(tmp_path: Path):
    p = ADVEDMPlusAttack()
    a = _run_attack(p, "A", artifacts_dir=str(tmp_path / "artifacts"))
    b = _run_attack(p, "B", artifacts_dir=str(tmp_path / "artifacts"))
    assert a.sample.metadata["attack_mode"] == "A"
    assert b.sample.metadata["attack_mode"] == "B"
    assert len(a.attack_trace) > 0
    assert len(b.attack_trace) > 0


# 验证 `advclip a b modes` 场景，防止相关行为在后续修改中退化。
def test_advclip_a_b_modes(tmp_path: Path):
    p = AdvCLIPPatchAttack()
    a = _run_attack(p, "A", artifacts_dir=str(tmp_path / "artifacts"))
    b = _run_attack(p, "B", artifacts_dir=str(tmp_path / "artifacts"))
    assert a.sample.metadata["attack_mode"] == "A"
    assert b.sample.metadata["attack_mode"] == "B"
    assert len(a.attack_trace) > 0
    assert len(b.attack_trace) > 0


# 验证 `tmm a b modes` 场景，防止相关行为在后续修改中退化。
def test_tmm_a_b_modes(tmp_path: Path):
    p = TMMAttack()
    a = _run_attack(p, "A", artifacts_dir=str(tmp_path / "artifacts"))
    b = _run_attack(p, "B", artifacts_dir=str(tmp_path / "artifacts"))
    assert a.sample.metadata["attack_mode"] == "A"
    assert b.sample.metadata["attack_mode"] == "B"
    assert len(a.attack_trace) > 0
    assert len(b.attack_trace) > 0
