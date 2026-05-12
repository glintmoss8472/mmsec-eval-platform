from __future__ import annotations

from pathlib import Path

import numpy as np

from mmsec_eval.attacks.advedm_plus.attack import ADVEDMPlusAttack, compute_adaptive_budget
from mmsec_eval.config.loader import load_config
from mmsec_eval.types import AttackContext, Sample


class _DummyJointAdapter:
    def __init__(self) -> None:
        self._device = "cpu"

    def score_pairs(self, pairs, batch_size: int = 8):
        vals = []
        for image, text in pairs:
            vals.append(float(np.asarray(image, dtype=np.float32).mean()) + 0.05 * len(str(text).split()))
        return np.asarray(vals, dtype=np.float32)

    def attention_map(self, image: np.ndarray, text: str):
        h, w = image.shape[:2]
        grid = np.linspace(0.1, 0.9, num=h * w, dtype=np.float32).reshape(h, w)
        return grid

    def score_pairs_torch(self, images_bchw, texts, *, output_attentions: bool = False):
        import torch

        del output_attentions
        base = images_bchw.mean(dim=(1, 2, 3))
        text_term = torch.tensor([0.05 * len(str(text).split()) for text in texts], device=images_bchw.device, dtype=images_bchw.dtype)
        return base + text_term

    def patch_text_similarity_torch(self, images_bchw, texts):
        import torch.nn.functional as F

        del texts
        sim = images_bchw.mean(dim=1, keepdim=True)
        return F.interpolate(sim, size=(8, 8), mode="bilinear", align_corners=False).squeeze(1)


class _CenterFocusAdapter(_DummyJointAdapter):
    def score_pairs(self, pairs, batch_size: int = 8):
        del batch_size
        return np.asarray([0.25 + 0.01 * len(str(text).split()) for _, text in pairs], dtype=np.float32)

    def attention_map(self, image: np.ndarray, text: str):
        del text
        h, w = image.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        cy = (h - 1) / 2.0
        cx = (w - 1) / 2.0
        dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        dist = dist / max(1.0, float(dist.max()))
        return np.clip(1.0 - dist, 0.0, 1.0).astype(np.float32)

    def patch_text_similarity_torch(self, images_bchw, texts):
        del texts
        import torch

        return torch.ones((images_bchw.shape[0], 8, 8), device=images_bchw.device, dtype=images_bchw.dtype)


def test_compute_adaptive_budget_prefers_image_when_semantics_are_focused():
    focused_scores = np.zeros((4, 4), dtype=np.float32)
    focused_scores[0, 0] = 1.0
    flat_scores = np.full((4, 4), 0.5, dtype=np.float32)
    fixation = np.full((32, 32), 0.2, dtype=np.float32)

    focused = compute_adaptive_budget(
        scores=focused_scores,
        fixation=fixation,
        epsilon=0.05,
        step_size=0.01,
        steps=8,
        eps_t=1,
        alpha=10.0,
        beta=5.0,
        gamma=1.0,
        text="a focused target object",
    )
    flat = compute_adaptive_budget(
        scores=flat_scores,
        fixation=fixation,
        epsilon=0.05,
        step_size=0.01,
        steps=8,
        eps_t=1,
        alpha=10.0,
        beta=5.0,
        gamma=1.0,
        text="a focused target object",
    )

    assert focused["image_scale"] > flat["image_scale"]
    assert focused["epsilon"] > flat["epsilon"]


def test_compute_adaptive_budget_rewards_fixation_focus():
    scores = np.zeros((4, 4), dtype=np.float32)
    scores[1:3, 1:3] = 1.0
    focused_fixation = np.zeros((32, 32), dtype=np.float32)
    focused_fixation[8:24, 8:24] = 1.0
    flat_fixation = np.full((32, 32), 0.15, dtype=np.float32)

    focused = compute_adaptive_budget(
        scores=scores,
        fixation=focused_fixation,
        epsilon=0.05,
        step_size=0.01,
        steps=8,
        eps_t=1,
        alpha=10.0,
        beta=5.0,
        gamma=1.0,
        text="target object in the center",
    )
    flat = compute_adaptive_budget(
        scores=scores,
        fixation=flat_fixation,
        epsilon=0.05,
        step_size=0.01,
        steps=8,
        eps_t=1,
        alpha=10.0,
        beta=5.0,
        gamma=1.0,
        text="target object in the center",
    )

    assert focused["beta"] > flat["beta"]
    assert focused["steps"] >= flat["steps"]
    assert focused["epsilon"] >= flat["epsilon"]


def test_compute_adaptive_budget_can_expand_text_budget():
    scores = np.zeros((4, 4), dtype=np.float32)
    scores[1:3, 1:3] = 1.0
    fixation = np.zeros((32, 32), dtype=np.float32)
    fixation[8:24, 8:24] = 1.0

    budget = compute_adaptive_budget(
        scores=scores,
        fixation=fixation,
        epsilon=0.05,
        step_size=0.01,
        steps=8,
        eps_t=1,
        alpha=10.0,
        beta=5.0,
        gamma=1.0,
        text="red square target object centered in the frame",
    )

    assert budget["eps_t"] >= 2.0


def test_advedm_plus_joint_attack_writes_debug_and_changes_text(tmp_path: Path):
    cfg = load_config("configs/mvp.yaml")
    cfg.artifacts_dir = str(tmp_path / "artifacts")
    cfg.attack.steps = 1
    cfg.attack.patch_size = 8
    cfg.attack.topk = 4
    cfg.attack.eps_t = 1
    cfg.task.kind = "pairwise"

    image = np.zeros((64, 64, 3), dtype=np.float32)
    image[16:48, 16:48] = 0.8
    sample = Sample(sample_id="joint-1", image=image, text="red square target object", target_text="target")

    debug_dir = tmp_path / "debug"
    adapter = _DummyJointAdapter()
    ctx = AttackContext(config=cfg, model_adapter=adapter, surrogate_model_adapter=adapter, sample_debug_dir=str(debug_dir))

    out = ADVEDMPlusAttack().attack(sample, ctx)

    assert out.sample.metadata["attack_variant"] == "ADVEDM+"
    assert out.sample.metadata["attack_scope"] == "joint"
    assert out.metadata["adaptive_budget"]["epsilon"] > 0
    assert out.metadata["adaptive_budget"]["steps"] >= 1
    assert out.metadata["text_edit"]["num_edits"] >= 1
    assert out.sample.text != sample.text
    assert (debug_dir / "advedm_plus_mask.png").exists()
    assert (debug_dir / "advedm_plus_attention.png").exists()
    assert (debug_dir / "advedm_plus_debug.json").exists()


def test_advedm_plus_adaptive_budget_is_stronger_than_fixed_budget(tmp_path: Path):
    cfg = load_config("configs/mvp.yaml")
    cfg.artifacts_dir = str(tmp_path / "artifacts")
    cfg.attack.steps = 2
    cfg.attack.patch_size = 8
    cfg.attack.topk = 4
    cfg.attack.eps_t = 1
    cfg.task.kind = "pairwise"

    image = np.zeros((64, 64, 3), dtype=np.float32)
    image[16:48, 16:48] = 0.8
    sample = Sample(sample_id="joint-budget", image=image, text="red square target object", target_text="target")
    adapter = _CenterFocusAdapter()

    full = ADVEDMPlusAttack().attack(
        sample,
        AttackContext(config=cfg, model_adapter=adapter, surrogate_model_adapter=adapter, sample_debug_dir=str(tmp_path / "full")),
    )

    cfg_fixed = load_config("configs/mvp.yaml")
    cfg_fixed.artifacts_dir = str(tmp_path / "artifacts")
    cfg_fixed.attack.steps = 2
    cfg_fixed.attack.patch_size = 8
    cfg_fixed.attack.topk = 4
    cfg_fixed.attack.eps_t = 1
    cfg_fixed.task.kind = "pairwise"
    cfg_fixed.extra = {"advedm_plus_ablation": {"disable_adaptive_budget": True}}
    fixed = ADVEDMPlusAttack().attack(
        sample,
        AttackContext(config=cfg_fixed, model_adapter=adapter, surrogate_model_adapter=adapter, sample_debug_dir=str(tmp_path / "fixed")),
    )

    assert full.metadata["adaptive_budget"]["epsilon"] >= fixed.metadata["adaptive_budget"]["epsilon"]
    assert full.metadata["adaptive_budget"]["steps"] >= fixed.metadata["adaptive_budget"]["steps"]


def test_advedm_plus_fixation_guides_perturbation_into_center(tmp_path: Path):
    cfg = load_config("configs/mvp.yaml")
    cfg.artifacts_dir = str(tmp_path / "artifacts")
    cfg.attack.steps = 3
    cfg.attack.patch_size = 8
    cfg.attack.topk = 8
    cfg.task.kind = "pairwise"

    image = np.zeros((64, 64, 3), dtype=np.float32)
    image[16:48, 16:48] = 0.8
    sample = Sample(sample_id="joint-fix", image=image, text="red square target object", target_text="target")
    adapter = _CenterFocusAdapter()

    full = ADVEDMPlusAttack().attack(
        sample,
        AttackContext(config=cfg, model_adapter=adapter, surrogate_model_adapter=adapter, sample_debug_dir=str(tmp_path / "full-fix")),
    )

    cfg_no_fix = load_config("configs/mvp.yaml")
    cfg_no_fix.artifacts_dir = str(tmp_path / "artifacts")
    cfg_no_fix.attack.steps = 3
    cfg_no_fix.attack.patch_size = 8
    cfg_no_fix.attack.topk = 8
    cfg_no_fix.task.kind = "pairwise"
    cfg_no_fix.extra = {"advedm_plus_ablation": {"disable_fixation_constraint": True}}
    no_fix = ADVEDMPlusAttack().attack(
        sample,
        AttackContext(config=cfg_no_fix, model_adapter=adapter, surrogate_model_adapter=adapter, sample_debug_dir=str(tmp_path / "no-fix")),
    )

    full_delta = np.abs(full.sample.image - image)
    no_fix_delta = np.abs(no_fix.sample.image - image)
    full_center_ratio = float(full_delta[16:48, 16:48].mean() / (full_delta.mean() + 1e-8))
    no_fix_center_ratio = float(no_fix_delta[16:48, 16:48].mean() / (no_fix_delta.mean() + 1e-8))

    assert full_center_ratio > no_fix_center_ratio
    assert full.metadata["adaptive_budget"]["epsilon"] >= no_fix.metadata["adaptive_budget"]["epsilon"]
    assert full.metadata["adaptive_budget"]["steps"] >= no_fix.metadata["adaptive_budget"]["steps"]
    assert int(full.metadata["adaptive_topk"]) >= int(no_fix.metadata["adaptive_topk"])
