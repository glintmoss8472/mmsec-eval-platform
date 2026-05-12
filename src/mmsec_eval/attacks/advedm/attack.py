from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from mmsec_eval.attacks.advedm.masks import attention_fixation_map, patch_similarity_scores, select_mask
from mmsec_eval.attacks.advedm.optimize import masked_pgd_optimize
from mmsec_eval.attacks.base import perturb_stats
from mmsec_eval.plugins.base import AttackPlugin
from mmsec_eval.types import AttackContext, AttackedSample, Sample


@dataclass(frozen=True)
class _ADVEDMModeConfig:
    variant: str
    objective: str
    mask_mode: str
    alpha: float
    beta: float
    gamma: float
    steps: int


def _mode_config(cfg, *, mode: str, steps: int) -> _ADVEDMModeConfig:
    if mode == "A":
        return _ADVEDMModeConfig(
            variant="ADVEDM-R",
            objective="remove",
            mask_mode="remove",
            alpha=float(cfg.alpha),
            beta=float(cfg.beta),
            gamma=float(cfg.gamma),
            steps=steps,
        )
    return _ADVEDMModeConfig(
        variant="ADVEDM-A",
        objective="add",
        mask_mode="add",
        alpha=float(cfg.alpha) * 1.1,
        beta=float(cfg.beta) * 0.9,
        gamma=float(cfg.gamma),
        steps=max(steps, 10),
    )


def _debug_artifacts(
    *,
    ctx: AttackContext,
    scores: np.ndarray,
    mask: np.ndarray,
    fixation: np.ndarray,
    mode: str,
    variant: str,
    target: str,
) -> tuple[str, str, str]:
    if not ctx.sample_debug_dir:
        return "", "", ""
    return _write_debug_artifacts(
        sample_debug_dir=ctx.sample_debug_dir,
        scores=scores,
        mask=mask,
        fixation=fixation,
        mode=mode,
        variant=variant,
        target_text=target,
    )


def _make_advedm_sample(sample: Sample, adv: np.ndarray, *, mode: str, variant: str, objective: str) -> Sample:
    adv_sample = Sample(
        sample_id=sample.sample_id,
        image=adv,
        text=sample.text,
        target_text=sample.target_text,
        metadata=dict(sample.metadata),
    )
    adv_sample.metadata["attack_name"] = "advedm"
    adv_sample.metadata["attack_mode"] = mode
    adv_sample.metadata["attack_variant"] = variant
    adv_sample.metadata["attack_objective"] = objective
    return adv_sample


def _attack_metadata(
    *,
    mode_cfg: _ADVEDMModeConfig,
    target: str,
    mode: str,
    mask: np.ndarray,
    patch_size: int,
    fixation: np.ndarray,
    debug_paths: tuple[str, str, str],
    loss_parts: dict[str, float],
) -> dict[str, object]:
    mask_debug_path, region_debug_path, attention_debug_path = debug_paths
    return {
        "variant": mode_cfg.variant,
        "objective": mode_cfg.objective,
        "target_text": target,
        "score_provider": "surrogate_semantic_score",
        "mode": mode,
        "mask_ratio": float((1.0 - mask).mean()),
        "selected_patches": int(((1.0 - mask).sum() / max(1, patch_size * patch_size))),
        "fixation_mean": float(np.mean(fixation)),
        "mask_debug_path": mask_debug_path,
        "region_debug_path": region_debug_path,
        "attention_debug_path": attention_debug_path,
        "loss_decomposition": loss_parts,
    }


class ADVEDMAttack(AttackPlugin):
    """AdvEDM-inspired fine-grained attack with A/B variants.

    - Mode A -> AdvEDM-R (semantic removal)
    - Mode B -> AdvEDM-A (semantic addition)
    """

    def attack(self, sample: Sample, ctx: AttackContext) -> AttackedSample:
        cfg = ctx.config.attack
        mode = str(cfg.mode).upper()
        patch_size = int(cfg.patch_size)
        topk = int(cfg.topk)
        epsilon = float(cfg.epsilon)
        step = float(cfg.step_size)
        steps = int(cfg.steps)

        image = sample.image.astype("float32")
        target = sample.target_text or "object"
        surrogate = ctx.surrogate_model_adapter or ctx.model_adapter

        scores = patch_similarity_scores(image, target, patch_size=patch_size, model_adapter=surrogate)
        fixation = attention_fixation_map(image, sample.text or target, surrogate)
        mode_cfg = _mode_config(cfg, mode=mode, steps=steps)

        mask = select_mask(
            scores,
            patch_size,
            topk,
            mode=mode_cfg.mask_mode,
            threshold=float(cfg.threshold),
            shape=image.shape[:2],
        )

        debug_paths = _debug_artifacts(
            ctx=ctx,
            scores=scores,
            mask=mask,
            fixation=fixation,
            mode=mode,
            variant=mode_cfg.variant,
            target=target,
        )

        adv, trace = masked_pgd_optimize(
            image=image,
            mask=mask,
            patch_size=patch_size,
            target_text=str(target),
            caption_text=str(sample.text or ""),
            model_adapter=surrogate,
            epsilon=epsilon,
            step_size=step,
            steps=mode_cfg.steps,
            alpha=mode_cfg.alpha,
            beta=mode_cfg.beta,
            gamma=mode_cfg.gamma,
            fixation_map=fixation,
            objective=mode_cfg.objective,
        )

        l0, l2, linf = perturb_stats(image, adv)

        return AttackedSample(
            sample=_make_advedm_sample(
                sample,
                adv,
                mode=mode,
                variant=mode_cfg.variant,
                objective=mode_cfg.objective,
            ),
            perturbation_l0=l0,
            perturbation_l2=l2,
            perturbation_linf=linf,
            attack_trace=trace,
            metadata=_attack_metadata(
                mode_cfg=mode_cfg,
                target=target,
                mode=mode,
                mask=mask,
                patch_size=patch_size,
                fixation=fixation,
                debug_paths=debug_paths,
                loss_parts=trace[-1].loss_parts if trace else {},
            ),
        )

def _write_debug_artifacts(
    sample_debug_dir: str,
    scores: np.ndarray,
    mask: np.ndarray,
    fixation: np.ndarray,
    mode: str,
    variant: str,
    target_text: str,
) -> tuple[str, str, str]:
    debug_dir = Path(sample_debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    mask_2d = mask[..., 0]
    mask_img = ((1.0 - np.clip(mask_2d, 0.0, 1.0)) * 255.0).astype(np.uint8)
    mask_path = debug_dir / "advedm_mask.png"
    Image.fromarray(mask_img).save(mask_path)

    att_img = (np.clip(fixation, 0.0, 1.0) * 255.0).astype(np.uint8)
    att_path = debug_dir / "advedm_attention.png"
    Image.fromarray(att_img).save(att_path)

    flat = scores.flatten()
    top_idx = np.argsort(flat)[-min(8, len(flat)) :].tolist()
    low_idx = np.argsort(flat)[: min(8, len(flat))].tolist()
    score_path = debug_dir / "advedm_regions.json"
    payload = {
        "mode": mode,
        "variant": variant,
        "target_text": target_text,
        "grid_shape": [int(scores.shape[0]), int(scores.shape[1])],
        "score_mean": float(scores.mean()),
        "score_std": float(scores.std()),
        "score_max": float(scores.max()),
        "score_min": float(scores.min()),
        "score_top_indices": [int(x) for x in top_idx],
        "score_low_indices": [int(x) for x in low_idx],
        "fixation_mean": float(fixation.mean()),
        "fixation_std": float(fixation.std()),
    }
    score_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(mask_path), str(score_path), str(att_path)
