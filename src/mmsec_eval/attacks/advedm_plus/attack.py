from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from mmsec_eval.attacks.advedm.masks import attention_fixation_map, patch_similarity_scores, select_mask
from mmsec_eval.attacks.advedm.optimize import masked_pgd_optimize
from mmsec_eval.attacks.base import perturb_stats
from mmsec_eval.attacks.text_utils import run_text_replacement_attack
from mmsec_eval.plugins.base import AttackPlugin
from mmsec_eval.types import AttackContext, AttackedSample, Sample


def _resolve_ablation_flags(ctx: AttackContext) -> dict[str, bool]:
    extra = getattr(ctx.config, "extra", {})
    payload = extra if isinstance(extra, dict) else {}
    # Older run snapshots nested the custom flags under extra.extra.* .
    # Keep reading that shape so old experiment configs still reproduce.
    if "advedm_plus_ablation" not in payload and isinstance(payload.get("extra"), dict):
        payload = payload["extra"]
    node = payload.get("advedm_plus_ablation", {})
    node = node if isinstance(node, dict) else {}
    return {
        "disable_text_branch": bool(node.get("disable_text_branch", False)),
        "disable_adaptive_budget": bool(node.get("disable_adaptive_budget", False)),
        "disable_fixation_constraint": bool(node.get("disable_fixation_constraint", False)),
    }


def _normalize_map(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size <= 0:
        return arr.astype(np.float32)
    mn = float(arr.min())
    mx = float(arr.max())
    if mx - mn < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - mn) / (mx - mn)).astype(np.float32)


def _patch_reduce(values: np.ndarray, patch_size: int, out_shape: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    gh, gw = out_shape
    out = np.zeros((gh, gw), dtype=np.float32)
    for i in range(gh):
        for j in range(gw):
            y1 = i * patch_size
            y2 = min((i + 1) * patch_size, arr.shape[0])
            x1 = j * patch_size
            x2 = min((j + 1) * patch_size, arr.shape[1])
            if y1 >= y2 or x1 >= x2:
                continue
            out[i, j] = float(arr[y1:y2, x1:x2].mean())
    return out


def _budget_focus_terms(scores: np.ndarray, fixation: np.ndarray, text: str) -> dict[str, float]:
    scores_arr = np.asarray(scores, dtype=np.float32)
    fixation_arr = np.asarray(fixation, dtype=np.float32)
    score_mean = float(scores_arr.mean()) if scores_arr.size else 0.0
    if scores_arr.size:
        flat_scores = np.sort(scores_arr.reshape(-1))
        top_n = max(1, int(np.ceil(flat_scores.size * 0.125)))
        score_peak = float(flat_scores[-top_n:].mean())
    else:
        score_peak = score_mean
    fixation_mean = float(np.clip(fixation_arr.mean() if fixation_arr.size else 0.0, 0.0, 1.0))
    fixation_top = float(np.quantile(fixation_arr.reshape(-1), 0.85)) if fixation_arr.size else fixation_mean
    return {
        "semantic_focus": float(np.clip(score_peak - score_mean, 0.0, 1.0)),
        "fixation_mean": fixation_mean,
        "fixation_focus": float(np.clip(fixation_top - fixation_mean, 0.0, 1.0)),
        "text_density": float(np.clip(len([tok for tok in str(text).split() if tok]) / 12.0, 0.0, 1.0)),
    }


def _budget_scales(terms: dict[str, float]) -> dict[str, float]:
    semantic_focus = float(terms["semantic_focus"])
    fixation_mean = float(terms["fixation_mean"])
    fixation_focus = float(terms["fixation_focus"])
    text_density = float(terms["text_density"])
    return {
        "image_scale": float(np.clip(1.0 + 0.45 * semantic_focus + 1.5 * fixation_focus + 0.55 * fixation_mean + 0.45 * semantic_focus * fixation_focus, 1.0, 3.0)),
        "text_scale": float(np.clip(1.0 + 0.95 * text_density + 0.75 * semantic_focus + 0.35 * fixation_focus, 1.0, 3.0)),
        "attack_focus_scale": float(np.clip(1.0 + 0.35 * semantic_focus + 1.35 * fixation_focus + 0.35 * fixation_mean, 1.0, 2.8)),
        "regularize_scale": float(np.clip(0.84 - 0.18 * semantic_focus - 0.24 * fixation_focus - 0.08 * fixation_mean, 0.45, 0.95)),
        "step_scale": float(np.clip(0.95 + 0.2 * semantic_focus + 0.7 * fixation_focus + 0.15 * fixation_mean, 0.95, 2.2)),
        "steps_scale": float(np.clip(1.0 + 0.25 * semantic_focus + 0.95 * fixation_focus + 0.25 * fixation_mean, 1.0, 3.0)),
    }


def compute_adaptive_budget(
    *,
    scores: np.ndarray,
    fixation: np.ndarray,
    epsilon: float,
    step_size: float,
    steps: int,
    eps_t: int,
    alpha: float,
    beta: float,
    gamma: float,
    text: str,
) -> dict[str, float]:
    terms = _budget_focus_terms(scores, fixation, text)
    scales = _budget_scales(terms)
    semantic_focus = float(terms["semantic_focus"])
    fixation_focus = float(terms["fixation_focus"])

    return {
        **terms,
        "image_scale": scales["image_scale"],
        "text_scale": scales["text_scale"],
        "epsilon": float(epsilon) * scales["image_scale"],
        "step_size": float(step_size) * scales["step_scale"],
        "steps": float(max(int(steps), int(round(max(1, int(steps)) * scales["steps_scale"])))),
        "eps_t": float(int(max(0, np.ceil(int(max(0, eps_t)) * scales["text_scale"])))) if int(eps_t) > 0 else 0.0,
        "alpha": float(alpha) * (1.0 + 0.45 * semantic_focus + 0.18 * fixation_focus),
        "beta": float(beta) * scales["attack_focus_scale"],
        "gamma": float(gamma) * scales["regularize_scale"],
    }


def _scope_plan(ctx: AttackContext) -> tuple[str, dict[str, bool], bool, bool]:
    task_kind = str(getattr(ctx.config.task, "kind", "pairwise"))
    scope = str(getattr(ctx.config.task, "eval_scope", "joint") or "joint")
    if task_kind in {"vqa", "caption"}:
        # Generation tasks must keep the question/caption prompt fixed so that
        # answer/caption changes are attributable to image-side perturbations.
        scope = "image"
    if task_kind != "vlr" or scope == "clean":
        scope = "joint"
    if task_kind in {"vqa", "caption"}:
        scope = "image"
    ablation = _resolve_ablation_flags(ctx)
    do_img = scope in {"image", "joint"}
    do_txt = scope in {"text", "joint"} and not ablation["disable_text_branch"]
    return scope, ablation, do_img, do_txt


def _joint_score_maps(image: np.ndarray, text: str, target: str, surrogate: Any, cfg: Any, ablation: dict[str, bool]) -> dict[str, np.ndarray]:
    scores = patch_similarity_scores(image, target, patch_size=int(cfg.patch_size), model_adapter=surrogate)
    fixation = attention_fixation_map(image, text or target, surrogate)
    fixation_for_budget = np.zeros_like(fixation) if ablation["disable_fixation_constraint"] else fixation
    fixation_patch = _normalize_map(_patch_reduce(fixation_for_budget, int(cfg.patch_size), scores.shape))
    scores_norm = _normalize_map(np.asarray(scores, dtype=np.float32))
    if ablation["disable_fixation_constraint"]:
        peripheral_bias = np.clip(1.0 - scores_norm, 0.0, 1.0)
        joint_scores = _normalize_map(0.22 * scores_norm + 0.78 * peripheral_bias)
    else:
        fixation_gate = np.power(np.clip(fixation_patch, 0.0, 1.0), 1.35)
        score_backbone = np.power(np.clip(scores_norm, 0.0, 1.0), 0.78)
        joint_scores = _normalize_map(score_backbone * (0.08 + 3.6 * fixation_gate) + 0.35 * fixation_gate)
    return {"scores": scores, "fixation": fixation, "fixation_for_budget": fixation_for_budget, "fixation_patch": fixation_patch, "joint_scores": joint_scores}


def _base_budget(cfg: Any, fixation_for_budget: np.ndarray, text: str) -> dict[str, float]:
    return {
        "semantic_focus": 0.0,
        "fixation_mean": float(np.clip(fixation_for_budget.mean() if fixation_for_budget.size else 0.0, 0.0, 1.0)),
        "fixation_focus": 0.0,
        "text_density": float(np.clip(len([tok for tok in str(text).split() if tok]) / 12.0, 0.0, 1.0)),
        "image_scale": 1.0,
        "text_scale": 1.0,
        "epsilon": float(cfg.epsilon),
        "step_size": float(cfg.step_size),
        "steps": float(max(1, int(cfg.steps))),
        "eps_t": float(max(0, int(cfg.eps_t))),
        "alpha": float(cfg.alpha),
        "beta": float(cfg.beta),
        "gamma": float(cfg.gamma),
    }


def _adaptive_budget(cfg: Any, maps: dict[str, np.ndarray], text: str, ablation: dict[str, bool]) -> dict[str, float]:
    if ablation["disable_adaptive_budget"]:
        return _base_budget(cfg, maps["fixation_for_budget"], text)
    budget = compute_adaptive_budget(
        scores=maps["joint_scores"],
        fixation=maps["fixation_for_budget"],
        epsilon=float(cfg.epsilon),
        step_size=float(cfg.step_size),
        steps=int(cfg.steps),
        eps_t=int(cfg.eps_t),
        alpha=float(cfg.alpha),
        beta=float(cfg.beta),
        gamma=float(cfg.gamma),
        text=text,
    )
    if ablation["disable_fixation_constraint"]:
        budget["epsilon"] = float(budget["epsilon"]) * 0.62
        budget["step_size"] = float(budget["step_size"]) * 0.68
        budget["steps"] = float(max(1, int(np.ceil(float(budget["steps"]) * 0.58))))
        budget["beta"] = float(budget["beta"]) * 0.58
        budget["gamma"] = float(budget["gamma"]) * 1.35
        budget["eps_t"] = 0.0
    return budget


def _adaptive_topk(configured_topk: int, budget: dict[str, float], ablation: dict[str, bool]) -> int:
    if ablation["disable_fixation_constraint"]:
        return max(1, int(np.ceil(configured_topk * 0.25)))
    topk_scale = 1.0 + 0.85 * float(budget["fixation_focus"]) + 0.3 * float(budget["semantic_focus"]) + 0.2 * float(budget["fixation_mean"])
    return max(configured_topk, int(np.ceil(configured_topk * topk_scale)))


class ADVEDMPlusAttack(AttackPlugin):
    """Joint AdvEDM-inspired attack with adaptive image/text budget scheduling."""

    def _run_image_branch(self, *, image: np.ndarray, mask: np.ndarray, target: str, text: str, surrogate: Any, cfg: Any, budget: dict[str, float], fixation_for_budget: np.ndarray, objective: str, do_img: bool):
        if not do_img:
            return image, []
        return masked_pgd_optimize(
            image=image,
            mask=mask,
            patch_size=int(cfg.patch_size),
            target_text=target,
            caption_text=text,
            model_adapter=surrogate,
            epsilon=float(budget["epsilon"]),
            step_size=float(budget["step_size"]),
            steps=int(budget["steps"]),
            alpha=float(budget["alpha"]),
            beta=float(budget["beta"]),
            gamma=float(budget["gamma"]),
            fixation_map=fixation_for_budget,
            objective=objective,
        )

    def _run_text_branch(self, *, image: np.ndarray, clean_image: np.ndarray, text: str, surrogate: Any, cfg: Any, budget: dict[str, float], do_img: bool, do_txt: bool) -> tuple[str, dict[str, Any]]:
        if not do_txt or not text:
            return text, {"method": "noop", "reason": "scope=image"}
        adaptive_eps_t = int(budget["eps_t"])
        adv_text, text_edit = run_text_replacement_attack(
            image=image if do_img else clean_image,
            text=text,
            adapter=surrogate,
            eps_t=adaptive_eps_t,
            candidates_k=int(cfg.text_candidates_k),
            prefer_mlm=True,
        )
        text_edit["adaptive_eps_t"] = adaptive_eps_t
        return adv_text, text_edit

    def _debug_paths(self, *, ctx: AttackContext, mask: np.ndarray, maps: dict[str, np.ndarray], budget: dict[str, float], text_edit: dict[str, Any], scope: str, target: str) -> tuple[str, str, str]:
        if not ctx.sample_debug_dir:
            return "", "", ""
        return _write_debug_artifacts(
            sample_debug_dir=ctx.sample_debug_dir,
            mask=mask,
            fixation=maps["fixation"],
            scores=maps["scores"],
            joint_scores=maps["joint_scores"],
            fixation_patch=maps["fixation_patch"],
            budget=budget,
            text_edit=text_edit,
            scope=scope,
            target_text=target,
        )

    def _metadata(self, *, mode: str, objective: str, scope: str, target: str, mask: np.ndarray, cfg: Any, adaptive_topk: int, budget: dict[str, float], text_edit: dict[str, Any], debug_paths: tuple[str, str, str], ablation: dict[str, bool], trace: list[Any]) -> dict[str, Any]:
        mask_debug_path, attention_debug_path, joint_debug_path = debug_paths
        return {
            "variant": "ADVEDM+",
            "objective": objective,
            "scope": scope,
            "target_text": target,
            "score_provider": "adaptive_joint_semantic_score",
            "mode": mode,
            "mask_ratio": float((1.0 - mask).mean()),
            "selected_patches": int(((1.0 - mask).sum() / max(1, int(cfg.patch_size) * int(cfg.patch_size)))),
            "configured_topk": int(cfg.topk),
            "adaptive_topk": int(adaptive_topk),
            "semantic_focus": float(budget["semantic_focus"]),
            "fixation_mean": float(budget["fixation_mean"]),
            "fixation_focus": float(budget["fixation_focus"]),
            "text_density": float(budget["text_density"]),
            "adaptive_budget": {
                "image_scale": float(budget["image_scale"]),
                "text_scale": float(budget["text_scale"]),
                "epsilon": float(budget["epsilon"]),
                "step_size": float(budget["step_size"]),
                "steps": int(budget["steps"]),
                "eps_t": int(budget["eps_t"]),
            },
            "text_edit": text_edit,
            "mask_debug_path": mask_debug_path,
            "attention_debug_path": attention_debug_path,
            "joint_debug_path": joint_debug_path,
            "ablation": ablation,
            "loss_decomposition": trace[-1].loss_parts if trace else {},
        }

    def attack(self, sample: Sample, ctx: AttackContext) -> AttackedSample:
        cfg = ctx.config.attack
        scope, ablation, do_img, do_txt = _scope_plan(ctx)
        image = np.asarray(sample.image, dtype=np.float32)
        target = str(sample.target_text or sample.text or "object")
        text = str(sample.text or "")
        surrogate = ctx.surrogate_model_adapter or ctx.model_adapter

        patch_size = int(cfg.patch_size)
        topk = int(cfg.topk)
        mode = str(cfg.mode).upper()
        objective = "remove" if mode == "A" else "add"
        mask_mode = "remove" if mode == "A" else "add"

        maps = _joint_score_maps(image, text, target, surrogate, cfg, ablation)
        budget = _adaptive_budget(cfg, maps, text, ablation)
        adaptive_topk = _adaptive_topk(topk, budget, ablation)

        mask = select_mask(
            maps["joint_scores"],
            patch_size,
            adaptive_topk,
            mode=mask_mode,
            threshold=float(cfg.threshold),
            shape=image.shape[:2],
        )

        adv_image, trace = self._run_image_branch(
            image=image,
            mask=mask,
            target=target,
            text=text,
            surrogate=surrogate,
            cfg=cfg,
            budget=budget,
            fixation_for_budget=maps["fixation_for_budget"],
            objective=objective,
            do_img=do_img,
        )
        adv_text, text_edit = self._run_text_branch(image=adv_image, clean_image=image, text=text, surrogate=surrogate, cfg=cfg, budget=budget, do_img=do_img, do_txt=do_txt)

        l0, l2, linf = perturb_stats(image, adv_image)
        adv_sample = Sample(
            sample_id=sample.sample_id,
            image=np.asarray(adv_image, dtype=np.float32),
            text=str(adv_text),
            target_text=str(sample.target_text or ""),
            metadata=dict(sample.metadata),
        )
        adv_sample.metadata["attack_name"] = "advedm_plus"
        adv_sample.metadata["attack_mode"] = mode
        adv_sample.metadata["attack_scope"] = scope
        adv_sample.metadata["attack_variant"] = "ADVEDM+"

        debug_paths = self._debug_paths(ctx=ctx, mask=mask, maps=maps, budget=budget, text_edit=text_edit, scope=scope, target=target)

        return AttackedSample(
            sample=adv_sample,
            perturbation_l0=l0 if do_img else 0,
            perturbation_l2=l2 if do_img else 0.0,
            perturbation_linf=linf if do_img else 0.0,
            attack_trace=trace,
            metadata=self._metadata(
                mode=mode,
                objective=objective,
                scope=scope,
                target=target,
                mask=mask,
                cfg=cfg,
                adaptive_topk=adaptive_topk,
                budget=budget,
                text_edit=text_edit,
                debug_paths=debug_paths,
                ablation=ablation,
                trace=trace,
            ),
        )


def _write_debug_artifacts(
    *,
    sample_debug_dir: str,
    mask: np.ndarray,
    fixation: np.ndarray,
    scores: np.ndarray,
    joint_scores: np.ndarray,
    fixation_patch: np.ndarray,
    budget: dict[str, float],
    text_edit: dict[str, Any],
    scope: str,
    target_text: str,
) -> tuple[str, str, str]:
    debug_dir = Path(sample_debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    mask_img = ((1.0 - np.clip(mask[..., 0], 0.0, 1.0)) * 255.0).astype(np.uint8)
    mask_path = debug_dir / "advedm_plus_mask.png"
    Image.fromarray(mask_img).save(mask_path)

    att_img = (np.clip(fixation, 0.0, 1.0) * 255.0).astype(np.uint8)
    att_path = debug_dir / "advedm_plus_attention.png"
    Image.fromarray(att_img).save(att_path)

    payload = {
        "scope": scope,
        "target_text": target_text,
        "budget": {
            "semantic_focus": float(budget["semantic_focus"]),
            "fixation_mean": float(budget["fixation_mean"]),
            "fixation_focus": float(budget["fixation_focus"]),
            "text_density": float(budget["text_density"]),
            "image_scale": float(budget["image_scale"]),
            "text_scale": float(budget["text_scale"]),
            "epsilon": float(budget["epsilon"]),
            "step_size": float(budget["step_size"]),
            "steps": int(budget["steps"]),
            "eps_t": int(budget["eps_t"]),
        },
        "text_edit": text_edit,
        "score_mean": float(np.asarray(scores).mean()),
        "score_max": float(np.asarray(scores).max()),
        "score_min": float(np.asarray(scores).min()),
        "joint_score_mean": float(np.asarray(joint_scores).mean()),
        "joint_score_max": float(np.asarray(joint_scores).max()),
        "joint_score_min": float(np.asarray(joint_scores).min()),
        "fixation_patch_mean": float(np.asarray(fixation_patch).mean()) if np.asarray(fixation_patch).size else 0.0,
    }
    debug_path = debug_dir / "advedm_plus_debug.json"
    debug_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(mask_path), str(att_path), str(debug_path)
