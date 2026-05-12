# 文件说明：该文件属于AdvEDM 攻击模块，集中实现 optimize 相关逻辑。
from __future__ import annotations

from typing import Any

import numpy as np

from mmsec_eval.types import AttackTraceStep


# 中文注释：封装 _resize_fixation_map 的内部步骤，让AdvEDM 攻击模块主流程保持清晰并隔离边界细节。
def _resize_fixation_map(fixation_map: np.ndarray | None, height: int, width: int) -> np.ndarray:
    if fixation_map is None:
        return np.zeros((height, width), dtype=np.float32)
    fix_np = np.asarray(fixation_map, dtype=np.float32)
    if fix_np.shape != (height, width):
        sh, sw = int(fix_np.shape[0]), int(fix_np.shape[1])
        ry = max(1, int(np.ceil(height / max(1, sh))))
        rx = max(1, int(np.ceil(width / max(1, sw))))
        fix_np = np.repeat(np.repeat(fix_np, ry, axis=0), rx, axis=1)[:height, :width]
    return np.clip(fix_np, 0.0, 1.0).astype(np.float32)


# 中文注释：封装 _torch_masks 的内部步骤，让AdvEDM 攻击模块主流程保持清晰并隔离边界细节。
def _torch_masks(image: np.ndarray, mask: np.ndarray, fixation_map: np.ndarray | None, device: str) -> tuple[Any, Any, Any, Any]:
    import torch

    height, width = int(image.shape[0]), int(image.shape[1])
    preserve = torch.from_numpy(np.asarray(mask[..., 0], dtype=np.float32)).to(device).view(1, 1, height, width)
    attack = 1.0 - preserve
    fix_np = _resize_fixation_map(fixation_map, height, width)
    fix = torch.from_numpy(fix_np).to(device).view(1, 1, height, width)
    return preserve.expand(1, 3, height, width), attack.expand(1, 3, height, width), fix.expand(1, 3, height, width), fix_np


# 中文注释：封装 _patch_grid_focus 的内部步骤，让AdvEDM 攻击模块主流程保持清晰并隔离边界细节。
def _patch_grid_focus(image: np.ndarray, mask: np.ndarray, fix_np_full: np.ndarray, patch_size: int, device: str) -> tuple[int, int, Any]:
    import torch

    height, width = int(image.shape[0]), int(image.shape[1])
    p = max(1, int(patch_size))
    gh = max(1, int(np.ceil(height / p)))
    gw = max(1, int(np.ceil(width / p)))
    preserve_np = np.asarray(mask[..., 0], dtype=np.float32)
    attack_grid_np = np.zeros((gh, gw), dtype=np.float32)
    fix_grid_np = np.zeros((gh, gw), dtype=np.float32)
    for i in range(gh):
        for j in range(gw):
            y1, y2 = i * p, min((i + 1) * p, height)
            x1, x2 = j * p, min((j + 1) * p, width)
            attack_grid_np[i, j] = float(1.0 - float(preserve_np[y1:y2, x1:x2].mean()))
            fix_grid_np[i, j] = float(fix_np_full[y1:y2, x1:x2].mean())
    attack_grid = torch.from_numpy(attack_grid_np).to(device=device, dtype=torch.float32).view(1, gh, gw)
    fix_grid = torch.from_numpy(np.clip(fix_grid_np, 0.0, 1.0)).to(device=device, dtype=torch.float32).view(1, gh, gw)
    return gh, gw, attack_grid * (0.1 + 2.8 * torch.pow(fix_grid, 1.55))


# 中文注释：封装 _patch_score 的内部步骤，让AdvEDM 攻击模块主流程保持清晰并隔离边界细节。
def _patch_score(model_adapter: Any, x_adv: Any, target_text: str, gh: int, gw: int, attack_grid_focus: Any) -> Any:
    import torch.nn.functional as F

    patch_sim = model_adapter.patch_text_similarity_torch(x_adv, [str(target_text)])
    if patch_sim.ndim == 2:
        n = int(patch_sim.shape[1])
        side = int(np.sqrt(n))
        if side * side != n:
            raise RuntimeError(f"invalid patch similarity shape: {tuple(patch_sim.shape)}")
        patch_sim = patch_sim.view(1, side, side)
    elif patch_sim.ndim != 3:
        raise RuntimeError(f"invalid patch similarity ndim: {patch_sim.ndim}")
    if (int(patch_sim.shape[1]) != gh) or (int(patch_sim.shape[2]) != gw):
        patch_sim = F.interpolate(patch_sim.unsqueeze(1), size=(gh, gw), mode="bilinear", align_corners=False).squeeze(1)
    return (patch_sim * attack_grid_focus).sum() / (attack_grid_focus.sum() + 1e-8)


# 中文注释：封装 _loss_parts 的内部步骤，让AdvEDM 攻击模块主流程保持清晰并隔离边界细节。
def _loss_parts(
    *,
    x0: Any,
    x_adv: Any,
    preserve3: Any,
    attack3: Any,
    fix3: Any,
    score_t: Any,
    score_patch: Any,
    objective: str,
    alpha: float,
    beta: float,
    gamma: float,
) -> tuple[Any, dict[str, Any]]:
    add_objective = str(objective).lower() == "add"
    loss_cls = -score_t if add_objective else score_t
    loss_patch = -score_patch if add_objective else score_patch
    delta = x_adv - x0
    loss_preserve = ((delta.abs()) * preserve3 * (1.0 + 1.1 * fix3)).mean()
    loss_focus = ((delta.abs()) * attack3 * (0.15 + 1.75 * (1.0 - fix3).pow(1.2))).mean()
    loss_total = float(alpha) * loss_cls + float(beta) * loss_patch + float(gamma) * (loss_preserve + 0.5 * loss_focus)
    return loss_total, {
        "loss_cls": loss_cls,
        "loss_patch": loss_patch,
        "loss_preserve": loss_preserve,
        "loss_focus": loss_focus,
        "score_t": score_t,
        "score_patch": score_patch,
    }


# 中文注释：封装 _trace_step 的内部步骤，让AdvEDM 攻击模块主流程保持清晰并隔离边界细节。
def _trace_step(k: int, parts: dict[str, Any], loss_total: Any, objective: str, caption_text: str, target_text: str) -> AttackTraceStep:
    return AttackTraceStep(
        step=k + 1,
        loss_total=float(loss_total.detach().cpu().item()),
        loss_parts={
            "loss_target": float(parts["loss_cls"].detach().cpu().item()),
            "loss_patch": float(parts["loss_patch"].detach().cpu().item()),
            "loss_preserve": float(parts["loss_preserve"].detach().cpu().item()),
            "loss_fixation": float(parts["loss_focus"].detach().cpu().item()),
            "score_target": float(parts["score_t"].detach().cpu().item()),
            "score_patch": float(parts["score_patch"].detach().cpu().item()),
        },
        metadata={"objective": str(objective), "caption_text": str(caption_text), "target_text": str(target_text)},
    )


# 中文注释：实现 masked_pgd_optimize 的核心流程，支撑AdvEDM 攻击模块中的业务语义和异常边界。
def masked_pgd_optimize(
    *,
    image: np.ndarray,
    mask: np.ndarray,
    patch_size: int,
    target_text: str,
    caption_text: str,
    model_adapter: Any,
    epsilon: float,
    step_size: float,
    steps: int,
    alpha: float,
    beta: float,
    gamma: float,
    fixation_map: np.ndarray | None = None,
    objective: str = "remove",
) -> tuple[np.ndarray, list[AttackTraceStep]]:
    """CUDA torch gradient optimizer for AdvEDM."""
    if not hasattr(model_adapter, "score_pairs_torch"):
        raise RuntimeError("AdvEDM optimization requires model_adapter.score_pairs_torch")
    if not hasattr(model_adapter, "patch_text_similarity_torch"):
        raise RuntimeError("AdvEDM optimization requires model_adapter.patch_text_similarity_torch")

    import torch

    device = getattr(model_adapter, "_device", "cuda")
    x0 = torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1).unsqueeze(0).to(device)
    x_adv = x0.clone().detach()
    preserve3, attack3, fix3, fix_np_full = _torch_masks(image, mask, fixation_map, device)
    attack_focus = attack3 * (0.15 + 3.0 * fix3)

    gh, gw, attack_grid_focus = _patch_grid_focus(image, mask, fix_np_full, patch_size, device)
    momentum = torch.zeros_like(x_adv)
    traces: list[AttackTraceStep] = []
    eps = float(epsilon)
    step = float(step_size)
    n_steps = max(1, int(steps))

    for k in range(n_steps):
        x_adv = x_adv.detach().clone().requires_grad_(True)

        score_t = model_adapter.score_pairs_torch(x_adv, [str(target_text)], output_attentions=False).mean()
        score_patch = _patch_score(model_adapter, x_adv, target_text, gh, gw, attack_grid_focus)
        loss_total, parts = _loss_parts(
            x0=x0,
            x_adv=x_adv,
            preserve3=preserve3,
            attack3=attack3,
            fix3=fix3,
            score_t=score_t,
            score_patch=score_patch,
            objective=objective,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )

        grad = torch.autograd.grad(loss_total, x_adv, retain_graph=False, create_graph=False)[0]
        grad = grad * attack_focus
        grad = grad / (grad.abs().mean() + 1e-8)
        momentum = 0.9 * momentum + grad

        x_next = x_adv - step * momentum.sign()
        delta_next = (x_next - x0).clamp(-eps, eps)
        x_adv = (x0 + delta_next).clamp(0.0, 1.0)
        traces.append(_trace_step(k, parts, loss_total, objective, caption_text, target_text))

    adv = x_adv[0].detach().cpu().permute(1, 2, 0).numpy().astype(np.float32)
    return adv, traces
