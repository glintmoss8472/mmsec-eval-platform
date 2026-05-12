from __future__ import annotations

import numpy as np


def _broadcast_map(weight_map: np.ndarray | None, shape: tuple[int, int, int]) -> np.ndarray:
    if weight_map is None:
        return np.ones(shape, dtype=np.float32)
    w = np.asarray(weight_map, dtype=np.float32)
    if w.ndim == 2:
        w = w[..., None]
    if w.shape[:2] != shape[:2]:
        h, w2 = shape[:2]
        sh, sw = int(w.shape[0]), int(w.shape[1])
        if sh <= 0 or sw <= 0:
            return np.ones(shape, dtype=np.float32)
        ry = max(1, int(np.ceil(h / sh)))
        rx = max(1, int(np.ceil(w2 / sw)))
        w = np.repeat(np.repeat(w, ry, axis=0), rx, axis=1)[:h, :w2]
        if w.ndim == 2:
            w = w[..., None]
    if w.shape[2] == 1 and shape[2] != 1:
        w = np.repeat(w, shape[2], axis=2)
    return np.clip(w, 0.0, 1.0).astype(np.float32)


def target_object_loss(clean: np.ndarray, adv: np.ndarray, mask: np.ndarray) -> float:
    # Encourage changes within selected regions.
    delta = np.abs(adv - clean)
    return float((delta * (1.0 - mask)).mean())


def preserve_semantics_loss(clean: np.ndarray, adv: np.ndarray, mask: np.ndarray, fixation_map: np.ndarray | None = None) -> float:
    # Keep non-target region close to clean image (attention-weighted).
    delta = clean - adv
    w = _broadcast_map(fixation_map, clean.shape)
    return float(((delta * delta) * mask * (1.0 + w)).mean())


def fixation_loss(clean: np.ndarray, adv: np.ndarray, mask: np.ndarray, fixation_map: np.ndarray | None = None) -> float:
    # Penalize leaking perturbation into preserved regions.
    leak = np.abs(adv - clean) * mask
    if fixation_map is not None:
        w = _broadcast_map(fixation_map, clean.shape)
        leak = leak * (1.0 + w)
    return float(leak.mean())


def total_loss(
    clean: np.ndarray,
    adv: np.ndarray,
    mask: np.ndarray,
    alpha: float,
    beta: float,
    gamma: float,
    fixation_map: np.ndarray | None = None,
    objective: str = "remove",
) -> tuple[float, dict[str, float]]:
    l_target = target_object_loss(clean, adv, mask)
    l_preserve = preserve_semantics_loss(clean, adv, mask, fixation_map=fixation_map)
    l_fix = fixation_loss(clean, adv, mask, fixation_map=fixation_map)
    if objective == "add":
        # AdvEDM-A focuses on semantic injection while preserving other regions.
        total = alpha * l_target - 0.8 * beta * l_preserve - gamma * l_fix
    else:
        # AdvEDM-R: stronger preserve/fixation constraints.
        total = alpha * l_target - beta * l_preserve - gamma * l_fix
    return float(total), {
        "target": float(l_target),
        "preserve": float(l_preserve),
        "fixation": float(l_fix),
    }
