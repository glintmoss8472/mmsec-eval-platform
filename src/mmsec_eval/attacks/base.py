from __future__ import annotations

import numpy as np


def clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def perturb_stats(clean: np.ndarray, adv: np.ndarray) -> tuple[int, float, float]:
    d = adv - clean
    l0 = int(np.count_nonzero(np.abs(d) > 1e-8))
    l2 = float(np.sqrt((d * d).sum()))
    linf = float(np.abs(d).max())
    return l0, l2, linf


def total_variation(x: np.ndarray) -> float:
    dx = np.abs(np.diff(x, axis=1)).mean()
    dy = np.abs(np.diff(x, axis=0)).mean()
    return float(dx + dy)
