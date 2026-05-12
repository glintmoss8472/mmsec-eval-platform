# 文件说明：该文件属于TMM 迁移攻击模块，集中实现 adfp 相关逻辑。
from __future__ import annotations

import numpy as np


# 中文注释：实现 attention_to_critical_mask 的核心流程，支撑TMM 迁移攻击模块中的业务语义和异常边界。
def attention_to_critical_mask(att_map: np.ndarray, threshold: float) -> np.ndarray:
    """Convert an attention map to a boolean critical-region mask."""
    m = np.asarray(att_map, dtype=np.float32)
    if m.ndim != 2:
        raise ValueError("att_map must be HxW")
    thr = float(threshold)
    return (m >= thr).astype(np.float32)


# 中文注释：实现 allocate_budget 的核心流程，支撑TMM 迁移攻击模块中的业务语义和异常边界。
def allocate_budget(mask_hw: np.ndarray, eps_v: float, ratio_r: float) -> tuple[float, float]:
    """ADFP budget split (approx).

    eps_crit = eps_v * (1 - r) * (P_image / P_crit)
    eps_noncrit = eps_v * r
    """
    eps_v = float(eps_v)
    r = float(ratio_r)
    h, w = int(mask_hw.shape[0]), int(mask_hw.shape[1])
    p_img = float(h * w)
    p_crit = float(mask_hw.mean() * p_img)
    if p_crit <= 0:
        return 0.0, float(max(0.0, eps_v))
    eps_crit = eps_v * max(0.0, 1.0 - r) * (p_img / max(1.0, p_crit))
    eps_noncrit = eps_v * max(0.0, r)
    # Keep bounds sane in [0,1].
    eps_crit = float(min(max(0.0, eps_crit), 1.0))
    eps_noncrit = float(min(max(0.0, eps_noncrit), 1.0))
    return eps_crit, eps_noncrit


# 中文注释：实现 input_diversity 的核心流程，支撑TMM 迁移攻击模块中的业务语义和异常边界。
def input_diversity(images, *, rng: np.random.Generator, p: float = 0.7, low: float = 0.9):
    """DI-FGSM style input diversity (differentiable).

    images: BCHW torch tensor in [0,1]
    """
    import torch
    import torch.nn.functional as F

    if float(rng.random()) > float(p):
        return images

    b, c, h, w = images.shape
    low = float(low)
    rh = int(rng.integers(int(low * h), h + 1))
    rw = int(rng.integers(int(low * w), w + 1))
    x = F.interpolate(images, size=(rh, rw), mode="bilinear", align_corners=False)
    pad_h = int(h - rh)
    pad_w = int(w - rw)
    top = int(rng.integers(0, pad_h + 1)) if pad_h > 0 else 0
    left = int(rng.integers(0, pad_w + 1)) if pad_w > 0 else 0
    bottom = pad_h - top
    right = pad_w - left
    x = F.pad(x, (left, right, top, bottom), mode="constant", value=0.0)
    return x

