# 文件说明：该文件属于AdvCLIP 攻击模块，集中实现 patch 相关逻辑。
from __future__ import annotations

import numpy as np


# 中文注释：实现 patch_initialization 的核心流程，支撑AdvCLIP 攻击模块中的业务语义和异常边界。
def patch_initialization(patch_size: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 1, size=(patch_size, patch_size, 3)).astype(np.float32)


# 中文注释：实现 random_location 的核心流程，支撑AdvCLIP 攻击模块中的业务语义和异常边界。
def random_location(
    image_shape: tuple[int, int, int],
    patch_size: int,
    *,
    rng: np.random.Generator | None = None,
    margin: int = 0,
) -> tuple[int, int]:
    """Sample a top-left (y,x) patch location inside the image."""
    h, w, _ = image_shape
    ph = min(h, int(patch_size))
    pw = min(w, int(patch_size))
    max_y = max(0, h - ph - int(margin))
    max_x = max(0, w - pw - int(margin))
    rng = rng or np.random.default_rng()
    y = int(rng.integers(0, max(1, max_y + 1)))
    x = int(rng.integers(0, max(1, max_x + 1)))
    return y, x


# 中文注释：实现 mask_generation 的核心流程，支撑AdvCLIP 攻击模块中的业务语义和异常边界。
def mask_generation(image_shape: tuple[int, int, int], patch_size: int, margin: int = 14) -> tuple[np.ndarray, tuple[int, int]]:
    h, w, _ = image_shape
    ph = min(h, patch_size)
    pw = min(w, patch_size)
    y = max(0, h - margin - ph)
    x = max(0, w - margin - pw)
    mask = np.zeros((h, w, 1), dtype=np.float32)
    mask[y : y + ph, x : x + pw, 0] = 1.0
    return mask, (y, x)


# 中文注释：实现 clamp_patch 的核心流程，支撑AdvCLIP 攻击模块中的业务语义和异常边界。
def clamp_patch(patch: np.ndarray) -> np.ndarray:
    return np.clip(patch, 0.0, 1.0)


# 中文注释：实现 apply_patch 的核心流程，支撑AdvCLIP 攻击模块中的业务语义和异常边界。
def apply_patch(image: np.ndarray, patch: np.ndarray, location: tuple[int, int]) -> np.ndarray:
    y, x = location
    h, w, _ = image.shape
    ph = min(patch.shape[0], h - y)
    pw = min(patch.shape[1], w - x)
    out = image.copy()
    out[y : y + ph, x : x + pw] = patch[:ph, :pw]
    return np.clip(out, 0.0, 1.0)


# 中文注释：实现 apply_patch_torch 的核心流程，支撑AdvCLIP 攻击模块中的业务语义和异常边界。
def apply_patch_torch(images, patch, locs: list[tuple[int, int]]):
    """Apply a universal patch to a BCHW image tensor.

    images: float tensor in [0,1], shape [B,3,H,W]
    patch: float tensor in [0,1], shape [3,Ph,Pw]
    locs: list of (y,x) per batch item
    """
    import torch

    if images.ndim != 4:
        raise ValueError("images must be BCHW")
    b, c, h, w = images.shape
    if c != 3:
        raise ValueError("images must have 3 channels")

    ph, pw = int(patch.shape[-2]), int(patch.shape[-1])
    out = images.clone()
    for i in range(int(b)):
        y, x = locs[i]
        y = int(max(0, min(h - 1, y)))
        x = int(max(0, min(w - 1, x)))
        y2 = int(min(h, y + ph))
        x2 = int(min(w, x + pw))
        ph2 = int(y2 - y)
        pw2 = int(x2 - x)
        if ph2 <= 0 or pw2 <= 0:
            continue
        out[i, :, y:y2, x:x2] = patch[:, :ph2, :pw2]
    return out.clamp(0.0, 1.0)


# 中文注释：实现 patch_tv 的核心流程，支撑AdvCLIP 攻击模块中的业务语义和异常边界。
def patch_tv(patch: np.ndarray) -> float:
    dx = np.abs(np.diff(patch, axis=1)).mean()
    dy = np.abs(np.diff(patch, axis=0)).mean()
    return float(dx + dy)


# 中文注释：实现 patch_tv_torch 的核心流程，支撑AdvCLIP 攻击模块中的业务语义和异常边界。
def patch_tv_torch(patch) -> float:
    """Total variation on a [3,Ph,Pw] patch tensor."""
    import torch

    if patch.ndim != 3:
        raise ValueError("patch must be CHW")
    dx = torch.abs(patch[:, :, 1:] - patch[:, :, :-1]).mean()
    dy = torch.abs(patch[:, 1:, :] - patch[:, :-1, :]).mean()
    return float((dx + dy).detach().cpu().item())
