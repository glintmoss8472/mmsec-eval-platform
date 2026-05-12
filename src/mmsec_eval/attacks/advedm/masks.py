# 文件说明：该文件属于AdvEDM 攻击模块，集中实现 masks 相关逻辑。
from __future__ import annotations

import math

import numpy as np


# 中文注释：封装 _normalize_scores 的内部步骤，让AdvEDM 攻击模块主流程保持清晰并隔离边界细节。
def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float32)
    if arr.size == 0:
        return arr.astype(np.float32)
    mn = float(arr.min())
    mx = float(arr.max())
    if mx - mn < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - mn) / (mx - mn)).astype(np.float32)


# 中文注释：实现 patch_similarity_scores 的核心流程，支撑AdvEDM 攻击模块中的业务语义和异常边界。
def patch_similarity_scores(
    image: np.ndarray,
    target_text: str,
    patch_size: int = 8,
    model_adapter: object | None = None,
    batch_size: int = 32,
) -> np.ndarray:
    """Compute semantic patch scores via model adapter pair scoring."""
    if model_adapter is None or not hasattr(model_adapter, "score_pairs"):
        raise RuntimeError("AdvEDM requires model_adapter.score_pairs for semantic patch scoring")

    h, w, _ = image.shape
    gh, gw = max(1, int(math.ceil(h / max(1, patch_size)))), max(1, int(math.ceil(w / max(1, patch_size))))
    scores = np.zeros((gh, gw), dtype=np.float32)

    # Prefer patch-token semantic map if adapter provides it (closer to paper Eq.(3)).
    if hasattr(model_adapter, "patch_text_similarity"):
        raw = np.asarray(model_adapter.patch_text_similarity(image, target_text), dtype=np.float32)
        if raw.ndim == 3:
            raw = raw.mean(axis=-1)
        if raw.shape != (h, w):
            sh, sw = int(raw.shape[0]), int(raw.shape[1])
            ry = max(1, int(np.ceil(h / max(1, sh))))
            rx = max(1, int(np.ceil(w / max(1, sw))))
            raw = np.repeat(np.repeat(raw, ry, axis=0), rx, axis=1)[:h, :w]
        for i in range(gh):
            for j in range(gw):
                y1 = i * patch_size
                y2 = min((i + 1) * patch_size, h)
                x1 = j * patch_size
                x2 = min((j + 1) * patch_size, w)
                scores[i, j] = float(raw[y1:y2, x1:x2].mean())
        return _normalize_scores(scores)

    # Fallback: crop-level scoring with score_pairs.
    patches: list[tuple[np.ndarray, str]] = []
    coords: list[tuple[int, int]] = []
    step = max(1, int(batch_size))
    for i in range(gh):
        for j in range(gw):
            y1 = i * patch_size
            y2 = min((i + 1) * patch_size, h)
            x1 = j * patch_size
            x2 = min((j + 1) * patch_size, w)
            crop = np.asarray(image[y1:y2, x1:x2], dtype=np.float32)
            patches.append((crop, str(target_text)))
            coords.append((i, j))
            if len(patches) >= step:
                vals = np.asarray(model_adapter.score_pairs(patches, batch_size=step), dtype=np.float32).reshape(-1)
                for (ii, jj), vv in zip(coords, vals.tolist()):
                    scores[ii, jj] = float(vv)
                patches.clear()
                coords.clear()

    if patches:
        vals = np.asarray(model_adapter.score_pairs(patches, batch_size=step), dtype=np.float32).reshape(-1)
        for (ii, jj), vv in zip(coords, vals.tolist()):
            scores[ii, jj] = float(vv)

    return _normalize_scores(scores)


# 中文注释：实现 attention_fixation_map 的核心流程，支撑AdvEDM 攻击模块中的业务语义和异常边界。
def attention_fixation_map(
    image: np.ndarray,
    text: str,
    model_adapter: object | None,
) -> np.ndarray:
    """Return normalized attention guidance map in [0,1]."""
    if model_adapter is None or not hasattr(model_adapter, "attention_map"):
        raise RuntimeError("AdvEDM requires model_adapter.attention_map")
    h, w, _ = image.shape
    att = np.asarray(model_adapter.attention_map(image, text), dtype=np.float32)
    if att.ndim == 3:
        att = att.mean(axis=-1)
    if att.shape != (h, w):
        sh, sw = int(att.shape[0]), int(att.shape[1])
        if sh <= 0 or sw <= 0:
            raise RuntimeError("invalid attention map shape")
        ry = max(1, int(np.ceil(h / sh)))
        rx = max(1, int(np.ceil(w / sw)))
        att = np.repeat(np.repeat(att, ry, axis=0), rx, axis=1)[:h, :w]
    return _normalize_scores(np.clip(att, 0.0, 1.0))


# 中文注释：实现 select_mask 的核心流程，支撑AdvEDM 攻击模块中的业务语义和异常边界。
def select_mask(
    scores: np.ndarray,
    patch_size: int,
    topk: int,
    mode: str = "remove",
    threshold: float = 0.5,
    shape: tuple[int, int] | None = None,
) -> np.ndarray:
    gh, gw = scores.shape
    flat = scores.flatten()
    idx = np.argsort(flat)
    if mode == "remove":
        chosen = idx[-max(1, topk) :]
    elif mode == "add":
        chosen = idx[: max(1, topk)]
    else:
        chosen = np.where(flat >= threshold)[0]

    mask_grid = np.ones_like(flat, dtype=np.float32)
    mask_grid[chosen] = 0.0
    mask_grid = mask_grid.reshape(gh, gw)
    mask = np.repeat(np.repeat(mask_grid, patch_size, axis=0), patch_size, axis=1)

    if shape is not None:
        h, w = shape
        out = np.ones((h, w), dtype=np.float32)
        mh = min(h, mask.shape[0])
        mw = min(w, mask.shape[1])
        out[:mh, :mw] = mask[:mh, :mw]
        mask = out
    return mask[..., None]
