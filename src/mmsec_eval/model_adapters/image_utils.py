# 文件说明：该文件属于模型适配层，集中实现 image utils 相关逻辑。
from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image


# 执行 `图像 to rgb01` 辅助逻辑，保持模型适配层中的输入处理和结果输出一致。
def image_to_rgb01(image: Any) -> np.ndarray:
    """Convert grayscale/RGB/RGBA image-like input to HWC RGB float32 in [0, 1]."""
    arr = np.asarray(image, dtype=np.float32)
    if arr.size == 0:
        raise ValueError("image array is empty")
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim != 3:
        raise ValueError(f"unsupported image shape: {arr.shape}")
    if int(arr.shape[2]) == 0:
        raise ValueError(f"unsupported image shape: {arr.shape}")
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    if arr.shape[2] >= 3:
        arr = arr[:, :, :3]
    if float(np.nanmax(arr)) > 1.0:
        arr = arr / 255.0
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


# 执行 `rgb01 to uint8` 辅助逻辑，保持模型适配层中的输入处理和结果输出一致。
def rgb01_to_uint8(image: Any) -> np.ndarray:
    return (image_to_rgb01(image) * 255.0).round().astype(np.uint8)


# 执行 `图像 to pil rgb` 辅助逻辑，保持模型适配层中的输入处理和结果输出一致。
def image_to_pil_rgb(image: Any) -> Image.Image:
    return Image.fromarray(rgb01_to_uint8(image))


# 执行 `resize rgb01` 辅助逻辑，保持模型适配层中的输入处理和结果输出一致。
def resize_rgb01(image: Any, *, size_hw: tuple[int, int], resample: int = Image.BICUBIC) -> np.ndarray:
    th, tw = int(size_hw[0]), int(size_hw[1])
    if th <= 0 or tw <= 0:
        raise ValueError(f"invalid target size: {size_hw}")
    arr = image_to_rgb01(image)
    if int(arr.shape[0]) == th and int(arr.shape[1]) == tw:
        return arr.astype(np.float32)
    pil = image_to_pil_rgb(arr)
    pil = pil.resize((tw, th), resample=resample)
    return np.asarray(pil, dtype=np.float32) / 255.0


# 执行 `processor target hw` 辅助逻辑，保持模型适配层中的输入处理和结果输出一致。
def processor_target_hw(
    processor: Any,
    *,
    default_hw: tuple[int, int],
    prefer_shortest_edge: bool = False,
) -> tuple[int, int]:
    image_processor = getattr(processor, "image_processor", None)
    size = getattr(image_processor, "size", None) or {}
    if prefer_shortest_edge and isinstance(size, dict) and "shortest_edge" in size:
        target = int(size["shortest_edge"])
        return target, target
    if isinstance(size, dict) and "height" in size and "width" in size:
        return int(size["height"]), int(size["width"])
    if isinstance(size, dict) and "shortest_edge" in size:
        target = int(size["shortest_edge"])
        return target, target
    return int(default_hw[0]), int(default_hw[1])


# 执行 `stack resized rgb01` 辅助逻辑，保持模型适配层中的输入处理和结果输出一致。
def stack_resized_rgb01(images: list[Any], *, size_hw: tuple[int, int]) -> np.ndarray:
    return np.stack([resize_rgb01(image, size_hw=size_hw).astype(np.float32) for image in images], axis=0).astype(np.float32)


# 执行 `bchw to pil 图像` 辅助逻辑，保持模型适配层中的输入处理和结果输出一致。
def bchw_to_pil_images(images_bchw: Any) -> list[Image.Image]:
    ndim = int(getattr(images_bchw, "ndim", -1))
    if ndim != 4:
        raise ValueError("images must be BCHW")
    if hasattr(images_bchw, "detach"):
        arr = images_bchw.detach().cpu().permute(0, 2, 3, 1).numpy()
    else:
        arr = np.asarray(images_bchw).transpose(0, 2, 3, 1)
    return [image_to_pil_rgb(image) for image in arr]
