# 文件说明：该文件属于项目工程，集中实现 serializer 相关逻辑。
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


# 写出 `图像 png`，保证后续报告、页面或复现实验能读取。
def save_image_png(path: str, image: np.ndarray) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(image, 0, 1)
    arr = (arr * 255).astype(np.uint8)
    Image.fromarray(arr).save(p)
    return str(p)


# 写出 `JSON`，保证后续报告、页面或复现实验能读取。
def write_json(path: str, data: dict[str, Any]) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(p)


# 写出 `JSONL`，保证后续报告、页面或复现实验能读取。
def write_jsonl(path: str, rows: list[dict[str, Any]]) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return str(p)
