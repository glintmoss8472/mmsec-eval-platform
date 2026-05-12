# 文件说明：该文件属于数据集加载层，集中实现 folder jsonl 相关逻辑。
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from mmsec_eval.io.jsonl_io import read_jsonl
from mmsec_eval.types import Sample


# 加载 `folder JSONL`，把外部文件、配置或运行产物转换为内存结构。
def load_folder_jsonl(path: str) -> list[Sample]:
    rows = read_jsonl(path)
    out: list[Sample] = []
    for i, row in enumerate(rows):
        image_path = Path(str(row.get("image_path", "")))
        if not image_path.exists():
            continue
        img = Image.open(image_path).convert("RGB")
        arr = np.asarray(img).astype(np.float32) / 255.0
        out.append(
            Sample(
                sample_id=str(row.get("id", f"row-{i:04d}")),
                image=arr,
                text=str(row.get("text", "")),
                target_text=str(row.get("target_text", "")),
                metadata={"source": str(image_path)},
            )
        )
    return out

