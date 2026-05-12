# 文件说明：该文件属于数据集加载层，集中实现 toy shapes 相关逻辑。
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
from PIL import Image, ImageDraw

from mmsec_eval.types import Sample


SHAPES = ["circle", "square", "triangle"]
COLORS = ["red", "green", "blue"]


# 定义 `ToyShapesDataset` 的不可变配置载体，集中保存后续计算需要的结构化字段。
@dataclass
class ToyShapesDataset:
    num_samples: int = 16
    image_size: int = 96
    seed: int = 42

    # 实现 `ToyShapesDataset.generate` 的对象行为，维护该类在数据集加载层中的调用契约。
    def generate(self) -> list[Sample]:
        rng = np.random.default_rng(self.seed)
        items: list[Sample] = []
        for i in range(self.num_samples):
            shape = SHAPES[i % len(SHAPES)]
            color = COLORS[(i // len(SHAPES)) % len(COLORS)]
            target = SHAPES[(i + 1) % len(SHAPES)]
            img = self._draw(shape, color, rng)
            text = f"a {color} {shape} on plain background"
            items.append(
                Sample(
                    sample_id=f"toy-{i:04d}",
                    image=img,
                    text=text,
                    target_text=target,
                    metadata={"shape": shape, "color": color},
                )
            )
        return items

    # 实现 `ToyShapesDataset._draw` 的对象行为，维护该类在数据集加载层中的调用契约。
    def _draw(self, shape: str, color: str, rng: np.random.Generator) -> np.ndarray:
        s = self.image_size
        image = Image.new("RGB", (s, s), (245, 245, 245))
        draw = ImageDraw.Draw(image)
        c = {
            "red": (220, 30, 30),
            "green": (30, 160, 60),
            "blue": (40, 80, 220),
        }[color]
        margin = int(s * 0.2)
        x1 = margin + int(rng.integers(-4, 5))
        y1 = margin + int(rng.integers(-4, 5))
        x2 = s - margin + int(rng.integers(-4, 5))
        y2 = s - margin + int(rng.integers(-4, 5))
        if shape == "circle":
            draw.ellipse((x1, y1, x2, y2), fill=c)
        elif shape == "square":
            draw.rectangle((x1, y1, x2, y2), fill=c)
        else:
            draw.polygon([(s // 2, y1), (x1, y2), (x2, y2)], fill=c)
        return np.asarray(image).astype(np.float32) / 255.0

