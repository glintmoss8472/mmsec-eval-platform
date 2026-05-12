# 文件说明：该文件属于数据集加载层，集中实现 flickr1k 相关逻辑。
from __future__ import annotations

from typing import Any

from mmsec_eval.datasets.flickr30k import load_flickr_like
from mmsec_eval.types import Sample


# 加载 `Flickr1k`，把外部文件、配置或运行产物转换为内存结构。
def load_flickr1k(dataset_cfg: Any) -> list[Sample]:
    return load_flickr_like(dataset_cfg, dataset_name="flickr1k")
