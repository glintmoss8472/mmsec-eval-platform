# 文件说明：该文件属于数据集加载层，集中实现 flickr1k 相关逻辑。
from __future__ import annotations

from typing import Any

from mmsec_eval.datasets.flickr30k import load_flickr_like
from mmsec_eval.types import Sample


# 中文注释：实现 load_flickr1k 的核心流程，支撑数据集加载层中的业务语义和异常边界。
def load_flickr1k(dataset_cfg: Any) -> list[Sample]:
    return load_flickr_like(dataset_cfg, dataset_name="flickr1k")
