# 文件说明：该文件属于数据集加载层，集中实现 registry 相关逻辑。
from __future__ import annotations

from mmsec_eval.config.schema import AppConfig
from mmsec_eval.datasets.coco_subset import load_coco_subset
from mmsec_eval.datasets.flickr1k import load_flickr1k
from mmsec_eval.datasets.flickr30k import load_flickr30k
from mmsec_eval.datasets.folder_jsonl import load_folder_jsonl
from mmsec_eval.datasets.mini_flickr import load_mini_flickr
from mmsec_eval.datasets.toy_shapes import ToyShapesDataset
from mmsec_eval.types import Sample


# 加载 `数据集`，把外部文件、配置或运行产物转换为内存结构。
def load_dataset(cfg: AppConfig) -> list[Sample]:
    kind = cfg.dataset.kind
    if kind == "toy_shapes":
        ds = ToyShapesDataset(
            num_samples=cfg.dataset.num_samples,
            image_size=cfg.dataset.image_size,
            seed=cfg.seed,
        )
        return ds.generate()
    if kind == "folder_jsonl":
        return load_folder_jsonl(cfg.dataset.folder_jsonl)
    if kind == "flickr30k":
        return load_flickr30k(cfg.dataset)
    if kind == "flickr1k":
        return load_flickr1k(cfg.dataset)
    if kind == "coco_subset":
        return load_coco_subset(cfg.dataset)
    if kind == "mini_flickr":
        return load_mini_flickr(cfg.dataset)
    raise ValueError(f"Unknown dataset kind: {kind}")
