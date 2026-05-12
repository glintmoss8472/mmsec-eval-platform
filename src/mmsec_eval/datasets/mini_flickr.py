from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from mmsec_eval.datasets.flickr30k import load_flickr30k


def load_mini_flickr(dataset_cfg: Any):
    root = str(getattr(dataset_cfg, "root", "") or "").strip()
    if not root:
        project_root = Path(__file__).resolve().parents[3]
        root = str(project_root / "seed" / "data" / "mini_flickr")

    image_dir = str(getattr(dataset_cfg, "image_dir", "") or "images")
    captions_file = str(getattr(dataset_cfg, "captions_file", "") or "captions_index.jsonl")
    split = str(getattr(dataset_cfg, "split", "") or "test")
    max_items = int(getattr(dataset_cfg, "max_items", 0) or 0)
    benchmark_tag = str(getattr(dataset_cfg, "benchmark_tag", "") or "mini_flickr")

    proxy_cfg = SimpleNamespace(
        root=root,
        image_dir=image_dir,
        captions_file=captions_file,
        split=split,
        max_items=max_items,
        benchmark_tag=benchmark_tag,
    )
    items = load_flickr30k(proxy_cfg)
    for item in items:
        item.metadata["dataset"] = "mini_flickr"
        item.metadata["benchmark_tag"] = benchmark_tag
    return items
