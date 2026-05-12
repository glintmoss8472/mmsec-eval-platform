from __future__ import annotations

from typing import Any

from mmsec_eval.datasets.flickr30k import load_flickr_like
from mmsec_eval.types import Sample


def load_flickr1k(dataset_cfg: Any) -> list[Sample]:
    return load_flickr_like(dataset_cfg, dataset_name="flickr1k")
