from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from mmsec_eval.datasets.coco_subset import load_coco_subset
from mmsec_eval.datasets.flickr1k import load_flickr1k
from mmsec_eval.datasets.flickr30k import load_flickr30k
from mmsec_eval.datasets.mini_flickr import load_mini_flickr


def _write_image(path: Path, value: int) -> None:
    arr = np.zeros((32, 32, 3), dtype=np.uint8)
    arr[..., value % 3] = 200
    Image.fromarray(arr).save(path)


def test_flickr30k_loader(tmp_path: Path):
    root = tmp_path / "flickr"
    images = root / "images"
    images.mkdir(parents=True, exist_ok=True)
    _write_image(images / "a.png", 0)
    _write_image(images / "b.png", 1)

    rows = [
        {"id": "1", "image": "a.png", "caption": "a red object", "split": "test"},
        {"id": "2", "image": "b.png", "caption": "a blue object", "split": "test"},
    ]
    cap = root / "captions_index.jsonl"
    cap.write_text("\n".join(json.dumps(x) for x in rows) + "\n", encoding="utf-8")

    cfg = SimpleNamespace(
        root=str(root),
        image_dir="images",
        captions_file="captions_index.jsonl",
        split="test",
        max_items=0,
        benchmark_tag="flickr_test",
    )
    out = load_flickr30k(cfg)
    assert len(out) == 2
    assert out[0].metadata["dataset"] == "flickr30k"


def test_flickr1k_loader(tmp_path: Path):
    root = tmp_path / "flickr1k"
    images = root / "images"
    images.mkdir(parents=True, exist_ok=True)
    _write_image(images / "a.png", 0)
    _write_image(images / "b.png", 1)

    rows = [
        {"id": "1", "image": "a.png", "caption": "a red object", "split": "test"},
        {"id": "2", "image": "b.png", "caption": "a blue object", "split": "test"},
    ]
    cap = root / "captions_index.jsonl"
    cap.write_text("\n".join(json.dumps(x) for x in rows) + "\n", encoding="utf-8")

    cfg = SimpleNamespace(
        root=str(root),
        image_dir="images",
        captions_file="captions_index.jsonl",
        split="test",
        max_items=0,
        benchmark_tag="flickr1k_test",
    )
    out = load_flickr1k(cfg)
    assert len(out) == 2
    assert out[0].metadata["dataset"] == "flickr1k"


def test_flickr30k_loader_rejects_placeholder_rows(tmp_path: Path):
    root = tmp_path / "flickr"
    images = root / "images"
    images.mkdir(parents=True, exist_ok=True)
    _write_image(images / "demo_0000.jpg", 0)
    cap = root / "captions_index.jsonl"
    cap.write_text(
        json.dumps(
            {
                "id": "demo-0000",
                "image": "demo_0000.jpg",
                "caption": "Demo caption 0000: multimodal retrieval placeholder sample.",
                "split": "test",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = SimpleNamespace(
        root=str(root),
        image_dir="images",
        captions_file="captions_index.jsonl",
        split="test",
        max_items=0,
        benchmark_tag="flickr_placeholder",
    )
    try:
        load_flickr30k(cfg)
    except RuntimeError as exc:
        assert "placeholder Flickr30k data detected" in str(exc)
    else:
        raise AssertionError("placeholder Flickr30k rows should be rejected")


def test_coco_subset_loader(tmp_path: Path):
    root = tmp_path / "coco"
    images = root / "val2017"
    ann = root / "annotations"
    images.mkdir(parents=True, exist_ok=True)
    ann.mkdir(parents=True, exist_ok=True)

    _write_image(images / "000000000001.jpg", 0)
    _write_image(images / "000000000002.jpg", 1)

    payload = {
        "images": [
            {"id": 1, "file_name": "000000000001.jpg", "width": 32, "height": 32},
            {"id": 2, "file_name": "000000000002.jpg", "width": 32, "height": 32},
        ],
        "annotations": [
            {"id": 10, "image_id": 1, "caption": "a sample one"},
            {"id": 11, "image_id": 2, "caption": "a sample two"},
        ],
    }
    (ann / "captions_val2017_subset.json").write_text(json.dumps(payload), encoding="utf-8")

    cfg = SimpleNamespace(
        root=str(root),
        image_dir="val2017",
        captions_file="annotations/captions_val2017_subset.json",
        split="val",
        max_items=0,
        benchmark_tag="coco_test",
    )
    out = load_coco_subset(cfg)
    assert len(out) == 2
    assert out[0].metadata["dataset"] == "coco_subset"


def test_coco_subset_loader_rejects_placeholder_rows(tmp_path: Path):
    root = tmp_path / "coco"
    images = root / "val2017"
    ann = root / "annotations"
    images.mkdir(parents=True, exist_ok=True)
    ann.mkdir(parents=True, exist_ok=True)
    _write_image(images / "demo_0000.jpg", 0)
    payload = {
        "images": [{"id": 1, "file_name": "demo_0000.jpg", "width": 32, "height": 32}],
        "annotations": [{"id": 10, "image_id": 1, "caption": "Demo caption 0000: multimodal retrieval placeholder sample."}],
    }
    (ann / "captions_val2017_subset.json").write_text(json.dumps(payload), encoding="utf-8")
    cfg = SimpleNamespace(
        root=str(root),
        image_dir="val2017",
        captions_file="annotations/captions_val2017_subset.json",
        split="val",
        max_items=0,
        benchmark_tag="coco_placeholder",
    )
    try:
        load_coco_subset(cfg)
    except RuntimeError as exc:
        assert "placeholder COCO subset data detected" in str(exc)
    else:
        raise AssertionError("placeholder COCO rows should be rejected")


def test_mini_flickr_loader_uses_seed_dataset():
    cfg = SimpleNamespace(
        root="",
        image_dir="",
        captions_file="",
        split="test",
        max_items=4,
        benchmark_tag="mini_flickr_test",
    )
    out = load_mini_flickr(cfg)
    assert len(out) == 4
    assert all(item.metadata["dataset"] == "mini_flickr" for item in out)
