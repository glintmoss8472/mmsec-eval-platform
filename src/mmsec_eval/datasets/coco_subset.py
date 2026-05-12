from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from mmsec_eval.io.jsonl_io import read_jsonl
from mmsec_eval.types import Sample


def _resolve_path(base: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else base / p


def _select_value(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for k in keys:
        if k in row and row[k] is not None:
            v = str(row[k]).strip()
            if v:
                return v
    return ""


def _guess_split(image_name: str) -> str:
    lower = image_name.lower()
    if "train" in lower:
        return "train"
    if "val" in lower:
        return "val"
    if "test" in lower:
        return "test"
    return "unknown"


def _looks_like_placeholder_rows(rows: list[dict[str, Any]]) -> bool:
    sample = rows[: min(8, len(rows))]
    captions = [_select_value(row, ("caption", "text")).lower() for row in sample]
    captions = [text for text in captions if text]
    if not captions:
        return False
    return all(("placeholder sample" in text) or text.startswith("demo caption ") for text in captions)


def _read_rows(captions_path: Path) -> list[dict[str, Any]]:
    suffix = captions_path.suffix.lower()
    if suffix == ".jsonl":
        return read_jsonl(str(captions_path))

    if suffix in {".csv", ".tsv"}:
        dialect = "excel-tab" if suffix == ".tsv" else "excel"
        with captions_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, dialect=dialect)
            return [dict(row) for row in reader]

    data = json.loads(captions_path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]

    if isinstance(data, dict) and "annotations" in data and "images" in data:
        image_map = {
            int(item["id"]): {
                "file_name": str(item.get("file_name", "")),
                "width": int(item.get("width", 0) or 0),
                "height": int(item.get("height", 0) or 0),
            }
            for item in data.get("images", [])
            if isinstance(item, dict) and "id" in item
        }
        rows: list[dict[str, Any]] = []
        for ann in data.get("annotations", []):
            if not isinstance(ann, dict):
                continue
            image_id = ann.get("image_id")
            if image_id is None:
                continue
            image_meta = image_map.get(int(image_id), {})
            image_name = str(image_meta.get("file_name", ""))
            rows.append(
                {
                    "id": ann.get("id", ""),
                    "image_id": int(image_id),
                    "image": image_name,
                    "caption": str(ann.get("caption", "")),
                    "split": _guess_split(image_name),
                    "width": int(image_meta.get("width", 0)),
                    "height": int(image_meta.get("height", 0)),
                }
            )
        return rows

    return []


def load_coco_subset(dataset_cfg: Any) -> list[Sample]:
    root = Path(dataset_cfg.root)
    image_dir = _resolve_path(root, dataset_cfg.image_dir)
    captions_path = _resolve_path(root, dataset_cfg.captions_file)
    split = str(dataset_cfg.split or "").lower().strip()
    max_items = int(dataset_cfg.max_items or 0)

    rows = _read_rows(captions_path)
    if _looks_like_placeholder_rows(rows) and os.getenv("MMSEC_ALLOW_PLACEHOLDER_DATA", "0").strip().lower() not in {"1", "true", "yes"}:
        raise RuntimeError(
            f"placeholder COCO subset data detected in {captions_path}; "
            "real benchmark data is required unless MMSEC_ALLOW_PLACEHOLDER_DATA=1 is set."
        )
    out: list[Sample] = []
    for idx, row in enumerate(rows):
        row_split = _select_value(row, ("split",)).lower()
        if row_split in {"", "unknown"} and split and split not in {"all", "*"}:
            row_split = split
        if split and split not in {"all", "*"} and row_split and row_split != split:
            continue

        image_name = _select_value(row, ("image", "image_path", "file_name", "filename"))
        if not image_name:
            continue
        img_path = _resolve_path(image_dir, image_name)
        if not img_path.exists():
            alt = _resolve_path(root, image_name)
            if not alt.exists():
                continue
            img_path = alt

        caption = _select_value(row, ("caption", "text"))
        sample_id = _select_value(row, ("sample_id", "id"))
        if not sample_id:
            sample_id = f"coco-{idx:06d}"
        target_text = _select_value(row, ("target_text",))

        img = Image.open(img_path).convert("RGB")
        arr = np.asarray(img).astype(np.float32) / 255.0
        metadata = {
            "dataset": "coco_subset",
            "split": row_split or split or _guess_split(image_name),
            "source_image": str(img_path),
            "image_id": row.get("image_id"),
            "benchmark_tag": str(dataset_cfg.benchmark_tag or "coco_subset"),
        }
        for key in ("target_text", "object_category", "category_id", "bbox", "area", "iscrowd"):
            if key in row:
                metadata[key] = row.get(key)

        out.append(
            Sample(
                sample_id=sample_id,
                image=arr,
                text=caption,
                target_text=target_text,
                metadata=metadata,
            )
        )
        if max_items > 0 and len(out) >= max_items:
            break
    return out
