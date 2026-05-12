# 文件说明：该文件属于数据集加载层，集中实现 flickr30k 相关逻辑。
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


# 解析 `路径` 的真实位置或配置值，减少调用方重复分支。
def _resolve_path(base: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else base / p


# 读取 `rows`，并对缺失或异常输入做边界处理。
def _read_rows(captions_path: Path) -> list[dict[str, Any]]:
    suffix = captions_path.suffix.lower()
    if suffix == ".jsonl":
        return read_jsonl(str(captions_path))

    if suffix == ".json":
        data = json.loads(captions_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [row for row in data if isinstance(row, dict)]
        if isinstance(data, dict) and "images" in data and "annotations" in data:
            id_to_name = {int(row["id"]): str(row["file_name"]) for row in data.get("images", []) if "id" in row and "file_name" in row}
            out: list[dict[str, Any]] = []
            for ann in data.get("annotations", []):
                if not isinstance(ann, dict):
                    continue
                image_id = ann.get("image_id")
                out.append(
                    {
                        "id": ann.get("id", ""),
                        "image": id_to_name.get(int(image_id), "") if image_id is not None else "",
                        "caption": ann.get("caption", ""),
                        "split": ann.get("split", ""),
                    }
                )
            return out
        return []

    if suffix in {".csv", ".tsv"}:
        dialect = "excel-tab" if suffix == ".tsv" else "excel"
        with captions_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, dialect=dialect)
            return [dict(row) for row in reader]

    # Common Flickr30k format: "<image>#<id>\t<caption>"
    rows: list[dict[str, Any]] = []
    for i, line in enumerate(captions_path.read_text(encoding="utf-8", errors="ignore").splitlines()):
        line = line.strip()
        if not line:
            continue
        if "\t" in line:
            left, caption = line.split("\t", 1)
            image = left.split("#", 1)[0].strip()
            rows.append({"id": f"line-{i:06d}", "image": image, "caption": caption.strip()})
            continue
        rows.append({"id": f"line-{i:06d}", "image": "", "caption": line})
    return rows


# 筛选 `value`，按配置条件保留可用于评测或展示的数据。
def _select_value(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for k in keys:
        if k in row and row[k] is not None:
            v = str(row[k]).strip()
            if v:
                return v
    return ""


# 整理 `looks like placeholder rows` 行记录，把原始结果转换成列表接口和报告可消费的结构。
def _looks_like_placeholder_rows(rows: list[dict[str, Any]]) -> bool:
    sample = rows[: min(8, len(rows))]
    captions = [_select_value(row, ("caption", "text", "sentence")).lower() for row in sample]
    captions = [text for text in captions if text]
    if not captions:
        return False
    return all(("placeholder sample" in text) or text.startswith("demo caption ") for text in captions)


# 加载 `Flickr like`，把外部文件、配置或运行产物转换为内存结构。
def load_flickr_like(dataset_cfg: Any, *, dataset_name: str = "flickr30k") -> list[Sample]:
    root = Path(dataset_cfg.root)
    image_dir = _resolve_path(root, dataset_cfg.image_dir)
    captions_path = _resolve_path(root, dataset_cfg.captions_file)
    split = str(dataset_cfg.split or "").lower().strip()
    max_items = int(dataset_cfg.max_items or 0)

    rows = _read_rows(captions_path)
    if _looks_like_placeholder_rows(rows) and os.getenv("MMSEC_ALLOW_PLACEHOLDER_DATA", "0").strip().lower() not in {"1", "true", "yes"}:
        raise RuntimeError(
            f"placeholder Flickr30k data detected in {captions_path}; "
            "real benchmark data is required unless MMSEC_ALLOW_PLACEHOLDER_DATA=1 is set."
        )
    out: list[Sample] = []

    for idx, row in enumerate(rows):
        row_split = _select_value(row, ("split", "subset")).lower()
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

        caption = _select_value(row, ("caption", "text", "sentence"))
        sample_id = _select_value(row, ("sample_id", "id"))
        if not sample_id:
            sample_id = f"flickr30k-{idx:06d}"
        target_text = _select_value(row, ("target_text",))

        img = Image.open(img_path).convert("RGB")
        arr = np.asarray(img).astype(np.float32) / 255.0
        out.append(
            Sample(
                sample_id=sample_id,
                image=arr,
                text=caption,
                target_text=target_text,
                metadata={
                    "dataset": dataset_name,
                    "split": row_split or split or "unknown",
                    "source_image": str(img_path),
                    "benchmark_tag": str(dataset_cfg.benchmark_tag or dataset_name),
                },
            )
        )
        if max_items > 0 and len(out) >= max_items:
            break

    return out


# 加载 `Flickr30k`，把外部文件、配置或运行产物转换为内存结构。
def load_flickr30k(dataset_cfg: Any) -> list[Sample]:
    return load_flickr_like(dataset_cfg, dataset_name="flickr30k")
