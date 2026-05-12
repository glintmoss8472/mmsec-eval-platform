from __future__ import annotations

from pathlib import Path
from typing import Any


_GENERATION_JSONL_FILES = {
    "vqa_v2_coco_val": "vqa_v2_coco_val.jsonl",
    "coco_object_probe_val": "coco_object_probe_val.jsonl",
    "coco_caption_object_val": "coco_caption_object_val.jsonl",
}


def _default_root_for(name: str, project_root: Path) -> Path:
    key = name.strip().lower()
    if key == "coco_subset":
        return project_root / "data" / "coco"
    if key in {"flickr30k", "flickr1k"}:
        return project_root / "data" / "flickr30k"
    if key == "mini_flickr":
        return project_root / "data" / "mini_flickr"
    if key in _GENERATION_JSONL_FILES:
        return project_root / "data" / "coco2014" / "generation"
    return project_root


def _resolve_root_path(name: str, root_path: str, project_root: Path) -> Path:
    raw = root_path.strip()
    if raw:
        path = Path(raw)
        return path if path.is_absolute() else project_root / path
    return _default_root_for(name, project_root)


def _required_paths(name: str, root: Path) -> list[tuple[Path, str]]:
    key = name.strip().lower()
    if key == "coco_subset":
        return [
            (root / "val2017", "缺少图像目录 val2017"),
            (root / "annotations" / "captions_val2017_subset.json", "缺少 captions_val2017_subset.json"),
        ]
    if key == "flickr1k":
        return [
            (root / "images", "缺少图像目录 images"),
            (root / "captions_index_single.jsonl", "缺少 captions_index_single.jsonl"),
        ]
    if key in {"flickr30k", "mini_flickr"}:
        return [
            (root / "images", "缺少图像目录 images"),
            (root / "captions_index.jsonl", "缺少 captions_index.jsonl"),
        ]
    if key in _GENERATION_JSONL_FILES:
        filename = _GENERATION_JSONL_FILES[key]
        return [(root / filename, f"缺少生成式评测 JSONL：{filename}")]
    return [(root, "当前 root_path 不存在")]


def enrich_dataset_registry_rows(rows: list[dict[str, Any]], project_root: Path) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        name = str(item.get("name", "") or "").strip()
        prepared = bool(item.get("prepared", False))
        item_count = int(item.get("item_count", 0) or 0)
        root_path = str(item.get("root_path", "") or "")
        resolved_root = _resolve_root_path(name, root_path, project_root)

        ready = False
        ready_reason = ""
        if not prepared:
            ready_reason = "数据库中尚未标记为 prepared=true"
        elif not resolved_root.exists():
            ready_reason = f"当前 root_path 不存在：{resolved_root}"
        else:
            missing_message = next((message for path, message in _required_paths(name, resolved_root) if not path.exists()), "")
            if missing_message:
                ready_reason = missing_message
            elif item_count <= 0:
                ready_reason = "item_count <= 0，不能判定为当前可用"
            else:
                ready = True

        item["prepared"] = prepared
        item["ready"] = ready
        item["ready_reason"] = ready_reason
        enriched.append(item)
    return enriched
