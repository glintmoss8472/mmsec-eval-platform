# 文件说明：该文件属于运维与实验脚本，集中实现 prepare coco object subset 相关逻辑。
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any


COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


# 加载 `JSON`，把外部文件、配置或运行产物转换为内存结构。
def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


# 执行 `download` 辅助逻辑，保持运维与实验脚本中的输入处理和结果输出一致。
def _download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as resp:
        out_path.write_bytes(resp.read())


# 确保 `instances` 已准备好，不满足条件时主动创建、下载或报错。
def _ensure_instances(coco_root: Path, *, allow_download: bool) -> Path:
    ann_dir = coco_root / "annotations"
    instances = ann_dir / "instances_val2017.json"
    if instances.exists():
        return instances

    archive = ann_dir / "annotations_trainval2017.zip"
    if not archive.exists():
        if not allow_download:
            raise FileNotFoundError(f"missing {instances} and {archive}; rerun with --download")
        print(f"[COCO-OBJECT] downloading {COCO_ANNOTATIONS_URL}", file=sys.stderr)
        _download(COCO_ANNOTATIONS_URL, archive)

    with zipfile.ZipFile(archive, "r") as zf:
        wanted = "annotations/instances_val2017.json"
        if wanted not in zf.namelist():
            raise RuntimeError(f"{archive} does not contain {wanted}")
        zf.extract(wanted, coco_root)
    if not instances.exists():
        raise RuntimeError(f"failed to extract {instances}")
    return instances


# 整理 `图像描述 rows` 行记录，把原始结果转换成列表接口和报告可消费的结构。
def _caption_rows(captions_path: Path) -> tuple[dict[int, str], dict[int, str]]:
    data = _load_json(captions_path)
    image_by_id: dict[int, str] = {}
    caption_by_id: dict[int, str] = {}

    if isinstance(data, dict) and "images" in data and "annotations" in data:
        image_by_id = {
            int(item["id"]): str(item.get("file_name", ""))
            for item in data.get("images", [])
            if isinstance(item, dict) and item.get("id") is not None
        }
        for ann in data.get("annotations", []):
            if not isinstance(ann, dict) or ann.get("image_id") is None:
                continue
            image_id = int(ann["image_id"])
            caption_by_id.setdefault(image_id, str(ann.get("caption", "")))
        return image_by_id, caption_by_id

    if isinstance(data, list):
        for row in data:
            if not isinstance(row, dict) or row.get("image_id") is None:
                continue
            image_id = int(row["image_id"])
            image_by_id.setdefault(image_id, str(row.get("image") or row.get("file_name") or row.get("filename") or ""))
            caption_by_id.setdefault(image_id, str(row.get("caption") or row.get("text") or ""))
    return image_by_id, caption_by_id


# 构建 `object subset` 数据，集中整理运维与实验脚本需要的输出结构。
def build_object_subset(
    *,
    coco_root: Path,
    captions_file: Path,
    output: Path,
    categories: set[str],
    max_items: int,
    min_area: float,
    allow_download: bool,
) -> list[dict[str, Any]]:
    instances_path = _ensure_instances(coco_root, allow_download=allow_download)
    instances = _load_json(instances_path)
    image_by_id, caption_by_id = _caption_rows(captions_file)

    cat_by_id = {
        int(item["id"]): str(item.get("name", ""))
        for item in instances.get("categories", [])
        if isinstance(item, dict) and item.get("id") is not None
    }
    images_meta = {
        int(item["id"]): item
        for item in instances.get("images", [])
        if isinstance(item, dict) and item.get("id") is not None
    }

    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for ann in instances.get("annotations", []):
        if not isinstance(ann, dict) or ann.get("image_id") is None:
            continue
        image_id = int(ann["image_id"])
        category = cat_by_id.get(int(ann.get("category_id", -1)), "")
        if categories and category not in categories:
            continue
        area = float(ann.get("area", 0.0) or 0.0)
        if area < float(min_area):
            continue
        image_name = image_by_id.get(image_id) or str(images_meta.get(image_id, {}).get("file_name", ""))
        caption = caption_by_id.get(image_id, "")
        if not image_name or not caption:
            continue
        key = (image_id, category)
        if key in seen:
            continue
        seen.add(key)
        bbox = ann.get("bbox", [])
        rows.append(
            {
                "sample_id": f"coco-obj-{image_id}-{category.replace(' ', '_')}",
                "id": f"coco-obj-{image_id}-{category.replace(' ', '_')}",
                "image_id": image_id,
                "image": image_name,
                "caption": caption,
                "target_text": category,
                "object_category": category,
                "category_id": int(ann.get("category_id", -1)),
                "bbox": bbox,
                "area": area,
                "iscrowd": int(ann.get("iscrowd", 0) or 0),
                "split": "val",
            }
        )
        if max_items > 0 and len(rows) >= max_items:
            break

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""), encoding="utf-8")
    return rows


# 作为 `prepare_coco_object_subset.py` 的执行入口，串联参数读取、核心处理和退出状态。
def main() -> int:
    parser = argparse.ArgumentParser(description="Build a real COCO object-level JSONL subset with target_text and bbox.")
    parser.add_argument("--coco-root", default="data/coco")
    parser.add_argument("--captions-file", default="data/coco/annotations/captions_val2017_subset.json")
    parser.add_argument("--output", default="data/coco/annotations/captions_val2017_object_subset.jsonl")
    parser.add_argument("--categories", default="person,dog,cat,cup,stop sign,car,bus,traffic light,bicycle")
    parser.add_argument("--max-items", type=int, default=1000)
    parser.add_argument("--min-area", type=float, default=400.0)
    parser.add_argument("--download", action="store_true", help="Download official COCO annotations if instances_val2017.json is absent.")
    args = parser.parse_args()

    categories = {x.strip() for x in str(args.categories or "").split(",") if x.strip()}
    rows = build_object_subset(
        coco_root=Path(args.coco_root).resolve(),
        captions_file=Path(args.captions_file).resolve(),
        output=Path(args.output).resolve(),
        categories=categories,
        max_items=int(args.max_items),
        min_area=float(args.min_area),
        allow_download=bool(args.download),
    )
    print(json.dumps({"output": str(Path(args.output).resolve()), "rows": len(rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
