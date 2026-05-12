from __future__ import annotations

import argparse
import json
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from PIL import Image, ImageDraw


def _truthy(value: str) -> bool:
    return str(value).strip().lower() not in {"0", "false", "no", "off", ""}


def _allow_synthetic_fallback(raw: str | None = None) -> bool:
    if raw is None:
        raw = os.getenv("MMSEC_ALLOW_PLACEHOLDER_DATA", "0")
    return _truthy(str(raw))


def _download(url: str, target: Path) -> bool:
    try:
        with requests.get(url, timeout=90, stream=True) as response:
            response.raise_for_status()
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        return True
    except (OSError, requests.RequestException):
        return False


def _download_image(split: str, file_name: str, target: Path) -> bool:
    url = f"http://images.cocodataset.org/{split}/{file_name}"
    return _download(url, target)


def _extract_member(zip_path: Path, member_name: str, target_root: Path) -> bool:
    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            info = archive.getinfo(member_name)
            archive.extract(info, target_root)
        return True
    except (KeyError, OSError, zipfile.BadZipFile):
        return False


def _placeholder(image_path: Path, text: str, idx: int) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    width = 224
    height = 224
    color = (
        (57 * (idx + 1)) % 255,
        (111 * (idx + 3)) % 255,
        (181 * (idx + 7)) % 255,
    )
    image = Image.new("RGB", (width, height), color)
    drawer = ImageDraw.Draw(image)
    drawer.rectangle((10, 158, 214, 214), fill=(255, 255, 255))
    drawer.text((18, 170), text[:46], fill=(24, 24, 24))
    image.save(image_path)


@dataclass(frozen=True)
class CocoPaths:
    dataset_root: Path
    ann_dir: Path
    img_dir: Path
    ann_file: Path
    subset_file: Path


def _resolve_paths(root: str, split: str) -> CocoPaths:
    dataset_root = Path(root)
    if not dataset_root.is_absolute():
        dataset_root = Path.cwd() / dataset_root
    ann_dir = dataset_root / "annotations"
    img_dir = dataset_root / split
    return CocoPaths(
        dataset_root=dataset_root,
        ann_dir=ann_dir,
        img_dir=img_dir,
        ann_file=ann_dir / f"captions_{split}.json",
        subset_file=ann_dir / f"captions_{split}_subset.json",
    )


def _maybe_download_annotations(paths: CocoPaths, *, split: str, need_annotations: bool) -> None:
    if not need_annotations or paths.ann_file.exists():
        return
    zip_path = paths.dataset_root / "annotations_trainval2017.zip"
    url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    if _download(url, zip_path):
        _extract_member(zip_path, f"annotations/captions_{split}.json", paths.dataset_root)


def _load_caption_payload(ann_file: Path) -> tuple[dict[str, Any], dict[int, dict[str, Any]], list[Any]]:
    data = json.loads(ann_file.read_text(encoding="utf-8"))
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    image_map = {int(item["id"]): item for item in images if isinstance(item, dict) and "id" in item}
    return data, image_map, annotations if isinstance(annotations, list) else []


def _select_subset(
    *,
    annotations: list[Any],
    image_map: dict[int, dict[str, Any]],
    img_dir: Path,
    split: str,
    target_count: int,
    download_images: bool,
    allow_download: bool,
    allow_synthetic_fallback: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    kept: list[dict[str, Any]] = []
    images_kept: list[dict[str, Any]] = []
    seen_image_ids: set[int] = set()
    skipped_images: list[str] = []

    for ann in annotations:
        if not isinstance(ann, dict) or "image_id" not in ann:
            continue
        image_id = int(ann["image_id"])
        if image_id in seen_image_ids:
            continue
        image_meta = image_map.get(image_id)
        if not image_meta:
            continue
        file_name = str(image_meta.get("file_name", "")).strip()
        if not file_name:
            continue
        image_path = img_dir / file_name
        image_ready = image_path.exists()
        if not image_ready and (download_images or allow_download):
            image_ready = _download_image(split, file_name, image_path)
        if not image_ready and allow_synthetic_fallback:
            _placeholder(image_path, file_name, len(images_kept))
            image_ready = True
        if not image_ready:
            skipped_images.append(file_name)
            continue
        kept.append({"id": ann.get("id", f"coco-{image_id}"), "image_id": image_id, "caption": ann.get("caption", "")})
        images_kept.append(image_meta)
        seen_image_ids.add(image_id)
        if len(images_kept) >= target_count:
            break
    return kept, images_kept, skipped_images


def _write_subset_files(paths: CocoPaths, data: dict[str, Any], kept: list[dict[str, Any]], images_kept: list[dict[str, Any]]) -> Path:
    out = {
        "info": data.get("info", {}),
        "licenses": data.get("licenses", []),
        "images": images_kept,
        "annotations": kept,
        "type": "captions",
    }
    paths.subset_file.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    jsonl_path = paths.ann_dir / f"{paths.subset_file.stem}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row, image_meta in zip(kept, images_kept):
            handle.write(
                json.dumps(
                    {
                        "id": row.get("id", ""),
                        "image": str(image_meta.get("file_name", "")),
                        "caption": str(row.get("caption", "")),
                        "split": "val",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return jsonl_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-Root", dest="root", default="data/coco")
    parser.add_argument("-Split", dest="split", default="val2017")
    parser.add_argument("-MaxItems", dest="max_items", type=int, default=500)
    parser.add_argument("-DownloadAnnotations", dest="download_annotations", action="store_true")
    parser.add_argument("-DownloadImages", dest="download_images", action="store_true")
    parser.add_argument("-AutoDownload", dest="auto_download", default="true")
    parser.add_argument("-AllowSyntheticFallback", dest="allow_synthetic_fallback", default="false")
    args = parser.parse_args()

    paths = _resolve_paths(args.root, args.split)
    paths.ann_dir.mkdir(parents=True, exist_ok=True)
    paths.img_dir.mkdir(parents=True, exist_ok=True)

    allow_download = _truthy(args.auto_download)
    allow_synthetic_fallback = _allow_synthetic_fallback(args.allow_synthetic_fallback)
    need_annotations = args.download_annotations or allow_download
    _maybe_download_annotations(paths, split=args.split, need_annotations=need_annotations)

    if not paths.ann_file.exists():
        print(f"[PREP] Missing annotation file: {paths.ann_file}")
        return 1

    data, image_map, annotations = _load_caption_payload(paths.ann_file)
    target_count = int(args.max_items or 0)
    if target_count <= 0:
        target_count = len(image_map)

    kept, images_kept, skipped_images = _select_subset(
        annotations=annotations,
        image_map=image_map,
        img_dir=paths.img_dir,
        split=args.split,
        target_count=target_count,
        download_images=bool(args.download_images),
        allow_download=allow_download,
        allow_synthetic_fallback=allow_synthetic_fallback,
    )

    if not allow_synthetic_fallback and len(images_kept) < target_count:
        print(
            f"[PREP] Unable to build a fully real COCO subset. requested={target_count} "
            f"ready={len(images_kept)} skipped={len(skipped_images)}"
        )
        if skipped_images:
            print(f"[PREP] first_missing={skipped_images[:5]}")
        return 1

    jsonl_path = _write_subset_files(paths, data, kept, images_kept)
    print(f"[PREP] subset_annotations={paths.subset_file}")
    print(f"[PREP] subset_jsonl={jsonl_path}")
    print(f"[PREP] images={len(images_kept)} annotations={len(kept)}")
    print("[PREP] source=real_public_coco" + ("+synthetic_fallback" if allow_synthetic_fallback else ""))
    print(f"[PREP] COCO subset ready: {paths.dataset_root}")
    return 0 if images_kept and kept else 1


if __name__ == "__main__":
    raise SystemExit(main())
