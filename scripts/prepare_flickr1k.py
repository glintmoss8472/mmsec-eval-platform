from __future__ import annotations

import argparse
import io
import json
import shutil
from pathlib import Path

from PIL import Image

from prepare_flickr30k import (
    _allow_synthetic_fallback,
    _limit_rows_by_unique_images,
    _pick,
    _read_rows,
    _seed_demo_rows,
    _truthy,
    _write_rows,
)


def _subset_from_existing_source(
    dataset_root: Path,
    image_dir: Path,
    out_path: Path,
    source_root: Path,
    source_image_dir_name: str,
    limit: int,
    *,
    allow_synthetic_fallback: bool,
) -> int:
    captions_path = source_root / "captions_index.jsonl"
    if not captions_path.exists():
        return 0

    rows = _limit_rows_by_unique_images(_read_rows(captions_path), limit)
    if not rows:
        return 0

    normalized: list[dict[str, str]] = []
    source_image_dir = source_root / source_image_dir_name
    for idx, row in enumerate(rows):
        image_name = _pick(row, "image", "image_path", "file_name", "filename")
        caption = _pick(row, "caption", "text", "sentence")
        if not image_name or not caption:
            continue

        source_candidates = [
            source_image_dir / image_name,
            source_image_dir / Path(image_name).name,
            source_root / image_name,
            source_root / Path(image_name).name,
        ]
        source_path = next((candidate for candidate in source_candidates if candidate.exists()), None)
        if source_path is None:
            if not allow_synthetic_fallback:
                continue
            target_name = Path(image_name).name or f"flickr1k_{idx:06d}.jpg"
            normalized.append(
                {
                    "id": _pick(row, "id", "sample_id") or f"flickr1k-{idx:06d}",
                    "image": target_name,
                    "caption": caption,
                    "split": _pick(row, "split", "subset") or "test",
                }
            )
            continue

        target_name = Path(source_path).name
        target_path = image_dir / target_name
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if not target_path.exists():
            shutil.copy2(source_path, target_path)
        normalized.append(
            {
                "id": _pick(row, "id", "sample_id") or f"flickr1k-{idx:06d}",
                "image": target_name,
                "caption": caption,
                "split": _pick(row, "split", "subset") or "test",
            }
        )

    return _write_rows(dataset_root, image_dir, out_path, normalized, synthesize_missing=allow_synthetic_fallback)


def _resolve_tmm_asset_paths(source_root: Path, tmm_json: str, tmm_image_dir: str) -> tuple[Path, Path] | None:
    json_candidates = [Path(tmm_json)] if tmm_json else []
    json_candidates.append(source_root / "flickr30k_test.json")
    image_candidates = [Path(tmm_image_dir)] if tmm_image_dir else []
    image_candidates.extend(
        [
            source_root / "flickr" / "flickr30k-images",
            source_root / "flickr30k-images",
        ]
    )
    def resolve_existing(path: Path) -> Path | None:
        candidate = path if path.is_absolute() else source_root / path
        return candidate if candidate.exists() else None

    json_path = next((candidate for path in json_candidates if (candidate := resolve_existing(path)) is not None), None)
    image_path = next((candidate for path in image_candidates if (candidate := resolve_existing(path)) is not None), None)
    if json_path is None or image_path is None:
        return None
    return json_path, image_path


def _subset_from_tmm_official_source(
    dataset_root: Path,
    image_dir: Path,
    out_path: Path,
    source_root: Path,
    limit: int,
    *,
    tmm_json: str = "",
    tmm_image_dir: str = "",
) -> int:
    resolved = _resolve_tmm_asset_paths(source_root, tmm_json, tmm_image_dir)
    if resolved is None:
        return 0
    json_path, source_image_dir = resolved
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return 0
    if not isinstance(payload, list):
        return 0

    normalized: list[dict[str, str]] = []
    for idx, row in enumerate(payload):
        if not isinstance(row, dict):
            continue
        if limit > 0 and len(normalized) >= limit:
            break
        image_name = Path(str(row.get("image", "")).strip()).name
        captions_raw = row.get("caption") or row.get("captions") or []
        if isinstance(captions_raw, str):
            captions = [captions_raw]
        elif isinstance(captions_raw, list):
            captions = [str(item).strip() for item in captions_raw if str(item).strip()]
        else:
            captions = []
        if not image_name or not captions:
            continue
        source_path = source_image_dir / image_name
        if not source_path.exists():
            continue
        target_path = image_dir / image_name
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if not target_path.exists():
            shutil.copy2(source_path, target_path)
        normalized.append(
            {
                "id": f"flickr1k-{idx:04d}",
                "image": image_name,
                "caption": captions[0],
                "split": "test",
            }
        )

    return _write_rows(dataset_root, image_dir, out_path, normalized, synthesize_missing=False)


def _try_hf_flickr1k_download(dataset_root: Path, image_dir: Path, out_path: Path, limit: int) -> int:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        return 0

    split = f"test[:{limit}]" if limit > 0 else "test"
    try:
        ds = load_dataset("nlphuji/flickr_1k_test_image_text_retrieval", split=split, trust_remote_code=True)
    except (OSError, RuntimeError, ValueError):
        return 0

    rows: list[dict[str, str]] = []
    saved = 0
    for idx, item in enumerate(ds):
        image_obj = None
        for key in ("image", "img", "jpg", "jpeg"):
            if key in item and item[key] is not None:
                image_obj = item[key]
                break
        captions = item.get("caption") or item.get("captions") or item.get("sentence") or item.get("sentences") or item.get("text")
        if isinstance(captions, str):
            captions_list = [captions]
        elif isinstance(captions, list):
            captions_list = [str(x).strip() for x in captions if str(x).strip()]
        else:
            captions_list = []
        if image_obj is None or not captions_list:
            continue

        image_name = f"hf1k_{idx:06d}.jpg"
        image_path = image_dir / image_name
        image_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if hasattr(image_obj, "save"):
                image_obj.save(image_path)
            elif isinstance(image_obj, dict) and image_obj.get("bytes"):
                Image.open(io.BytesIO(image_obj["bytes"])).convert("RGB").save(image_path)
            else:
                continue
        except (OSError, TypeError, ValueError):
            continue

        saved += 1
        for cap_idx, caption in enumerate(captions_list[:5]):
            rows.append(
                {
                    "id": f"hf1k-{idx:06d}-{cap_idx:02d}",
                    "image": image_name,
                    "caption": caption,
                    "split": "test",
                }
            )

    if saved <= 0:
        return 0
    return _write_rows(dataset_root, image_dir, out_path, rows, synthesize_missing=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-Root", dest="root", default="data/flickr1k")
    parser.add_argument("-SourceRoot", dest="source_root", default="data/flickr30k")
    parser.add_argument("-SourceImageDir", dest="source_image_dir", default="images")
    parser.add_argument("-ImageDir", dest="image_dir", default="images")
    parser.add_argument("-OutputFile", dest="output_file", default="captions_index.jsonl")
    parser.add_argument("-AutoDownload", dest="auto_download", default="true")
    parser.add_argument("-SkipDownload", dest="skip_download", action="store_true")
    parser.add_argument("-MaxItems", dest="max_items", type=int, default=1000)
    parser.add_argument("-AllowSyntheticFallback", dest="allow_synthetic_fallback", default="false")
    parser.add_argument("-TmmJson", dest="tmm_json", default="")
    parser.add_argument("-TmmImageDir", dest="tmm_image_dir", default="")
    args = parser.parse_args()

    cwd = Path.cwd()
    dataset_root = Path(args.root)
    if not dataset_root.is_absolute():
        dataset_root = cwd / dataset_root
    source_root = Path(args.source_root)
    if not source_root.is_absolute():
        source_root = cwd / source_root
    image_dir = dataset_root / args.image_dir
    out_path = dataset_root / args.output_file
    dataset_root.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    allow_synthetic_fallback = _allow_synthetic_fallback(args.allow_synthetic_fallback)

    written = _subset_from_existing_source(
        dataset_root=dataset_root,
        image_dir=image_dir,
        out_path=out_path,
        source_root=source_root,
        source_image_dir_name=str(args.source_image_dir or "images"),
        limit=int(args.max_items or 1000),
        allow_synthetic_fallback=allow_synthetic_fallback,
    )
    if written > 0:
        print(f"[PREP] source=flickr30k_subset:{source_root}")
        print(f"[PREP] rows={written}")
        print(f"[PREP] Flickr1k ready: {dataset_root}")
        return 0

    written = _subset_from_tmm_official_source(
        dataset_root=dataset_root,
        image_dir=image_dir,
        out_path=out_path,
        source_root=source_root,
        limit=int(args.max_items or 1000),
        tmm_json=str(args.tmm_json or ""),
        tmm_image_dir=str(args.tmm_image_dir or ""),
    )
    if written > 0:
        print(f"[PREP] source=tmm_official_flickr1k:{source_root}")
        print(f"[PREP] rows={written}")
        print(f"[PREP] Flickr1k ready: {dataset_root}")
        return 0

    if not args.skip_download and _truthy(args.auto_download):
        written = _try_hf_flickr1k_download(dataset_root, image_dir, out_path, int(args.max_items or 1000))
        if written > 0:
            print("[PREP] source=hf_dataset_flickr1k")
            print(f"[PREP] rows={written}")
            print(f"[PREP] Flickr1k ready: {dataset_root}")
            return 0

    if allow_synthetic_fallback:
        written = _seed_demo_rows(image_dir, out_path, int(args.max_items or 1000))
        if written <= 0:
            print("[PREP] captions index is empty.")
            return 1
        print("[PREP] source=synthetic_demo")
        print(f"[PREP] rows={written}")
        print(f"[PREP] Flickr1k ready: {dataset_root}")
        return 0

    print("[PREP] Real Flickr1k data not available. Synthetic fallback is disabled by default.")
    print("[PREP] Provide an existing Flickr30k root to subset from, or enable auto download for the 1k test split.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
