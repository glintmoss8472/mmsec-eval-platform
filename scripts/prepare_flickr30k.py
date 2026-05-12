from __future__ import annotations

import argparse
import csv
import io
import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Any

import requests
from PIL import Image, ImageDraw


FLICKR30K_GITHUB_PARTS = [
    "https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr30k_part00",
    "https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr30k_part01",
    "https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr30k_part02",
]


def _truthy(value: str) -> bool:
    return str(value).strip().lower() not in {"0", "false", "no", "off", ""}


def _read_rows(src: Path) -> list[dict[str, Any]]:
    suffix = src.suffix.lower()
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        for line in src.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                rows.append(data)
        return rows

    if suffix == ".json":
        data = json.loads(src.read_text(encoding="utf-8", errors="ignore"))
        if isinstance(data, list):
            return [row for row in data if isinstance(row, dict)]
        if isinstance(data, dict) and "images" in data and "annotations" in data:
            id_to_name = {
                int(item["id"]): str(item["file_name"])
                for item in data.get("images", [])
                if isinstance(item, dict) and "id" in item and "file_name" in item
            }
            out: list[dict[str, Any]] = []
            for ann in data.get("annotations", []):
                if not isinstance(ann, dict):
                    continue
                image_id = ann.get("image_id")
                try:
                    image_name = id_to_name.get(int(image_id), "")
                except (TypeError, ValueError):
                    image_name = ""
                out.append(
                    {
                        "id": ann.get("id", ""),
                        "image": image_name,
                        "caption": ann.get("caption", ""),
                        "split": ann.get("split", "test"),
                    }
                )
            return out
        return []

    if suffix in {".csv", ".tsv"}:
        dialect = "excel-tab" if suffix == ".tsv" else "excel"
        with src.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, dialect=dialect)
            return [dict(row) for row in reader]

    return _read_token_text(src.read_text(encoding="utf-8", errors="ignore"))


def _read_token_text(content: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, line in enumerate(content.splitlines()):
        line = line.strip()
        if not line:
            continue
        if "\t" in line:
            left, caption = line.split("\t", 1)
            image = left.split("#", 1)[0].strip()
            rows.append({"id": f"line-{idx:06d}", "image": image, "caption": caption.strip(), "split": "test"})
        else:
            rows.append({"id": f"line-{idx:06d}", "image": "", "caption": line, "split": "test"})
    return rows


def _read_caption_blob(name: str, content: str) -> list[dict[str, Any]]:
    first_line = next((line.strip() for line in content.splitlines() if line.strip()), "")
    suffix = Path(name).suffix.lower()
    if suffix in {".csv", ".tsv"} or first_line.lower().startswith("image,caption"):
        delimiter = "\t" if suffix == ".tsv" else ","
        reader = csv.DictReader(io.StringIO(content), delimiter=delimiter)
        rows: list[dict[str, Any]] = []
        for idx, row in enumerate(reader):
            image_name = str(row.get("image") or row.get("filename") or "").strip()
            caption = str(row.get("caption") or row.get("text") or "").strip()
            if not image_name or not caption:
                continue
            rows.append({"id": f"csv-{idx:06d}", "image": image_name, "caption": caption, "split": "test"})
        return rows
    return _read_token_text(content)


def _pick(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        if key in row and row[key] is not None:
            value = str(row[key]).strip()
            if value:
                return value
    return ""


def _looks_like_placeholder_rows(rows: list[dict[str, Any]]) -> bool:
    if not rows:
        return False
    sample = rows[: min(8, len(rows))]
    captions = [str(_pick(row, "caption", "text", "sentence")).strip().lower() for row in sample]
    if not captions:
        return False
    return all(("placeholder sample" in text) or text.startswith("demo caption ") for text in captions if text)


def _allow_synthetic_fallback(raw: str | None = None) -> bool:
    if raw is None:
        raw = os.getenv("MMSEC_ALLOW_PLACEHOLDER_DATA", "0")
    return _truthy(str(raw))


def _request_get(url: str, *, timeout: int = 45, stream: bool = False) -> requests.Response:
    session = requests.Session()
    session.trust_env = False
    response = session.get(url, timeout=timeout, stream=stream, allow_redirects=True)
    response.raise_for_status()
    return response


def _placeholder(image_path: Path, caption: str, idx: int) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    width = 224
    height = 224
    color = (
        (37 * (idx + 1)) % 255,
        (91 * (idx + 3)) % 255,
        (153 * (idx + 5)) % 255,
    )
    image = Image.new("RGB", (width, height), color)
    drawer = ImageDraw.Draw(image)
    snippet = caption[:48] if caption else f"sample-{idx:03d}"
    drawer.rectangle((12, 156, 212, 212), fill=(255, 255, 255))
    drawer.text((18, 168), snippet, fill=(20, 28, 36))
    image.save(image_path)


def _resolve_existing_image(dataset_root: Path, image_dir: Path, raw_image: str) -> str:
    raw = str(raw_image or "").strip().replace("\\", "/")
    if not raw:
        return ""
    raw_name = Path(raw).name
    candidates = [raw, raw_name, f"{image_dir.name}/{raw_name}", f"images/{raw_name}", f"val2017/{raw_name}"]
    seen: set[str] = set()
    for rel in candidates:
        rel = rel.strip("/").replace("\\", "/")
        if not rel or rel in seen:
            continue
        seen.add(rel)
        if (dataset_root / rel).exists():
            return rel
    return ""


def _write_rows(dataset_root: Path, image_dir: Path, out_path: Path, rows: list[dict[str, Any]], *, synthesize_missing: bool) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    norm: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        image_name = _pick(row, "image", "image_path", "file_name", "filename")
        caption = _pick(row, "caption", "text", "sentence")
        if not image_name or not caption:
            continue
        rel_image = _resolve_existing_image(dataset_root, image_dir, image_name)
        if not rel_image and synthesize_missing:
            target_name = Path(image_name).name or f"flickr_{idx:06d}.jpg"
            rel_image = f"{image_dir.name}/{target_name}"
            _placeholder(dataset_root / rel_image, caption, idx)
        if not rel_image:
            continue
        rid = _pick(row, "id", "sample_id") or f"flickr30k-{idx:06d}"
        split = _pick(row, "split", "subset") or "test"
        norm.append({"id": rid, "image": rel_image, "caption": caption, "split": split})

    with out_path.open("w", encoding="utf-8") as handle:
        for row in norm:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(norm)


def _limit_rows_by_unique_images(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return rows
    kept: list[dict[str, Any]] = []
    seen_images: set[str] = set()
    for row in rows:
        image_name = _pick(row, "image", "image_path", "file_name", "filename")
        if not image_name:
            continue
        if image_name not in seen_images and len(seen_images) >= limit:
            continue
        seen_images.add(image_name)
        kept.append(row)
    return kept


def _try_hf_download(dataset_root: Path, image_dir: Path, out_path: Path, limit: int) -> int:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        return 0

    attempts = [
        ("nlphuji/flickr30k", "test"),
        ("nlphuji/flickr_1k_test_image_text_retrieval", "test"),
    ]
    for dataset_name, split_name in attempts:
        try:
            split = f"{split_name}[:{limit}]" if limit > 0 else split_name
            ds = load_dataset(dataset_name, split=split, trust_remote_code=True)
        except (OSError, RuntimeError, ValueError):
            continue

        rows: list[dict[str, Any]] = []
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

            image_name = f"hf_{idx:06d}.jpg"
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
                        "id": f"hf-{idx:06d}-{cap_idx:02d}",
                        "image": image_name,
                        "caption": caption,
                        "split": "test",
                    }
                )
        if saved > 0:
            return _write_rows(dataset_root, image_dir, out_path, rows, synthesize_missing=False)
    return 0


def _download_flickr30k_release(dataset_root: Path) -> tuple[Path, Path] | None:
    cache_dir = dataset_root / ".download_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    part_paths: list[Path] = []
    for idx, url in enumerate(FLICKR30K_GITHUB_PARTS):
        part_path = cache_dir / f"flickr30k_part{idx:02d}"
        if not part_path.exists() or part_path.stat().st_size == 0:
            with _request_get(url, timeout=120, stream=True) as response:
                with part_path.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
        part_paths.append(part_path)

    archive_path = cache_dir / "flickr30k.zip"
    if not archive_path.exists() or archive_path.stat().st_size == 0:
        with archive_path.open("wb") as out_handle:
            for part_path in part_paths:
                with part_path.open("rb") as in_handle:
                    while True:
                        chunk = in_handle.read(1024 * 1024)
                        if not chunk:
                            break
                        out_handle.write(chunk)
    return archive_path, cache_dir


def _extract_flickr30k_release(dataset_root: Path, image_dir: Path, limit: int) -> list[dict[str, Any]]:
    download = _download_flickr30k_release(dataset_root)
    if not download:
        return []
    archive_path, cache_dir = download
    with zipfile.ZipFile(archive_path, "r") as zf:
        caption_name = next((name for name in zf.namelist() if name.lower().endswith("captions.txt")), "")
        if not caption_name:
            return []
        rows = _limit_rows_by_unique_images(
            _read_caption_blob(caption_name, zf.read(caption_name).decode("utf-8", errors="ignore")),
            limit,
        )
        if not rows:
            return []

        archive_images: dict[str, str] = {}
        for name in zf.namelist():
            if not name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            archive_images.setdefault(Path(name).name, name)

        image_dir.mkdir(parents=True, exist_ok=True)
        for image_name in sorted({_pick(row, "image") for row in rows if _pick(row, "image")}):
            member = archive_images.get(Path(image_name).name)
            if not member:
                continue
            target = image_dir / Path(image_name).name
            if target.exists() and target.stat().st_size > 0:
                continue
            with zf.open(member, "r") as src, target.open("wb") as dst:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)

    extracted_rows = [
        {
            **row,
            "image": Path(_pick(row, "image")).name,
        }
        for row in rows
        if _pick(row, "image") and (image_dir / Path(_pick(row, "image")).name).exists()
    ]
    if extracted_rows:
        shutil.rmtree(cache_dir, ignore_errors=True)
    return extracted_rows


def _download_token_file(dataset_root: Path, download_url: str) -> Path | None:
    urls = [download_url] if download_url else []
    urls.append("https://raw.githubusercontent.com/BryanPlummer/flickr30k_entities/master/results_20130124.token")
    for url in urls:
        try:
            target = dataset_root / "results_20130124.token"
            target.write_bytes(_request_get(url, timeout=45).content)
            return target
        except (OSError, requests.RequestException):
            continue
    return None


def _seed_demo_rows(image_dir: Path, out_path: Path, count: int) -> int:
    rows: list[dict[str, Any]] = []
    for idx in range(max(16, count or 64)):
        image_name = f"demo_{idx:04d}.jpg"
        caption = f"Demo caption {idx:04d}: multimodal retrieval placeholder sample."
        _placeholder(image_dir / image_name, caption, idx)
        rows.append({"id": f"demo-{idx:04d}", "image": image_name, "caption": caption, "split": "test"})
    return _write_rows(out_path.parent, image_dir, out_path, rows, synthesize_missing=False)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-Root", dest="root", default="data/flickr30k")
    parser.add_argument("-ImageDir", dest="image_dir", default="images")
    parser.add_argument("-CaptionsSource", dest="captions_source", default="")
    parser.add_argument("-OutputFile", dest="output_file", default="captions_index.jsonl")
    parser.add_argument("-DownloadUrl", dest="download_url", default="")
    parser.add_argument("-AutoDownload", dest="auto_download", default="true")
    parser.add_argument("-SkipDownload", dest="skip_download", action="store_true")
    parser.add_argument("-MaxItems", dest="max_items", type=int, default=256)
    parser.add_argument("-AllowSyntheticFallback", dest="allow_synthetic_fallback", default="false")
    return parser.parse_args()


def _prepare_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    cwd = Path.cwd()
    dataset_root = Path(args.root)
    if not dataset_root.is_absolute():
        dataset_root = cwd / dataset_root
    image_dir = dataset_root / args.image_dir
    out_path = dataset_root / args.output_file
    dataset_root.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    return dataset_root, image_dir, out_path


def _local_candidates(dataset_root: Path, captions_source: str) -> list[Path]:
    captions_source = str(captions_source or "").strip()
    local_candidates = [
        dataset_root / "annotations" / "captions_val2017_subset.json",
        dataset_root / "annotations" / "captions_val2017.json",
        dataset_root / "results_20130124.token",
        dataset_root / "captions.txt",
        dataset_root / "captions.csv",
        dataset_root / "captions.tsv",
        dataset_root / "captions.jsonl",
        dataset_root / "captions.json",
    ]
    if captions_source:
        src = Path(captions_source)
        if not src.is_absolute():
            src = dataset_root / src
        if src.exists():
            local_candidates.insert(0, src)
    return local_candidates


def _try_local_sources(dataset_root: Path, image_dir: Path, out_path: Path, candidates: list[Path], allow_synthetic_fallback: bool) -> int:
    for candidate in candidates:
        if candidate.exists():
            rows = _read_rows(candidate)
            if rows and _looks_like_placeholder_rows(rows):
                if allow_synthetic_fallback:
                    print(f"[PREP] placeholder source accepted via allow_synthetic_fallback: {candidate}")
                else:
                    print(f"[PREP] synthetic placeholder source detected at {candidate}, rejecting it by default.")
                    continue
            written = _write_rows(dataset_root, image_dir, out_path, rows, synthesize_missing=allow_synthetic_fallback)
            if written > 0:
                print(f"[PREP] captions_source={candidate}")
                print(f"[PREP] rows={written}")
                print(f"[PREP] Flickr30k ready: {dataset_root}")
                return 0
    return -1


def _try_remote_sources(args: argparse.Namespace, dataset_root: Path, image_dir: Path, out_path: Path, allow_synthetic_fallback: bool) -> int:
    if not args.skip_download and _truthy(args.auto_download):
        release_rows = _extract_flickr30k_release(dataset_root, image_dir, args.max_items)
        written = _write_rows(dataset_root, image_dir, out_path, release_rows, synthesize_missing=False)
        if written > 0:
            unique_images = len({_pick(row, "image") for row in release_rows if _pick(row, "image")})
            print("[PREP] source=github_release_mirror")
            print(f"[PREP] rows={written}")
            print(f"[PREP] unique_images={unique_images}")
            print(f"[PREP] Flickr30k ready: {dataset_root}")
            return 0

        written = _try_hf_download(dataset_root, image_dir, out_path, args.max_items)
        if written > 0:
            print("[PREP] source=hf_dataset")
            print(f"[PREP] rows={written}")
            print(f"[PREP] Flickr30k ready: {dataset_root}")
            return 0

        token_file = _download_token_file(dataset_root, args.download_url)
        if token_file and token_file.exists():
            rows = _limit_rows_by_unique_images(_read_rows(token_file), args.max_items)
            written = _write_rows(dataset_root, image_dir, out_path, rows, synthesize_missing=allow_synthetic_fallback)
            if written > 0:
                print(f"[PREP] captions_source={token_file}")
                print("[PREP] source=flickr30k_entities_token" + ("+synthetic_images" if allow_synthetic_fallback else "+real_images"))
                print(f"[PREP] rows={written}")
                print(f"[PREP] Flickr30k ready: {dataset_root}")
                return 0
    return -1


def main() -> int:
    args = _parse_args()
    dataset_root, image_dir, out_path = _prepare_paths(args)
    allow_synthetic_fallback = _allow_synthetic_fallback(args.allow_synthetic_fallback)
    local_status = _try_local_sources(
        dataset_root,
        image_dir,
        out_path,
        _local_candidates(dataset_root, str(args.captions_source or "").strip()),
        allow_synthetic_fallback,
    )
    if local_status >= 0:
        return local_status
    remote_status = _try_remote_sources(args, dataset_root, image_dir, out_path, allow_synthetic_fallback)
    if remote_status >= 0:
        return remote_status
    if allow_synthetic_fallback:
        written = _seed_demo_rows(image_dir, out_path, args.max_items)
        if written <= 0:
            print("[PREP] captions index is empty.")
            return 1
        print("[PREP] source=synthetic_demo")
        print(f"[PREP] rows={written}")
        print(f"[PREP] Flickr30k ready: {dataset_root}")
        return 0

    print("[PREP] Real Flickr30k data not available. Synthetic fallback is disabled by default.")
    print("[PREP] Provide a real captions source with matching images, or pass -AllowSyntheticFallback true explicitly.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
