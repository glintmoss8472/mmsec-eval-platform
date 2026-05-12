# 文件说明：该文件属于运维与实验脚本，集中实现 prepare coco2014 generation 相关逻辑。
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


VAL2014_URL = "http://images.cocodataset.org/zips/val2014.zip"
COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
VQA_Q_URL = "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip"
VQA_A_URL = "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip"


# 执行 `download` 辅助逻辑，保持运维与实验脚本中的输入处理和结果输出一致。
def _download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=180) as response, target.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)


# 确保 `zip` 已准备好，不满足条件时主动创建、下载或报错。
def _ensure_zip(path: Path, url: str, *, allow_download: bool) -> Path:
    if path.exists() and path.stat().st_size > 0:
        return path
    if not allow_download:
        raise FileNotFoundError(f"missing {path}; rerun with the matching download flag")
    print(f"[COCO2014] downloading {url}", file=sys.stderr)
    _download(url, path)
    return path


# 提取 `member`，从归档、结果或响应中取出后续流程需要的字段。
def _extract_member(zip_path: Path, member: str, target: Path) -> Path:
    if target.exists() and target.stat().st_size > 0:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        with archive.open(member, "r") as src, target.open("wb") as dst:
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)
    return target


# 加载 `JSON`，把外部文件、配置或运行产物转换为内存结构。
def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


# 确保 `annotations` 已准备好，不满足条件时主动创建、下载或报错。
def _ensure_annotations(root: Path, *, allow_download: bool) -> dict[str, Path]:
    archives = root / "archives"
    ann_dir = root / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)

    # 确保 `JSON` 已准备好，不满足条件时主动创建、下载或报错。
    def ensure_json(target: Path, zip_name: str, url: str, member: str) -> Path:
        if target.exists() and target.stat().st_size > 0:
            return target
        zip_path = _ensure_zip(archives / zip_name, url, allow_download=allow_download)
        return _extract_member(zip_path, member, target)

    return {
        "captions": ensure_json(
            ann_dir / "captions_val2014.json",
            "annotations_trainval2014.zip",
            COCO_ANN_URL,
            "annotations/captions_val2014.json",
        ),
        "instances": ensure_json(
            ann_dir / "instances_val2014.json",
            "annotations_trainval2014.zip",
            COCO_ANN_URL,
            "annotations/instances_val2014.json",
        ),
        "vqa_questions": ensure_json(
            ann_dir / "v2_OpenEnded_mscoco_val2014_questions.json",
            "v2_Questions_Val_mscoco.zip",
            VQA_Q_URL,
            "v2_OpenEnded_mscoco_val2014_questions.json",
        ),
        "vqa_annotations": ensure_json(
            ann_dir / "v2_mscoco_val2014_annotations.json",
            "v2_Annotations_Val_mscoco.zip",
            VQA_A_URL,
            "v2_mscoco_val2014_annotations.json",
        ),
    }


# 执行 `图像 名称` 辅助逻辑，保持运维与实验脚本中的输入处理和结果输出一致。
def _image_name(image_id: int) -> str:
    return f"COCO_val2014_{image_id:012d}.jpg"


# 整理 `行记录 图像 引用路径` 路径信息，把本地文件或产物引用转换成统一表示。
def _row_image_ref(image_id: int) -> str:
    return f"../val2014/{_image_name(image_id)}"


# 确保 `图像` 已准备好，不满足条件时主动创建、下载或报错。
def _ensure_images(root: Path, image_ids: set[int], *, allow_download: bool) -> None:
    if not image_ids:
        return
    image_dir = root / "val2014"
    missing = [image_id for image_id in sorted(image_ids) if not (image_dir / _image_name(image_id)).exists()]
    if not missing:
        return

    zip_path = _ensure_zip(root / "archives" / "val2014.zip", VAL2014_URL, allow_download=allow_download)
    image_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        names = set(archive.namelist())
        for image_id in missing:
            file_name = _image_name(image_id)
            member = f"val2014/{file_name}"
            if member not in names:
                raise FileNotFoundError(f"{zip_path} does not contain {member}")
            _extract_member(zip_path, member, image_dir / file_name)


# 执行 `answer aliases` 辅助逻辑，保持运维与实验脚本中的输入处理和结果输出一致。
def _answer_aliases(annotation: dict[str, Any]) -> list[str]:
    answers = [str(item.get("answer", "")).strip() for item in annotation.get("answers", []) if isinstance(item, dict)]
    aliases = [answer for answer, _ in Counter(answer for answer in answers if answer).most_common()]
    main = str(annotation.get("multiple_choice_answer") or "").strip()
    if main and main not in aliases:
        aliases.insert(0, main)
    return aliases[:8]


# 构建 `视觉问答 rows` 数据，集中整理运维与实验脚本需要的输出结构。
def _build_vqa_rows(paths: dict[str, Path], *, max_items: int) -> tuple[list[dict[str, Any]], set[int]]:
    questions = _load_json(paths["vqa_questions"]).get("questions", [])
    annotations = {
        int(item["question_id"]): item
        for item in _load_json(paths["vqa_annotations"]).get("annotations", [])
        if isinstance(item, dict) and item.get("question_id") is not None
    }
    rows: list[dict[str, Any]] = []
    image_ids: set[int] = set()
    for question in questions:
        if max_items > 0 and len(rows) >= max_items:
            break
        if not isinstance(question, dict):
            continue
        question_id = int(question.get("question_id", -1))
        image_id = int(question.get("image_id", -1))
        annotation = annotations.get(question_id)
        answer = str((annotation or {}).get("multiple_choice_answer") or "").strip()
        prompt = str(question.get("question") or "").strip()
        if image_id < 0 or not prompt or not answer:
            continue
        image_ids.add(image_id)
        rows.append(
            {
                "id": f"vqa2-val2014-{question_id}",
                "image": _row_image_ref(image_id),
                "question": prompt,
                "answer": answer,
                "answer_aliases": _answer_aliases(annotation or {}),
                "target_object": answer,
                "attack_goal": "answer_change",
                "question_type": str((annotation or {}).get("question_type") or ""),
                "answer_type": str((annotation or {}).get("answer_type") or ""),
            }
        )
    return rows, image_ids


# 执行 `图像描述 maps` 辅助逻辑，保持运维与实验脚本中的输入处理和结果输出一致。
def _caption_maps(captions_payload: dict[str, Any]) -> tuple[dict[int, str], dict[int, list[str]]]:
    image_names = {
        int(item["id"]): str(item.get("file_name", ""))
        for item in captions_payload.get("images", [])
        if isinstance(item, dict) and item.get("id") is not None
    }
    captions: dict[int, list[str]] = defaultdict(list)
    for ann in captions_payload.get("annotations", []):
        if not isinstance(ann, dict) or ann.get("image_id") is None:
            continue
        caption = str(ann.get("caption") or "").strip()
        if caption:
            captions[int(ann["image_id"])].append(caption)
    return image_names, captions


# 构建 `图像描述 rows` 数据，集中整理运维与实验脚本需要的输出结构。
def _build_caption_rows(paths: dict[str, Path], *, categories: set[str], max_items: int, min_area: float) -> tuple[list[dict[str, Any]], set[int]]:
    captions_payload = _load_json(paths["captions"])
    image_names, captions = _caption_maps(captions_payload)
    instances = _load_json(paths["instances"])
    cat_by_id = {
        int(item["id"]): str(item.get("name", ""))
        for item in instances.get("categories", [])
        if isinstance(item, dict) and item.get("id") is not None
    }

    objects: dict[int, list[tuple[str, float]]] = defaultdict(list)
    for ann in instances.get("annotations", []):
        if not isinstance(ann, dict) or ann.get("image_id") is None:
            continue
        category = cat_by_id.get(int(ann.get("category_id", -1)), "")
        if categories and category not in categories:
            continue
        area = float(ann.get("area", 0.0) or 0.0)
        if area < min_area:
            continue
        objects[int(ann["image_id"])].append((category, area))

    rows: list[dict[str, Any]] = []
    image_ids: set[int] = set()
    for image_id, object_items in sorted(objects.items()):
        if max_items > 0 and len(rows) >= max_items:
            break
        refs = captions.get(image_id, [])
        if not refs or image_id not in image_names:
            continue
        object_items.sort(key=lambda item: item[1], reverse=True)
        target_object = object_items[0][0]
        other_objects = [name for name, _ in object_items[1:6] if name != target_object]
        image_ids.add(image_id)
        rows.append(
            {
                "id": f"caption-val2014-{image_id}",
                "image": _row_image_ref(image_id),
                "reference_captions": refs[:5],
                "clean_caption": refs[0],
                "target_object": target_object,
                "target_aliases": [target_object],
                "non_target_objects": other_objects,
                "attack_goal": "remove_object",
            }
        )
    return rows, image_ids


# 写出 `JSONL`，保证后续报告、页面或复现实验能读取。
def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


# 作为 `prepare_coco2014_generation.py` 的执行入口，串联参数读取、核心处理和退出状态。
def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare COCO2014/VQA-v2 JSONL files for generation-style evaluation.")
    parser.add_argument("--root", default="data/coco2014")
    parser.add_argument("--download", action="store_true", help="Download official annotation archives when absent.")
    parser.add_argument("--download-images", action="store_true", help="Download val2014.zip when selected images are absent.")
    parser.add_argument("--max-vqa", type=int, default=1000)
    parser.add_argument("--max-caption", type=int, default=1000)
    parser.add_argument("--min-area", type=float, default=400.0)
    parser.add_argument("--categories", default="person,dog,cat,cup,stop sign,car,bus,traffic light,bicycle")
    parser.add_argument("--vqa-output", default="generation/vqa_v2_coco_val.jsonl")
    parser.add_argument("--caption-output", default="generation/coco_caption_object_val.jsonl")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    paths = _ensure_annotations(root, allow_download=bool(args.download))
    vqa_rows, vqa_image_ids = _build_vqa_rows(paths, max_items=int(args.max_vqa))
    categories = {item.strip() for item in str(args.categories or "").split(",") if item.strip()}
    caption_rows, caption_image_ids = _build_caption_rows(
        paths,
        categories=categories,
        max_items=int(args.max_caption),
        min_area=float(args.min_area),
    )
    _ensure_images(root, vqa_image_ids | caption_image_ids, allow_download=bool(args.download_images))

    _write_jsonl(root / str(args.vqa_output), vqa_rows)
    _write_jsonl(root / str(args.caption_output), caption_rows)
    print(json.dumps({"vqa_rows": len(vqa_rows), "caption_rows": len(caption_rows)}, ensure_ascii=False))
    return 0 if vqa_rows and caption_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
