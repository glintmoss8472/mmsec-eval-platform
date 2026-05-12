from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mmsec_eval.model_adapters.local_vlm_catalog import local_vlm_calibration_map
from mmsec_eval.model_adapters.openai_compat_adapter import OpenAICompatAdapter


ADAPTER_VARIANTS = local_vlm_calibration_map()


def _load_coco_pairs(max_items: int) -> list[tuple[np.ndarray, str, str]]:
    root = Path("data/coco")
    data = json.loads((root / "annotations/captions_val2017_subset.json").read_text(encoding="utf-8"))
    image_map = {int(item["id"]): str(item["file_name"]) for item in data.get("images", [])}
    anns = [ann for ann in data.get("annotations", []) if int(ann.get("image_id", -1)) in image_map]
    anns = anns[: max(2, int(max_items) + 1)]
    out: list[tuple[np.ndarray, str, str]] = []
    for idx, ann in enumerate(anns[:max_items]):
        wrong_ann = anns[(idx + 1) % len(anns)]
        img_path = root / "val2017" / image_map[int(ann["image_id"])]
        image = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
        out.append((image, str(ann.get("caption", "")), str(wrong_ann.get("caption", ""))))
    return out


def _score_order(adapter_name: str, order: str, pairs: list[tuple[np.ndarray, str, str]]) -> dict[str, Any]:
    variant, model_name, base_url = ADAPTER_VARIANTS[adapter_name]
    os.environ[f"MMSEC_OPENAI_{variant}_MODEL_NAME"] = model_name
    os.environ[f"MMSEC_OPENAI_{variant}_BASE_URL"] = base_url
    os.environ[f"MMSEC_OPENAI_{variant}_TIMEOUT"] = os.getenv(f"MMSEC_OPENAI_{variant}_TIMEOUT", "120")
    os.environ[f"MMSEC_OPENAI_{variant}_PROMPT_ORDER"] = order
    adapter = OpenAICompatAdapter(variant=variant)

    correct: list[float] = []
    wrong: list[float] = []
    for image, correct_text, wrong_text in pairs:
        correct.append(float(adapter._request_pair(image, correct_text)["score"]))
        wrong.append(float(adapter._request_pair(image, wrong_text)["score"]))
    all_scores = np.asarray(correct + wrong, dtype=np.float32)
    correct_arr = np.asarray(correct, dtype=np.float32)
    wrong_arr = np.asarray(wrong, dtype=np.float32)
    margin = float(correct_arr.mean() - wrong_arr.mean()) if correct_arr.size else 0.0
    pairwise_win_rate = float((correct_arr > wrong_arr).mean()) if correct_arr.size else 0.0
    unique_rounded = int(len(set(float(x) for x in np.round(all_scores, 4).tolist())))
    return {
        "order": order,
        "mean_correct": float(correct_arr.mean()) if correct_arr.size else 0.0,
        "mean_wrong": float(wrong_arr.mean()) if wrong_arr.size else 0.0,
        "margin": margin,
        "pairwise_win_rate": pairwise_win_rate,
        "std": float(all_scores.std()) if all_scores.size else 0.0,
        "unique_rounded": unique_rounded,
        "valid": bool(unique_rounded > 1 and margin > 0.0 and pairwise_win_rate >= 0.5),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate image/text message order for OpenAI-compatible local VLM scorers.")
    parser.add_argument("--adapter", required=True, choices=sorted(ADAPTER_VARIANTS))
    parser.add_argument("--max-items", type=int, default=6)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    pairs = _load_coco_pairs(args.max_items)
    results = [_score_order(args.adapter, order, pairs) for order in ("image_first", "text_first")]
    best = max(results, key=lambda row: (bool(row["valid"]), float(row["margin"]), float(row["pairwise_win_rate"]), float(row["std"])))
    payload = {
        "adapter": args.adapter,
        "selected_prompt_order": best["order"],
        "valid": bool(best["valid"]),
        "results": results,
        "note": "Use selected_prompt_order only when valid=true; otherwise exclude this model from attack-strength claims.",
    }
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    print(best["order"])
    return 0 if bool(best["valid"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
