from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mmsec_eval.model_adapters.local_vlm_catalog import local_vlm_model_map


MODEL_MAP = local_vlm_model_map()
DEFAULT_MODELS = ",".join(MODEL_MAP)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", default="artifacts/local_vlm")
    parser.add_argument("--models", default=DEFAULT_MODELS)
    args = parser.parse_args()

    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    selected = [item.strip() for item in str(args.models or "").split(",") if item.strip()]
    ignore_patterns = [
        "*.onnx",
        "*.ot",
        "*.tflite",
        "*.msgpack",
        "*.h5",
        "*.ckpt",
        "original/*",
        "demo/*",
        "assets/*",
    ]

    for key in selected:
        model_id = MODEL_MAP.get(key)
        if not model_id:
            raise KeyError(f"unknown model key: {key}")
        target = out_root / key
        target.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=model_id,
            local_dir=str(target),
            local_dir_use_symlinks=False,
            ignore_patterns=ignore_patterns,
            resume_download=True,
        )
        (target / ".source_model").write_text(f"{model_id}\n", encoding="utf-8")
        print(f"{key} -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
