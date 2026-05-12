#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
PROFILE="${MMSEC_PREFETCH_PROFILE:-validated}"
if [[ -z "${LOCAL_VLM_MODELS:-}" ]]; then
  LOCAL_VLM_MODELS="$(PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}" "${PYTHON_BIN}" - <<'PY'
from mmsec_eval.model_adapters.local_vlm_catalog import LOCAL_OPENAI_COMPAT_LOCAL_DIRS

print(",".join(LOCAL_OPENAI_COMPAT_LOCAL_DIRS))
PY
)"
fi

cd "${PROJECT_ROOT}"
mkdir -p artifacts/hf_models artifacts/local_vlm
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/artifacts/hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${HF_DATASETS_CACHE}"

"${PYTHON_BIN}" - "${PROFILE}" <<'PY'
from pathlib import Path
import sys

from huggingface_hub import snapshot_download

profile = str(sys.argv[1] or "validated").strip().lower()
model_sets = {
    "validated": {
        "clip": "openai/clip-vit-base-patch32",
        "blip_itm": "Salesforce/blip-itm-base-coco",
        "vilt_itm": "dandelin/vilt-b32-finetuned-coco",
        "bert_mlm": "bert-base-uncased",
    },
    "all": {
        "clip": "openai/clip-vit-base-patch32",
        "blip_itm": "Salesforce/blip-itm-base-coco",
        "vilt_itm": "dandelin/vilt-b32-finetuned-coco",
        "bert_mlm": "bert-base-uncased",
    },
}

selected = model_sets.get(profile, model_sets["validated"])
root = Path("artifacts/hf_models")

for local_name, repo_id in selected.items():
    target = root / local_name
    if (target / "config.json").exists():
        print(f"[prefetch] skip existing HF model: {local_name} -> {target}")
        continue
    print(f"[prefetch] download HF model: {repo_id} -> {target}")
    snapshot_download(repo_id=repo_id, local_dir=str(target), local_dir_use_symlinks=False)
PY

"${PYTHON_BIN}" scripts/prefetch_local_vlm_assets.py --models "${LOCAL_VLM_MODELS}"
