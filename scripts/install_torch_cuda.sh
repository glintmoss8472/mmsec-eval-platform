#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"

echo "[INFO] Using Python: ${PYTHON_BIN}"
echo "[INFO] Using torch index: ${TORCH_INDEX_URL}"
"${PYTHON_BIN}" -m pip install --upgrade pip
"${PYTHON_BIN}" -m pip install torch torchvision torchaudio --index-url "${TORCH_INDEX_URL}"
echo "[INFO] CUDA-enabled torch install finished."
