#!/usr/bin/env bash
# 文件说明：该文件属于运维与实验脚本，集中实现 install torch cuda 相关逻辑。
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"

echo "[INFO] Using Python: ${PYTHON_BIN}"
echo "[INFO] Using torch index: ${TORCH_INDEX_URL}"
"${PYTHON_BIN}" -m pip install --upgrade pip
"${PYTHON_BIN}" -m pip install torch torchvision torchaudio --index-url "${TORCH_INDEX_URL}"
echo "[INFO] CUDA-enabled torch install finished."
