#!/usr/bin/env bash
# 文件说明：该文件属于运维与实验脚本，集中实现 install local model env 相关逻辑。
set -euo pipefail

STACK_ROOT="${STACK_ROOT:-/HARD-DATA/bks/aat-model-stack}"
MINIFORGE_DIR="${MINIFORGE_DIR:-${STACK_ROOT}/miniforge3}"
ENV_DIR="${ENV_DIR:-${STACK_ROOT}/vlm-py310}"
TMP_WORK="${TMPDIR:-/HARD-DATA/bks/tmp}"
INSTALLER="${STACK_ROOT}/Miniforge3-Linux-x86_64.sh"

mkdir -p "${STACK_ROOT}" "${TMP_WORK}"
export TMPDIR="${TMP_WORK}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

if [[ ! -x "${MINIFORGE_DIR}/bin/conda" ]]; then
  curl -L "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" -o "${INSTALLER}"
  bash "${INSTALLER}" -b -p "${MINIFORGE_DIR}"
fi

"${MINIFORGE_DIR}/bin/conda" create -y -p "${ENV_DIR}" python=3.10
"${ENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel
"${ENV_DIR}/bin/python" -m pip install --index-url "https://download.pytorch.org/whl/cu121" torch torchvision
"${ENV_DIR}/bin/python" -m pip install "transformers>=4.57.0" accelerate fastapi "uvicorn[standard]" pillow requests sentencepiece safetensors
