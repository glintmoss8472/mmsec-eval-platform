#!/usr/bin/env bash
# 文件说明：该文件属于运维与实验脚本，集中实现 install attack toolkit 相关逻辑。
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "[INFO] Installing optional attack toolkit with existing project dependencies."
"${PYTHON_BIN}" -m pip install --upgrade pip
"${PYTHON_BIN}" -m pip install --no-deps "torchattacks==3.5.1"
echo "[INFO] torchattacks installed without overriding project requests/torch pins."
