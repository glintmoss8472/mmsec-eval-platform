#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "[INFO] Installing optional attack toolkit with existing project dependencies."
"${PYTHON_BIN}" -m pip install --upgrade pip
"${PYTHON_BIN}" -m pip install --no-deps "torchattacks==3.5.1"
echo "[INFO] torchattacks installed without overriding project requests/torch pins."
