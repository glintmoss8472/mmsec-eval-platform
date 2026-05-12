#!/usr/bin/env bash
# 文件说明：该文件属于运维与实验脚本，集中实现 start local model fleet 相关逻辑。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

while IFS= read -r launch_script; do
  [[ -n "${launch_script}" ]] || continue
  bash "${PROJECT_ROOT}/${launch_script}"
done < <(
  PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}" "${PYTHON_BIN}" - <<'PY'
from mmsec_eval.model_adapters.local_vlm_catalog import LOCAL_OPENAI_COMPAT_MODEL_SPECS

for spec in LOCAL_OPENAI_COMPAT_MODEL_SPECS:
    print(spec.launch_script)
PY
)
