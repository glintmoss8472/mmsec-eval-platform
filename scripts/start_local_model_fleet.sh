#!/usr/bin/env bash
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
