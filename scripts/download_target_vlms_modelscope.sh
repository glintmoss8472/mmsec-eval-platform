#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/artifacts/local_vlm}"
MODELSCOPE_BIN="${MODELSCOPE_BIN:-modelscope}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "${OUT_ROOT}"

download_model() {
  local repo_id="$1"
  local local_name="$2"
  local target="${OUT_ROOT}/${local_name}"
  mkdir -p "${target}"
  if [[ -f "${target}/.source_model" && -f "${target}/config.json" ]]; then
    echo "[modelscope] skip completed ${local_name}: $(cat "${target}/.source_model")"
    return 0
  fi
  echo "[modelscope] ${repo_id} -> ${target}"
  "${MODELSCOPE_BIN}" download --model "${repo_id}" --local_dir "${target}"
  printf '%s\n' "${repo_id}" > "${target}/.source_model"
}

while IFS=$'\t' read -r repo_id local_name; do
  [[ -n "${repo_id}" && -n "${local_name}" ]] || continue
  download_model "${repo_id}" "${local_name}"
done < <(
  PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}" "${PYTHON_BIN}" - <<'PY'
from mmsec_eval.model_adapters.local_vlm_catalog import LOCAL_OPENAI_COMPAT_MODEL_SPECS

for spec in LOCAL_OPENAI_COMPAT_MODEL_SPECS:
    print(f"{spec.model_name_default}\t{spec.local_dir}")
PY
)
