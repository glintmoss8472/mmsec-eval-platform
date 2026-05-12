#!/usr/bin/env bash
# 文件说明：该文件属于运维与实验脚本，集中实现 run backend 相关逻辑。
set -euo pipefail

LISTEN_HOST="${LISTEN_HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
BOOTSTRAP_ENABLED="${MMSEC_BOOTSTRAP_ENABLED:-0}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFERRED_PYTHON="${PREFERRED_PYTHON:-/HARD-DATA/bks/aat-model-stack/vlm-py310-pip/bin/python}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "${PREFERRED_PYTHON}" ]]; then
    PYTHON_BIN="${PREFERRED_PYTHON}"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    PYTHON_BIN="$(command -v python)"
  fi
fi
VENV_DIR="${PROJECT_ROOT}/.venv"
RUNTIME_PYTHON_BIN=""

cd "${PROJECT_ROOT}"

# 中文注释：实现 python_mm 的核心流程，支撑运维与实验脚本中的业务语义和异常边界。
python_mm() {
  local bin_path="$1"
  "${bin_path}" - <<'PY'
import sys
print(f"{sys.version_info[0]}.{sys.version_info[1]}")
PY
}

# 中文注释：实现 python_ok 的核心流程，支撑运维与实验脚本中的业务语义和异常边界。
python_ok() {
  local version
  version="$(python_mm "$1" 2>/dev/null || echo '0.0')"
  local major="${version%%.*}"
  local minor="${version##*.}"
  [[ "${major}" =~ ^[0-9]+$ ]] || return 1
  [[ "${minor}" =~ ^[0-9]+$ ]] || return 1
  (( major > 3 || (major == 3 && minor >= 10) ))
}

# 中文注释：实现 python_has_runtime_deps 的核心流程，支撑运维与实验脚本中的业务语义和异常边界。
python_has_runtime_deps() {
  local bin_path="$1"
  "${bin_path}" - <<'PY' >/dev/null 2>&1
import fastapi  # noqa: F401
import pydantic  # noqa: F401
import torch  # noqa: F401
import uvicorn  # noqa: F401
import yaml  # noqa: F401
PY
}

if ! python_ok "${PYTHON_BIN}"; then
  echo "[RUN_BACKEND] Python >= 3.10 is required, got: $(python_mm "${PYTHON_BIN}" 2>/dev/null || echo unknown)" >&2
  exit 1
fi

if [[ -x "${VENV_DIR}/bin/python" ]] && ! python_ok "${VENV_DIR}/bin/python"; then
  if [[ "$(cd "${PROJECT_ROOT}" && pwd)" == "${PROJECT_ROOT}" && "${VENV_DIR}" == "${PROJECT_ROOT}/.venv" ]]; then
    rm -rf "${VENV_DIR}"
  else
    echo "[RUN_BACKEND] Refusing to remove unexpected venv path: ${VENV_DIR}" >&2
    exit 1
  fi
fi

# The server session currently exports a dead localhost proxy. Clear all proxy
# variables before any pip or runtime network calls so local model services and
# offline HF assets do not try to route through 127.0.0.1:7890.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"

USE_EXISTING_PYTHON="${USE_EXISTING_PYTHON:-auto}"
if [[ "${USE_EXISTING_PYTHON}" != "0" && "${USE_EXISTING_PYTHON}" != "false" && "${USE_EXISTING_PYTHON}" != "False" ]]; then
  if python_has_runtime_deps "${PYTHON_BIN}"; then
    RUNTIME_PYTHON_BIN="${PYTHON_BIN}"
    echo "[RUN_BACKEND] Using existing runtime Python: ${RUNTIME_PYTHON_BIN}" >&2
  fi
fi

if [[ -z "${RUNTIME_PYTHON_BIN}" ]]; then
  if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  fi
  "${VENV_DIR}/bin/python" -m pip install --upgrade pip
  "${VENV_DIR}/bin/python" -m pip install -r requirements.txt
  RUNTIME_PYTHON_BIN="${VENV_DIR}/bin/python"
fi

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export MMSEC_BOOTSTRAP_ENABLED="${BOOTSTRAP_ENABLED}"
export MMSEC_ARTIFACTS_DIR="${MMSEC_ARTIFACTS_DIR:-${PROJECT_ROOT}/artifacts}"
export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/.hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export MMSEC_HF_LOCAL_ONLY="${MMSEC_HF_LOCAL_ONLY:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
mkdir -p "${MMSEC_ARTIFACTS_DIR}/hf_models" "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${HF_DATASETS_CACHE}"

case "${MMSEC_HF_LOCAL_ONLY}" in
  0|false|False|no|No)
    export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
    ;;
  *)
    unset HF_ENDPOINT
    ;;
esac

exec "${RUNTIME_PYTHON_BIN}" -m uvicorn mmsec_api.main:app --host "${LISTEN_HOST}" --port "${PORT}"
