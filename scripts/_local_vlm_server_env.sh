#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export STACK_ROOT="${STACK_ROOT:-${PROJECT_ROOT}}"
export APP_ROOT="${APP_ROOT:-${PROJECT_ROOT}}"

mmsec_truthy() {
  local value="${1:-}"
  case "${value}" in
    1|true|True|yes|Yes|on|On)
      return 0
      ;;
  esac
  return 1
}

mmsec_local_model_ready() {
  local model_dir="${1:-}"
  [[ -n "${model_dir}" ]] || return 1
  [[ -d "${model_dir}" ]] || return 1
  [[ -f "${model_dir}/config.json" ]] || return 1
  [[ -f "${model_dir}/.source_model" ]] || return 1
  find "${model_dir}" \
    \( -path "*/._____temp" -o -path "*/._____temp/*" -o -path "*/.cache" -o -path "*/.cache/*" \) -prune \
    -o -type f \( -name "*.safetensors" -o -name "pytorch_model*.bin" -o -name "model*.bin" \) \
    -print -quit 2>/dev/null | grep -q .
}

mmsec_pick_best_gpu() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "0"
    return
  fi
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t',' -k2,2nr \
    | head -n1 \
    | cut -d',' -f1 \
    | tr -d ' '
}

mmsec_gpu_count() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "0"
    return
  fi
  nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' '
}

mmsec_single_tenant_local_vlm() {
  local mode="${MMSEC_LOCAL_VLM_SINGLE_TENANT:-auto}"
  case "${mode}" in
    1|true|True|yes|Yes|on|On)
      return 0
      ;;
    0|false|False|no|No|off|Off)
      return 1
      ;;
  esac

  local gpu_count
  gpu_count="$(mmsec_gpu_count)"
  [[ "${gpu_count}" =~ ^[0-9]+$ ]] || gpu_count="0"
  (( gpu_count <= 1 ))
}

mmsec_default_cleanup_ports() {
  local target_port="$1"
  local all_ports="${MMSEC_LOCAL_VLM_ALL_PORTS:-8011 8012 8013 8014 8015 8016 8017}"
  if mmsec_single_tenant_local_vlm; then
    echo "${all_ports}"
    return
  fi
  echo "${target_port}"
}

mmsec_prepare_local_vlm_env() {
  export ENV_DIR="${ENV_DIR:-${PROJECT_ROOT}/.venv}"
  export LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/model_servers}"
  export TMP_WORK="${TMPDIR:-${PROJECT_ROOT}/tmp/model_servers}"
  export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/artifacts/hf-cache}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
  export LOCAL_VLM_ROOT="${LOCAL_VLM_ROOT:-${PROJECT_ROOT}/artifacts/local_vlm}"
  export PYTHON_BIN="${PYTHON_BIN:-${ENV_DIR}/bin/python}"
  export SERVER_SCRIPT="${SERVER_SCRIPT:-${PROJECT_ROOT}/scripts/local_openai_mm_server.py}"
  export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
  if [[ -n "${HF_ENDPOINT:-}" ]]; then
    export HF_ENDPOINT
  else
    unset HF_ENDPOINT
  fi

  if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="$(mmsec_pick_best_gpu)"
  else
    export CUDA_VISIBLE_DEVICES
  fi

  mkdir -p "${LOG_DIR}" "${TMP_WORK}" "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${HF_DATASETS_CACHE}" "${LOCAL_VLM_ROOT}"
  export TMPDIR="${TMP_WORK}"

  if [[ ! -f "${PYTHON_BIN}" ]]; then
    echo "missing python runtime: ${PYTHON_BIN}" >&2
    return 1
  fi
  if [[ ! -f "${SERVER_SCRIPT}" ]]; then
    echo "missing local server entrypoint: ${SERVER_SCRIPT}" >&2
    return 1
  fi
}

mmsec_launch_local_openai_mm_server() {
  local log_file_name="$1"

  mmsec_prepare_local_vlm_env

  if [[ -n "${EXTRA_PYTHONPATH_DIR:-}" && -d "${EXTRA_PYTHONPATH_DIR}" ]]; then
    export PYTHONPATH="${EXTRA_PYTHONPATH_DIR}:${PYTHONPATH:-}"
  fi

  if mmsec_truthy "${MMSEC_LOCAL_VLM_REQUIRE_OFFLINE:-0}"; then
    if ! mmsec_local_model_ready "${LOCAL_MODEL_DIR:-}"; then
      echo "offline local VLM assets are incomplete: ${LOCAL_MODEL_DIR:-missing}" >&2
      return 1
    fi
    MODEL_ID="${LOCAL_MODEL_DIR}"
    export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
    export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
    export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
  elif mmsec_local_model_ready "${LOCAL_MODEL_DIR:-}"; then
    MODEL_ID="${LOCAL_MODEL_DIR}"
    export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
    export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
  fi

  if [[ "${MMSEC_MODEL_SERVER_PREFLIGHT:-0}" == "1" ]]; then
    echo "python=${PYTHON_BIN} server=${SERVER_SCRIPT} gpu=${CUDA_VISIBLE_DEVICES} model=${MODEL_ID} port=${TARGET_PORT}" >&2
    return 0
  fi

  TARGET_PORT="${TARGET_PORT:?TARGET_PORT is required}"
  CLEANUP_PORTS="${CLEANUP_PORTS:-$(mmsec_default_cleanup_ports "${TARGET_PORT}")}"
  for port in ${CLEANUP_PORTS}; do
    pkill -f "local_openai_mm_server.py.*--port ${port}" || true
  done

  nohup "${PYTHON_BIN}" "${SERVER_SCRIPT}" \
    --model-id "${MODEL_ID}" \
    --public-model-id "${PUBLIC_MODEL_ID}" \
    --host "127.0.0.1" \
    --port "${TARGET_PORT}" \
    --dtype "${DTYPE_NAME:-float16}" \
    > "${LOG_DIR}/${log_file_name}" 2>&1 &

  echo $!
}
