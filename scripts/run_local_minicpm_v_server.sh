#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_local_vlm_server_env.sh"

MODEL_ID="${MODEL_ID:-openbmb/MiniCPM-V-4_5}"
PUBLIC_MODEL_ID="${PUBLIC_MODEL_ID:-openbmb/MiniCPM-V-4_5}"
LOCAL_MODEL_DIR="${LOCAL_MODEL_DIR:-${APP_ROOT}/artifacts/local_vlm/minicpm_v}"
MINICPM_PYDEPS_DIR="${MINICPM_PYDEPS_DIR:-${STACK_ROOT}/vendor/minicpm_pydeps}"
if [[ "${MODEL_ID}" != *"MiniCPM-V-4_5"* && "${PUBLIC_MODEL_ID}" != *"MiniCPM-V-4_5"* && -d "${MINICPM_PYDEPS_DIR}/transformers" ]]; then
  EXTRA_PYTHONPATH_DIR="${MINICPM_PYDEPS_DIR}"
fi
TARGET_PORT="${TARGET_PORT:-8015}"
mmsec_launch_local_openai_mm_server "minicpm_v_45.log"
