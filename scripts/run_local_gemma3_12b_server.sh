#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_local_vlm_server_env.sh"

MODEL_ID="${MODEL_ID:-google/gemma-3-12b-it}"
PUBLIC_MODEL_ID="${PUBLIC_MODEL_ID:-google/gemma-3-12b-it}"
LOCAL_MODEL_DIR="${LOCAL_MODEL_DIR:-${APP_ROOT:-${SCRIPT_DIR}/..}/artifacts/local_vlm/gemma3_12b}"
TARGET_PORT="${TARGET_PORT:-8017}"
DTYPE_NAME="${DTYPE_NAME:-bfloat16}"
mmsec_launch_local_openai_mm_server "gemma3_12b.log"
