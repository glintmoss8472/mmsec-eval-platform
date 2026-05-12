#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_local_vlm_server_env.sh"

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-VL-7B-Instruct}"
PUBLIC_MODEL_ID="${PUBLIC_MODEL_ID:-Qwen/Qwen2.5-VL-7B-Instruct}"
LOCAL_MODEL_DIR="${LOCAL_MODEL_DIR:-${APP_ROOT:-${SCRIPT_DIR}/..}/artifacts/local_vlm/qwen25_vl}"
TARGET_PORT="${TARGET_PORT:-8013}"
mmsec_launch_local_openai_mm_server "qwen25_vl_7b.log"
