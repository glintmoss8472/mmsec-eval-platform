#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_local_vlm_server_env.sh"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-9B}"
PUBLIC_MODEL_ID="${PUBLIC_MODEL_ID:-Qwen/Qwen3.5-9B}"
LOCAL_MODEL_DIR="${LOCAL_MODEL_DIR:-${APP_ROOT:-${SCRIPT_DIR}/..}/artifacts/local_vlm/qwen35_9b}"
TARGET_PORT="${TARGET_PORT:-8011}"
mmsec_launch_local_openai_mm_server "qwen35_9b.log"
