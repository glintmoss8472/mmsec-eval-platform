#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_local_vlm_server_env.sh"

MODEL_ID="${MODEL_ID:-OpenGVLab/InternVL3_5-8B-HF}"
PUBLIC_MODEL_ID="${PUBLIC_MODEL_ID:-OpenGVLab/InternVL3_5-8B-HF}"
LOCAL_MODEL_DIR="${LOCAL_MODEL_DIR:-${APP_ROOT:-${SCRIPT_DIR}/..}/artifacts/local_vlm/internvl35}"
TARGET_PORT="${TARGET_PORT:-8014}"
mmsec_launch_local_openai_mm_server "internvl35_8b.log"
