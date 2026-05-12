#!/usr/bin/env bash
# 文件说明：该文件属于运维与实验脚本，集中实现 run local ovis25 server 相关逻辑。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_local_vlm_server_env.sh"

MODEL_ID="${MODEL_ID:-AIDC-AI/Ovis2.5-9B}"
PUBLIC_MODEL_ID="${PUBLIC_MODEL_ID:-AIDC-AI/Ovis2.5-9B}"
LOCAL_MODEL_DIR="${LOCAL_MODEL_DIR:-${APP_ROOT:-${SCRIPT_DIR}/..}/artifacts/local_vlm/ovis25}"
TARGET_PORT="${TARGET_PORT:-8016}"
DTYPE_NAME="${DTYPE_NAME:-bfloat16}"
mmsec_launch_local_openai_mm_server "ovis25_9b.log"
