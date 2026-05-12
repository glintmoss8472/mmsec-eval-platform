#!/usr/bin/env bash
# 文件说明：该文件属于运维与实验脚本，集中实现 run local qwen3 vl server 相关逻辑。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_local_vlm_server_env.sh"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}"
PUBLIC_MODEL_ID="${PUBLIC_MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}"
LOCAL_MODEL_DIR="${LOCAL_MODEL_DIR:-${APP_ROOT:-${SCRIPT_DIR}/..}/artifacts/local_vlm/qwen3_vl}"
TARGET_PORT="${TARGET_PORT:-8012}"
mmsec_launch_local_openai_mm_server "qwen3_vl_8b.log"
