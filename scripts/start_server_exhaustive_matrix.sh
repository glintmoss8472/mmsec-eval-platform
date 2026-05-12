#!/usr/bin/env bash
# 文件说明：该文件属于运维与实验脚本，集中实现 start server exhaustive matrix 相关逻辑。
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACTS_DIR="${MMSEC_ARTIFACTS_DIR:-$PROJECT_ROOT/artifacts}"
if [ "$#" -gt 0 ]; then
  OUT_ROOT="$1"
  shift
else
  OUT_ROOT="$ARTIFACTS_DIR/exhaustive_matrix_$(date -u +%Y%m%d_%H%M%S)"
fi

mkdir -p "$OUT_ROOT"
mkdir -p "/HARD-DATA/bks/tmp"

export TMPDIR="${TMPDIR:-/HARD-DATA/bks/tmp}"
export MMSEC_ARTIFACTS_DIR="$ARTIFACTS_DIR"
export HF_HOME="${HF_HOME:-$ARTIFACTS_DIR/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/HARD-DATA/bks/.cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/HARD-DATA/bks/.cache/pip}"

cd "$PROJECT_ROOT"
nohup "$PROJECT_ROOT/.venv/bin/python" "$PROJECT_ROOT/scripts/run_server_exhaustive_matrix.py" \
  --api-base "http://127.0.0.1:8000/api/v1" \
  --out-dir "$OUT_ROOT" \
  --resume \
  --timeout-seconds "${MMSEC_EXHAUSTIVE_TIMEOUT_SECONDS:-86400}" \
  --poll-seconds "${MMSEC_EXHAUSTIVE_POLL_SECONDS:-15}" \
  "$@" \
  > "$OUT_ROOT/exhaustive_matrix.log" 2>&1 &

echo $! > "$OUT_ROOT/exhaustive_matrix.pid"
echo "$OUT_ROOT"
