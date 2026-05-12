#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACTS_DIR="${MMSEC_ARTIFACTS_DIR:-$PROJECT_ROOT/artifacts}"
if [ "$#" -gt 0 ]; then
  OUT_ROOT="$1"
  shift
else
  OUT_ROOT="$ARTIFACTS_DIR/exhaustive_shards_$(date -u +%Y%m%d_%H%M%S)"
fi

mkdir -p "$OUT_ROOT" "/HARD-DATA/bks/tmp"

export TMPDIR="${TMPDIR:-/HARD-DATA/bks/tmp}"
export HF_HOME="${HF_HOME:-$ARTIFACTS_DIR/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/HARD-DATA/bks/.cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/HARD-DATA/bks/.cache/pip}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"

BACKEND_TIMEOUT="${MMSEC_EXHAUSTIVE_TIMEOUT_SECONDS:-86400}"
POLL_SECONDS="${MMSEC_EXHAUSTIVE_POLL_SECONDS:-15}"
RUNNER_BIN="${PROJECT_ROOT}/.venv/bin/python"

wait_backend() {
  local port="$1"
  "${RUNNER_BIN}" - <<PY
import sys, time, requests
session = requests.Session()
session.trust_env = False
url = "http://127.0.0.1:${port}/api/v1/health"
deadline = time.time() + 180
last_error = None
while time.time() < deadline:
    try:
        resp = session.get(url, timeout=10)
        if resp.status_code == 200:
            sys.exit(0)
    except requests.RequestException as exc:
        last_error = exc
    time.sleep(2)
if last_error is not None:
    print(f"backend health probe failed on port ${port}: {last_error}", file=sys.stderr)
raise SystemExit(1)
PY
}

start_backend() {
  local port="$1"
  local backend_artifacts="$2"
  local log_file="$3"
  pkill -f "uvicorn mmsec_api.main:app --host 127.0.0.1 --port ${port}" || true
  mkdir -p "$backend_artifacts"
  nohup env \
    LISTEN_HOST="127.0.0.1" \
    PORT="${port}" \
    MMSEC_BOOTSTRAP_ENABLED="0" \
    MMSEC_HF_LOCAL_ONLY="1" \
    MMSEC_OPENAI_COMPAT_CONCURRENCY="1" \
    MMSEC_ARTIFACTS_DIR="${backend_artifacts}" \
    HF_HOME="${backend_artifacts}/hf_cache" \
    TRANSFORMERS_CACHE="${backend_artifacts}/hf_cache/transformers" \
    HF_DATASETS_CACHE="${backend_artifacts}/hf_cache/datasets" \
    bash "${PROJECT_ROOT}/scripts/run_backend.sh" \
    > "${log_file}" 2>&1 &
  echo $! > "${log_file}.pid"
  wait_backend "${port}"
}

start_runner() {
  local api_base="$1"
  local shard_name="$2"
  local models="$3"
  shift 3
  local runner_out="${OUT_ROOT}/${shard_name}"
  mkdir -p "${runner_out}"
  nohup "${RUNNER_BIN}" "${PROJECT_ROOT}/scripts/run_server_exhaustive_matrix.py" \
    --api-base "${api_base}" \
    --out-dir "${runner_out}" \
    --models "${models}" \
    --resume \
    --timeout-seconds "${BACKEND_TIMEOUT}" \
    --poll-seconds "${POLL_SECONDS}" \
    "$@" \
    > "${runner_out}/exhaustive_matrix.log" 2>&1 &
  echo $! > "${runner_out}/exhaustive_matrix.pid"
}

cd "$PROJECT_ROOT"

start_backend "8001" "${PROJECT_ROOT}/artifacts_port8001" "${OUT_ROOT}/backend_8001.log"
start_backend "8002" "${PROJECT_ROOT}/artifacts_port8002" "${OUT_ROOT}/backend_8002.log"

start_runner "http://127.0.0.1:8000/api/v1" "classic_models" "clip_hf,blip_itm,vilt_itm" "$@"
start_runner "http://127.0.0.1:8001/api/v1" "vlm_group_a" "openai_qwen35_9b,openai_qwen3_vl,openai_qwen25_vl,openai_internvl35" "$@"
start_runner "http://127.0.0.1:8002/api/v1" "vlm_group_b" "openai_minicpm_v,openai_ovis25,openai_gemma3_12b" "$@"

echo "$OUT_ROOT"
