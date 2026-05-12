#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACTS_DIR="${MMSEC_ARTIFACTS_DIR:-${PROJECT_ROOT}/artifacts}"
OUT_ROOT="${1:-${ARTIFACTS_DIR}/autodl_full_matrix_$(date -u +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
API_BASE="${MMSEC_MATRIX_API_BASE:-http://127.0.0.1:8000/api/v1}"
DATASETS="${MMSEC_MATRIX_DATASETS:-coco_subset}"
MODES="${MMSEC_MATRIX_MODES:-standard}"
ATTACKS="${MMSEC_MATRIX_ATTACKS:-}"
REPEATS="${MMSEC_MATRIX_REPEATS:-3}"
SEED_BASE="${MMSEC_MATRIX_SEED_BASE:-20260429}"
MAX_ITEMS="${MMSEC_MATRIX_MAX_ITEMS:-5000}"
CLIP_MAX_PAIRS="${MMSEC_CLIP_MAX_PAIRS:-0}"
ITM_MAX_PAIRS="${MMSEC_ITM_MAX_PAIRS:-8192}"
VLM_MAX_PAIRS="${MMSEC_VLM_MAX_PAIRS:-1024}"
TIMEOUT_SECONDS="${MMSEC_MATRIX_TIMEOUT_SECONDS:-86400}"
POLL_SECONDS="${MMSEC_MATRIX_POLL_SECONDS:-15}"

mkdir -p "${OUT_ROOT}" "${PROJECT_ROOT}/logs"
cd "${PROJECT_ROOT}"

export MMSEC_HF_LOCAL_ONLY="${MMSEC_HF_LOCAL_ONLY:-1}"
export MMSEC_OPENAI_COMPAT_CONCURRENCY="${MMSEC_OPENAI_COMPAT_CONCURRENCY:-1}"
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

wait_backend() {
  "${PYTHON_BIN}" - <<'PY'
import requests, time, sys
s = requests.Session(); s.trust_env = False
url = "http://127.0.0.1:8000/api/v1/health"
deadline = time.time() + 180
last_error = None
while time.time() < deadline:
    try:
        r = s.get(url, timeout=10)
        if r.status_code == 200:
            sys.exit(0)
    except requests.RequestException as exc:
        last_error = exc
    time.sleep(2)
if last_error is not None:
    print(f"backend health probe failed: {last_error}", file=sys.stderr)
raise SystemExit(1)
PY
}

ensure_backend() {
  if wait_backend >/dev/null 2>&1; then
    return 0
  fi
  nohup env LISTEN_HOST=127.0.0.1 PORT=8000 MMSEC_BOOTSTRAP_ENABLED=0 \
    MMSEC_HF_LOCAL_ONLY="${MMSEC_HF_LOCAL_ONLY}" \
    MMSEC_ARTIFACTS_DIR="${ARTIFACTS_DIR}" \
    bash "${PROJECT_ROOT}/scripts/run_backend.sh" \
    > "${OUT_ROOT}/backend_8000.log" 2>&1 &
  echo $! > "${OUT_ROOT}/backend_8000.pid"
  wait_backend
}

stop_local_vlm_ports() {
  ps -eo pid,args \
    | awk '/local_openai_mm_server.py/ && /--port 801[1-7]/ && !/awk/ {print $1}' \
    | xargs -r kill
  sleep 3
}

wait_local_model() {
  local port="$1"
  "${PYTHON_BIN}" - <<PY
import requests, time, sys
s = requests.Session(); s.trust_env = False
url = "http://127.0.0.1:${port}/v1/models"
deadline = time.time() + 900
last_error = None
while time.time() < deadline:
    try:
        r = s.get(url, timeout=10)
        if r.status_code == 200:
            sys.exit(0)
    except requests.RequestException as exc:
        last_error = exc
    time.sleep(5)
if last_error is not None:
    print(f"local model probe failed on port ${port}: {last_error}", file=sys.stderr)
raise SystemExit(1)
PY
}

export_prompt_order() {
  local adapter="$1"
  local order="$2"
  case "${adapter}" in
    openai_qwen35_9b) export MMSEC_OPENAI_QWEN35_9B_PROMPT_ORDER="${order}" ;;
    openai_qwen3_vl) export MMSEC_OPENAI_QWEN3_VL_PROMPT_ORDER="${order}" ;;
    openai_qwen25_vl) export MMSEC_OPENAI_QWEN25_VL_PROMPT_ORDER="${order}" ;;
    openai_internvl35) export MMSEC_OPENAI_INTERNVL35_PROMPT_ORDER="${order}" ;;
    openai_minicpm_v) export MMSEC_OPENAI_MINICPM_V_PROMPT_ORDER="${order}" ;;
    openai_ovis25) export MMSEC_OPENAI_OVIS25_PROMPT_ORDER="${order}" ;;
    openai_gemma3_12b) export MMSEC_OPENAI_GEMMA3_12B_PROMPT_ORDER="${order}" ;;
  esac
}

calibrate_prompt_order() {
  local adapter="$1"
  local out_json="${OUT_ROOT}/${adapter}_prompt_order_calibration.json"
  local out_log="${OUT_ROOT}/${adapter}_prompt_order_calibration.log"
  local order="image_first"
  set +e
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/calibrate_openai_prompt_order.py"     --adapter "${adapter}"     --max-items "${MMSEC_PROMPT_CALIBRATION_ITEMS:-6}"     --out "${out_json}"     > "${out_log}" 2>&1
  local status=$?
  set -e
  local maybe_order
  maybe_order="$(tail -n 1 "${out_log}" 2>/dev/null | tr -d '
 ')"
  if [[ "${maybe_order}" == "image_first" || "${maybe_order}" == "text_first" ]]; then
    order="${maybe_order}"
  fi
  export_prompt_order "${adapter}" "${order}"
  if [[ "${status}" -ne 0 ]]; then
    echo "[$(date -u +%FT%TZ)] warning ${adapter} prompt-order calibration invalid; using ${order} and relying on metric_quality flags" | tee -a "${OUT_ROOT}/matrix.log"
  else
    echo "[$(date -u +%FT%TZ)] ${adapter} prompt-order=${order}" | tee -a "${OUT_ROOT}/matrix.log"
  fi
}

run_matrix() {
  local shard_name="$1"
  local models="$2"
  local max_pairs="$3"
  local shard_dir="${OUT_ROOT}/${shard_name}"
  mkdir -p "${shard_dir}"
  echo "[$(date -u +%FT%TZ)] start ${shard_name} models=${models} max_pairs=${max_pairs}" | tee -a "${OUT_ROOT}/matrix.log"
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/run_server_exhaustive_matrix.py" \
    --api-base "${API_BASE}" \
    --out-dir "${shard_dir}" \
    --datasets "${DATASETS}" \
    --modes "${MODES}" \
    ${ATTACKS:+--attacks "${ATTACKS}"} \
    --models "${models}" \
    --repeats "${REPEATS}" \
    --seed-base "${SEED_BASE}" \
    --max-items "${MAX_ITEMS}" \
    --max-pairs "${max_pairs}" \
    --timeout-seconds "${TIMEOUT_SECONDS}" \
    --poll-seconds "${POLL_SECONDS}" \
    --resume \
    > "${shard_dir}/exhaustive_matrix.log" 2>&1
  echo "[$(date -u +%FT%TZ)] done ${shard_name}" | tee -a "${OUT_ROOT}/matrix.log"
}

ensure_backend
stop_local_vlm_ports

run_matrix "clip_full" "clip_hf" "${CLIP_MAX_PAIRS}"
run_matrix "itm_pair_budget" "blip_itm,vilt_itm" "${ITM_MAX_PAIRS}"

mapfile -t VLM_SPECS < <(
  PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}" "${PYTHON_BIN}" - <<'PY'
from mmsec_eval.model_adapters.local_vlm_catalog import local_vlm_launch_matrix

for adapter, launch_script, port in local_vlm_launch_matrix():
    print(f"{adapter}|{launch_script}|{port}")
PY
)

for spec in "${VLM_SPECS[@]}"; do
  IFS='|' read -r adapter launch_script port <<< "${spec}"
  stop_local_vlm_ports
  echo "[$(date -u +%FT%TZ)] launch ${adapter} on port ${port}" | tee -a "${OUT_ROOT}/matrix.log"
  bash "${PROJECT_ROOT}/${launch_script}" >> "${OUT_ROOT}/matrix.log" 2>&1
  wait_local_model "${port}"
  calibrate_prompt_order "${adapter}"
  run_matrix "${adapter}" "${adapter}" "${VLM_MAX_PAIRS}"
done

stop_local_vlm_ports
echo "${OUT_ROOT}" | tee "${OUT_ROOT}/DONE"
