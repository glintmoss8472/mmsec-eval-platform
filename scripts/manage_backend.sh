#!/usr/bin/env bash
# 文件说明：该文件属于运维与实验脚本，集中实现 manage backend 相关逻辑。
set -euo pipefail

ACTION="${1:-status}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${LISTEN_HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
LOG_DIR="${PROJECT_ROOT}/logs"
PID_FILE="${PID_FILE:-${LOG_DIR}/backend-local.pid}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/backend-local.log}"
HEALTH_URL="http://${HOST}:${PORT}/api/v1/health"

mkdir -p "${LOG_DIR}"

# 处理 `read pid` 步骤，封装脚本中的可复用命令片段。
read_pid() {
  if [[ -s "${PID_FILE}" ]]; then
    tr -d '[:space:]' < "${PID_FILE}"
  fi
}

# 处理 `pid alive` 步骤，封装脚本中的可复用命令片段。
pid_alive() {
  local pid="${1:-}"
  [[ -n "${pid}" ]] && kill -0 "${pid}" >/dev/null 2>&1
}

# 处理 `pid cmdline` 步骤，封装脚本中的可复用命令片段。
pid_cmdline() {
  local pid="${1:-}"
  if [[ -n "${pid}" && -r "/proc/${pid}/cmdline" ]]; then
    tr '\0' ' ' < "/proc/${pid}/cmdline"
  fi
}

# 处理 `pid looks like backend` 步骤，封装脚本中的可复用命令片段。
pid_looks_like_backend() {
  local pid="${1:-}"
  local cmd
  cmd="$(pid_cmdline "${pid}")"
  [[ "${cmd}" == *"mmsec_api.main:app"* || "${cmd}" == *"scripts/run_backend.sh"* ]]
}

# 处理 `port pid` 步骤，封装脚本中的可复用命令片段。
port_pid() {
  if command -v ss >/dev/null 2>&1; then
    ss -ltnp 2>/dev/null \
      | awk -v port=":${PORT}" '$4 ~ port {print $0}' \
      | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' \
      | head -n 1
    return 0
  fi
  ps -eo pid=,comm=,args= \
    | awk -v port="${PORT}" '$2 ~ /python/ && $0 ~ /uvicorn mmsec_api.main:app/ && $0 ~ ("--port " port) {print $1; exit}'
}

# 处理 `health ok` 步骤，封装脚本中的可复用命令片段。
health_ok() {
  curl -fsS "${HEALTH_URL}" >/dev/null 2>&1
}

# 启动 `start backend` 进程，并准备日志、端口或运行目录。
start_backend() {
  local pid
  pid="$(read_pid || true)"
  if pid_alive "${pid}"; then
    echo "backend already running pid=${pid}"
    return 0
  fi

  local existing_pid
  existing_pid="$(port_pid || true)"
  if pid_alive "${existing_pid}"; then
    if pid_looks_like_backend "${existing_pid}"; then
      echo "${existing_pid}" > "${PID_FILE}"
      echo "adopted backend pid=${existing_pid}"
      return 0
    fi
    echo "port ${PORT} is already used by pid=${existing_pid}; refusing to start" >&2
    return 1
  fi

  cd "${PROJECT_ROOT}"
  nohup env LISTEN_HOST="${HOST}" PORT="${PORT}" MMSEC_BOOTSTRAP_ENABLED="${MMSEC_BOOTSTRAP_ENABLED:-0}" \
    bash "${PROJECT_ROOT}/scripts/run_backend.sh" >> "${LOG_FILE}" 2>&1 &
  pid="$!"
  echo "${pid}" > "${PID_FILE}"

  for _ in $(seq 1 60); do
    if health_ok; then
      echo "backend started pid=${pid} ${HEALTH_URL}"
      return 0
    fi
    if ! pid_alive "${pid}"; then
      echo "backend exited during startup; see ${LOG_FILE}" >&2
      return 1
    fi
    sleep 1
  done
  echo "backend did not become healthy in time; see ${LOG_FILE}" >&2
  return 1
}

# 停止 `stop backend` 进程，避免残留服务占用资源。
stop_backend() {
  local pid
  pid="$(read_pid || true)"
  if ! pid_alive "${pid}"; then
    pid="$(port_pid || true)"
  fi
  if ! pid_alive "${pid}"; then
    rm -f "${PID_FILE}"
    echo "backend not running"
    return 0
  fi
  if ! pid_looks_like_backend "${pid}"; then
    echo "pid=${pid} does not look like this backend; refusing to stop it" >&2
    return 1
  fi

  kill "${pid}" >/dev/null 2>&1 || true
  for _ in $(seq 1 30); do
    if ! pid_alive "${pid}"; then
      rm -f "${PID_FILE}"
      echo "backend stopped pid=${pid}"
      return 0
    fi
    sleep 1
  done
  echo "backend pid=${pid} did not stop after SIGTERM" >&2
  return 1
}

# 处理 `状态 backend` 步骤，封装脚本中的可复用命令片段。
status_backend() {
  local pid
  pid="$(read_pid || true)"
  if ! pid_alive "${pid}"; then
    pid="$(port_pid || true)"
  fi
  if pid_alive "${pid}"; then
    if health_ok; then
      echo "backend running pid=${pid} healthy ${HEALTH_URL}"
    else
      echo "backend running pid=${pid} but health check failed ${HEALTH_URL}"
    fi
    return 0
  fi
  echo "backend stopped"
}

case "${ACTION}" in
  start)
    start_backend
    ;;
  stop)
    stop_backend
    ;;
  restart)
    stop_backend
    start_backend
    ;;
  status)
    status_backend
    ;;
  logs)
    tail -n "${LINES:-120}" "${LOG_FILE}"
    ;;
  *)
    echo "usage: $0 {start|stop|restart|status|logs}" >&2
    exit 2
    ;;
esac
