#!/usr/bin/env bash
# 文件说明：该文件属于运维与实验脚本，集中实现 run fullstack 相关逻辑。
set -euo pipefail

API_PORT="${API_PORT:-8000}"
WEB_PORT="${WEB_PORT:-5173}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 中文注释：实现 cleanup 的核心流程，支撑运维与实验脚本中的业务语义和异常边界。
cleanup() {
  if [[ -n "${BACKEND_PID:-}" ]]; then
    kill "${BACKEND_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

"${PROJECT_ROOT}/scripts/run_backend.sh" &
BACKEND_PID=$!

for _ in $(seq 1 30); do
  if curl -fsS "http://127.0.0.1:${API_PORT}/api/v1/health" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

echo "Backend: http://127.0.0.1:${API_PORT}"
echo "Frontend: http://127.0.0.1:${WEB_PORT}"
HOST=0.0.0.0 PORT="${WEB_PORT}" "${PROJECT_ROOT}/scripts/run_frontend.sh"
