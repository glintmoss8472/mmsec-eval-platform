#!/usr/bin/env bash
# 文件说明：该文件属于运维与实验脚本，集中实现 run frontend 相关逻辑。
set -euo pipefail

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-5173}"
MODE="${MODE:-dev}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${PROJECT_ROOT}"

if ! command -v pnpm >/dev/null 2>&1; then
  corepack enable >/dev/null 2>&1 || true
  corepack prepare pnpm@9.12.3 --activate >/dev/null 2>&1 || true
fi

pnpm -C frontend install

if [[ "${MODE}" == "preview" ]]; then
  pnpm -C frontend build
  exec pnpm -C frontend preview --host "${HOST}" --port "${PORT}"
fi

exec pnpm -C frontend dev --host "${HOST}" --port "${PORT}"
