#!/usr/bin/env bash
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
