#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python3 "${PROJECT_ROOT}/scripts/check_portable_assets.py" --artifacts-root "${PROJECT_ROOT}/artifacts"
docker build -t "${IMAGE_NAME:-att-project}:${TAG:-portable-latest}" "${PROJECT_ROOT}"
