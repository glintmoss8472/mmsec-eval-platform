#!/usr/bin/env bash
set -euo pipefail
VALIDATE_BEFORE_PUSH="${VALIDATE_BEFORE_PUSH:-1}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${VALIDATE_BEFORE_PUSH}" == "1" ]]; then
  bash "${PROJECT_ROOT}/scripts/docker_run_offline_validation.sh"
fi
docker push "${IMAGE_REF:?IMAGE_REF is required}"
