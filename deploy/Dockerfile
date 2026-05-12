FROM node:22-bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/app/.venv \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    COREPACK_ENABLE_DOWNLOAD_PROMPT=0 \
    PNPM_HOME=/pnpm \
    PATH=/app/.venv/bin:/pnpm:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    PYTHON_BIN=/app/.venv/bin/python \
    USE_EXISTING_PYTHON=1 \
    PYTHONPATH=/app/src \
    LISTEN_HOST=0.0.0.0 \
    PORT=8000 \
    MMSEC_BOOTSTRAP_ENABLED=1 \
    MMSEC_ARTIFACTS_DIR=/app/artifacts \
    MMSEC_APP_DB=/app/artifacts/app.db \
    MMSEC_HF_LOCAL_ONLY=1 \
    HF_HOME=/app/.hf-cache \
    TRANSFORMERS_CACHE=/app/.hf-cache/transformers \
    HF_DATASETS_CACHE=/app/.hf-cache/datasets \
    HF_HUB_DISABLE_XET=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    build-essential \
    ca-certificates \
    curl \
    git \
    rsync \
    procps \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /app/.venv \
    && /app/.venv/bin/pip install --upgrade pip setuptools wheel \
    && corepack enable \
    && corepack prepare pnpm@9.12.3 --activate

COPY requirements.txt requirements-gpu.txt pyproject.toml ./
RUN /app/.venv/bin/pip install -r requirements.txt

ARG INSTALL_CUDA_TORCH=0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126
RUN if [ "$INSTALL_CUDA_TORCH" = "1" ]; then \
      /app/.venv/bin/python -m pip install torch torchvision torchaudio --index-url "$TORCH_INDEX_URL"; \
      /app/.venv/bin/pip install -r requirements-gpu.txt; \
    fi

COPY frontend/package.json frontend/pnpm-lock.yaml ./frontend/
RUN pnpm -C frontend install --frozen-lockfile

COPY . .
RUN mkdir -p /app/artifacts /app/data /app/logs /app/tmp /app/.hf-cache \
    && pnpm -C frontend build \
    && chmod +x /app/scripts/*.sh || true

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=5 \
  CMD curl -fsS http://127.0.0.1:8000/api/v1/health || exit 1

CMD ["bash", "scripts/run_backend.sh"]
