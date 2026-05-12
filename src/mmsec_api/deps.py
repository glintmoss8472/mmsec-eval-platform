from __future__ import annotations

from fastapi import Request

from mmsec_api.runtime import ensure_app_runtime
from mmsec_api.services.bootstrap_orchestrator import BootstrapOrchestrator
from mmsec_api.store.sqlite import SQLiteStore
from mmsec_api.worker.queue import JobQueue


def get_store(request: Request) -> SQLiteStore:
    ensure_app_runtime(request.app, start_queue=True, start_bootstrap=True)
    return request.app.state.store  # type: ignore[return-value]


def get_queue(request: Request) -> JobQueue:
    ensure_app_runtime(request.app, start_queue=True, start_bootstrap=True)
    return request.app.state.job_queue  # type: ignore[return-value]


def get_bootstrap(request: Request) -> BootstrapOrchestrator:
    ensure_app_runtime(request.app, start_queue=True, start_bootstrap=True)
    return request.app.state.bootstrap_orchestrator  # type: ignore[return-value]
