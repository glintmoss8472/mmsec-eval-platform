# 文件说明：该文件属于项目工程，集中实现 deps 相关逻辑。
from __future__ import annotations

from fastapi import Request

from mmsec_api.runtime import ensure_app_runtime
from mmsec_api.services.bootstrap_orchestrator import BootstrapOrchestrator
from mmsec_api.store.sqlite import SQLiteStore
from mmsec_api.worker.queue import JobQueue


# 中文注释：实现 get_store 的核心流程，支撑项目工程中的业务语义和异常边界。
def get_store(request: Request) -> SQLiteStore:
    ensure_app_runtime(request.app, start_queue=True, start_bootstrap=True)
    return request.app.state.store  # type: ignore[return-value]


# 中文注释：实现 get_queue 的核心流程，支撑项目工程中的业务语义和异常边界。
def get_queue(request: Request) -> JobQueue:
    ensure_app_runtime(request.app, start_queue=True, start_bootstrap=True)
    return request.app.state.job_queue  # type: ignore[return-value]


# 中文注释：实现 get_bootstrap 的核心流程，支撑项目工程中的业务语义和异常边界。
def get_bootstrap(request: Request) -> BootstrapOrchestrator:
    ensure_app_runtime(request.app, start_queue=True, start_bootstrap=True)
    return request.app.state.bootstrap_orchestrator  # type: ignore[return-value]
