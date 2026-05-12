# 文件说明：该文件属于项目工程，集中实现 deps 相关逻辑。
from __future__ import annotations

from fastapi import Request

from mmsec_api.runtime import ensure_app_runtime
from mmsec_api.services.bootstrap_orchestrator import BootstrapOrchestrator
from mmsec_api.store.sqlite import SQLiteStore
from mmsec_api.worker.queue import JobQueue


# 获取 `store`，封装存储查询或状态读取细节。
def get_store(request: Request) -> SQLiteStore:
    ensure_app_runtime(request.app, start_queue=True, start_bootstrap=True)
    return request.app.state.store  # type: ignore[return-value]


# 获取 `队列`，封装存储查询或状态读取细节。
def get_queue(request: Request) -> JobQueue:
    ensure_app_runtime(request.app, start_queue=True, start_bootstrap=True)
    return request.app.state.job_queue  # type: ignore[return-value]


# 获取 `bootstrap`，封装存储查询或状态读取细节。
def get_bootstrap(request: Request) -> BootstrapOrchestrator:
    ensure_app_runtime(request.app, start_queue=True, start_bootstrap=True)
    return request.app.state.bootstrap_orchestrator  # type: ignore[return-value]
