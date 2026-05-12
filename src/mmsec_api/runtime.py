# 文件说明：该文件属于项目工程，集中实现 runtime 相关逻辑。
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI

from mmsec_api.services.bootstrap_orchestrator import BootstrapOrchestrator
from mmsec_api.services.job_executor import JobExecutor
from mmsec_api.store.sqlite import SQLiteStore
from mmsec_api.worker.queue import JobQueue
from mmsec_eval.config.loader import load_config
from mmsec_eval.config.schema import BootstrapConfig
from mmsec_eval.plugins.builtin import register_builtin_plugins


# 加载 `bootstrap 配置`，把外部文件、配置或运行产物转换为内存结构。
def _load_bootstrap_config() -> BootstrapConfig:
    if os.getenv("MMSEC_BOOTSTRAP_ENABLED", "").strip() in {"0", "false", "False"}:
        cfg = BootstrapConfig()
        cfg.enabled = False
        return cfg
    cfg_path = os.getenv("MMSEC_BOOTSTRAP_CONFIG", "configs/mvp.yaml")
    try:
        return load_config(cfg_path).bootstrap
    except (KeyError, OSError, TypeError, ValueError):
        return BootstrapConfig()


# 确保 `应用 runtime` 已准备好，不满足条件时主动创建、下载或报错。
def ensure_app_runtime(
    app: FastAPI,
    *,
    start_queue: bool = True,
    start_bootstrap: bool = True,
) -> dict[str, Any]:
    # The API exposes plugin-backed capabilities in overview endpoints before any
    # job runs, so builtin plugins must be registered during app startup too.
    register_builtin_plugins()

    store = getattr(app.state, "store", None)
    queue = getattr(app.state, "job_queue", None)
    bootstrap = getattr(app.state, "bootstrap_orchestrator", None)
    if store is not None and queue is not None and bootstrap is not None:
        if start_queue:
            queue.start()
            if not bool(getattr(app.state, "queue_recovery_done", False)):
                queue.recover_unfinished_jobs()
                app.state.queue_recovery_done = True
        if start_bootstrap and getattr(bootstrap, "bootstrap", None) and bootstrap.bootstrap.enabled:
            bootstrap.start_async()
        return {
            "store": store,
            "executor": getattr(app.state, "executor", None),
            "job_queue": queue,
            "bootstrap_orchestrator": bootstrap,
            "artifacts_dir": getattr(app.state, "artifacts_dir", "artifacts"),
        }

    artifacts_dir = os.getenv("MMSEC_ARTIFACTS_DIR", "artifacts")
    db_path = os.getenv("MMSEC_APP_DB", str(Path(artifacts_dir) / "app.db"))

    store = SQLiteStore(db_path)
    store.init_db()
    executor = JobExecutor(store=store, artifacts_dir=artifacts_dir)
    job_queue = JobQueue(store=store, executor=executor, workers=1)
    bootstrap_cfg = _load_bootstrap_config()
    bootstrap = BootstrapOrchestrator(
        store=store,
        queue=job_queue,
        artifacts_dir=artifacts_dir,
        bootstrap=bootstrap_cfg,
    )
    if start_queue:
        job_queue.start()
        job_queue.recover_unfinished_jobs()
        app.state.queue_recovery_done = True
    if start_bootstrap and bootstrap_cfg.enabled:
        bootstrap.start_async()

    app.state.store = store
    app.state.executor = executor
    app.state.job_queue = job_queue
    app.state.bootstrap_orchestrator = bootstrap
    app.state.artifacts_dir = artifacts_dir

    return {
        "store": store,
        "executor": executor,
        "job_queue": job_queue,
        "bootstrap_orchestrator": bootstrap,
        "artifacts_dir": artifacts_dir,
    }
