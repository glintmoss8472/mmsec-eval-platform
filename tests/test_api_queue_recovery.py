# 文件说明：该文件属于自动化测试，集中实现 test api queue recovery 相关逻辑。
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from mmsec_api.store.sqlite import SQLiteStore
from mmsec_api.worker.queue import JobQueue
from mmsec_eval.exceptions import ConfigError


# 实现 `_NoopExecutor.execute` 的对象行为，维护该类在自动化测试中的调用契约。
class _NoopExecutor:
    # 实现 _NoopExecutor.execute 的核心行为，维护自动化测试在该对象上的调用契约。
    def execute(
        self,
        job: dict[str, Any],
        log: Callable[[str, str], None],
        progress: Callable[[str, str, float | None, str], None],
    ) -> dict[str, str]:
        progress("config_validation", "success", 30, "recovered job executed")
        log("info", f"executed {job['id']}")
        return {"run_id": "recovered-run"}


# 实现 `_ConfigErrorExecutor.execute` 的对象行为，维护该类在自动化测试中的调用契约。
class _ConfigErrorExecutor:
    # 实现 _ConfigErrorExecutor.execute 的核心行为，维护自动化测试在该对象上的调用契约。
    def execute(
        self,
        job: dict[str, Any],
        log: Callable[[str, str], None],
        progress: Callable[[str, str, float | None, str], None],
    ) -> dict[str, str]:
        progress("config_validation", "failed", 18, "configuration rejected")
        log("error", f"config rejected {job['id']}")
        raise ConfigError("runtime.device must be cuda")


# 验证 `队列 recovers queued 任务 after process restart` 场景，防止相关行为在后续修改中退化。
def test_queue_recovers_queued_job_after_process_restart(tmp_path: Path):
    store = SQLiteStore(str(tmp_path / "app.db"))
    store.init_db()
    job = store.create_job(
        job_type="run_eval",
        config_path="configs/mvp.yaml",
        override={},
        benchmark_mode=False,
        payload={},
    )
    store.init_job_progress(job["id"], "run_eval")

    queue = JobQueue(store=store, executor=_NoopExecutor(), workers=1)  # type: ignore[arg-type]
    queue.start()
    try:
        assert queue.recover_unfinished_jobs() == 1
        deadline = time.time() + 5
        while time.time() < deadline:
            current = store.get_job(job["id"]) or {}
            if current.get("status") == "success":
                break
            time.sleep(0.05)
        recovered = store.get_job(job["id"]) or {}
        assert recovered["status"] == "success"
        assert recovered["run_id"] == "recovered-run"
        logs = store.list_job_logs(job["id"], page=1, page_size=20)[1]
        assert any("recovered after API worker startup" in str(item["message"]) for item in logs)
    finally:
        queue.stop()


# 验证 `队列 marks running 任务 interrupted after process restart` 场景，防止相关行为在后续修改中退化。
def test_queue_marks_running_job_interrupted_after_process_restart(tmp_path: Path):
    store = SQLiteStore(str(tmp_path / "app.db"))
    store.init_db()
    job = store.create_job(
        job_type="run_eval",
        config_path="configs/mvp.yaml",
        override={},
        benchmark_mode=False,
        payload={},
    )
    store.init_job_progress(job["id"], "run_eval")
    store.set_job_running(job["id"])

    queue = JobQueue(store=store, executor=_NoopExecutor(), workers=1)  # type: ignore[arg-type]
    queue.start()
    try:
        assert queue.recover_unfinished_jobs() == 0
        recovered = store.get_job(job["id"]) or {}
        assert recovered["status"] == "failed"
        assert recovered["error_code"] == "interrupted_by_restart"
        logs = store.list_job_logs(job["id"], page=1, page_size=20)[1]
        assert any("cannot be resumed automatically" in str(item["message"]) for item in logs)
    finally:
        queue.stop()


# 验证 `队列 marks project 配置 error failed` 场景，防止相关行为在后续修改中退化。
def test_queue_marks_project_config_error_failed(tmp_path: Path):
    store = SQLiteStore(str(tmp_path / "app.db"))
    store.init_db()
    job = store.create_job(
        job_type="run_eval",
        config_path="configs/mvp.yaml",
        override={},
        benchmark_mode=False,
        payload={},
    )
    store.init_job_progress(job["id"], "run_eval")

    queue = JobQueue(store=store, executor=_ConfigErrorExecutor(), workers=1)  # type: ignore[arg-type]
    queue.start()
    try:
        queue.enqueue(job["id"])
        deadline = time.time() + 5
        while time.time() < deadline:
            current = store.get_job(job["id"]) or {}
            if current.get("status") == "failed":
                break
            time.sleep(0.05)
        failed = store.get_job(job["id"]) or {}
        assert failed["status"] == "failed"
        assert failed["error_message"] == "runtime.device must be cuda"
        stages = store.list_job_progress(job["id"])
        assert any(item["stage_key"] == "completed" and item["state"] == "failed" for item in stages)
    finally:
        queue.stop()
