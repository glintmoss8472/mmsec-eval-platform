# 文件说明：该文件属于任务队列，集中实现 queue 相关逻辑。
from __future__ import annotations

import queue
import threading
from typing import Callable

from mmsec_api.services.job_executor import JobExecutor
from mmsec_api.store.sqlite import SQLiteStore
from mmsec_eval.exceptions import MmsecError


# 实现 `JobQueue.__init__` 的对象行为，维护该类在任务队列中的调用契约。
class JobQueue:
    # 封装 JobQueue.__init__ 的内部步骤，让任务队列主流程保持清晰并隔离边界细节。
    def __init__(self, store: SQLiteStore, executor: JobExecutor, workers: int = 1) -> None:
        self.store = store
        self.executor = executor
        self.workers = max(1, int(workers))
        self._q: queue.Queue[str] = queue.Queue()
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()

    # 实现 `JobQueue.start` 的对象行为，维护该类在任务队列中的调用契约。
    def start(self) -> None:
        if self._threads:
            return
        for i in range(self.workers):
            t = threading.Thread(target=self._worker_loop, name=f"job-worker-{i}", daemon=True)
            t.start()
            self._threads.append(t)

    # 实现 `JobQueue.stop` 的对象行为，维护该类在任务队列中的调用契约。
    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        for _ in self._threads:
            self._q.put_nowait("__stop__")
        for t in self._threads:
            t.join(timeout=timeout)
        self._threads.clear()

    # 实现 `JobQueue.enqueue` 的对象行为，维护该类在任务队列中的调用契约。
    def enqueue(self, job_id: str) -> None:
        self._q.put_nowait(job_id)

    # 实现 `JobQueue.recover_unfinished_jobs` 的对象行为，维护该类在任务队列中的调用契约。
    def recover_unfinished_jobs(self) -> int:
        self.store.mark_running_jobs_interrupted()
        recovered = 0
        for job in self.store.list_recoverable_jobs():
            job_id = str(job.get("id") or "").strip()
            if not job_id:
                continue
            self.enqueue(job_id)
            self.store.add_job_log(job_id, level="warn", message="job recovered after API worker startup")
            recovered += 1
        return recovered

    # 执行 `日志` 辅助逻辑，保持任务队列中的输入处理和结果输出一致。
    def _mk_logger(self, job_id: str) -> Callable[[str, str], None]:
        # 封装 _log 的内部步骤，让任务队列主流程保持清晰并隔离边界细节。
        def _log(level: str, message: str) -> None:
            self.store.add_job_log(job_id, level=level, message=message)

        return _log

    # 执行 `进度` 辅助逻辑，保持任务队列中的输入处理和结果输出一致。
    def _mk_progress(self, job_id: str) -> Callable[[str, str, float | None, str], None]:
        # 封装 _progress 的内部步骤，让任务队列主流程保持清晰并隔离边界细节。
        def _progress(stage_key: str, state: str, progress_percent: float | None = None, message: str = "") -> None:
            self.store.update_job_stage(job_id, stage_key=stage_key, state=state, progress_percent=progress_percent, message=message)

        return _progress

    # 实现 `JobQueue._worker_loop` 的对象行为，维护该类在任务队列中的调用契约。
    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            try:
                job_id = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            if job_id == "__stop__":
                return

            log = self._mk_logger(job_id)
            job = self.store.get_job(job_id)
            if not job:
                continue

            if job.get("status") == "cancelled":
                log("warn", "job already cancelled before execution")
                continue

            try:
                self.store.set_job_running(job_id)
                now = self.store.get_job(job_id) or {}
                if now.get("status") != "running":
                    log("warn", "job skipped because status is no longer queued/running")
                    continue
                progress = self._mk_progress(job_id)
                progress("queued", "success", 10, "等待结束，任务开始执行。")
                log("info", "job started")
                result = self.executor.execute(job, log, progress)
                current = self.store.get_job(job_id) or {}
                if current.get("status") == "cancelled":
                    log("warn", "job marked cancelled; skip success commit")
                    continue
                self.store.set_job_success(job_id, run_id=str(result.get("run_id", "")))
                progress("completed", "success", 100, f"任务执行完成，运行编号：{str(result.get('run_id', '') or '未生成')}")
                log("info", "job finished")
            except (
                AttributeError,
                ImportError,
                KeyError,
                MmsecError,
                OSError,
                RuntimeError,
                TimeoutError,
                TypeError,
                ValueError,
            ) as e:
                error_code = str(getattr(e, "error_code", "job_failed"))
                self.store.set_job_failed(job_id, error_code=error_code, error_message=str(e))
                self.store.update_job_stage(job_id, stage_key="completed", state="failed", progress_percent=100, message=f"任务失败：{e}")
                log("error", f"job failed: {e}")
