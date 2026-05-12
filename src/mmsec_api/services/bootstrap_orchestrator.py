# 文件说明：该文件属于后端业务服务，集中实现 bootstrap orchestrator 相关逻辑。
from __future__ import annotations

import json
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from mmsec_api.schemas.models import BootstrapState, BootstrapStepState
from mmsec_api.store.sqlite import SQLiteStore
from mmsec_api.utils import utc_now_iso
from mmsec_api.worker.queue import JobQueue
from mmsec_eval.config.schema import BootstrapConfig


FAKE_MODEL_ADAPTERS = {"dummy", "fixture_vlm"}


# 推断 `iter 模型 markers`，从样本、配置或运行记录中提取统一名称。
def _iter_model_markers(payload: Any) -> list[str]:
    if isinstance(payload, dict):
        markers: list[str] = []
        for key in (
            "model_adapter",
            "victim_model_adapter",
            "victim_model_adapters",
            "target_model_adapter",
            "adapter",
        ):
            markers.extend(_iter_model_markers(payload.get(key)))
        for key in ("model", "plugins", "config", "summary"):
            markers.extend(_iter_model_markers(payload.get(key)))
        return markers
    if isinstance(payload, (list, tuple, set)):
        markers = []
        for item in payload:
            markers.extend(_iter_model_markers(item))
        return markers
    if isinstance(payload, str):
        value = payload.strip()
        return [value] if value else []
    return []


# 判断 `是否 fake 模型 载荷` 条件是否成立，为调用方提供布尔决策。
def _is_fake_model_payload(payload: Any) -> bool:
    return any(marker.lower() in FAKE_MODEL_ADAPTERS for marker in _iter_model_markers(payload))


# 定义 `_Step` 的不可变配置载体，集中保存后续计算需要的结构化字段。
@dataclass
class _Step:
    name: str
    state: BootstrapStepState = "pending"
    message: str = ""
    updated_at: str = ""

    # 实现 `_Step.as_dict` 的对象行为，维护该类在后端业务服务中的调用契约。
    def as_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "state": self.state,
            "message": self.message,
            "updated_at": self.updated_at or utc_now_iso(),
        }


# 定义 `BootstrapOrchestrator` 的状态和行为边界，供后端业务服务在固定职责内复用。
class BootstrapOrchestrator:
    """Prepare seed artifacts immediately and queue warm-up jobs in background."""

    # 实现 `BootstrapOrchestrator.__init__` 的对象行为，维护该类在后端业务服务中的调用契约。
    def __init__(
        self,
        *,
        store: SQLiteStore,
        queue: JobQueue,
        artifacts_dir: str,
        bootstrap: BootstrapConfig | None = None,
    ) -> None:
        self.store = store
        self.queue = queue
        self.artifacts_dir = Path(artifacts_dir)
        self.project_root = Path(__file__).resolve().parents[3]
        self.bootstrap = bootstrap or BootstrapConfig()

        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._state: BootstrapState = "pending"
        self._degraded_reason = ""
        self._started_at = ""
        self._updated_at = utc_now_iso()
        self._logs: list[dict[str, str]] = []
        self._max_logs = 500
        self._steps: dict[str, _Step] = {
            "seed_sync": _Step(name="seed_sync", updated_at=utc_now_iso()),
            "seed_docs": _Step(name="seed_docs", updated_at=utc_now_iso()),
            "seed_runs": _Step(name="seed_runs", updated_at=utc_now_iso()),
            "seed_data": _Step(name="seed_data", updated_at=utc_now_iso()),
            "queue_docs_ingest": _Step(name="queue_docs_ingest", updated_at=utc_now_iso()),
            "queue_dataset_prepare": _Step(name="queue_dataset_prepare", updated_at=utc_now_iso()),
            "queue_benchmark_demo": _Step(name="queue_benchmark_demo", updated_at=utc_now_iso()),
            "queue_benchmark_public": _Step(name="queue_benchmark_public", updated_at=utc_now_iso()),
            "model_warmup": _Step(name="model_warmup", updated_at=utc_now_iso()),
        }
        self._artifacts: dict[str, Any] = {
            "docs_index": "",
            "docs_snippets": "",
            "seeded_runs": [],
            "seeded_data": [],
        }
        if not self.bootstrap.enabled:
            self._state = "ready"
            now = utc_now_iso()
            self._started_at = now
            self._updated_at = now
            for step in self._steps.values():
                step.state = "skipped"
                step.message = "disabled by config"
                step.updated_at = now

    # 启动 `async`，完成运行前的环境、端口或队列准备。
    def start_async(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            if self._state in {"ready", "degraded"}:
                return
            self._thread = threading.Thread(target=self._run, name="bootstrap-orchestrator", daemon=True)
            self._thread.start()

    # 实现 `BootstrapOrchestrator.retry` 的对象行为，维护该类在后端业务服务中的调用契约。
    def retry(self) -> bool:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return False
            self._state = "pending"
            self._degraded_reason = ""
            self._started_at = ""
            self._updated_at = utc_now_iso()
            self._artifacts = {
                "docs_index": "",
                "docs_snippets": "",
                "seeded_runs": [],
                "seeded_data": [],
            }
            for step in self._steps.values():
                step.state = "pending"
                step.message = ""
                step.updated_at = utc_now_iso()
            self._thread = threading.Thread(target=self._run, name="bootstrap-orchestrator", daemon=True)
            self._thread.start()
            return True

    # 获取 `状态`，封装存储查询或状态读取细节。
    def get_status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "state": self._state,
                "started_at": self._started_at,
                "updated_at": self._updated_at,
                "degraded_reason": self._degraded_reason,
                "steps": [s.as_dict() for s in self._steps.values()],
                "artifacts": {
                    "docs_index": str(self._artifacts.get("docs_index", "")),
                    "docs_snippets": str(self._artifacts.get("docs_snippets", "")),
                    "seeded_runs": list(self._artifacts.get("seeded_runs", [])),
                    "seeded_data": list(self._artifacts.get("seeded_data", [])),
                },
            }

    # 获取 `日志`，封装存储查询或状态读取细节。
    def get_logs(self, limit: int = 200) -> list[dict[str, str]]:
        with self._lock:
            return list(self._logs[-max(1, limit) :])

    # 实现 `BootstrapOrchestrator._log` 的对象行为，维护该类在后端业务服务中的调用契约。
    def _log(self, level: str, message: str) -> None:
        row = {"ts": utc_now_iso(), "level": level, "message": message}
        with self._lock:
            self._logs.append(row)
            if len(self._logs) > self._max_logs:
                self._logs = self._logs[-self._max_logs :]
            self._updated_at = row["ts"]

    # 判断或归一 `set state` 状态，让调用方可以稳定渲染能力和可用性。
    def _set_state(self, state: BootstrapState, reason: str = "") -> None:
        with self._lock:
            self._state = state
            if reason:
                self._degraded_reason = reason
            self._updated_at = utc_now_iso()

    # 实现 `BootstrapOrchestrator._set_step` 的对象行为，维护该类在后端业务服务中的调用契约。
    def _set_step(self, name: str, *, state: BootstrapStepState, message: str = "") -> None:
        with self._lock:
            step = self._steps[name]
            step.state = state
            step.message = message
            step.updated_at = utc_now_iso()
            self._updated_at = step.updated_at

    # 执行 `step` 流程，按配置驱动后端业务服务完成一次任务。
    def _run_step(self, name: str, fn: Callable[[], None], *, critical: bool = False) -> None:
        self._set_step(name, state="running")
        try:
            fn()
            with self._lock:
                now_state = self._steps[name].state
            if now_state == "running":
                self._set_step(name, state="success")
        except (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError) as e:  # pragma: no cover - defensive path
            self._set_step(name, state="failed", message=str(e))
            self._log("error", f"{name} failed: {e}")
            if critical:
                self._set_state("degraded", reason=f"{name}: {e}")
            elif not self._degraded_reason:
                with self._lock:
                    self._degraded_reason = f"{name}: {e}"

    # 实现 `BootstrapOrchestrator._run` 的对象行为，维护该类在后端业务服务中的调用契约。
    def _run(self) -> None:
        with self._lock:
            if not self._started_at:
                self._started_at = utc_now_iso()
            self._updated_at = self._started_at

        self._log("info", "bootstrap started")
        self._set_state("seeding")
        self._run_step("seed_sync", self._seed_sync, critical=True)

        if self._state == "degraded":
            self._log("warn", "bootstrap stopped in degraded state during seeding")
            return

        self._set_state("warming")
        self._run_step("queue_docs_ingest", self._queue_docs_ingest, critical=False)
        self._run_step("queue_dataset_prepare", self._queue_dataset_prepare, critical=False)
        self._run_step("queue_benchmark_demo", self._queue_benchmark_demo, critical=False)
        self._run_step("queue_benchmark_public", self._queue_benchmark_public, critical=False)
        self._run_step("model_warmup", self._model_warmup, critical=False)

        if self._degraded_reason:
            self._set_state("degraded")
            self._log("warn", f"bootstrap completed with degraded state: {self._degraded_reason}")
            return

        self._set_state("ready")
        self._log("info", "bootstrap ready")

    # 实现 `BootstrapOrchestrator._seed_sync` 的对象行为，维护该类在后端业务服务中的调用契约。
    def _seed_sync(self) -> None:
        seed_root = self.project_root / self.bootstrap.seed_dir
        if not seed_root.exists():
            self._log("warn", f"seed directory not found: {seed_root}")
            self._set_step("seed_docs", state="skipped", message="seed dir missing")
            self._set_step("seed_runs", state="skipped", message="seed dir missing")
            self._set_step("seed_data", state="skipped", message="seed dir missing")
            return

        self._set_step("seed_docs", state="running")
        self._seed_docs(seed_root)
        self._set_step("seed_docs", state="success")

        self._set_step("seed_runs", state="running")
        self._seed_runs(seed_root)
        self._set_step("seed_runs", state="success")

        self._set_step("seed_data", state="running")
        self._seed_data(seed_root)
        self._set_step("seed_data", state="success")

    # 复制 `if missing` 对应的文件引用，并返回可写入结果记录的路径。
    def _copy_if_missing(self, src: Path, dst: Path) -> bool:
        if not src.exists():
            return False
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and dst.stat().st_size > 0:
            return False
        shutil.copy2(src, dst)
        return True

    # 实现 `BootstrapOrchestrator._seed_docs` 的对象行为，维护该类在后端业务服务中的调用契约。
    def _seed_docs(self, seed_root: Path) -> None:
        src_docs = seed_root / "docs"
        if not src_docs.exists():
            self._log("warn", "seed/docs missing; docs seed skipped")
            return

        dst_index = self.artifacts_dir / "docs_index.json"
        dst_snips = self.artifacts_dir / "docs_snippets.jsonl"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        copied_index = self._copy_if_missing(src_docs / "docs_index.json", dst_index)
        copied_snips = self._copy_if_missing(src_docs / "docs_snippets.jsonl", dst_snips)

        if dst_index.exists():
            self._artifacts["docs_index"] = str(dst_index)
        if dst_snips.exists():
            self._artifacts["docs_snippets"] = str(dst_snips)

        if copied_index or copied_snips:
            self._log("info", "seed docs imported")
        else:
            self._log("info", "seed docs already present")

    # 实现 `BootstrapOrchestrator._seed_runs` 的对象行为，维护该类在后端业务服务中的调用契约。
    def _seed_runs(self, seed_root: Path) -> None:
        src_runs = seed_root / "runs"
        if not src_runs.exists():
            self._log("warn", "seed/runs missing; run seed skipped")
            return

        dst_runs = self.artifacts_dir / "runs"
        dst_runs.mkdir(parents=True, exist_ok=True)
        seeded: list[str] = []
        for run_dir in sorted([p for p in src_runs.iterdir() if p.is_dir()]):
            src_summary_path = run_dir / "summary.json"
            if src_summary_path.exists():
                try:
                    src_summary = json.loads(src_summary_path.read_text(encoding="utf-8"))
                    if _is_fake_model_payload(src_summary):
                        self._log("warn", f"seed run skipped for fake model adapter: {run_dir.name}")
                        continue
                except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
                    self._log("warn", f"seed run precheck skipped for {run_dir.name}: {exc}")

            dst = dst_runs / run_dir.name
            if not dst.exists():
                shutil.copytree(run_dir, dst, dirs_exist_ok=True)
                seeded.append(run_dir.name)

            summary_path = dst / "summary.json"
            if summary_path.exists():
                try:
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    if isinstance(summary, dict):
                        if _is_fake_model_payload(summary):
                            self._log("warn", f"seed run cache skipped for fake model adapter: {run_dir.name}")
                            continue
                        self.store.upsert_run_cache(summary, str(dst))
                except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
                    self._log("warn", f"seed run cache skipped for {run_dir.name}: {exc}")

        self._artifacts["seeded_runs"] = seeded
        if seeded:
            self._log("info", f"seed runs imported: {', '.join(seeded)}")
        else:
            self._log("info", "seed runs already present")

    # 实现 `BootstrapOrchestrator._seed_data` 的对象行为，维护该类在后端业务服务中的调用契约。
    def _seed_data(self, seed_root: Path) -> None:
        src_data = seed_root / "data"
        if not src_data.exists():
            self._log("warn", "seed/data missing; data seed skipped")
            return

        dst_root = self.project_root / "data"
        dst_root.mkdir(parents=True, exist_ok=True)

        seeded: list[str] = []
        for item in sorted(src_data.iterdir()):
            dst = dst_root / item.name
            if item.is_dir():
                if not dst.exists():
                    shutil.copytree(item, dst, dirs_exist_ok=True)
                    seeded.append(item.name)
            else:
                if self._copy_if_missing(item, dst):
                    seeded.append(item.name)

        mini_flickr = dst_root / "mini_flickr"
        if mini_flickr.exists():
            image_dir = mini_flickr / "images"
            item_count = len(list(image_dir.glob("*"))) if image_dir.exists() else 0
            self.store.upsert_dataset(
                name="mini_flickr",
                root_path=str(mini_flickr),
                prepared=True,
                item_count=item_count,
                note="seed mini dataset",
            )

        self._artifacts["seeded_data"] = seeded
        if seeded:
            self._log("info", f"seed data imported: {', '.join(seeded)}")
        else:
            self._log("info", "seed data already present")

    # 判断 `是否包含 任务` 条件是否成立，为调用方提供布尔决策。
    def _has_job(self, *, job_type: str, config_path: str = "", statuses: set[str] | None = None) -> bool:
        statuses = statuses or {"queued", "running", "success", "failed", "cancelled"}
        page = 1
        while True:
            total, items = self.store.list_jobs(page=page, page_size=200, status="")
            for row in items:
                if row.get("job_type") != job_type:
                    continue
                if config_path and row.get("config_path") != config_path:
                    continue
                if row.get("status") in statuses:
                    return True
            if page * 200 >= total:
                break
            page += 1
        return False

    # 实现 `BootstrapOrchestrator._enqueue_job` 的对象行为，维护该类在后端业务服务中的调用契约。
    def _enqueue_job(
        self,
        *,
        job_type: str,
        config_path: str,
        payload: dict[str, Any] | None = None,
        benchmark_mode: bool = False,
    ) -> str:
        job = self.store.create_job(
            job_type=job_type,
            config_path=config_path,
            override={},
            benchmark_mode=benchmark_mode,
            payload=payload or {},
        )
        self.store.init_job_progress(job["id"], job_type)
        self.queue.enqueue(job["id"])
        self._log("info", f"queued {job_type} ({config_path})")
        return str(job["id"])

    # 实现 `BootstrapOrchestrator._queue_docs_ingest` 的对象行为，维护该类在后端业务服务中的调用契约。
    def _queue_docs_ingest(self) -> None:
        if not self.bootstrap.auto_ingest_docs:
            self._set_step("queue_docs_ingest", state="skipped", message="配置已禁用")
            return
        if self._has_job(job_type="docs_ingest", config_path=self.bootstrap.docs_config):
            self._set_step("queue_docs_ingest", state="skipped", message="已存在相同作业")
            return
        self._enqueue_job(job_type="docs_ingest", config_path=self.bootstrap.docs_config)

    # 推断 `队列 数据集 prepare`，从样本、配置或运行记录中提取统一名称。
    def _queue_dataset_prepare(self) -> None:
        if not self.bootstrap.auto_prepare_datasets:
            self._set_step("queue_dataset_prepare", state="skipped", message="配置已禁用")
            return

        if not self._has_job(job_type="dataset_prepare", config_path="configs/mvp.yaml"):
            self._enqueue_job(
                job_type="dataset_prepare",
                config_path="configs/mvp.yaml",
                payload={
                    "name": "flickr30k",
                    "root_path": self.bootstrap.flickr_root,
                    "image_dir": "images",
                    "auto_download": self.bootstrap.dataset_auto_download,
                },
            )
            self._enqueue_job(
                job_type="dataset_prepare",
                config_path="configs/mvp.yaml",
                payload={
                    "name": "flickr1k",
                    "root_path": self.bootstrap.flickr_root,
                    "auto_download": self.bootstrap.dataset_auto_download,
                    "max_items": 1000,
                },
            )
            self._enqueue_job(
                job_type="dataset_prepare",
                config_path="configs/mvp.yaml",
                payload={
                    "name": "coco_subset",
                    "root_path": self.bootstrap.coco_root,
                    "split": "val2017",
                    "max_items": 5000,
                    "download_annotations": self.bootstrap.dataset_auto_download,
                    "download_images": False,
                    "auto_download": self.bootstrap.dataset_auto_download,
                },
            )
        else:
            self._set_step("queue_dataset_prepare", state="skipped", message="已存在相同作业")

    # 实现 `BootstrapOrchestrator._queue_benchmark_demo` 的对象行为，维护该类在后端业务服务中的调用契约。
    def _queue_benchmark_demo(self) -> None:
        cfg = self.bootstrap.demo_benchmark_config
        if self._has_job(job_type="run_benchmark", config_path=cfg):
            self._set_step("queue_benchmark_demo", state="skipped", message="已存在相同作业")
            return
        self._enqueue_job(job_type="run_benchmark", config_path=cfg, benchmark_mode=True)

    # 实现 `BootstrapOrchestrator._queue_benchmark_public` 的对象行为，维护该类在后端业务服务中的调用契约。
    def _queue_benchmark_public(self) -> None:
        if not self.bootstrap.auto_run_benchmark:
            self._set_step("queue_benchmark_public", state="skipped", message="配置已禁用")
            return
        cfg = self.bootstrap.public_benchmark_config
        if self._has_job(job_type="run_benchmark", config_path=cfg):
            self._set_step("queue_benchmark_public", state="skipped", message="已存在相同作业")
            return
        self._enqueue_job(job_type="run_benchmark", config_path=cfg, benchmark_mode=True)

    # 推断 `模型 warmup`，从样本、配置或运行记录中提取统一名称。
    def _model_warmup(self) -> None:
        if not self.bootstrap.model_warmup:
            self._set_step("model_warmup", state="skipped", message="配置已禁用")
            return
        self._log("info", "模型预热已交由基准任务处理")
        self._set_step("model_warmup", state="success", message="已交由基准任务处理")
