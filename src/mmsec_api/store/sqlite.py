from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from pathlib import Path
from typing import Any

from mmsec_api.services.job_progress import initial_stage_rows
from mmsec_api.utils import utc_now_iso


_FAKE_SAMPLE_RUN_IDS = (
    "20260101_000000_demo",
    "20260503_221636_400614",
    "20260503_221707_322497",
    "20260503_230826_803479",
    "20260503_230832_982578",
)
_FAKE_SAMPLE_MODEL_ADAPTERS = ("dummy", "fixture_vlm")


def _sample_asset_visibility_filter() -> tuple[list[str], list[Any]]:
    return (
        [
            f"COALESCE(source_run_id, '') NOT IN ({','.join('?' for _ in _FAKE_SAMPLE_RUN_IDS)})",
            f"lower(COALESCE(model_adapter, '')) NOT IN ({','.join('?' for _ in _FAKE_SAMPLE_MODEL_ADAPTERS)})",
        ],
        [*_FAKE_SAMPLE_RUN_IDS, *_FAKE_SAMPLE_MODEL_ADAPTERS],
    )


def _is_fake_sample_asset(source_run_id: str, model_adapter: str) -> bool:
    return source_run_id in _FAKE_SAMPLE_RUN_IDS or model_adapter.strip().lower() in _FAKE_SAMPLE_MODEL_ADAPTERS


_BASE_SCHEMA_DDL = (
    """
    CREATE TABLE IF NOT EXISTS jobs (
        id TEXT PRIMARY KEY,
        job_type TEXT NOT NULL,
        status TEXT NOT NULL,
        created_at TEXT NOT NULL,
        started_at TEXT,
        finished_at TEXT,
        config_path TEXT NOT NULL,
        override_json TEXT,
        benchmark_mode INTEGER NOT NULL DEFAULT 0,
        run_id TEXT,
        error_code TEXT,
        error_message TEXT,
        payload_json TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS job_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id TEXT NOT NULL,
        ts TEXT NOT NULL,
        level TEXT NOT NULL,
        message TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS job_progress_stages (
        job_id TEXT NOT NULL,
        stage_key TEXT NOT NULL,
        stage_label TEXT NOT NULL,
        stage_order INTEGER NOT NULL,
        state TEXT NOT NULL,
        progress_percent REAL NOT NULL DEFAULT 0,
        message TEXT NOT NULL DEFAULT '',
        updated_at TEXT NOT NULL,
        PRIMARY KEY(job_id, stage_key)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS runs_cache (
        run_id TEXT PRIMARY KEY,
        created_at TEXT NOT NULL,
        task_kind TEXT,
        dataset_name TEXT,
        benchmark_tag TEXT,
        attack TEXT,
        mode TEXT,
        defense TEXT,
        experiment_id TEXT,
        model_adapter TEXT,
        asr REAL,
        asr_attack REAL,
        asr_defended REAL,
        defense_gain REAL,
        risk_score REAL,
        risk_level TEXT,
        risk_scenario TEXT,
        avg_l2 REAL,
        path TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS dataset_registry (
        name TEXT PRIMARY KEY,
        root_path TEXT,
        prepared INTEGER NOT NULL DEFAULT 0,
        prepared_at TEXT,
        item_count INTEGER NOT NULL DEFAULT 0,
        note TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS sample_assets (
        asset_id TEXT PRIMARY KEY,
        variant_id TEXT NOT NULL DEFAULT '',
        source_run_id TEXT NOT NULL,
        source_case_id TEXT NOT NULL,
        task_kind TEXT NOT NULL DEFAULT '',
        dataset_name TEXT NOT NULL DEFAULT '',
        benchmark_tag TEXT NOT NULL DEFAULT '',
        model_adapter TEXT NOT NULL DEFAULT '',
        attack TEXT NOT NULL DEFAULT '',
        attack_scope TEXT NOT NULL DEFAULT '',
        source_text TEXT NOT NULL DEFAULT '',
        target_text TEXT NOT NULL DEFAULT '',
        clean_image_ref TEXT NOT NULL DEFAULT '',
        adv_image_ref TEXT NOT NULL DEFAULT '',
        artifact_status TEXT NOT NULL DEFAULT '',
        reusable_status TEXT NOT NULL DEFAULT '',
        reusable_note TEXT NOT NULL DEFAULT '',
        judge_success INTEGER NOT NULL DEFAULT 0,
        risk_level TEXT NOT NULL DEFAULT '',
        risk_score REAL NOT NULL DEFAULT 0,
        perturbation_l2 REAL NOT NULL DEFAULT 0,
        perturbation_linf REAL NOT NULL DEFAULT 0,
        semantic_score REAL NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        used_count INTEGER NOT NULL DEFAULT 0,
        last_used_at TEXT,
        metadata_json TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS sample_asset_usages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        asset_id TEXT NOT NULL,
        evaluation_run_id TEXT NOT NULL,
        job_id TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL,
        note TEXT NOT NULL DEFAULT ''
    )
    """,
)


class SQLiteStore:
    def __init__(self, path: str) -> None:
        self.path = str(Path(path))
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self) -> None:
        with self._connect() as conn:
            for stmt in _BASE_SCHEMA_DDL:
                conn.execute(stmt)
            self._ensure_runs_cache_columns(conn)
            self._ensure_sample_asset_columns(conn)
            self._backfill_runs_cache_defaults(conn)
            conn.commit()

    @staticmethod
    def _ensure_runs_cache_columns(conn: sqlite3.Connection) -> None:
        cols = {str(r[1]) for r in conn.execute("PRAGMA table_info(runs_cache)").fetchall()}
        desired = {
            "task_kind": "TEXT",
            "defense": "TEXT",
            "experiment_id": "TEXT",
            "asr_attack": "REAL",
            "asr_defended": "REAL",
            "defense_gain": "REAL",
            "risk_score": "REAL",
            "risk_level": "TEXT",
            "risk_scenario": "TEXT",
        }
        for name, typ in desired.items():
            if name not in cols:
                conn.execute(f"ALTER TABLE runs_cache ADD COLUMN {name} {typ}")

    @staticmethod
    def _backfill_runs_cache_defaults(conn: sqlite3.Connection) -> None:
        # Older cache rows may contain NULLs for newly-added fields; normalize once at startup.
        conn.execute(
            """
            UPDATE runs_cache
            SET
                task_kind = COALESCE(task_kind, ''),
                dataset_name = COALESCE(dataset_name, ''),
                benchmark_tag = COALESCE(benchmark_tag, ''),
                attack = COALESCE(attack, ''),
                mode = COALESCE(mode, ''),
                defense = COALESCE(defense, ''),
                experiment_id = COALESCE(experiment_id, ''),
                model_adapter = COALESCE(model_adapter, ''),
                asr = COALESCE(asr, 0.0),
                asr_attack = COALESCE(asr_attack, COALESCE(asr, 0.0)),
                asr_defended = COALESCE(asr_defended, COALESCE(asr, 0.0)),
                defense_gain = COALESCE(defense_gain, 0.0),
                risk_score = COALESCE(risk_score, 0.0),
                risk_level = COALESCE(risk_level, ''),
                risk_scenario = COALESCE(risk_scenario, ''),
                avg_l2 = COALESCE(avg_l2, 0.0),
                path = COALESCE(path, '')
            """
        )

    @staticmethod
    def _ensure_sample_asset_columns(conn: sqlite3.Connection) -> None:
        cols = {str(r[1]) for r in conn.execute("PRAGMA table_info(sample_assets)").fetchall()}
        desired = {
            "variant_id": "TEXT NOT NULL DEFAULT ''",
            "source_run_id": "TEXT NOT NULL DEFAULT ''",
            "source_case_id": "TEXT NOT NULL DEFAULT ''",
            "task_kind": "TEXT NOT NULL DEFAULT ''",
            "dataset_name": "TEXT NOT NULL DEFAULT ''",
            "benchmark_tag": "TEXT NOT NULL DEFAULT ''",
            "model_adapter": "TEXT NOT NULL DEFAULT ''",
            "attack": "TEXT NOT NULL DEFAULT ''",
            "attack_scope": "TEXT NOT NULL DEFAULT ''",
            "source_text": "TEXT NOT NULL DEFAULT ''",
            "target_text": "TEXT NOT NULL DEFAULT ''",
            "clean_image_ref": "TEXT NOT NULL DEFAULT ''",
            "adv_image_ref": "TEXT NOT NULL DEFAULT ''",
            "artifact_status": "TEXT NOT NULL DEFAULT ''",
            "reusable_status": "TEXT NOT NULL DEFAULT ''",
            "reusable_note": "TEXT NOT NULL DEFAULT ''",
            "judge_success": "INTEGER NOT NULL DEFAULT 0",
            "risk_level": "TEXT NOT NULL DEFAULT ''",
            "risk_score": "REAL NOT NULL DEFAULT 0",
            "perturbation_l2": "REAL NOT NULL DEFAULT 0",
            "perturbation_linf": "REAL NOT NULL DEFAULT 0",
            "semantic_score": "REAL NOT NULL DEFAULT 0",
            "created_at": "TEXT NOT NULL DEFAULT ''",
            "updated_at": "TEXT NOT NULL DEFAULT ''",
            "used_count": "INTEGER NOT NULL DEFAULT 0",
            "last_used_at": "TEXT",
            "metadata_json": "TEXT",
        }
        for name, typ in desired.items():
            if name not in cols:
                conn.execute(f"ALTER TABLE sample_assets ADD COLUMN {name} {typ}")

    @staticmethod
    def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        return dict(row)

    def create_job(
        self,
        *,
        job_type: str,
        config_path: str,
        override: dict[str, Any] | None,
        benchmark_mode: bool,
        payload: dict[str, Any] | None,
    ) -> dict[str, Any]:
        job_id = str(uuid.uuid4())
        now = utc_now_iso()
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO jobs(
                        id, job_type, status, created_at, config_path,
                        override_json, benchmark_mode, payload_json,
                        run_id, error_code, error_message
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        job_id,
                        job_type,
                        "queued",
                        now,
                        config_path,
                        json.dumps(override or {}, ensure_ascii=False),
                        1 if benchmark_mode else 0,
                        json.dumps(payload or {}, ensure_ascii=False),
                        "",
                        "",
                        "",
                    ),
                )
                conn.commit()
        return self.get_job(job_id) or {}

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            item = self._row_to_dict(row)
        if not item:
            return None
        item["benchmark_mode"] = bool(item.get("benchmark_mode", 0))
        item["run_id"] = str(item.get("run_id") or "")
        item["error_code"] = str(item.get("error_code") or "")
        item["error_message"] = str(item.get("error_message") or "")
        return item

    def list_jobs(self, page: int, page_size: int, status: str = "") -> tuple[int, list[dict[str, Any]]]:
        offset = (page - 1) * page_size
        where = ""
        params: list[Any] = []
        if status:
            where = "WHERE status = ?"
            params.append(status)

        with self._connect() as conn:
            total = conn.execute(f"SELECT COUNT(1) FROM jobs {where}", params).fetchone()[0]
            rows = conn.execute(
                f"SELECT * FROM jobs {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
                [*params, page_size, offset],
            ).fetchall()

        out = [dict(r) for r in rows]
        for row in out:
            row["benchmark_mode"] = bool(row.get("benchmark_mode", 0))
            row["run_id"] = str(row.get("run_id") or "")
            row["error_code"] = str(row.get("error_code") or "")
            row["error_message"] = str(row.get("error_message") or "")
        return int(total), out

    def list_recoverable_jobs(self, limit: int = 100) -> list[dict[str, Any]]:
        """Return queued jobs that can be safely reattached to an in-process worker.

        The API queue is intentionally lightweight and lives inside the FastAPI
        process. Queued jobs have not started executing and can be re-enqueued
        after a restart. Running jobs cannot be resumed safely because their
        Python call stack and intermediate artifacts were owned by the previous
        process, so startup code marks them as interrupted instead.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM jobs
                WHERE status = 'queued'
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        out = [dict(r) for r in rows]
        for row in out:
            row["benchmark_mode"] = bool(row.get("benchmark_mode", 0))
            row["run_id"] = str(row.get("run_id") or "")
            row["error_code"] = str(row.get("error_code") or "")
            row["error_message"] = str(row.get("error_message") or "")
        return out

    def mark_running_jobs_interrupted(self, reason: str = "") -> int:
        message = (
            reason.strip()
            or "后端工作进程重启时该任务仍在运行，进程内任务无法自动续跑，请重新提交任务。（API worker restarted while the job was running; in-process jobs cannot be resumed automatically. Please resubmit the task.）"
        )
        now = utc_now_iso()
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute("SELECT id FROM jobs WHERE status = 'running' ORDER BY created_at ASC").fetchall()
                for row in rows:
                    job_id = str(row["id"])
                    conn.execute(
                        """
                        UPDATE jobs
                        SET status = ?, finished_at = ?, error_code = ?, error_message = ?
                        WHERE id = ? AND status = 'running'
                        """,
                        ("failed", now, "interrupted_by_restart", message[:2000], job_id),
                    )
                    conn.execute(
                        """
                        UPDATE job_progress_stages
                        SET state = 'failed', message = ?, updated_at = ?
                        WHERE job_id = ? AND state = 'running'
                        """,
                        (message[:2000], now, job_id),
                    )
                    conn.execute(
                        """
                        UPDATE job_progress_stages
                        SET state = 'failed', progress_percent = 100, message = ?, updated_at = ?
                        WHERE job_id = ? AND stage_key = 'completed'
                        """,
                        (message[:2000], now, job_id),
                    )
                    conn.execute(
                        "INSERT INTO job_logs(job_id, ts, level, message) VALUES(?, ?, ?, ?)",
                        (job_id, now, "error", message[:4000]),
                    )
                conn.commit()
        return len(rows)

    def set_job_running(self, job_id: str) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "UPDATE jobs SET status = ?, started_at = ? WHERE id = ? AND status = ?",
                    ("running", utc_now_iso(), job_id, "queued"),
                )
                conn.commit()

    def set_job_success(self, job_id: str, run_id: str = "") -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "UPDATE jobs SET status = ?, finished_at = ?, run_id = ?, error_code = '', error_message = '' WHERE id = ?",
                    ("success", utc_now_iso(), run_id, job_id),
                )
                conn.commit()

    def set_job_failed(self, job_id: str, error_code: str, error_message: str) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "UPDATE jobs SET status = ?, finished_at = ?, error_code = ?, error_message = ? WHERE id = ?",
                    ("failed", utc_now_iso(), error_code, error_message, job_id),
                )
                conn.commit()

    def set_job_cancelled(self, job_id: str, note: str = "") -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "UPDATE jobs SET status = ?, finished_at = ?, error_code = ?, error_message = ? WHERE id = ?",
                    ("cancelled", utc_now_iso(), "cancelled", note, job_id),
                )
                conn.commit()

    def init_job_progress(self, job_id: str, job_type: str) -> None:
        now = utc_now_iso()
        rows = initial_stage_rows(job_type, now)
        with self._lock:
            with self._connect() as conn:
                conn.execute("DELETE FROM job_progress_stages WHERE job_id = ?", (job_id,))
                for row in rows:
                    conn.execute(
                        """
                        INSERT INTO job_progress_stages(
                            job_id, stage_key, stage_label, stage_order,
                            state, progress_percent, message, updated_at
                        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            job_id,
                            row["stage_key"],
                            row["stage_label"],
                            row["stage_order"],
                            row["state"],
                            row["progress_percent"],
                            row["message"],
                            row["updated_at"],
                        ),
                    )
                conn.commit()

    def list_job_progress(self, job_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT stage_key, stage_label, state, progress_percent, message, updated_at
                FROM job_progress_stages
                WHERE job_id = ?
                ORDER BY stage_order ASC
                """,
                (job_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def update_job_stage(self, job_id: str, stage_key: str, state: str, progress_percent: float | None = None, message: str = "") -> None:
        now = utc_now_iso()
        with self._lock:
            with self._connect() as conn:
                if state == "running":
                    conn.execute(
                        """
                        UPDATE job_progress_stages
                        SET state = CASE WHEN state = 'running' THEN 'success' ELSE state END,
                            updated_at = ?
                        WHERE job_id = ? AND stage_key <> ? AND state = 'running'
                        """,
                        (now, job_id, stage_key),
                    )
                if progress_percent is None:
                    conn.execute(
                        """
                        UPDATE job_progress_stages
                        SET state = ?, message = ?, updated_at = ?
                        WHERE job_id = ? AND stage_key = ?
                        """,
                        (state, message[:2000], now, job_id, stage_key),
                    )
                else:
                    conn.execute(
                        """
                        UPDATE job_progress_stages
                        SET state = ?, progress_percent = ?, message = ?, updated_at = ?
                        WHERE job_id = ? AND stage_key = ?
                        """,
                        (state, float(progress_percent), message[:2000], now, job_id, stage_key),
                    )
                if stage_key == "completed" and state == "success":
                    conn.execute(
                        """
                        UPDATE job_progress_stages
                        SET state = 'success', updated_at = ?
                        WHERE job_id = ? AND stage_key <> 'completed' AND state IN ('pending', 'running')
                        """,
                        (now, job_id),
                    )
                conn.commit()

    def get_latest_job_log(self, job_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM job_logs WHERE job_id = ? ORDER BY id DESC LIMIT 1",
                (job_id,),
            ).fetchone()
        return dict(row) if row is not None else None

    def get_queue_position(self, job_id: str) -> int:
        row = self.get_job(job_id)
        if not row:
            return 0
        if str(row.get("status", "")) != "queued":
            return 0
        created_at = str(row.get("created_at", ""))
        with self._connect() as conn:
            count = conn.execute(
                """
                SELECT COUNT(1)
                FROM jobs
                WHERE status = 'queued' AND created_at <= ?
                """,
                (created_at,),
            ).fetchone()[0]
        return int(count)

    def list_success_durations(self, job_type: str, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT started_at, finished_at
                FROM jobs
                WHERE job_type = ? AND status = 'success'
                ORDER BY finished_at DESC
                LIMIT ?
                """,
                (job_type, int(limit)),
            ).fetchall()
        return [dict(row) for row in rows]

    def add_job_log(self, job_id: str, level: str, message: str) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO job_logs(job_id, ts, level, message) VALUES(?, ?, ?, ?)",
                    (job_id, utc_now_iso(), level, message[:4000]),
                )
                conn.commit()

    def list_job_logs(self, job_id: str, page: int, page_size: int) -> tuple[int, list[dict[str, Any]]]:
        offset = (page - 1) * page_size
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(1) FROM job_logs WHERE job_id = ?", (job_id,)).fetchone()[0]
            rows = conn.execute(
                "SELECT * FROM job_logs WHERE job_id = ? ORDER BY id ASC LIMIT ? OFFSET ?",
                (job_id, page_size, offset),
            ).fetchall()
        return int(total), [dict(r) for r in rows]

    def upsert_run_cache(self, summary: dict[str, Any], run_dir: str) -> None:
        run_id = str(summary.get("run_id", "")).strip()
        if not run_id:
            return
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO runs_cache(
                        run_id, created_at, task_kind, dataset_name, benchmark_tag, attack, mode, defense, experiment_id,
                        model_adapter, asr, asr_attack, asr_defended, defense_gain, risk_score, risk_level, risk_scenario, avg_l2, path
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_id) DO UPDATE SET
                        created_at=excluded.created_at,
                        task_kind=excluded.task_kind,
                        dataset_name=excluded.dataset_name,
                        benchmark_tag=excluded.benchmark_tag,
                        attack=excluded.attack,
                        mode=excluded.mode,
                        defense=excluded.defense,
                        experiment_id=excluded.experiment_id,
                        model_adapter=excluded.model_adapter,
                        asr=excluded.asr,
                        asr_attack=excluded.asr_attack,
                        asr_defended=excluded.asr_defended,
                        defense_gain=excluded.defense_gain,
                        risk_score=excluded.risk_score,
                        risk_level=excluded.risk_level,
                        risk_scenario=excluded.risk_scenario,
                        avg_l2=excluded.avg_l2,
                        path=excluded.path
                    """,
                    (
                        run_id,
                        utc_now_iso(),
                        str(summary.get("task_kind", "")),
                        str(summary.get("dataset_name", "")),
                        str(summary.get("benchmark_tag", "")),
                        str(summary.get("attack", "")),
                        str(summary.get("attack_mode", "")),
                        str(summary.get("defense", "")),
                        str(summary.get("experiment_id", "")),
                        str(summary.get("model_adapter", "")),
                        float(summary.get("asr", 0.0) or 0.0),
                        float(summary.get("asr_attack", summary.get("asr", 0.0)) or 0.0),
                        float(summary.get("asr_defended", summary.get("asr", 0.0)) or 0.0),
                        float(summary.get("defense_gain", 0.0) or 0.0),
                        float(summary.get("risk_score", 0.0) or 0.0),
                        str(summary.get("risk_level", "")),
                        str(summary.get("risk_scenario", "")),
                        float(summary.get("avg_l2", 0.0) or 0.0),
                        str(run_dir),
                    ),
                )
                conn.commit()

    def list_runs_cache(self, page: int, page_size: int) -> tuple[int, list[dict[str, Any]]]:
        offset = (page - 1) * page_size
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(1) FROM runs_cache").fetchone()[0]
            rows = conn.execute(
                "SELECT * FROM runs_cache ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (page_size, offset),
            ).fetchall()
        out = [dict(r) for r in rows]
        for row in out:
            row["task_kind"] = str(row.get("task_kind") or "")
            row["dataset_name"] = str(row.get("dataset_name") or "")
            row["benchmark_tag"] = str(row.get("benchmark_tag") or "")
            row["attack"] = str(row.get("attack") or "")
            row["mode"] = str(row.get("mode") or "")
            row["defense"] = str(row.get("defense") or "")
            row["experiment_id"] = str(row.get("experiment_id") or "")
            row["model_adapter"] = str(row.get("model_adapter") or "")
            row["asr"] = float(row.get("asr", 0.0) or 0.0)
            row["asr_attack"] = float(row.get("asr_attack", row["asr"]) or 0.0)
            row["asr_defended"] = float(row.get("asr_defended", row["asr"]) or 0.0)
            row["defense_gain"] = float(row.get("defense_gain", 0.0) or 0.0)
            row["risk_score"] = float(row.get("risk_score", 0.0) or 0.0)
            row["risk_level"] = str(row.get("risk_level") or "")
            row["risk_scenario"] = str(row.get("risk_scenario") or "")
            row["avg_l2"] = float(row.get("avg_l2", 0.0) or 0.0)
            row["path"] = str(row.get("path") or "")
        return int(total), out

    def upsert_dataset(self, name: str, root_path: str, prepared: bool, item_count: int = 0, note: str = "") -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO dataset_registry(name, root_path, prepared, prepared_at, item_count, note)
                    VALUES(?, ?, ?, ?, ?, ?)
                    ON CONFLICT(name) DO UPDATE SET
                        root_path=excluded.root_path,
                        prepared=excluded.prepared,
                        prepared_at=excluded.prepared_at,
                        item_count=excluded.item_count,
                        note=excluded.note
                    """,
                    (
                        name,
                        root_path,
                        1 if prepared else 0,
                        utc_now_iso(),
                        int(item_count),
                        note,
                    ),
                )
                conn.commit()

    def list_datasets(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM dataset_registry ORDER BY name ASC").fetchall()
        out = [dict(r) for r in rows]
        for row in out:
            row["prepared"] = bool(row.get("prepared", 0))
        return out

    def count_sample_assets(self) -> int:
        where, params = _sample_asset_visibility_filter()
        where_sql = "WHERE " + " AND ".join(where)
        with self._connect() as conn:
            row = conn.execute(f"SELECT COUNT(1) FROM sample_assets {where_sql}", params).fetchone()
        return int(row[0] if row else 0)

    def upsert_sample_assets(self, assets: list[dict[str, Any]]) -> int:
        if not assets:
            return 0
        now = utc_now_iso()
        written = 0
        with self._lock:
            with self._connect() as conn:
                for asset in assets:
                    asset_id = str(asset.get("asset_id") or "").strip()
                    source_run_id = str(asset.get("source_run_id") or asset.get("run_id") or "").strip()
                    source_case_id = str(asset.get("source_case_id") or asset.get("sample_id") or "").strip()
                    if not asset_id or not source_run_id or not source_case_id:
                        continue
                    if _is_fake_sample_asset(source_run_id, str(asset.get("model_adapter") or "")):
                        continue
                    metadata = asset.get("metadata")
                    metadata_json = json.dumps(metadata if isinstance(metadata, dict) else {}, ensure_ascii=False)
                    conn.execute(
                        """
                        INSERT INTO sample_assets(
                            asset_id, variant_id, source_run_id, source_case_id, task_kind,
                            dataset_name, benchmark_tag, model_adapter, attack, attack_scope,
                            source_text, target_text, clean_image_ref, adv_image_ref,
                            artifact_status, reusable_status, reusable_note, judge_success,
                            risk_level, risk_score, perturbation_l2, perturbation_linf,
                            semantic_score, created_at, updated_at, metadata_json
                        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(asset_id) DO UPDATE SET
                            variant_id=excluded.variant_id,
                            source_run_id=excluded.source_run_id,
                            source_case_id=excluded.source_case_id,
                            task_kind=excluded.task_kind,
                            dataset_name=excluded.dataset_name,
                            benchmark_tag=excluded.benchmark_tag,
                            model_adapter=excluded.model_adapter,
                            attack=excluded.attack,
                            attack_scope=excluded.attack_scope,
                            source_text=excluded.source_text,
                            target_text=excluded.target_text,
                            clean_image_ref=excluded.clean_image_ref,
                            adv_image_ref=excluded.adv_image_ref,
                            artifact_status=excluded.artifact_status,
                            reusable_status=excluded.reusable_status,
                            reusable_note=excluded.reusable_note,
                            judge_success=excluded.judge_success,
                            risk_level=excluded.risk_level,
                            risk_score=excluded.risk_score,
                            perturbation_l2=excluded.perturbation_l2,
                            perturbation_linf=excluded.perturbation_linf,
                            semantic_score=excluded.semantic_score,
                            updated_at=excluded.updated_at,
                            metadata_json=excluded.metadata_json
                        """,
                        (
                            asset_id,
                            str(asset.get("variant_id") or ""),
                            source_run_id,
                            source_case_id,
                            str(asset.get("task_kind") or ""),
                            str(asset.get("dataset_name") or ""),
                            str(asset.get("benchmark_tag") or ""),
                            str(asset.get("model_adapter") or ""),
                            str(asset.get("attack") or ""),
                            str(asset.get("attack_scope") or ""),
                            str(asset.get("source_text") or ""),
                            str(asset.get("target_text") or ""),
                            str(asset.get("clean_image_ref") or ""),
                            str(asset.get("adv_image_ref") or ""),
                            str(asset.get("artifact_status") or ""),
                            str(asset.get("reusable_status") or ""),
                            str(asset.get("reusable_note") or ""),
                            1 if bool(asset.get("judge_success")) else 0,
                            str(asset.get("risk_level") or ""),
                            float(asset.get("risk_score") or 0.0),
                            float(asset.get("perturbation_l2") or 0.0),
                            float(asset.get("perturbation_linf") or 0.0),
                            float(asset.get("semantic_score") or 0.0),
                            str(asset.get("created_at") or now),
                            now,
                            metadata_json,
                        ),
                    )
                    written += 1
                conn.commit()
        return written

    @staticmethod
    def _asset_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        out = dict(row)
        out["run_id"] = str(out.get("source_run_id") or "")
        out["sample_id"] = str(out.get("source_case_id") or "")
        out["judge_success"] = bool(out.get("judge_success", 0))
        out["risk_score"] = float(out.get("risk_score", 0.0) or 0.0)
        out["perturbation_l2"] = float(out.get("perturbation_l2", 0.0) or 0.0)
        out["perturbation_linf"] = float(out.get("perturbation_linf", 0.0) or 0.0)
        out["semantic_score"] = float(out.get("semantic_score", 0.0) or 0.0)
        out["linked_evaluation_count"] = int(out.get("used_count", 0) or 0)
        pending_only = str(out.get("reusable_status") or "").lower() == "pending_evaluation" or str(out.get("artifact_status") or "").lower() == "generated_only"
        out["report_url"] = "" if pending_only else (f"/reports/{out['run_id']}" if out["run_id"] else "")
        out["case_url"] = "" if pending_only else (f"/reports/{out['run_id']}/cases/{out['sample_id']}" if out["run_id"] and out["sample_id"] else "")
        raw_meta = str(out.pop("metadata_json", "") or "{}")
        try:
            out["metadata"] = json.loads(raw_meta)
        except json.JSONDecodeError:
            out["metadata"] = {}
        return out

    def list_sample_assets(
        self,
        *,
        page: int,
        page_size: int,
        task_kind: str = "",
        attack: str = "",
        scope: str = "",
        reusable_status: str = "",
        model: str = "",
        dataset: str = "",
        search: str = "",
    ) -> tuple[int, list[dict[str, Any]], dict[str, Any]]:
        where, params = _sample_asset_visibility_filter()
        if task_kind:
            where.append("task_kind = ?")
            params.append(task_kind)
        if attack:
            where.append("attack = ?")
            params.append(attack)
        if scope:
            where.append("attack_scope LIKE ?")
            params.append(f"%{scope}%")
        if reusable_status:
            where.append("reusable_status = ?")
            params.append(reusable_status)
        if model:
            where.append("model_adapter = ?")
            params.append(model)
        if dataset:
            where.append("(benchmark_tag = ? OR dataset_name = ?)")
            params.extend([dataset, dataset])
        if search:
            like = f"%{search.lower()}%"
            where.append(
                """
                (
                    lower(asset_id) LIKE ? OR lower(source_case_id) LIKE ? OR lower(source_run_id) LIKE ?
                    OR lower(source_text) LIKE ? OR lower(benchmark_tag) LIKE ? OR lower(dataset_name) LIKE ?
                    OR lower(model_adapter) LIKE ? OR lower(attack) LIKE ?
                )
                """
            )
            params.extend([like] * 8)
        where_sql = "WHERE " + " AND ".join(where) if where else ""
        offset = (int(page) - 1) * int(page_size)
        with self._connect() as conn:
            total = conn.execute(f"SELECT COUNT(1) FROM sample_assets {where_sql}", params).fetchone()[0]
            rows = conn.execute(
                f"""
                SELECT *
                FROM sample_assets
                {where_sql}
                ORDER BY created_at DESC, asset_id DESC
                LIMIT ? OFFSET ?
                """,
                [*params, int(page_size), int(offset)],
            ).fetchall()

            def grouped(column: str) -> list[dict[str, Any]]:
                group_rows = conn.execute(
                    f"""
                    SELECT {column} AS value, COUNT(1) AS count
                    FROM sample_assets
                    {where_sql}
                    GROUP BY {column}
                    ORDER BY {column} ASC
                    """,
                    params,
                ).fetchall()
                return [{"value": str(r["value"] or ""), "count": int(r["count"] or 0)} for r in group_rows if str(r["value"] or "")]

            status_rows = conn.execute(
                f"""
                SELECT reusable_status AS value, COUNT(1) AS count
                FROM sample_assets
                {where_sql}
                GROUP BY reusable_status
                """,
                params,
            ).fetchall()
            reusable_counts = {str(r["value"] or ""): int(r["count"] or 0) for r in status_rows}
            options = {
                "task_kinds": grouped("task_kind"),
                "attacks": grouped("attack"),
                "scopes": grouped("attack_scope"),
                "models": grouped("model_adapter"),
                "datasets": grouped("benchmark_tag"),
                "reusable_statuses": [{"value": key, "count": value} for key, value in sorted(reusable_counts.items()) if key],
            }
        summary = {
            "total_assets": int(total),
            "ready_assets": reusable_counts.get("ready", 0),
            "summary_only_assets": reusable_counts.get("summary_only", 0),
            "legacy_assets": reusable_counts.get("legacy", 0),
            "pending_evaluation_assets": reusable_counts.get("pending_evaluation", 0),
            "task_distribution": {item["value"]: item["count"] for item in options["task_kinds"]},
            "attack_distribution": {item["value"]: item["count"] for item in options["attacks"]},
            "scope_distribution": {item["value"]: item["count"] for item in options["scopes"]},
        }
        return int(total), [self._asset_row_to_dict(row) for row in rows], {"summary": summary, "options": options}

    def list_sample_asset_batches(
        self,
        *,
        page: int,
        page_size: int,
        task_kind: str = "",
        attack: str = "",
        scope: str = "",
        reusable_status: str = "",
        model: str = "",
        dataset: str = "",
        search: str = "",
        sort_by: str = "created_at",
        sort_dir: str = "desc",
        include_asset_ids: bool = True,
    ) -> tuple[int, list[dict[str, Any]], dict[str, Any]]:
        where, params = _sample_asset_visibility_filter()
        if task_kind:
            where.append("task_kind = ?")
            params.append(task_kind)
        if attack:
            where.append("attack = ?")
            params.append(attack)
        if scope:
            where.append("attack_scope LIKE ?")
            params.append(f"%{scope}%")
        if reusable_status:
            where.append("reusable_status = ?")
            params.append(reusable_status)
        if model:
            where.append("model_adapter = ?")
            params.append(model)
        if dataset:
            where.append("(benchmark_tag = ? OR dataset_name = ?)")
            params.extend([dataset, dataset])
        if search:
            like = f"%{search.lower()}%"
            where.append(
                """
                (
                    lower(asset_id) LIKE ? OR lower(source_case_id) LIKE ? OR lower(source_run_id) LIKE ?
                    OR lower(source_text) LIKE ? OR lower(benchmark_tag) LIKE ? OR lower(dataset_name) LIKE ?
                    OR lower(model_adapter) LIKE ? OR lower(attack) LIKE ?
                )
                """
            )
            params.extend([like] * 8)
        where_sql = "WHERE " + " AND ".join(where) if where else ""
        offset = (int(page) - 1) * int(page_size)
        callable_expr = "reusable_status = 'ready' AND COALESCE(clean_image_ref, '') <> '' AND COALESCE(adv_image_ref, '') <> ''"
        sort_columns = {
            "created_at": "created_at",
            "batch_id": "batch_id",
            "task_dataset": "task_kind",
            "attack": "attack",
            "sample_count": "callable_assets",
            "evidence": "evidence_complete_count",
            "avg_l2": "avg_l2",
            "risk": "avg_risk_score",
            "used_count": "batch_call_count",
            "batch_call_count": "batch_call_count",
        }
        sort_key = str(sort_by or "created_at").strip()
        sort_expr = sort_columns.get(sort_key, "created_at")
        sort_direction = "ASC" if str(sort_dir or "desc").lower() == "asc" else "DESC"
        sort_secondary = "source_run_id DESC" if sort_expr == "created_at" else "created_at DESC, source_run_id DESC"
        order_sql = f"{sort_expr} {sort_direction}, {sort_secondary}"

        with self._connect() as conn:
            total = conn.execute(
                f"""
                SELECT COUNT(1) FROM (
                    SELECT source_run_id
                    FROM sample_assets
                    {where_sql}
                    GROUP BY source_run_id
                ) batches
                """,
                params,
            ).fetchone()[0]
            status_rows = conn.execute(
                f"""
                SELECT reusable_status AS value, COUNT(1) AS count
                FROM sample_assets
                {where_sql}
                GROUP BY reusable_status
                """,
                params,
            ).fetchall()
            reusable_counts = {str(r["value"] or ""): int(r["count"] or 0) for r in status_rows}
            total_assets_count = conn.execute(f"SELECT COUNT(1) FROM sample_assets {where_sql}", params).fetchone()[0]
            callable_assets = conn.execute(
                f"SELECT COUNT(1) FROM sample_assets {where_sql + (' AND ' if where_sql else 'WHERE ') + callable_expr}",
                params,
            ).fetchone()[0]
            callable_batches = conn.execute(
                f"""
                SELECT COUNT(1) FROM (
                    SELECT source_run_id
                    FROM sample_assets
                    {where_sql}
                    GROUP BY source_run_id
                    HAVING SUM(CASE WHEN {callable_expr} THEN 1 ELSE 0 END) >= 1
                ) batches
                """,
                params,
            ).fetchone()[0]
            rows = conn.execute(
                f"""
                SELECT
                    source_run_id AS batch_id,
                    source_run_id,
                    MAX(created_at) AS created_at,
                    MAX(updated_at) AS updated_at,
                    MAX(task_kind) AS task_kind,
                    COUNT(DISTINCT NULLIF(task_kind, '')) AS task_kind_count,
                    MAX(dataset_name) AS dataset_name,
                    COUNT(DISTINCT NULLIF(dataset_name, '')) AS dataset_name_count,
                    MAX(benchmark_tag) AS benchmark_tag,
                    COUNT(DISTINCT NULLIF(benchmark_tag, '')) AS benchmark_tag_count,
                    MAX(model_adapter) AS model_adapter,
                    COUNT(DISTINCT NULLIF(model_adapter, '')) AS model_adapter_count,
                    MAX(attack) AS attack,
                    COUNT(DISTINCT NULLIF(attack, '')) AS attack_count,
                    MAX(attack_scope) AS attack_scope,
                    COUNT(DISTINCT NULLIF(attack_scope, '')) AS attack_scope_count,
                    COUNT(1) AS total_assets,
                    SUM(CASE WHEN reusable_status = 'ready' THEN 1 ELSE 0 END) AS ready_assets,
                    SUM(CASE WHEN reusable_status = 'summary_only' THEN 1 ELSE 0 END) AS summary_only_assets,
                    SUM(CASE WHEN reusable_status = 'legacy' THEN 1 ELSE 0 END) AS legacy_assets,
                    SUM(CASE WHEN reusable_status = 'pending_evaluation' THEN 1 ELSE 0 END) AS pending_evaluation_assets,
                    SUM(CASE WHEN {callable_expr} THEN 1 ELSE 0 END) AS callable_assets,
                    SUM(CASE WHEN COALESCE(clean_image_ref, '') <> '' AND COALESCE(adv_image_ref, '') <> '' THEN 1 ELSE 0 END) AS evidence_complete_count,
                    SUM(CASE WHEN judge_success THEN 1 ELSE 0 END) AS successful_assets,
                    AVG(risk_score) AS avg_risk_score,
                    AVG(perturbation_l2) AS avg_l2,
                    AVG(perturbation_linf) AS avg_linf,
                    SUM(COALESCE(used_count, 0)) AS sample_usage_count,
                    (
                        CASE WHEN COALESCE(sample_assets.source_run_id, '') <> '' THEN 1 ELSE 0 END
                        +
                        (
                            SELECT COUNT(DISTINCT sau.evaluation_run_id)
                            FROM sample_asset_usages sau
                            JOIN sample_assets used_assets ON used_assets.asset_id = sau.asset_id
                            WHERE used_assets.source_run_id = sample_assets.source_run_id
                              AND COALESCE(sau.evaluation_run_id, '') <> ''
                              AND sau.evaluation_run_id <> sample_assets.source_run_id
                        )
                    ) AS batch_call_count,
                    MAX(last_used_at) AS last_used_at
                FROM sample_assets
                {where_sql}
                GROUP BY source_run_id
                ORDER BY {order_sql}
                LIMIT ? OFFSET ?
                """,
                [*params, int(page_size), int(offset)],
            ).fetchall()

            def grouped(column: str) -> list[dict[str, Any]]:
                group_rows = conn.execute(
                    f"""
                    SELECT {column} AS value, COUNT(DISTINCT source_run_id) AS count
                    FROM sample_assets
                    {where_sql}
                    GROUP BY {column}
                    ORDER BY {column} ASC
                    """,
                    params,
                ).fetchall()
                return [{"value": str(r["value"] or ""), "count": int(r["count"] or 0)} for r in group_rows if str(r["value"] or "")]

            options = {
                "task_kinds": grouped("task_kind"),
                "attacks": grouped("attack"),
                "scopes": grouped("attack_scope"),
                "models": grouped("model_adapter"),
                "datasets": grouped("benchmark_tag"),
                "reusable_statuses": [{"value": key, "count": value} for key, value in sorted(reusable_counts.items()) if key],
            }

            items: list[dict[str, Any]] = []
            for row in rows:
                batch_id = str(row["batch_id"] or "")
                callable_where = list(where)
                callable_params = list(params)
                row_callable_count = int(row["callable_assets"] or 0)
                row_pending_count = int(row["pending_evaluation_assets"] or 0)
                preview_pending = str(reusable_status or "") == "pending_evaluation" or (row_callable_count == 0 and row_pending_count >= 1)
                selectable_status = "pending_evaluation" if preview_pending else "ready"
                callable_where.extend(["source_run_id = ?", "reusable_status = ?", "COALESCE(clean_image_ref, '') <> ''", "COALESCE(adv_image_ref, '') <> ''"])
                callable_params.extend([batch_id, selectable_status])
                callable_where_sql = "WHERE " + " AND ".join(callable_where)
                asset_rows = conn.execute(
                    f"""
                    SELECT *
                    FROM sample_assets
                    {callable_where_sql}
                    ORDER BY created_at DESC, source_case_id ASC, asset_id ASC
                    LIMIT 6
                    """,
                    callable_params,
                ).fetchall()
                asset_id_rows = []
                if include_asset_ids:
                    asset_id_rows = conn.execute(
                        f"""
                        SELECT asset_id
                        FROM sample_assets
                        {callable_where_sql}
                        ORDER BY created_at DESC, source_case_id ASC, asset_id ASC
                        """,
                        callable_params,
                    ).fetchall()
                total_assets = int(row["total_assets"] or 0)
                evidence_count = int(row["evidence_complete_count"] or 0)
                callable_count = row_callable_count
                pending_count = row_pending_count

                def value_or_mixed(name: str) -> str:
                    count = int(row[f"{name}_count"] or 0)
                    value = str(row[name] or "")
                    if count > 1:
                        return "mixed"
                    return value

                item = {
                    "batch_id": batch_id,
                    "source_run_id": batch_id,
                    "task_kind": value_or_mixed("task_kind"),
                    "dataset_name": value_or_mixed("dataset_name"),
                    "benchmark_tag": value_or_mixed("benchmark_tag"),
                    "model_adapter": value_or_mixed("model_adapter"),
                    "attack": value_or_mixed("attack"),
                    "attack_scope": value_or_mixed("attack_scope"),
                    "created_at": str(row["created_at"] or ""),
                    "updated_at": str(row["updated_at"] or ""),
                    "total_assets": total_assets,
                    "ready_assets": int(row["ready_assets"] or 0),
                    "summary_only_assets": int(row["summary_only_assets"] or 0),
                    "legacy_assets": int(row["legacy_assets"] or 0),
                    "pending_evaluation_assets": pending_count,
                    "callable_assets": callable_count,
                    "evidence_complete_count": evidence_count,
                    "evidence_integrity": float(evidence_count / total_assets) if total_assets else 0.0,
                    "successful_assets": int(row["successful_assets"] or 0),
                    "avg_risk_score": float(row["avg_risk_score"] or 0.0),
                    "avg_l2": float(row["avg_l2"] or 0.0),
                    "avg_linf": float(row["avg_linf"] or 0.0),
                    "used_count": int(row["batch_call_count"] or 0),
                    "batch_call_count": int(row["batch_call_count"] or 0),
                    "sample_usage_count": int(row["sample_usage_count"] or 0),
                    "last_used_at": str(row["last_used_at"] or "") if row["last_used_at"] else "",
                    "asset_ids": [str(asset["asset_id"] or "") for asset in asset_id_rows if str(asset["asset_id"] or "")],
                    "preview_assets": [self._asset_row_to_dict(asset) for asset in asset_rows],
                    "report_url": "" if pending_count >= 1 and callable_count == 0 else (f"/reports/{batch_id}" if batch_id else ""),
                    "batch_status": "callable" if callable_count >= 1 else ("pending_evaluation" if pending_count >= 1 else "not_callable"),
                }
                items.append(item)

        summary = {
            "total_batches": int(total),
            "callable_batches": int(callable_batches or 0),
            "total_assets": int(total_assets_count or 0),
            "ready_assets": reusable_counts.get("ready", 0),
            "callable_assets": int(callable_assets or 0),
            "summary_only_assets": reusable_counts.get("summary_only", 0),
            "legacy_assets": reusable_counts.get("legacy", 0),
            "pending_evaluation_assets": reusable_counts.get("pending_evaluation", 0),
            "task_distribution": {item["value"]: item["count"] for item in options["task_kinds"]},
            "attack_distribution": {item["value"]: item["count"] for item in options["attacks"]},
            "scope_distribution": {item["value"]: item["count"] for item in options["scopes"]},
        }
        return int(total), items, {"summary": summary, "options": options}

    def get_sample_assets(self, asset_ids: list[str]) -> list[dict[str, Any]]:
        ids = [str(item).strip() for item in asset_ids if str(item).strip()]
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        visibility_where, visibility_params = _sample_asset_visibility_filter()
        where_sql = " AND ".join([f"asset_id IN ({placeholders})", *visibility_where])
        with self._connect() as conn:
            rows = conn.execute(f"SELECT * FROM sample_assets WHERE {where_sql}", [*ids, *visibility_params]).fetchall()
        by_id = {str(row["asset_id"]): self._asset_row_to_dict(row) for row in rows}
        return [by_id[item] for item in ids if item in by_id]

    def record_sample_asset_usage(self, *, asset_ids: list[str], evaluation_run_id: str, job_id: str = "", note: str = "") -> None:
        ids = [str(item).strip() for item in asset_ids if str(item).strip()]
        if not ids or not evaluation_run_id:
            return
        now = utc_now_iso()
        with self._lock:
            with self._connect() as conn:
                for asset_id in ids:
                    conn.execute(
                        """
                        INSERT INTO sample_asset_usages(asset_id, evaluation_run_id, job_id, created_at, note)
                        VALUES(?, ?, ?, ?, ?)
                        """,
                        (asset_id, evaluation_run_id, job_id, now, note[:1000]),
                    )
                    conn.execute(
                        """
                        UPDATE sample_assets
                        SET used_count = COALESCE(used_count, 0) + 1,
                            last_used_at = ?,
                            updated_at = ?
                        WHERE asset_id = ?
                        """,
                        (now, now, asset_id),
                    )
                conn.commit()
