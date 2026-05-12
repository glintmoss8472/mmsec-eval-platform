from __future__ import annotations

import importlib
import json
import time
from pathlib import Path

import pytest
import yaml

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from mmsec_api.services.bootstrap_orchestrator import BootstrapOrchestrator
from mmsec_api.store.sqlite import SQLiteStore
from mmsec_api.worker.queue import JobQueue
from mmsec_eval.config.schema import BootstrapConfig


def make_client(tmp_path: Path, monkeypatch) -> TestClient:
    art = tmp_path / "artifacts"
    cfg = tmp_path / "bootstrap.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "bootstrap": {
                    "enabled": True,
                    "auto_prepare_datasets": False,
                    "auto_ingest_docs": False,
                    "auto_run_benchmark": False,
                    "model_warmup": False,
                    "seed_dir": "seed",
                    "demo_benchmark_config": "configs/bench/bootstrap_quick.yaml",
                }
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("MMSEC_ARTIFACTS_DIR", str(art))
    monkeypatch.setenv("MMSEC_APP_DB", str(art / "app.db"))
    monkeypatch.setenv("MMSEC_BOOTSTRAP_CONFIG", str(cfg))
    monkeypatch.setenv("MMSEC_BOOTSTRAP_ENABLED", "1")

    import mmsec_api.main as api_main

    importlib.reload(api_main)
    return TestClient(api_main.app)


def test_bootstrap_status_and_logs(tmp_path: Path, monkeypatch):
    with make_client(tmp_path, monkeypatch) as client:
        final = {}
        for _ in range(120):
            time.sleep(0.1)
            r = client.get("/api/v1/bootstrap/status")
            assert r.status_code == 200
            final = r.json()
            if final.get("state") in {"ready", "degraded"}:
                break

        assert final.get("state") in {"ready", "degraded", "warming"}
        assert isinstance(final.get("steps"), list)
        assert "artifacts" in final

        artifacts = final["artifacts"]
        docs_index = artifacts.get("docs_index", "")
        docs_snips = artifacts.get("docs_snippets", "")
        assert isinstance(docs_index, str)
        assert isinstance(docs_snips, str)
        if docs_index:
            assert Path(docs_index).exists()
        if docs_snips:
            assert Path(docs_snips).exists()

        logs = client.get("/api/v1/bootstrap/logs")
        assert logs.status_code == 200
        assert len(logs.json()["items"]) >= 1

        retry = client.post("/api/v1/bootstrap/retry")
        assert retry.status_code == 200
        assert retry.json()["state"] in {"pending", "seeding", "warming", "ready", "degraded"}

        h = client.get("/api/v1/health")
        assert h.status_code == 200
        assert "bootstrap_state" in h.json()


def test_seed_runs_skip_fake_model_adapters(tmp_path: Path):
    seed_root = tmp_path / "seed"
    run_dir = seed_root / "runs" / "fake_seed"
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "run_id": "fake_seed",
                "model_adapter": "dummy",
                "benchmark_tag": "seed_bootstrap",
                "dataset_name": "demo",
                "attack": "fgsm",
            }
        ),
        encoding="utf-8",
    )

    store = SQLiteStore(str(tmp_path / "artifacts" / "app.db"))
    store.init_db()
    orchestrator = BootstrapOrchestrator(
        store=store,
        queue=JobQueue(store=store, executor=object()),
        artifacts_dir=str(tmp_path / "artifacts"),
        bootstrap=BootstrapConfig(
            seed_dir="seed",
            auto_prepare_datasets=False,
            auto_ingest_docs=False,
            auto_run_benchmark=False,
            model_warmup=False,
        ),
    )
    orchestrator.project_root = tmp_path

    orchestrator._seed_runs(seed_root)

    assert not (tmp_path / "artifacts" / "runs" / "fake_seed").exists()
    assert orchestrator.get_status()["artifacts"]["seeded_runs"] == []
    total, rows = store.list_runs_cache(page=1, page_size=20)
    assert total == 0
    assert rows == []
