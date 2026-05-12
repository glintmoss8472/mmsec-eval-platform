from __future__ import annotations

from pathlib import Path

from api_test_utils import make_client, wait_job_done, write_toy_eval_cfg


def write_cfg(path: Path) -> None:
    write_toy_eval_cfg(path)


def test_job_progress_endpoint_tracks_stage_and_run_id(tmp_path: Path, monkeypatch):
    with make_client(tmp_path, monkeypatch, skip_model_preflight=True) as client:
        cfg = tmp_path / "cfg.yaml"
        write_cfg(cfg)

        resp = client.post(
            "/api/v1/jobs",
            json={
                "job_type": "run_eval",
                "config_path": str(cfg),
                "override": {},
                "benchmark_mode": False,
            },
        )
        assert resp.status_code == 200
        job_id = resp.json()["id"]

        initial_progress = client.get(f"/api/v1/jobs/{job_id}/progress")
        assert initial_progress.status_code == 200
        initial_data = initial_progress.json()
        assert initial_data["job_id"] == job_id
        assert initial_data["status"] in {"queued", "running"}
        assert isinstance(initial_data["stages"], list)
        assert initial_data["stages"]

        status = wait_job_done(client, job_id, timeout_s=240.0)
        assert status == "success"

        final_progress = client.get(f"/api/v1/jobs/{job_id}/progress")
        assert final_progress.status_code == 200
        final_data = final_progress.json()
        assert final_data["status"] == "success"
        assert final_data["run_id"]
        assert final_data["progress_percent"] >= 97
        assert any(stage["stage_key"] == "completed" and stage["state"] == "success" for stage in final_data["stages"])
