from __future__ import annotations

from pathlib import Path

from api_test_utils import make_client


def test_api_health(tmp_path: Path, monkeypatch):
    with make_client(tmp_path, monkeypatch) as client:
        r = client.get("/api/v1/health")
        assert r.status_code == 200
        payload = r.json()
        assert payload["status"] == "ok"
        assert payload["bootstrap_state"] == "ready"
