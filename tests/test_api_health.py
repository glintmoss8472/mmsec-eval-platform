# 文件说明：该文件属于自动化测试，集中实现 test api health 相关逻辑。
from __future__ import annotations

from pathlib import Path

from api_test_utils import make_client


# 中文注释：验证 test_api_health 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_api_health(tmp_path: Path, monkeypatch):
    with make_client(tmp_path, monkeypatch) as client:
        r = client.get("/api/v1/health")
        assert r.status_code == 200
        payload = r.json()
        assert payload["status"] == "ok"
        assert payload["bootstrap_state"] == "ready"
