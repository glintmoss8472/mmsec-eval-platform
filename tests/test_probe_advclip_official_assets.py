from __future__ import annotations

import json
from io import BytesIO

from scripts.probe_advclip_official_assets import probe_cstcloud_share


class _FakeResponse(BytesIO):
    status = 200

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None


def test_probe_cstcloud_share_marks_expired(monkeypatch) -> None:
    body = json.dumps({"stat": "ERR_SHARE_EXPIRED", "errText": "分享已过期"}).encode()

    def fake_urlopen(request, timeout):  # noqa: ANN001
        return _FakeResponse(body)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    result = probe_cstcloud_share("JqKbqGfTRs", timeout=1)

    assert result["status"] == "expired"
    assert "raw_data.rar" in result["required_payload"]
    assert "blocker" in result


def test_probe_cstcloud_share_marks_available(monkeypatch) -> None:
    body = json.dumps({"stat": "OK", "name": "raw_data.rar"}).encode()

    def fake_urlopen(request, timeout):  # noqa: ANN001
        return _FakeResponse(body)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    result = probe_cstcloud_share("JqKbqGfTRs", timeout=1)

    assert result["status"] == "available"
    assert result["parsed_response"]["stat"] == "OK"
