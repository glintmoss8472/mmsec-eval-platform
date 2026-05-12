from __future__ import annotations

import numpy as np
import pytest
import requests

from mmsec_eval.model_adapters.http_adapter import HttpAdapter, HttpAdapterError
from mmsec_eval.types import Sample


class _Resp:
    def __init__(self, status_code: int, payload):
        self.status_code = status_code
        self._payload = payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status={self.status_code}")

    def json(self):
        return self._payload


def _sample() -> Sample:
    return Sample(sample_id="s1", image=np.zeros((16, 16, 3), dtype=np.float32), text="hello world")


def test_http_adapter_contract_success(monkeypatch):
    monkeypatch.setenv("MMSEC_HTTP_ADAPTER_ENDPOINT", "http://localhost/mock")
    monkeypatch.setenv("MMSEC_HTTP_ADAPTER_RETRIES", "1")
    monkeypatch.setenv("MMSEC_HTTP_ADAPTER_TIMEOUT", "3")

    def _post(url, json, timeout):  # noqa: ANN001
        assert url == "http://localhost/mock"
        assert set(json.keys()) == {"text", "image_b64", "metadata"}
        assert timeout == 3.0
        return _Resp(200, {"text": "ok", "score": 0.7, "embedding": [0.1, 0.2], "raw": {"x": 1}})

    monkeypatch.setattr("mmsec_eval.model_adapters.http_adapter.requests.post", _post)
    out = HttpAdapter().predict(_sample())
    assert out.text == "ok"
    assert out.score == 0.7
    assert out.embedding is not None
    assert out.raw["attempt"] == 1


def test_http_adapter_schema_error(monkeypatch):
    monkeypatch.setenv("MMSEC_HTTP_ADAPTER_ENDPOINT", "http://localhost/mock")

    def _post(url, json, timeout):  # noqa: ANN001
        return _Resp(200, {"text": "ok"})

    monkeypatch.setattr("mmsec_eval.model_adapters.http_adapter.requests.post", _post)
    with pytest.raises(HttpAdapterError) as e:
        HttpAdapter().predict(_sample())
    assert e.value.error_code == "http_schema_invalid"


def test_http_adapter_retry_on_timeout(monkeypatch):
    monkeypatch.setenv("MMSEC_HTTP_ADAPTER_ENDPOINT", "http://localhost/mock")
    monkeypatch.setenv("MMSEC_HTTP_ADAPTER_RETRIES", "2")
    calls = {"n": 0}

    def _post(url, json, timeout):  # noqa: ANN001
        calls["n"] += 1
        if calls["n"] == 1:
            raise requests.Timeout("timeout")
        return _Resp(200, {"text": "ok", "score": 0.5})

    monkeypatch.setattr("mmsec_eval.model_adapters.http_adapter.requests.post", _post)
    out = HttpAdapter().predict(_sample())
    assert out.text == "ok"
    assert calls["n"] == 2
