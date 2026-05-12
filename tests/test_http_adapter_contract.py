# 文件说明：该文件属于自动化测试，集中实现 test http adapter contract 相关逻辑。
from __future__ import annotations

import numpy as np
import pytest
import requests

from mmsec_eval.model_adapters.http_adapter import HttpAdapter, HttpAdapterError
from mmsec_eval.types import Sample


# 实现 `_Resp.__init__` 的对象行为，维护该类在自动化测试中的调用契约。
class _Resp:
    # 封装 _Resp.__init__ 的内部步骤，让自动化测试主流程保持清晰并隔离边界细节。
    def __init__(self, status_code: int, payload):
        self.status_code = status_code
        self._payload = payload

    # 判断或归一 `抛错 所属 状态` 状态，让调用方可以稳定渲染能力和可用性。
    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status={self.status_code}")

    # 实现 `_Resp.json` 的对象行为，维护该类在自动化测试中的调用契约。
    def json(self):
        return self._payload


# 执行 `样本` 辅助逻辑，保持自动化测试中的输入处理和结果输出一致。
def _sample() -> Sample:
    return Sample(sample_id="s1", image=np.zeros((16, 16, 3), dtype=np.float32), text="hello world")


# 验证 `HTTP adapter contract success` 场景，防止相关行为在后续修改中退化。
def test_http_adapter_contract_success(monkeypatch):
    monkeypatch.setenv("MMSEC_HTTP_ADAPTER_ENDPOINT", "http://localhost/mock")
    monkeypatch.setenv("MMSEC_HTTP_ADAPTER_RETRIES", "1")
    monkeypatch.setenv("MMSEC_HTTP_ADAPTER_TIMEOUT", "3")

    # 执行 `post` 辅助逻辑，保持自动化测试中的输入处理和结果输出一致。
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


# 验证 `HTTP adapter 数据结构 error` 场景，防止相关行为在后续修改中退化。
def test_http_adapter_schema_error(monkeypatch):
    monkeypatch.setenv("MMSEC_HTTP_ADAPTER_ENDPOINT", "http://localhost/mock")

    # 执行 `post` 辅助逻辑，保持自动化测试中的输入处理和结果输出一致。
    def _post(url, json, timeout):  # noqa: ANN001
        return _Resp(200, {"text": "ok"})

    monkeypatch.setattr("mmsec_eval.model_adapters.http_adapter.requests.post", _post)
    with pytest.raises(HttpAdapterError) as e:
        HttpAdapter().predict(_sample())
    assert e.value.error_code == "http_schema_invalid"


# 验证 `HTTP adapter retry on timeout` 场景，防止相关行为在后续修改中退化。
def test_http_adapter_retry_on_timeout(monkeypatch):
    monkeypatch.setenv("MMSEC_HTTP_ADAPTER_ENDPOINT", "http://localhost/mock")
    monkeypatch.setenv("MMSEC_HTTP_ADAPTER_RETRIES", "2")
    calls = {"n": 0}

    # 执行 `post` 辅助逻辑，保持自动化测试中的输入处理和结果输出一致。
    def _post(url, json, timeout):  # noqa: ANN001
        calls["n"] += 1
        if calls["n"] == 1:
            raise requests.Timeout("timeout")
        return _Resp(200, {"text": "ok", "score": 0.5})

    monkeypatch.setattr("mmsec_eval.model_adapters.http_adapter.requests.post", _post)
    out = HttpAdapter().predict(_sample())
    assert out.text == "ok"
    assert calls["n"] == 2
