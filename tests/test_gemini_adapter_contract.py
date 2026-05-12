# 文件说明：该文件属于自动化测试，集中实现 test gemini adapter contract 相关逻辑。
from __future__ import annotations

import numpy as np

from mmsec_eval.model_adapters.gemini_adapter import GeminiVisionAdapter
from mmsec_eval.types import Sample


# 中文注释：定义 _Resp 的结构化职责，作为自动化测试中状态、配置或行为的边界。
class _Resp:
    # 中文注释：封装 _Resp.__init__ 的内部步骤，让自动化测试主流程保持清晰并隔离边界细节。
    def __init__(self, payload):
        self._payload = payload

    # 中文注释：实现 _Resp.raise_for_status 的核心行为，维护自动化测试在该对象上的调用契约。
    def raise_for_status(self):
        return None

    # 中文注释：实现 _Resp.json 的核心行为，维护自动化测试在该对象上的调用契约。
    def json(self):
        return self._payload


# 中文注释：封装 _sample 的内部步骤，让自动化测试主流程保持清晰并隔离边界细节。
def _sample() -> Sample:
    return Sample(sample_id="s1", image=np.zeros((16, 16, 3), dtype=np.float32), text="a blue circle")


# 中文注释：验证 test_gemini_predict 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_gemini_predict(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("MMSEC_GEMINI_MODEL_NAME", "gemini-2.5-pro")
    monkeypatch.setenv("MMSEC_GEMINI_TIMEOUT", "18")

    # 中文注释：封装 _post 的内部步骤，让自动化测试主流程保持清晰并隔离边界细节。
    def _post(url, json, timeout):  # noqa: ANN001
        assert "gemini-2.5-pro:generateContent" in url
        assert timeout == 18.0
        return _Resp(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": '{"score": 0.74, "reason": "caption mostly matches the image"}'}
                            ]
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr("mmsec_eval.model_adapters.gemini_adapter.requests.post", _post)
    out = GeminiVisionAdapter().predict(_sample())
    assert out.score == 0.74
    assert "mostly matches" in out.text

