# 文件说明：该文件属于自动化测试，集中实现 test gemini adapter contract 相关逻辑。
from __future__ import annotations

import numpy as np

from mmsec_eval.model_adapters.gemini_adapter import GeminiVisionAdapter
from mmsec_eval.types import Sample


# 实现 `_Resp.__init__` 的对象行为，维护该类在自动化测试中的调用契约。
class _Resp:
    # 封装 _Resp.__init__ 的内部步骤，让自动化测试主流程保持清晰并隔离边界细节。
    def __init__(self, payload):
        self._payload = payload

    # 判断或归一 `抛错 所属 状态` 状态，让调用方可以稳定渲染能力和可用性。
    def raise_for_status(self):
        return None

    # 实现 `_Resp.json` 的对象行为，维护该类在自动化测试中的调用契约。
    def json(self):
        return self._payload


# 执行 `样本` 辅助逻辑，保持自动化测试中的输入处理和结果输出一致。
def _sample() -> Sample:
    return Sample(sample_id="s1", image=np.zeros((16, 16, 3), dtype=np.float32), text="a blue circle")


# 验证 `gemini predict` 场景，防止相关行为在后续修改中退化。
def test_gemini_predict(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("MMSEC_GEMINI_MODEL_NAME", "gemini-2.5-pro")
    monkeypatch.setenv("MMSEC_GEMINI_TIMEOUT", "18")

    # 执行 `post` 辅助逻辑，保持自动化测试中的输入处理和结果输出一致。
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

