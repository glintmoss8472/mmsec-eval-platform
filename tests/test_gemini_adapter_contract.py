from __future__ import annotations

import numpy as np

from mmsec_eval.model_adapters.gemini_adapter import GeminiVisionAdapter
from mmsec_eval.types import Sample


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _sample() -> Sample:
    return Sample(sample_id="s1", image=np.zeros((16, 16, 3), dtype=np.float32), text="a blue circle")


def test_gemini_predict(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("MMSEC_GEMINI_MODEL_NAME", "gemini-2.5-pro")
    monkeypatch.setenv("MMSEC_GEMINI_TIMEOUT", "18")

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

