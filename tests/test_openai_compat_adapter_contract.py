# 文件说明：该文件属于自动化测试，集中实现 test openai compat adapter contract 相关逻辑。
from __future__ import annotations

import time

import numpy as np
import pytest

from mmsec_eval.model_adapters.openai_compat_adapter import OpenAICompatAdapter
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
    return Sample(sample_id="s1", image=np.zeros((16, 16, 3), dtype=np.float32), text="a red square")


# 验证 `OpenAI compat predict` 场景，防止相关行为在后续修改中退化。
def test_openai_compat_predict(monkeypatch):
    monkeypatch.setenv("MMSEC_OPENAI_COMPAT_BASE_URL", "http://localhost:8001/v1")
    monkeypatch.setenv("MMSEC_OPENAI_COMPAT_MODEL_NAME", "chatgpt-4o-latest")
    monkeypatch.setenv("MMSEC_OPENAI_COMPAT_TIMEOUT", "12")

    # 执行 `post` 辅助逻辑，保持自动化测试中的输入处理和结果输出一致。
    def _post(self, url, headers, json, timeout):  # noqa: ANN001
        assert url == "http://localhost:8001/v1/chat/completions"
        assert json["model"] == "chatgpt-4o-latest"
        assert json["max_tokens"] == 48
        assert timeout == 12.0
        return _Resp(
            {
                "choices": [
                    {
                        "message": {
                            "content": '{"score": 0.82, "reason": "image and caption are strongly aligned"}'
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr("requests.sessions.Session.post", _post)
    out = OpenAICompatAdapter().predict(_sample())
    assert out.score == 0.82
    assert "strongly aligned" in out.text


# 验证 `OpenAI compat variant predict` 场景，防止相关行为在后续修改中退化。
def test_openai_compat_variant_predict(monkeypatch):
    monkeypatch.setenv("MMSEC_OPENAI_QWEN3_VL_BASE_URL", "http://127.0.0.1:8012/v1")
    monkeypatch.setenv("MMSEC_OPENAI_QWEN3_VL_MODEL_NAME", "Qwen/Qwen3-VL-8B-Instruct")
    monkeypatch.setenv("MMSEC_OPENAI_QWEN3_VL_TIMEOUT", "90")
    monkeypatch.setenv("MMSEC_OPENAI_QWEN3_VL_API_KEY_ENV", "LOCAL_QWEN3_VL_API_KEY")

    # 执行 `post` 辅助逻辑，保持自动化测试中的输入处理和结果输出一致。
    def _post(self, url, headers, json, timeout):  # noqa: ANN001
        assert url == "http://127.0.0.1:8012/v1/chat/completions"
        assert json["model"] == "Qwen/Qwen3-VL-8B-Instruct"
        assert json["max_tokens"] == 48
        assert timeout == 90.0
        return _Resp({"choices": [{"message": {"content": '{"score": 0.61, "reason": "variant ok"}'}}]})

    monkeypatch.setattr("requests.sessions.Session.post", _post)
    out = OpenAICompatAdapter(variant="QWEN3_VL").predict(_sample())
    assert out.score == 0.61
    assert out.raw["adapter"] == "openai_qwen3_vl"


# 验证 `OpenAI compat additional variant predict` 场景，防止相关行为在后续修改中退化。
def test_openai_compat_additional_variant_predict(monkeypatch):
    monkeypatch.setenv("MMSEC_OPENAI_GEMMA3_12B_BASE_URL", "http://127.0.0.1:8017/v1")
    monkeypatch.setenv("MMSEC_OPENAI_GEMMA3_12B_MODEL_NAME", "google/gemma-3-12b-it")

    # 执行 `post` 辅助逻辑，保持自动化测试中的输入处理和结果输出一致。
    def _post(self, url, headers, json, timeout):  # noqa: ANN001
        assert url == "http://127.0.0.1:8017/v1/chat/completions"
        assert json["model"] == "google/gemma-3-12b-it"
        assert timeout == 45.0
        return _Resp({"choices": [{"message": {"content": '{"score": 0.57, "reason": "gemma ok"}'}}]})

    monkeypatch.setattr("requests.sessions.Session.post", _post)
    out = OpenAICompatAdapter(variant="GEMMA3_12B").predict(_sample())
    assert out.score == 0.57
    assert out.raw["adapter"] == "openai_gemma3_12b"


# 验证 `OpenAI compat max tokens 环境 override` 场景，防止相关行为在后续修改中退化。
def test_openai_compat_max_tokens_env_override(monkeypatch):
    monkeypatch.setenv("MMSEC_OPENAI_COMPAT_BASE_URL", "http://127.0.0.1:8001/v1")
    monkeypatch.setenv("MMSEC_OPENAI_COMPAT_MODEL_NAME", "local-vlm")
    monkeypatch.setenv("MMSEC_OPENAI_COMPAT_MAX_TOKENS", "24")

    # 执行 `post` 辅助逻辑，保持自动化测试中的输入处理和结果输出一致。
    def _post(self, url, headers, json, timeout):  # noqa: ANN001
        del self, url, headers, timeout
        assert json["max_tokens"] == 24
        return _Resp({"choices": [{"message": {"content": '{"score": 0.5, "reason": "ok"}'}}]})

    monkeypatch.setattr("requests.sessions.Session.post", _post)

    assert OpenAICompatAdapter().predict(_sample()).score == 0.5


# 验证 `OpenAI compat 分数 pairs preserves order under concurrency` 场景，防止相关行为在后续修改中退化。
def test_openai_compat_score_pairs_preserves_order_under_concurrency(monkeypatch):
    monkeypatch.setenv("MMSEC_OPENAI_COMPAT_BASE_URL", "http://127.0.0.1:8001/v1")
    monkeypatch.setenv("MMSEC_OPENAI_COMPAT_MODEL_NAME", "local-vlm")
    monkeypatch.setenv("MMSEC_OPENAI_COMPAT_CONCURRENCY", "4")

    # 执行 `post` 辅助逻辑，保持自动化测试中的输入处理和结果输出一致。
    def _post(self, url, headers, json, timeout):  # noqa: ANN001
        del headers, timeout
        assert url == "http://127.0.0.1:8001/v1/chat/completions"
        content_parts = json["messages"][0]["content"]
        text_parts = [part for part in content_parts if part.get("type") == "text"]
        assert text_parts
        content = text_parts[0]["text"]
        marker = int(content.split("pair-")[1].split("\n", 1)[0])
        time.sleep(0.01 * float(4 - marker))
        score = 0.1 * float(marker + 1)
        return _Resp({"choices": [{"message": {"content": f'{{"score": {score}, "reason": "ok"}}'}}]})

    monkeypatch.setattr("requests.sessions.Session.post", _post)
    adapter = OpenAICompatAdapter()
    pairs = [
        (np.zeros((8, 8, 3), dtype=np.float32), "pair-0"),
        (np.zeros((8, 8, 3), dtype=np.float32), "pair-1"),
        (np.zeros((8, 8, 3), dtype=np.float32), "pair-2"),
        (np.zeros((8, 8, 3), dtype=np.float32), "pair-3"),
    ]

    scores = adapter.score_pairs(pairs, batch_size=4)

    assert scores.tolist() == pytest.approx([0.1, 0.2, 0.3, 0.4])
