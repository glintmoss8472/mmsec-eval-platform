# 文件说明：该文件属于模型适配层，集中实现 http adapter 相关逻辑。
from __future__ import annotations

import os
from typing import Any

import numpy as np
import requests

from mmsec_eval.model_adapters.remote_common import encode_image_b64
from mmsec_eval.plugins.base import ModelAdapter
from mmsec_eval.types import ModelOutput, Sample


# 实现 `HttpAdapterError.__init__` 的对象行为，维护该类在模型适配层中的调用契约。
class HttpAdapterError(RuntimeError):
    # 封装 HttpAdapterError.__init__ 的内部步骤，让模型适配层主流程保持清晰并隔离边界细节。
    def __init__(self, error_code: str, message: str, *, status_code: int = 0) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.status_code = status_code


# 实现 `HttpAdapter.__init__` 的对象行为，维护该类在模型适配层中的调用契约。
class HttpAdapter(ModelAdapter):
    # 封装 HttpAdapter.__init__ 的内部步骤，让模型适配层主流程保持清晰并隔离边界细节。
    def __init__(self) -> None:
        self.endpoint = os.getenv("MMSEC_HTTP_ADAPTER_ENDPOINT", "").strip()
        self.timeout = float(os.getenv("MMSEC_HTTP_ADAPTER_TIMEOUT", "15"))
        self.retries = int(os.getenv("MMSEC_HTTP_ADAPTER_RETRIES", "2"))
        if not self.endpoint:
            raise HttpAdapterError("http_endpoint_missing", "MMSEC_HTTP_ADAPTER_ENDPOINT is required for HttpAdapter")

    # 实现 `HttpAdapter._encode_image` 的对象行为，维护该类在模型适配层中的调用契约。
    def _encode_image(self, image: np.ndarray) -> str:
        return encode_image_b64(image, image_format="PNG")

    # 组装 `载荷`，把分散字段整理成后端任务或风险评分使用的载荷。
    def _payload(self, sample: Sample) -> dict[str, Any]:
        return {"text": sample.text, "image_b64": self._encode_image(sample.image), "metadata": dict(sample.metadata)}

    # 组装 `生成式评测 载荷`，把分散字段整理成后端任务或风险评分使用的载荷。
    def _generation_payload(self, sample: Sample, *, task: str, prompt: str, question: str = "", object_name: str = "", max_tokens: int = 64) -> dict[str, Any]:
        return {
            "task": task,
            "prompt": prompt,
            "question": question,
            "object_name": object_name,
            "max_tokens": int(max_tokens),
            "text": sample.text,
            "image_b64": self._encode_image(sample.image),
            "metadata": dict(sample.metadata),
        }

    # 解析 `响应`，把文本或载荷转换成可校验的字段。
    def _parse_response(self, data: Any) -> ModelOutput:
        if not isinstance(data, dict):
            raise HttpAdapterError("http_schema_invalid", "response must be a JSON object")
        if "text" not in data or "score" not in data:
            raise HttpAdapterError("http_schema_invalid", "response requires keys: text, score")
        text = str(data.get("text", ""))
        try:
            score = float(data.get("score", 0.0))
        except (TypeError, ValueError) as e:
            raise HttpAdapterError("http_schema_invalid", f"score must be numeric: {e}") from e
        embedding_raw = data.get("embedding")
        embedding = None
        if embedding_raw is not None:
            if not isinstance(embedding_raw, list):
                raise HttpAdapterError("http_schema_invalid", "embedding must be a list when present")
            embedding = np.asarray(embedding_raw, dtype=np.float32)
        raw_field = data.get("raw", {})
        if raw_field is None:
            raw_field = {}
        if not isinstance(raw_field, dict):
            raw_field = {"raw_value": raw_field}
        return ModelOutput(
            text=text,
            score=score,
            embedding=embedding,
            error_code="",
            raw={"adapter": "http", "payload": data, "raw": raw_field},
        )

    # 实现 `HttpAdapter.predict` 的对象行为，维护该类在模型适配层中的调用契约。
    def predict(self, sample: Sample) -> ModelOutput:
        payload = self._payload(sample)
        return self._post_payload(payload)

    # 组装 `post 载荷`，把分散字段整理成后端任务或风险评分使用的载荷。
    def _post_payload(self, payload: dict[str, Any]) -> ModelOutput:
        attempts = max(0, self.retries) + 1
        last_error: Exception | None = None
        for idx in range(attempts):
            try:
                resp = requests.post(self.endpoint, json=payload, timeout=self.timeout)
                status_code = int(resp.status_code)
                if status_code >= 500:
                    raise HttpAdapterError("http_server_error", f"server returned status {status_code}", status_code=status_code)
                resp.raise_for_status()
                data = resp.json()
                out = self._parse_response(data)
                out.raw["status_code"] = status_code
                out.raw["attempt"] = idx + 1
                return out
            except HttpAdapterError as e:
                last_error = e
                retryable = e.error_code in {"http_server_error"}
                if idx + 1 < attempts and retryable:
                    continue
                raise
            except requests.Timeout as e:
                last_error = e
                if idx + 1 < attempts:
                    continue
                raise HttpAdapterError("http_timeout", f"request timed out after {attempts} attempts") from e
            except requests.RequestException as e:
                last_error = e
                if idx + 1 < attempts:
                    continue
                raise HttpAdapterError("http_request_failed", f"request failed: {e}") from e
            except ValueError as e:
                last_error = e
                raise HttpAdapterError("http_invalid_json", f"response is not valid JSON: {e}") from e
        raise HttpAdapterError("http_unknown_error", f"http adapter failed: {last_error}")

    # 生成 `answer`，补齐前端展示或后续评测需要的样本资产。
    def generate_answer(self, sample: Sample, question: str, *, prompt: str = "", max_tokens: int = 64) -> ModelOutput:
        template = str(prompt or "Answer the question about the image. Use a short answer.\nQuestion: {question}")
        rendered = template.format(question=str(question))
        return self._post_payload(self._generation_payload(sample, task="vqa", prompt=rendered, question=str(question), max_tokens=max_tokens))

    # 生成 `图像描述`，补齐前端展示或后续评测需要的样本资产。
    def generate_caption(self, sample: Sample, *, prompt: str = "", max_tokens: int = 96) -> ModelOutput:
        rendered = str(prompt or "Describe only the visible content of this image in one concise sentence.")
        return self._post_payload(self._generation_payload(sample, task="caption", prompt=rendered, max_tokens=max_tokens))

    # 实现 `HttpAdapter.object_probe` 的对象行为，维护该类在模型适配层中的调用契约。
    def object_probe(self, sample: Sample, object_name: str, *, prompt: str = "", max_tokens: int = 8) -> ModelOutput:
        template = str(prompt or "Is there a {object_name} in the image? Answer yes or no.")
        rendered = template.format(object_name=str(object_name))
        return self._post_payload(
            self._generation_payload(sample, task="object_probe", prompt=rendered, object_name=str(object_name), max_tokens=max_tokens)
        )
