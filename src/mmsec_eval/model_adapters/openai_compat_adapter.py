# 文件说明：该文件属于模型适配层，集中实现 openai compat adapter 相关逻辑。
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import ipaddress
import os
import re
from threading import local
from typing import Any
from urllib.parse import urlparse

import numpy as np
import requests
from requests.adapters import HTTPAdapter

from mmsec_eval.model_adapters.remote_common import encode_image_b64, extract_json_payload, normalize_score
from mmsec_eval.plugins.base import ModelAdapter
from mmsec_eval.types import ModelOutput, Sample


SYSTEM_PROMPT = (
    "You are a strict image-text retrieval scorer. Inspect the image, compare it with the candidate text, "
    "and score only visible cross-modal alignment from 0.0 to 1.0. Wrong main object or scene: below 0.2; "
    "partial match: 0.3-0.6; strong visual match: above 0.8. "
    'Return compact JSON only: {"score": 0.0, "reason": "short reason"}.'
)


# 判断 `是否 loopback 基础 URL` 条件是否成立，为调用方提供布尔决策。
def _is_loopback_base_url(url: str) -> bool:
    host = str(urlparse(str(url or "")).hostname or "").strip().lower()
    if not host:
        return False
    if host == "localhost":
        return True
    try:
        return bool(ipaddress.ip_address(host).is_loopback)
    except ValueError:
        return False


# 定义 `OpenAICompatAdapter` 的插件适配边界，把模型、攻击或评测能力暴露为统一接口。
class OpenAICompatAdapter(ModelAdapter):
    """Vision scorer for OpenAI and OpenAI-compatible endpoints such as vLLM."""

    # 实现 `OpenAICompatAdapter.__init__` 的对象行为，维护该类在模型适配层中的调用契约。
    def __init__(self, *, variant: str = "") -> None:
        self.variant = str(variant or "").strip().upper()
        self.adapter_name = "openai_compat" if not self.variant else f"openai_{self.variant.lower()}"

        self.base_url = self._read_setting(
            "BASE_URL",
            generic_env="MMSEC_OPENAI_COMPAT_BASE_URL",
            legacy_env="MMSEC_OPENAI_BASE_URL",
            default="https://api.openai.com/v1",
        ).rstrip("/")
        self.model_name = self._read_setting(
            "MODEL_NAME",
            generic_env="MMSEC_OPENAI_COMPAT_MODEL_NAME",
            legacy_env="MMSEC_OPENAI_MODEL_NAME",
            default="chatgpt-4o-latest",
        ).strip()
        self.timeout = float(
            self._read_setting(
                "TIMEOUT",
                generic_env="MMSEC_OPENAI_COMPAT_TIMEOUT",
                legacy_env="MMSEC_OPENAI_TIMEOUT",
                default="45",
            )
        )
        self.api_key_env = self._read_setting(
            "API_KEY_ENV",
            generic_env="MMSEC_OPENAI_COMPAT_API_KEY_ENV",
            legacy_env="MMSEC_OPENAI_API_KEY_ENV",
            default="OPENAI_API_KEY",
        ).strip()
        self.api_key = os.getenv(self.api_key_env, "").strip()
        raw_prompt_order = self._read_setting(
            "PROMPT_ORDER",
            generic_env="MMSEC_OPENAI_COMPAT_PROMPT_ORDER",
            legacy_env="MMSEC_OPENAI_PROMPT_ORDER",
            default="image_first",
        ).strip().lower().replace("-", "_")
        self.prompt_order = raw_prompt_order if raw_prompt_order in {"image_first", "text_first"} else "image_first"
        self.loopback_base_url = _is_loopback_base_url(self.base_url)
        default_max_tokens = "48" if self.loopback_base_url else "120"
        self.max_tokens = max(
            16,
            int(
                self._read_setting(
                    "MAX_TOKENS",
                    generic_env="MMSEC_OPENAI_COMPAT_MAX_TOKENS",
                    legacy_env="MMSEC_OPENAI_MAX_TOKENS",
                    default=default_max_tokens,
                )
                or default_max_tokens
            ),
        )
        self.max_concurrency = max(
            1,
            int(
                self._read_setting(
                    "CONCURRENCY",
                    generic_env="MMSEC_OPENAI_COMPAT_CONCURRENCY",
                    legacy_env="MMSEC_OPENAI_CONCURRENCY",
                    # A single local RTX 4090 can host the 7B-12B VLMs used by
                    # this project, but concurrent multimodal generations can
                    # push Gemma 3 and similar models into partial offload/meta
                    # tensor failures. Keep local scoring serial by default;
                    # multi-GPU deployments can raise this explicitly via env.
                    default="1",
                )
                or "1"
            ),
        )
        self._thread_local = local()
        self.session = self._build_session()

    # 读取 `setting`，并对缺失或异常输入做边界处理。
    def _read_setting(self, suffix: str, *, generic_env: str, legacy_env: str, default: str) -> str:
        if self.variant:
            variant_key = f"MMSEC_OPENAI_{self.variant}_{suffix}"
            variant_val = os.getenv(variant_key, "").strip()
            if variant_val:
                return variant_val
        return os.getenv(generic_env, os.getenv(legacy_env, default))

    # 构建 `session` 数据，集中整理模型适配层需要的输出结构。
    def _build_session(self) -> requests.Session:
        session = requests.Session()
        if self.loopback_base_url:
            # Local model servers are usually on 127.0.0.1. If requests inherits
            # system proxy settings here, calls may loop out to VPN/proxy by mistake.
            session.trust_env = False
        pool_size = max(8, int(self.max_concurrency) * 2)
        adapter = HTTPAdapter(pool_connections=pool_size, pool_maxsize=pool_size, max_retries=0)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    # 实现 `OpenAICompatAdapter._worker_session` 的对象行为，维护该类在模型适配层中的调用契约。
    def _worker_session(self) -> requests.Session:
        session = getattr(self._thread_local, "session", None)
        if session is None:
            session = self._build_session()
            self._thread_local.session = session
        return session

    # 实现 `OpenAICompatAdapter._headers` 的对象行为，维护该类在模型适配层中的调用契约。
    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    # 组装 `载荷`，把分散字段整理成后端任务或风险评分使用的载荷。
    def _payload(self, image: np.ndarray, text: str) -> dict[str, Any]:
        image_b64 = encode_image_b64(image)
        # We use chat/completions as a generic multimodal scoring interface:
        # send image + caption candidate, then force the model to answer with
        # compact JSON so the retrieval runner can sort pairs by one score.
        image_part = {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
        text_part = {
            "type": "text",
            "text": (
                f"{SYSTEM_PROMPT}\n\n"
                "Candidate text:\n"
                f"{text}\n\n"
                "Return JSON with score and reason only."
            ),
        }
        content = [text_part, image_part] if self.prompt_order == "text_first" else [image_part, text_part]
        return {
            "model": self.model_name,
            "temperature": 0,
            "max_tokens": int(self.max_tokens),
            "messages": [
                {"role": "user", "content": content},
            ],
        }

    # 组装 `生成式评测 载荷`，把分散字段整理成后端任务或风险评分使用的载荷。
    def _generation_payload(self, image: np.ndarray, prompt: str, *, max_tokens: int) -> dict[str, Any]:
        image_b64 = encode_image_b64(image)
        image_part = {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
        text_part = {"type": "text", "text": str(prompt)}
        content = [text_part, image_part] if self.prompt_order == "text_first" else [image_part, text_part]
        return {
            "model": self.model_name,
            "temperature": 0,
            "max_tokens": int(max(8, max_tokens)),
            "messages": [{"role": "user", "content": content}],
        }

    # 实现 `OpenAICompatAdapter._content_to_text` 的对象行为，维护该类在模型适配层中的调用契约。
    @staticmethod
    def _content_to_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    txt = part.get("text")
                    if txt is not None:
                        chunks.append(str(txt))
            return "\n".join(chunks).strip()
        return str(content or "")

    # 实现 `OpenAICompatAdapter._fallback_score_from_text` 的对象行为，维护该类在模型适配层中的调用契约。
    @staticmethod
    def _fallback_score_from_text(text: str) -> float:
        raw = str(text or "").strip()
        if not raw:
            return 0.0

        ratio = re.search(r"(?<!\d)(10(?:\.0+)?|[0-9](?:\.\d+)?)\s*/\s*10(?!\d)", raw)
        if ratio:
            return normalize_score(ratio.group(1), default=0.0)

        decimal = re.search(r"(?<![\d.])(?:0(?:\.\d+)?|1(?:\.0+)?)(?![\d.])", raw)
        if decimal:
            return normalize_score(decimal.group(0), default=0.0)

        lowered = raw.lower()
        negative_terms = ("not match", "does not match", "mismatch", "irrelevant", "unrelated", "incorrect", "no")
        positive_terms = ("perfect match", "strong match", "matches", "relevant", "correct", "yes")
        if any(term in lowered for term in negative_terms):
            return 0.1
        if any(term in lowered for term in positive_terms):
            return 0.8
        return 0.0

    # 实现 `OpenAICompatAdapter._request_pair` 的对象行为，维护该类在模型适配层中的调用契约。
    def _request_pair(self, image: np.ndarray, text: str) -> dict[str, Any]:
        resp = self._worker_session().post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=self._payload(image, text),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        choice = ((data.get("choices") or [{}])[0] or {}).get("message", {})
        content = self._content_to_text(choice.get("content", ""))
        parsed = extract_json_payload(content)
        score_value = parsed.get("score", parsed.get("similarity", None))
        score = (
            normalize_score(score_value, default=0.0)
            if score_value is not None
            else self._fallback_score_from_text(content)
        )
        reason = str(parsed.get("reason", parsed.get("text", content))).strip()
        return {
            "score": score,
            "reason": reason,
            "raw": data,
        }

    # 实现 `OpenAICompatAdapter._request_generation` 的对象行为，维护该类在模型适配层中的调用契约。
    def _request_generation(self, sample: Sample, prompt: str, *, max_tokens: int) -> ModelOutput:
        resp = self._worker_session().post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=self._generation_payload(sample.image, prompt, max_tokens=max_tokens),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        choice = ((data.get("choices") or [{}])[0] or {}).get("message", {})
        content = self._content_to_text(choice.get("content", "")).strip()
        return ModelOutput(
            text=content,
            score=0.0,
            raw={
                "adapter": self.adapter_name,
                "base_url": self.base_url,
                "model_name": self.model_name,
                "api_key_env": self.api_key_env,
                "prompt_order": self.prompt_order,
                "prompt": prompt,
                "response": data,
            },
        )

    # 计算 `pairs`，为指标、风险或调度决策提供数值依据。
    def score_pairs(self, pairs: list[tuple[np.ndarray, str]], batch_size: int = 1) -> np.ndarray:
        if not pairs:
            return np.zeros((0,), dtype=np.float32)
        # The OpenAI-compatible path is currently a cross-encoder style scorer,
        # so each image-text pair is scored independently and then used for ranking.
        worker_count = min(max(1, int(batch_size or 1)), int(self.max_concurrency), len(pairs))
        if worker_count <= 1:
            scores = [self._request_pair(image, text)["score"] for image, text in pairs]
            return np.asarray(scores, dtype=np.float32)

        scores = [0.0] * len(pairs)
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            futures = {
                pool.submit(self._request_pair, image, text): idx
                for idx, (image, text) in enumerate(pairs)
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                scores[idx] = float(fut.result()["score"])
        return np.asarray(scores, dtype=np.float32)

    # 实现 `OpenAICompatAdapter.predict` 的对象行为，维护该类在模型适配层中的调用契约。
    def predict(self, sample: Sample) -> ModelOutput:
        out = self._request_pair(sample.image, sample.text)
        score = float(out["score"])
        reason = str(out["reason"] or "").strip()
        return ModelOutput(
            text=reason or f"openai_compat score={score:.4f}",
            score=score,
            raw={
                "adapter": self.adapter_name,
                "base_url": self.base_url,
                "model_name": self.model_name,
                "api_key_env": self.api_key_env,
                "prompt_order": self.prompt_order,
                "response": out["raw"],
            },
        )

    # 生成 `answer`，补齐前端展示或后续评测需要的样本资产。
    def generate_answer(self, sample: Sample, question: str, *, prompt: str = "", max_tokens: int = 64) -> ModelOutput:
        template = str(prompt or "Answer the question about the image. Use a short answer.\nQuestion: {question}")
        rendered = template.format(question=str(question))
        return self._request_generation(sample, rendered, max_tokens=max_tokens)

    # 生成 `图像描述`，补齐前端展示或后续评测需要的样本资产。
    def generate_caption(self, sample: Sample, *, prompt: str = "", max_tokens: int = 96) -> ModelOutput:
        rendered = str(prompt or "Describe only the visible content of this image in one concise sentence.")
        return self._request_generation(sample, rendered, max_tokens=max_tokens)

    # 实现 `OpenAICompatAdapter.object_probe` 的对象行为，维护该类在模型适配层中的调用契约。
    def object_probe(self, sample: Sample, object_name: str, *, prompt: str = "", max_tokens: int = 8) -> ModelOutput:
        template = str(prompt or "Is there a {object_name} in the image? Answer yes or no.")
        rendered = template.format(object_name=str(object_name))
        return self._request_generation(sample, rendered, max_tokens=max_tokens)
