from __future__ import annotations

import os
from typing import Any

import numpy as np
import requests

from mmsec_eval.model_adapters.remote_common import encode_image_b64, extract_json_payload, normalize_score
from mmsec_eval.plugins.base import ModelAdapter
from mmsec_eval.types import ModelOutput, Sample


PROMPT_TEXT = (
    "Score how well this image matches the candidate caption on a 0 to 1 scale. "
    'Return only JSON: {"score": 0.0, "reason": "short reason"}.'
)


class GeminiVisionAdapter(ModelAdapter):
    def __init__(self) -> None:
        self.base_url = os.getenv("MMSEC_GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
        self.model_name = os.getenv("MMSEC_GEMINI_MODEL_NAME", "gemini-2.5-pro").strip()
        self.timeout = float(os.getenv("MMSEC_GEMINI_TIMEOUT", "45"))
        self.api_key_env = os.getenv("MMSEC_GEMINI_API_KEY_ENV", "GEMINI_API_KEY").strip()
        self.api_key = os.getenv(self.api_key_env, "").strip()
        if not self.api_key:
            raise RuntimeError(f"GeminiVisionAdapter requires API key env: {self.api_key_env}")

    def _payload(self, image: np.ndarray, text: str) -> dict[str, Any]:
        image_b64 = encode_image_b64(image)
        return {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": f"{PROMPT_TEXT}\n\nCaption candidate:\n{text}"},
                        {"inline_data": {"mime_type": "image/png", "data": image_b64}},
                    ],
                }
            ],
            "generationConfig": {
                "temperature": 0,
                "responseMimeType": "application/json",
            },
        }

    def _generation_payload(self, image: np.ndarray, prompt: str, *, max_tokens: int) -> dict[str, Any]:
        image_b64 = encode_image_b64(image)
        return {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": str(prompt)},
                        {"inline_data": {"mime_type": "image/png", "data": image_b64}},
                    ],
                }
            ],
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": int(max(8, max_tokens)),
            },
        }

    def _request_pair(self, image: np.ndarray, text: str) -> dict[str, Any]:
        url = f"{self.base_url}/models/{self.model_name}:generateContent?key={self.api_key}"
        resp = requests.post(url, json=self._payload(image, text), timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        candidates = data.get("candidates") or []
        content = ""
        if candidates:
            parts = (((candidates[0] or {}).get("content") or {}).get("parts") or [])
            if parts:
                content = str((parts[0] or {}).get("text", "")).strip()
        parsed = extract_json_payload(content)
        score = normalize_score(parsed.get("score", parsed.get("similarity", 0.0)))
        reason = str(parsed.get("reason", parsed.get("text", content))).strip()
        return {
            "score": score,
            "reason": reason,
            "raw": data,
        }

    @staticmethod
    def _candidate_text(data: dict[str, Any]) -> str:
        candidates = data.get("candidates") or []
        if not candidates:
            return ""
        parts = (((candidates[0] or {}).get("content") or {}).get("parts") or [])
        chunks = []
        for part in parts:
            if isinstance(part, dict) and part.get("text") is not None:
                chunks.append(str(part.get("text", "")))
        return "\n".join(chunks).strip()

    def _request_generation(self, sample: Sample, prompt: str, *, max_tokens: int) -> ModelOutput:
        url = f"{self.base_url}/models/{self.model_name}:generateContent?key={self.api_key}"
        resp = requests.post(url, json=self._generation_payload(sample.image, prompt, max_tokens=max_tokens), timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return ModelOutput(
            text=self._candidate_text(data),
            score=0.0,
            raw={
                "adapter": "gemini_vision",
                "base_url": self.base_url,
                "model_name": self.model_name,
                "api_key_env": self.api_key_env,
                "prompt": prompt,
                "response": data,
            },
        )

    def score_pairs(self, pairs: list[tuple[np.ndarray, str]], batch_size: int = 1) -> np.ndarray:
        del batch_size
        if not pairs:
            return np.zeros((0,), dtype=np.float32)
        scores = [self._request_pair(image, text)["score"] for image, text in pairs]
        return np.asarray(scores, dtype=np.float32)

    def predict(self, sample: Sample) -> ModelOutput:
        out = self._request_pair(sample.image, sample.text)
        score = float(out["score"])
        reason = str(out["reason"] or "").strip()
        return ModelOutput(
            text=reason or f"gemini_vision score={score:.4f}",
            score=score,
            raw={
                "adapter": "gemini_vision",
                "base_url": self.base_url,
                "model_name": self.model_name,
                "api_key_env": self.api_key_env,
                "response": out["raw"],
            },
        )

    def generate_answer(self, sample: Sample, question: str, *, prompt: str = "", max_tokens: int = 64) -> ModelOutput:
        template = str(prompt or "Answer the question about the image. Use a short answer.\nQuestion: {question}")
        rendered = template.format(question=str(question))
        return self._request_generation(sample, rendered, max_tokens=max_tokens)

    def generate_caption(self, sample: Sample, *, prompt: str = "", max_tokens: int = 96) -> ModelOutput:
        rendered = str(prompt or "Describe only the visible content of this image in one concise sentence.")
        return self._request_generation(sample, rendered, max_tokens=max_tokens)

    def object_probe(self, sample: Sample, object_name: str, *, prompt: str = "", max_tokens: int = 8) -> ModelOutput:
        template = str(prompt or "Is there a {object_name} in the image? Answer yes or no.")
        rendered = template.format(object_name=str(object_name))
        return self._request_generation(sample, rendered, max_tokens=max_tokens)
