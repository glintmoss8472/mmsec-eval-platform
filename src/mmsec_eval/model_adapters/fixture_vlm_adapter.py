# 文件说明：该文件属于模型适配层，集中实现 fixture vlm adapter 相关逻辑。
from __future__ import annotations

import re
from typing import Any

import numpy as np

from mmsec_eval.plugins.base import ModelAdapter
from mmsec_eval.types import ModelOutput, Sample


# 中文注释：封装 _norm 的内部步骤，让模型适配层主流程保持清晰并隔离边界细节。
def _norm(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


# 中文注释：封装 _first_text 的内部步骤，让模型适配层主流程保持清晰并隔离边界细节。
def _first_text(*values: object, default: str = "") -> str:
    for value in values:
        if isinstance(value, list):
            for item in value:
                text = str(item or "").strip()
                if text:
                    return text
        else:
            text = str(value or "").strip()
            if text:
                return text
    return default


# 中文注释：封装 _stage 的内部步骤，让模型适配层主流程保持清晰并隔离边界细节。
def _stage(sample: Sample) -> str:
    text = _norm(sample.metadata.get("generation_stage"))
    if text in {"clean", "attacked", "defended"}:
        return text
    return "clean"


# 中文注释：封装 _contains_object 的内部步骤，让模型适配层主流程保持清晰并隔离边界细节。
def _contains_object(text: str, object_name: str, aliases: list[str]) -> bool:
    candidates = [object_name, *aliases]
    haystack = f" {_norm(text)} "
    for candidate in candidates:
        token = _norm(candidate)
        if token and re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", haystack):
            return True
    return False


# 中文注释：定义 FixtureVLMAdapter 的结构化职责，作为模型适配层中状态、配置或行为的边界。
class FixtureVLMAdapter(ModelAdapter):
    """Deterministic generation adapter for CI and offline UI smoke tests.

    It is intentionally labeled as a fixture, not a scientific victim model.
    Real VQA/caption runs should use OpenAI-compatible, Gemini, or local VLM adapters.
    """

    adapter_name = "fixture_vlm"

    # 中文注释：实现 FixtureVLMAdapter.predict 的核心行为，维护模型适配层在该对象上的调用契约。
    def predict(self, sample: Sample) -> ModelOutput:
        text = str(sample.text or "")
        answer = self.generate_answer(sample, text).text
        return ModelOutput(text=answer, score=1.0, raw={"adapter": self.adapter_name, "task": "predict_fixture"})

    # 中文注释：实现 FixtureVLMAdapter.generate_answer 的核心行为，维护模型适配层在该对象上的调用契约。
    def generate_answer(self, sample: Sample, question: str, *, prompt: str = "", max_tokens: int = 64) -> ModelOutput:
        del prompt, max_tokens
        meta = dict(sample.metadata)
        stage = _stage(sample)
        if stage == "attacked":
            answer = _first_text(meta.get("attacked_answer"), meta.get("target_wrong_answer"), meta.get("wrong_answers"), default="")
        elif stage == "defended":
            answer = _first_text(meta.get("defended_answer"), meta.get("answer"), meta.get("answer_aliases"), meta.get("acceptable_answers"), default="")
        else:
            answer = _first_text(meta.get("clean_answer"), meta.get("answer"), meta.get("answer_aliases"), meta.get("acceptable_answers"), default="")
        if not answer:
            answer = "yes" if "yes" in _norm(question) else "unknown"
        return ModelOutput(
            text=str(answer),
            score=1.0,
            raw={
                "adapter": self.adapter_name,
                "task": "vqa",
                "stage": stage,
                "question": str(question),
                "fixture": True,
            },
        )

    # 中文注释：实现 FixtureVLMAdapter.generate_caption 的核心行为，维护模型适配层在该对象上的调用契约。
    def generate_caption(self, sample: Sample, *, prompt: str = "", max_tokens: int = 96) -> ModelOutput:
        del prompt, max_tokens
        meta = dict(sample.metadata)
        stage = _stage(sample)
        if stage == "attacked":
            caption = _first_text(meta.get("attacked_caption"), default="")
        elif stage == "defended":
            caption = _first_text(meta.get("defended_caption"), meta.get("clean_caption"), meta.get("reference_captions"), default="")
        else:
            caption = _first_text(meta.get("clean_caption"), meta.get("reference_captions"), default="")
        if not caption:
            target = _first_text(meta.get("target_object"), default="object")
            caption = f"A photo containing {target}."
        return ModelOutput(
            text=str(caption),
            score=1.0,
            raw={"adapter": self.adapter_name, "task": "caption", "stage": stage, "fixture": True},
        )

    # 中文注释：实现 FixtureVLMAdapter.object_probe 的核心行为，维护模型适配层在该对象上的调用契约。
    def object_probe(self, sample: Sample, object_name: str, *, prompt: str = "", max_tokens: int = 8) -> ModelOutput:
        del prompt, max_tokens
        meta = dict(sample.metadata)
        stage = _stage(sample)
        target = _norm(meta.get("target_object"))
        added = _norm(meta.get("added_object"))
        attack_goal = _norm(meta.get("attack_goal") or meta.get("goal"))
        aliases = [str(x) for x in list(meta.get("target_aliases") or [])]
        object_norm = _norm(object_name)

        if stage == "attacked" and object_norm == target and attack_goal in {"remove_object", "remove", "hide_object"}:
            present = False
        elif stage == "attacked" and object_norm == added and attack_goal in {"add_object", "add"}:
            present = True
        elif stage == "defended" and object_norm == target and attack_goal in {"remove_object", "remove", "hide_object"}:
            present = True
        else:
            caption = self.generate_caption(sample).text
            present = _contains_object(caption, object_norm, aliases)

        answer = "yes" if present else "no"
        return ModelOutput(
            text=answer,
            score=1.0 if present else 0.0,
            raw={
                "adapter": self.adapter_name,
                "task": "object_probe",
                "stage": stage,
                "object_name": object_name,
                "present": present,
                "fixture": True,
            },
        )

    # 中文注释：实现 FixtureVLMAdapter.score_pairs 的核心行为，维护模型适配层在该对象上的调用契约。
    def score_pairs(self, pairs: list[tuple[np.ndarray, str]], batch_size: int = 1) -> np.ndarray:
        del batch_size
        return np.ones((len(pairs),), dtype=np.float32)
