# 文件说明：该文件属于模型适配层，集中实现 fixture vlm adapter 相关逻辑。
from __future__ import annotations

import re
from typing import Any

import numpy as np

from mmsec_eval.plugins.base import ModelAdapter
from mmsec_eval.types import ModelOutput, Sample


# 执行 `norm` 辅助逻辑，保持模型适配层中的输入处理和结果输出一致。
def _norm(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


# 查找第一个可用的 `文本`，用于页面预览或结果补全。
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


# 执行 `stage` 辅助逻辑，保持模型适配层中的输入处理和结果输出一致。
def _stage(sample: Sample) -> str:
    text = _norm(sample.metadata.get("generation_stage"))
    if text in {"clean", "attacked", "defended"}:
        return text
    return "clean"


# 执行 `contains object` 辅助逻辑，保持模型适配层中的输入处理和结果输出一致。
def _contains_object(text: str, object_name: str, aliases: list[str]) -> bool:
    candidates = [object_name, *aliases]
    haystack = f" {_norm(text)} "
    for candidate in candidates:
        token = _norm(candidate)
        if token and re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", haystack):
            return True
    return False


# 定义 `FixtureVLMAdapter` 的插件适配边界，把模型、攻击或评测能力暴露为统一接口。
class FixtureVLMAdapter(ModelAdapter):
    """Deterministic generation adapter for CI and offline UI smoke tests.

    It is intentionally labeled as a fixture, not a scientific victim model.
    Real VQA/caption runs should use OpenAI-compatible, Gemini, or local VLM adapters.
    """

    adapter_name = "fixture_vlm"

    # 实现 `FixtureVLMAdapter.predict` 的对象行为，维护该类在模型适配层中的调用契约。
    def predict(self, sample: Sample) -> ModelOutput:
        text = str(sample.text or "")
        answer = self.generate_answer(sample, text).text
        return ModelOutput(text=answer, score=1.0, raw={"adapter": self.adapter_name, "task": "predict_fixture"})

    # 生成 `answer`，补齐前端展示或后续评测需要的样本资产。
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

    # 生成 `图像描述`，补齐前端展示或后续评测需要的样本资产。
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

    # 实现 `FixtureVLMAdapter.object_probe` 的对象行为，维护该类在模型适配层中的调用契约。
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

    # 计算 `pairs`，为指标、风险或调度决策提供数值依据。
    def score_pairs(self, pairs: list[tuple[np.ndarray, str]], batch_size: int = 1) -> np.ndarray:
        del batch_size
        return np.ones((len(pairs),), dtype=np.float32)
