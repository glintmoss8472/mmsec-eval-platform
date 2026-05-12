from __future__ import annotations

from abc import ABC, abstractmethod

from mmsec_eval.types import (
    AttackContext,
    AttackedSample,
    DefenseContext,
    DefendedSample,
    EvalRecord,
    JudgeResult,
    ModelOutput,
    Sample,
)


class ModelAdapter(ABC):
    @abstractmethod
    def predict(self, sample: Sample) -> ModelOutput:
        raise NotImplementedError

    def generate_answer(self, sample: Sample, question: str, *, prompt: str = "", max_tokens: int = 64) -> ModelOutput:
        raise NotImplementedError(f"{self.__class__.__name__} does not support VQA generation")

    def generate_caption(self, sample: Sample, *, prompt: str = "", max_tokens: int = 96) -> ModelOutput:
        raise NotImplementedError(f"{self.__class__.__name__} does not support image caption generation")

    def object_probe(self, sample: Sample, object_name: str, *, prompt: str = "", max_tokens: int = 8) -> ModelOutput:
        question = (prompt or "Is there a {object_name} in the image? Answer yes or no.").format(object_name=object_name)
        return self.generate_answer(sample, question, prompt=question, max_tokens=max_tokens)


class AttackPlugin(ABC):
    @abstractmethod
    def attack(self, sample: Sample, ctx: AttackContext) -> AttackedSample:
        raise NotImplementedError


class DefensePlugin(ABC):
    @abstractmethod
    def defend(self, sample: Sample, ctx: DefenseContext) -> DefendedSample:
        raise NotImplementedError


class MetricPlugin(ABC):
    @abstractmethod
    def compute(self, record: EvalRecord) -> dict[str, float]:
        raise NotImplementedError


class Judge(ABC):
    @abstractmethod
    def judge(self, record: EvalRecord) -> JudgeResult:
        raise NotImplementedError
