# 文件说明：该文件属于项目工程，集中实现 base 相关逻辑。
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


# 中文注释：定义 ModelAdapter 的结构化职责，作为项目工程中状态、配置或行为的边界。
class ModelAdapter(ABC):
    # 中文注释：实现 ModelAdapter.predict 的核心行为，维护项目工程在该对象上的调用契约。
    @abstractmethod
    def predict(self, sample: Sample) -> ModelOutput:
        raise NotImplementedError

    # 中文注释：实现 ModelAdapter.generate_answer 的核心行为，维护项目工程在该对象上的调用契约。
    def generate_answer(self, sample: Sample, question: str, *, prompt: str = "", max_tokens: int = 64) -> ModelOutput:
        raise NotImplementedError(f"{self.__class__.__name__} does not support VQA generation")

    # 中文注释：实现 ModelAdapter.generate_caption 的核心行为，维护项目工程在该对象上的调用契约。
    def generate_caption(self, sample: Sample, *, prompt: str = "", max_tokens: int = 96) -> ModelOutput:
        raise NotImplementedError(f"{self.__class__.__name__} does not support image caption generation")

    # 中文注释：实现 ModelAdapter.object_probe 的核心行为，维护项目工程在该对象上的调用契约。
    def object_probe(self, sample: Sample, object_name: str, *, prompt: str = "", max_tokens: int = 8) -> ModelOutput:
        question = (prompt or "Is there a {object_name} in the image? Answer yes or no.").format(object_name=object_name)
        return self.generate_answer(sample, question, prompt=question, max_tokens=max_tokens)


# 中文注释：定义 AttackPlugin 的结构化职责，作为项目工程中状态、配置或行为的边界。
class AttackPlugin(ABC):
    # 中文注释：实现 AttackPlugin.attack 的核心行为，维护项目工程在该对象上的调用契约。
    @abstractmethod
    def attack(self, sample: Sample, ctx: AttackContext) -> AttackedSample:
        raise NotImplementedError


# 中文注释：定义 DefensePlugin 的结构化职责，作为项目工程中状态、配置或行为的边界。
class DefensePlugin(ABC):
    # 中文注释：实现 DefensePlugin.defend 的核心行为，维护项目工程在该对象上的调用契约。
    @abstractmethod
    def defend(self, sample: Sample, ctx: DefenseContext) -> DefendedSample:
        raise NotImplementedError


# 中文注释：定义 MetricPlugin 的结构化职责，作为项目工程中状态、配置或行为的边界。
class MetricPlugin(ABC):
    # 中文注释：实现 MetricPlugin.compute 的核心行为，维护项目工程在该对象上的调用契约。
    @abstractmethod
    def compute(self, record: EvalRecord) -> dict[str, float]:
        raise NotImplementedError


# 中文注释：定义 Judge 的结构化职责，作为项目工程中状态、配置或行为的边界。
class Judge(ABC):
    # 中文注释：实现 Judge.judge 的核心行为，维护项目工程在该对象上的调用契约。
    @abstractmethod
    def judge(self, record: EvalRecord) -> JudgeResult:
        raise NotImplementedError
