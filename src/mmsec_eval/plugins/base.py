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


# 实现 `ModelAdapter.predict` 的对象行为，维护该类在项目工程中的调用契约。
class ModelAdapter(ABC):
    # 实现 ModelAdapter.predict 的核心行为，维护项目工程在该对象上的调用契约。
    @abstractmethod
    def predict(self, sample: Sample) -> ModelOutput:
        raise NotImplementedError

    # 生成 `answer`，补齐前端展示或后续评测需要的样本资产。
    def generate_answer(self, sample: Sample, question: str, *, prompt: str = "", max_tokens: int = 64) -> ModelOutput:
        raise NotImplementedError(f"{self.__class__.__name__} does not support VQA generation")

    # 生成 `图像描述`，补齐前端展示或后续评测需要的样本资产。
    def generate_caption(self, sample: Sample, *, prompt: str = "", max_tokens: int = 96) -> ModelOutput:
        raise NotImplementedError(f"{self.__class__.__name__} does not support image caption generation")

    # 实现 `ModelAdapter.object_probe` 的对象行为，维护该类在项目工程中的调用契约。
    def object_probe(self, sample: Sample, object_name: str, *, prompt: str = "", max_tokens: int = 8) -> ModelOutput:
        question = (prompt or "Is there a {object_name} in the image? Answer yes or no.").format(object_name=object_name)
        return self.generate_answer(sample, question, prompt=question, max_tokens=max_tokens)


# 推断 `攻击`，从样本、配置或运行记录中提取统一名称。
class AttackPlugin(ABC):
    # 实现 AttackPlugin.attack 的核心行为，维护项目工程在该对象上的调用契约。
    @abstractmethod
    def attack(self, sample: Sample, ctx: AttackContext) -> AttackedSample:
        raise NotImplementedError


# 实现 `DefensePlugin.defend` 的对象行为，维护该类在项目工程中的调用契约。
class DefensePlugin(ABC):
    # 实现 DefensePlugin.defend 的核心行为，维护项目工程在该对象上的调用契约。
    @abstractmethod
    def defend(self, sample: Sample, ctx: DefenseContext) -> DefendedSample:
        raise NotImplementedError


# 实现 `MetricPlugin.compute` 的对象行为，维护该类在项目工程中的调用契约。
class MetricPlugin(ABC):
    # 实现 MetricPlugin.compute 的核心行为，维护项目工程在该对象上的调用契约。
    @abstractmethod
    def compute(self, record: EvalRecord) -> dict[str, float]:
        raise NotImplementedError


# 实现 `Judge.judge` 的对象行为，维护该类在项目工程中的调用契约。
class Judge(ABC):
    # 实现 Judge.judge 的核心行为，维护项目工程在该对象上的调用契约。
    @abstractmethod
    def judge(self, record: EvalRecord) -> JudgeResult:
        raise NotImplementedError
