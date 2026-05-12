# 文件说明：该文件属于项目工程，集中实现 schema 相关逻辑。
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# 定义 `SampleAsset` 的不可变配置载体，集中保存后续计算需要的结构化字段。
@dataclass
class SampleAsset:
    sample_id: str
    text: str
    target_text: str
    metadata: dict[str, Any] = field(default_factory=dict)


# 定义 `AdversarialAsset` 的不可变配置载体，集中保存后续计算需要的结构化字段。
@dataclass
class AdversarialAsset:
    sample_id: str
    perturbation_l0: int
    perturbation_l2: float
    perturbation_linf: float
    metadata: dict[str, Any] = field(default_factory=dict)


# 定义 `AttackTrace` 的不可变配置载体，集中保存后续计算需要的结构化字段。
@dataclass
class AttackTrace:
    steps: list[dict[str, Any]] = field(default_factory=list)


# 定义 `CaseBundle` 的不可变配置载体，集中保存后续计算需要的结构化字段。
@dataclass
class CaseBundle:
    sample: SampleAsset
    adversarial: AdversarialAsset
    dataset_tag: str = ""
    model_tag: str = ""
    outputs: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    judge: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    artifact_refs: dict[str, str] = field(default_factory=dict)
