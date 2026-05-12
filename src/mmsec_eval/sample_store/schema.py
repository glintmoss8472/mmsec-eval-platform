# 文件说明：该文件属于项目工程，集中实现 schema 相关逻辑。
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# 中文注释：定义 SampleAsset 的结构化职责，作为项目工程中状态、配置或行为的边界。
@dataclass
class SampleAsset:
    sample_id: str
    text: str
    target_text: str
    metadata: dict[str, Any] = field(default_factory=dict)


# 中文注释：定义 AdversarialAsset 的结构化职责，作为项目工程中状态、配置或行为的边界。
@dataclass
class AdversarialAsset:
    sample_id: str
    perturbation_l0: int
    perturbation_l2: float
    perturbation_linf: float
    metadata: dict[str, Any] = field(default_factory=dict)


# 中文注释：定义 AttackTrace 的结构化职责，作为项目工程中状态、配置或行为的边界。
@dataclass
class AttackTrace:
    steps: list[dict[str, Any]] = field(default_factory=list)


# 中文注释：定义 CaseBundle 的结构化职责，作为项目工程中状态、配置或行为的边界。
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
