from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SampleAsset:
    sample_id: str
    text: str
    target_text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AdversarialAsset:
    sample_id: str
    perturbation_l0: int
    perturbation_l2: float
    perturbation_linf: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AttackTrace:
    steps: list[dict[str, Any]] = field(default_factory=list)


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
