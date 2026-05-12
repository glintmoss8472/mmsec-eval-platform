from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Sample:
    sample_id: str
    image: np.ndarray
    text: str
    target_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelOutput:
    text: str
    score: float = 0.0
    embedding: np.ndarray | None = None
    text_embedding: np.ndarray | None = None
    attention: np.ndarray | None = None
    raw_logits: np.ndarray | None = None
    error_code: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class AttackTraceStep:
    step: int
    loss_total: float
    loss_parts: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AttackedSample:
    sample: Sample
    perturbation_l2: float
    perturbation_linf: float
    perturbation_l0: int = 0
    attack_trace: list[AttackTraceStep] = field(default_factory=list)
    artifact_refs: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class JudgeResult:
    success: bool
    reason: str
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalRecord:
    sample: Sample
    attacked: AttackedSample
    pred_clean: ModelOutput
    pred_adv: ModelOutput
    judge: JudgeResult | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    error_code: str = ""


@dataclass
class RunArtifacts:
    run_id: str
    run_dir: str
    results_path: str
    summary_path: str
    report_path: str
    run_index_path: str = ""
    benchmark_summary_path: str = ""


@dataclass
class AttackContext:
    config: Any
    model_adapter: Any
    surrogate_model_adapter: Any | None = None
    run_dir: str = ""
    sample_debug_dir: str = ""


@dataclass
class DefendedSample:
    sample: Sample
    metadata: dict[str, Any] = field(default_factory=dict)
    artifact_refs: dict[str, str] = field(default_factory=dict)


@dataclass
class DefenseContext:
    config: Any
    model_adapter: Any
    stage: str = ""
    run_dir: str = ""
    sample_debug_dir: str = ""
