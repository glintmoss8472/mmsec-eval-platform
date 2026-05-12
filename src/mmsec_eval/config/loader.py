# 文件说明：该文件属于配置系统，集中实现 loader 相关逻辑。
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from mmsec_eval.config.schema import (
    AppConfig,
    AttackConfig,
    BootstrapConfig,
    DatasetConfig,
    DefenseConfig,
    DocsConfig,
    JudgeConfig,
    ModelConfig,
    PluginsConfig,
    RiskConfig,
    ReportConfig,
    RunnerConfig,
    RuntimeConfig,
    SampleStoreConfig,
    SweepConfig,
    TaskConfig,
)
from mmsec_eval.io.yaml_io import read_yaml


KNOWN_TOP_LEVEL = {
    "seed",
    "device_preference",
    "artifacts_dir",
    "docs",
    "plugins",
    "runtime",
    "model",
    "dataset",
    "task",
    "attack",
    "defense",
    "risk",
    "runner",
    "report",
    "sample_store",
    "judge",
    "sweep",
    "bootstrap",
    "extra",
}


# 中文注释：封装 _deep_merge 的内部步骤，让配置系统主流程保持清晰并隔离边界细节。
def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


# 中文注释：封装 _migrate_legacy_runtime_fields 的内部步骤，让配置系统主流程保持清晰并隔离边界细节。
def _migrate_legacy_runtime_fields(raw: dict[str, Any]) -> dict[str, Any]:
    """Keep API-submitted legacy overrides compatible with the typed config.

    Older frontends placed retrieval pair limits under ``task``. The runner is
    the real owner of those execution limits, so migrate them before dataclass
    construction instead of failing with a low-level ``unexpected keyword``
    error.
    """

    migrated = dict(raw)
    task = dict(migrated.get("task", {}) or {})
    runner = dict(migrated.get("runner", {}) or {})
    report = dict(migrated.get("report", {}) or {})
    extra = dict(migrated.get("extra", {}) or {})
    for legacy_key in ("max_pairs", "max_samples"):
        if legacy_key in task:
            legacy_value = task.pop(legacy_key)
            if legacy_key not in runner or runner.get(legacy_key) in {None, "", 0}:
                runner[legacy_key] = legacy_value
    for legacy_key in ("task_name", "note"):
        if legacy_key in report:
            extra.setdefault(f"report_{legacy_key}", report.pop(legacy_key))
    migrated["task"] = task
    migrated["runner"] = runner
    migrated["report"] = report
    migrated["extra"] = extra
    return migrated


# 中文注释：封装 _to_config 的内部步骤，让配置系统主流程保持清晰并隔离边界细节。
def _to_config(raw: dict[str, Any]) -> AppConfig:
    raw = _migrate_legacy_runtime_fields(raw)
    extra_payload = dict(raw.get("extra", {}) or {})
    for key, value in raw.items():
        if key in KNOWN_TOP_LEVEL:
            continue
        extra_payload[key] = value

    return AppConfig(
        seed=int(raw.get("seed", 42)),
        device_preference=str(raw.get("device_preference", "cuda")),
        artifacts_dir=str(raw.get("artifacts_dir", "artifacts")),
        docs=DocsConfig(**raw.get("docs", {})),
        plugins=PluginsConfig(**raw.get("plugins", {})),
        runtime=RuntimeConfig(**raw.get("runtime", {})),
        model=ModelConfig(**raw.get("model", {})),
        dataset=DatasetConfig(**raw.get("dataset", {})),
        task=TaskConfig(**raw.get("task", {})),
        attack=AttackConfig(**raw.get("attack", {})),
        defense=DefenseConfig(**raw.get("defense", {})),
        risk=RiskConfig(**raw.get("risk", {})),
        runner=RunnerConfig(**raw.get("runner", {})),
        report=ReportConfig(**raw.get("report", {})),
        sample_store=SampleStoreConfig(**raw.get("sample_store", {})),
        judge=JudgeConfig(**raw.get("judge", {})),
        sweep=SweepConfig(**raw.get("sweep", {})),
        bootstrap=BootstrapConfig(**raw.get("bootstrap", {})),
        extra=extra_payload,
    )


# 中文注释：实现 load_config 的核心流程，支撑配置系统中的业务语义和异常边界。
def load_config(path: str) -> AppConfig:
    default_path = Path("configs/default.yaml")
    base = read_yaml(str(default_path)) if default_path.exists() else asdict(AppConfig())
    user = read_yaml(path)
    merged = _deep_merge(base, user)
    return _to_config(merged)
