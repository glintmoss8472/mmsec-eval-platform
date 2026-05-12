from __future__ import annotations

from dataclasses import asdict
from typing import Any

from mmsec_eval.config.loader import _deep_merge
from mmsec_eval.config.schema import AppConfig


def apply_override(base: AppConfig, override: dict[str, Any]) -> AppConfig:
    from mmsec_eval.config.loader import _to_config

    merged = _deep_merge(asdict(base), override)
    return _to_config(merged)

