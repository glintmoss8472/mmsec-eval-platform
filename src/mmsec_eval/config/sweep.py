# 文件说明：该文件属于配置系统，集中实现 sweep 相关逻辑。
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from mmsec_eval.config.loader import _deep_merge
from mmsec_eval.config.schema import AppConfig


# 应用 `override` 规则，把兼容字段写回报告或风险载荷。
def apply_override(base: AppConfig, override: dict[str, Any]) -> AppConfig:
    from mmsec_eval.config.loader import _to_config

    merged = _deep_merge(asdict(base), override)
    return _to_config(merged)

