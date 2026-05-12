# 文件说明：该文件属于配置系统，集中实现 sweep 相关逻辑。
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from mmsec_eval.config.loader import _deep_merge
from mmsec_eval.config.schema import AppConfig


# 中文注释：实现 apply_override 的核心流程，支撑配置系统中的业务语义和异常边界。
def apply_override(base: AppConfig, override: dict[str, Any]) -> AppConfig:
    from mmsec_eval.config.loader import _to_config

    merged = _deep_merge(asdict(base), override)
    return _to_config(merged)

