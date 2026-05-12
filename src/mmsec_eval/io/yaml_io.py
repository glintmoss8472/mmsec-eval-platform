# 文件说明：该文件属于项目工程，集中实现 yaml io 相关逻辑。
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


# 中文注释：实现 read_yaml 的核心流程，支撑项目工程中的业务语义和异常边界。
def read_yaml(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)  # required by design
    return data or {}


# 中文注释：实现 write_yaml 的核心流程，支撑项目工程中的业务语义和异常边界。
def write_yaml(path: str, data: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

