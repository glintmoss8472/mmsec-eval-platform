# 文件说明：该文件属于项目工程，集中实现 yaml io 相关逻辑。
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


# 读取 `YAML`，并对缺失或异常输入做边界处理。
def read_yaml(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)  # required by design
    return data or {}


# 写出 `YAML`，保证后续报告、页面或复现实验能读取。
def write_yaml(path: str, data: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

