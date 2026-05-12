# 文件说明：该文件属于项目工程，集中实现 jsonl io 相关逻辑。
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# 读取 `JSONL`，并对缺失或异常输入做边界处理。
def read_jsonl(path: str) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    rows: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


# 写出 `JSONL`，保证后续报告、页面或复现实验能读取。
def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

