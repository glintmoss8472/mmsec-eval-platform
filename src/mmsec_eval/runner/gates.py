# 文件说明：该文件属于评测运行器，集中实现 gates 相关逻辑。
from __future__ import annotations

from pathlib import Path


# 中文注释：实现 gate1_ok 的核心流程，支撑评测运行器中的业务语义和异常边界。
def gate1_ok(artifacts_dir: str) -> bool:
    return Path(artifacts_dir).exists()


# 中文注释：实现 gate2_ok 的核心流程，支撑评测运行器中的业务语义和异常边界。
def gate2_ok(run_dir: str) -> bool:
    p = Path(run_dir)
    return (p / "results.jsonl").exists() and (p / "summary.json").exists() and (p / "report.html").exists()


# 中文注释：实现 gate3_ok 的核心流程，支撑评测运行器中的业务语义和异常边界。
def gate3_ok(artifacts_dir: str) -> bool:
    p = Path(artifacts_dir) / "runs" / "run_index.jsonl"
    return p.exists()
