# 文件说明：该文件属于评测运行器，集中实现 gates 相关逻辑。
from __future__ import annotations

from pathlib import Path


# 执行 `gate1 ok` 辅助逻辑，保持评测运行器中的输入处理和结果输出一致。
def gate1_ok(artifacts_dir: str) -> bool:
    return Path(artifacts_dir).exists()


# 执行 `gate2 ok` 辅助逻辑，保持评测运行器中的输入处理和结果输出一致。
def gate2_ok(run_dir: str) -> bool:
    p = Path(run_dir)
    return (p / "results.jsonl").exists() and (p / "summary.json").exists() and (p / "report.html").exists()


# 执行 `gate3 ok` 辅助逻辑，保持评测运行器中的输入处理和结果输出一致。
def gate3_ok(artifacts_dir: str) -> bool:
    p = Path(artifacts_dir) / "runs" / "run_index.jsonl"
    return p.exists()
