from __future__ import annotations

from pathlib import Path


def gate1_ok(artifacts_dir: str) -> bool:
    return Path(artifacts_dir).exists()


def gate2_ok(run_dir: str) -> bool:
    p = Path(run_dir)
    return (p / "results.jsonl").exists() and (p / "summary.json").exists() and (p / "report.html").exists()


def gate3_ok(artifacts_dir: str) -> bool:
    p = Path(artifacts_dir) / "runs" / "run_index.jsonl"
    return p.exists()
