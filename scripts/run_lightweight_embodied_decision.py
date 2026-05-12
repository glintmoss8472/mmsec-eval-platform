# 文件说明：该文件属于运维与实验脚本，集中实现 run lightweight embodied decision 相关逻辑。
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mmsec_eval.embodied.decision_benchmark import evaluate_decision_cases, load_decision_cases, summarize_decision_cases


# 中文注释：实现 parse_args 的核心流程，支撑运维与实验脚本中的业务语义和异常边界。
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a lightweight embodied decision benchmark from JSONL cases.")
    parser.add_argument("--cases-jsonl", default="seed/data/embodied_decision/cases.jsonl")
    parser.add_argument("--out-dir", default="artifacts/embodied_decision")
    return parser.parse_args()


# 中文注释：串联 main 的主流程，集中处理运维与实验脚本的初始化、执行和退出条件。
def main() -> int:
    args = parse_args()
    rows = load_decision_cases(args.cases_jsonl)
    results = evaluate_decision_cases(rows)
    summary = summarize_decision_cases(results)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "decision_results.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in results) + ("\n" if results else ""),
        encoding="utf-8",
    )
    (out_dir / "decision_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": True, "summary": summary, "out_dir": str(out_dir)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
