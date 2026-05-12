from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mmsec_eval.interaction.vqa_dialogue_benchmark import (
    evaluate_interaction_cases,
    load_interaction_cases,
    summarize_interaction_cases,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate lightweight VQA and multimodal dialogue cases from JSONL records.")
    parser.add_argument("--cases-jsonl", default="seed/data/vqa_dialogue/cases.jsonl")
    parser.add_argument("--out-dir", default="artifacts/vqa_dialogue")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_interaction_cases(args.cases_jsonl)
    results = evaluate_interaction_cases(rows)
    summary = summarize_interaction_cases(results)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "interaction_results.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in results) + ("\n" if results else ""),
        encoding="utf-8",
    )
    (out_dir / "interaction_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": True, "summary": summary, "out_dir": str(out_dir)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
