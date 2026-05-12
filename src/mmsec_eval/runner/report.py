from __future__ import annotations

from pathlib import Path
from typing import Any

from mmsec_eval.viz.render_report import render_report_html


def write_report(run_dir: str, summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    out = Path(run_dir) / "report.html"
    html = render_report_html(summary=summary, rows=rows, run_dir=run_dir)
    out.write_text(html, encoding="utf-8")
    return str(out)

