# 文件说明：该文件属于评测运行器，集中实现 report 相关逻辑。
from __future__ import annotations

from pathlib import Path
from typing import Any

from mmsec_eval.viz.render_report import render_report_html


# 写出 `报告`，保证后续报告、页面或复现实验能读取。
def write_report(run_dir: str, summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    out = Path(run_dir) / "report.html"
    html = render_report_html(summary=summary, rows=rows, run_dir=run_dir)
    out.write_text(html, encoding="utf-8")
    return str(out)

