# 文件说明：该文件属于评测运行器，集中实现 report 相关逻辑。
from __future__ import annotations

from pathlib import Path
from typing import Any

from mmsec_eval.viz.render_report import render_report_html


# 中文注释：实现 write_report 的核心流程，支撑评测运行器中的业务语义和异常边界。
def write_report(run_dir: str, summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    out = Path(run_dir) / "report.html"
    html = render_report_html(summary=summary, rows=rows, run_dir=run_dir)
    out.write_text(html, encoding="utf-8")
    return str(out)

