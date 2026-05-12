# 文件说明：该文件属于Streamlit 辅助界面，集中实现 Browse Artifacts 相关逻辑。
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "artifacts" / "runs"

st.title("Browse Artifacts")

if not RUNS.exists():
    st.info("No runs found.")
else:
    dirs = sorted([p for p in RUNS.iterdir() if p.is_dir()], reverse=True)
    selected = st.selectbox("Run", [p.name for p in dirs])
    run_dir = RUNS / selected
    summary_path = run_dir / "summary.json"
    results_path = run_dir / "results.jsonl"
    report_path = run_dir / "report.html"
    cases_index = run_dir / "cases_index.jsonl"

    if summary_path.exists():
        st.subheader("Summary")
        st.json(json.loads(summary_path.read_text(encoding="utf-8")))
    if results_path.exists():
        st.subheader("Results (head)")
        rows = results_path.read_text(encoding="utf-8").splitlines()[:10]
        st.code("\n".join(rows))

    if cases_index.exists():
        st.subheader("Cases")
        case_rows = cases_index.read_text(encoding="utf-8").splitlines()[:20]
        st.code("\n".join(case_rows))

    if report_path.exists():
        st.subheader("Report")
        st.markdown(f"[Open report]({report_path.as_uri()})")
