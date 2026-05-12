# 文件说明：该文件属于Streamlit 辅助界面，集中实现 streamlit app 相关逻辑。
from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

st.set_page_config(page_title="mmsec-eval", layout="wide")
st.title("mmsec-eval-platform")
st.caption("Multimodal adversarial security evaluation")

docs_index = ROOT / "artifacts" / "docs_index.json"
runs_dir = ROOT / "artifacts" / "runs"

col1, col2 = st.columns(2)
with col1:
    st.subheader("Docs Index")
    if docs_index.exists():
        data = json.loads(docs_index.read_text(encoding="utf-8"))
        st.metric("Indexed files", len(data))
        st.json(data[:3] if isinstance(data, list) else data)
    else:
        st.info("No docs index yet. Run ingest-docs.")

with col2:
    st.subheader("Runs")
    if runs_dir.exists():
        runs = [p.name for p in runs_dir.iterdir() if p.is_dir()]
        st.metric("Run count", len(runs))
        st.write(runs[:10])

        case_count = 0
        for r in runs[:20]:
            idx = runs_dir / r / "cases_index.jsonl"
            if idx.exists():
                case_count += len(idx.read_text(encoding="utf-8").splitlines())
        st.metric("Indexed cases (recent)", case_count)
    else:
        st.info("No runs yet. Run run-eval.")
