# 文件说明：该文件属于Streamlit 辅助界面，集中实现 Docs Ingest 相关逻辑。
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
INDEX = ROOT / "artifacts" / "docs_index.json"
SNIPPETS = ROOT / "artifacts" / "docs_snippets.jsonl"

st.title("Docs Ingest")
if INDEX.exists():
    data = json.loads(INDEX.read_text(encoding="utf-8"))
    st.write(f"Indexed: {len(data)} files")
    st.json(data)
else:
    st.info("No docs index found.")

if SNIPPETS.exists():
    st.subheader("Snippets")
    st.code("\n".join(SNIPPETS.read_text(encoding="utf-8").splitlines()[:20]))
else:
    st.info("No snippet index found.")

