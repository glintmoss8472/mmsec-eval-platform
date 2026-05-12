from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

st.title("Run MVP")
cfg = st.text_input("Config path", value="configs/mvp.yaml")

if st.button("Run ingest-docs"):
    p = subprocess.run(
        [sys.executable, "-m", "mmsec_eval", "ingest-docs", "--config", cfg],
        capture_output=True,
        text=True,
    )
    st.code((p.stdout or "") + "\n" + (p.stderr or ""))
    st.write("Return code:", p.returncode)

if st.button("Run eval"):
    p = subprocess.run(
        [sys.executable, "-m", "mmsec_eval", "run-eval", "--config", cfg],
        capture_output=True,
        text=True,
    )
    st.code((p.stdout or "") + "\n" + (p.stderr or ""))
    st.write("Return code:", p.returncode)

