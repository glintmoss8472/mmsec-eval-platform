import os

import numpy as np

from mmsec_eval.judges.llm_judge import LLMJudge
from mmsec_eval.types import AttackedSample, EvalRecord, ModelOutput, Sample


def test_llm_judge_disabled(monkeypatch):
    monkeypatch.setenv("MMSEC_LLM_JUDGE_ENABLED", "0")
    j = LLMJudge()
    s = Sample("1", np.zeros((8, 8, 3), dtype=np.float32), "a")
    rec = EvalRecord(s, AttackedSample(s, 0.0, 0.0), ModelOutput("a"), ModelOutput("b"))
    out = j.judge(rec)
    assert out.raw.get("skipped") is True

