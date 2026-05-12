# 文件说明：该文件属于自动化测试，集中实现 test llm judge skip 相关逻辑。
import os

import numpy as np

from mmsec_eval.judges.llm_judge import LLMJudge
from mmsec_eval.types import AttackedSample, EvalRecord, ModelOutput, Sample


# 中文注释：验证 test_llm_judge_disabled 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_llm_judge_disabled(monkeypatch):
    monkeypatch.setenv("MMSEC_LLM_JUDGE_ENABLED", "0")
    j = LLMJudge()
    s = Sample("1", np.zeros((8, 8, 3), dtype=np.float32), "a")
    rec = EvalRecord(s, AttackedSample(s, 0.0, 0.0), ModelOutput("a"), ModelOutput("b"))
    out = j.judge(rec)
    assert out.raw.get("skipped") is True

