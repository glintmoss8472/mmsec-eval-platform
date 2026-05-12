from __future__ import annotations

import os

from mmsec_eval.plugins.base import Judge
from mmsec_eval.types import EvalRecord, JudgeResult


class LLMJudge(Judge):
    """Optional LLM scaffold.

    Default behavior is disabled unless env variables are configured.
    """

    def __init__(self) -> None:
        self.enabled = os.getenv("MMSEC_LLM_JUDGE_ENABLED", "0") in {"1", "true", "True"}
        self.provider = os.getenv("MMSEC_LLM_PROVIDER", "none")
        self.endpoint = os.getenv("MMSEC_LLM_ENDPOINT", "")

    def judge(self, record: EvalRecord) -> JudgeResult:
        if not self.enabled:
            return JudgeResult(success=False, reason="llm_judge_disabled", raw={"skipped": True})
        # Skeleton only: keep behavior deterministic and non-breaking.
        target = (record.sample.target_text or "").lower()
        adv = record.pred_adv.text.lower()
        success = bool(target and target in adv)
        return JudgeResult(
            success=success,
            reason="llm_scaffold_rule_proxy",
            raw={"provider": self.provider, "endpoint": self.endpoint},
        )

