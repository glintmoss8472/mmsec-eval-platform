# 文件说明：该文件属于项目工程，集中实现 llm judge 相关逻辑。
from __future__ import annotations

import os

from mmsec_eval.plugins.base import Judge
from mmsec_eval.types import EvalRecord, JudgeResult


# 中文注释：定义 LLMJudge 的结构化职责，作为项目工程中状态、配置或行为的边界。
class LLMJudge(Judge):
    """Optional LLM scaffold.

    Default behavior is disabled unless env variables are configured.
    """

    # 中文注释：封装 LLMJudge.__init__ 的内部步骤，让项目工程主流程保持清晰并隔离边界细节。
    def __init__(self) -> None:
        self.enabled = os.getenv("MMSEC_LLM_JUDGE_ENABLED", "0") in {"1", "true", "True"}
        self.provider = os.getenv("MMSEC_LLM_PROVIDER", "none")
        self.endpoint = os.getenv("MMSEC_LLM_ENDPOINT", "")

    # 中文注释：实现 LLMJudge.judge 的核心行为，维护项目工程在该对象上的调用契约。
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

