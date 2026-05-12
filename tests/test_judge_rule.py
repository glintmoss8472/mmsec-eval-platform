# 文件说明：该文件属于自动化测试，集中实现 test judge rule 相关逻辑。
import numpy as np

from mmsec_eval.judges.rule_judge import RuleJudge
from mmsec_eval.types import AttackedSample, EvalRecord, ModelOutput, Sample


# 验证 `rule judge` 场景，防止相关行为在后续修改中退化。
def test_rule_judge():
    judge = RuleJudge()
    s = Sample("1", np.zeros((16, 16, 3), dtype=np.float32), "a red circle", target_text="square")
    a = Sample("1", np.zeros((16, 16, 3), dtype=np.float32), "a red circle", target_text="square")
    rec = EvalRecord(s, AttackedSample(a, 0, 0), ModelOutput("detected object: circle"), ModelOutput("detected object: square"))
    res = judge.judge(rec)
    assert res.success is True

