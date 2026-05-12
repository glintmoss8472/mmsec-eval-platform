# 文件说明：该文件属于自动化测试，集中实现 test metrics basic 相关逻辑。
import numpy as np

from mmsec_eval.metrics.basic_metrics import BasicMetrics
from mmsec_eval.types import AttackedSample, EvalRecord, JudgeResult, ModelOutput, Sample


# 中文注释：验证 test_basic_metrics_compute 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_basic_metrics_compute():
    metric = BasicMetrics()
    s = Sample("1", np.zeros((16, 16, 3), dtype=np.float32), "a circle")
    adv = Sample("1", np.ones((16, 16, 3), dtype=np.float32) * 0.1, "a square")
    rec = EvalRecord(
        sample=s,
        attacked=AttackedSample(adv, 1.0, 0.1),
        pred_clean=ModelOutput("detected object: circle"),
        pred_adv=ModelOutput("detected object: square"),
        judge=JudgeResult(True, "ok"),
    )
    m = metric.compute(rec)
    assert "perturbation_l2" in m

