import numpy as np

from mmsec_eval.metrics.ssim_metric import SSIMMetric
from mmsec_eval.types import AttackedSample, EvalRecord, ModelOutput, Sample


def test_ssim_metric():
    metric = SSIMMetric()
    s = Sample("1", np.zeros((16, 16, 3), dtype=np.float32), "a")
    a = Sample("1", np.zeros((16, 16, 3), dtype=np.float32), "b")
    rec = EvalRecord(s, AttackedSample(a, 0.0, 0.0), ModelOutput("a"), ModelOutput("b"))
    m = metric.compute(rec)
    assert "ssim" in m

