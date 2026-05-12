from __future__ import annotations

import numpy as np

from mmsec_eval.plugins.base import MetricPlugin
from mmsec_eval.types import EvalRecord


def ssim_simple(x: np.ndarray, y: np.ndarray) -> float:
    ux, uy = float(x.mean()), float(y.mean())
    sx, sy = float(x.std()), float(y.std())
    cov = float(((x - ux) * (y - uy)).mean())
    c1, c2 = 1e-4, 9e-4
    num = (2 * ux * uy + c1) * (2 * cov + c2)
    den = (ux * ux + uy * uy + c1) * (sx * sx + sy * sy + c2)
    return float(num / (den + 1e-8))


class SSIMMetric(MetricPlugin):
    def compute(self, record: EvalRecord) -> dict[str, float]:
        score = ssim_simple(record.sample.image, record.attacked.sample.image)
        return {"ssim": score}

