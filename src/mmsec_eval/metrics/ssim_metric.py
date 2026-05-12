# 文件说明：该文件属于指标计算层，集中实现 ssim metric 相关逻辑。
from __future__ import annotations

import numpy as np

from mmsec_eval.plugins.base import MetricPlugin
from mmsec_eval.types import EvalRecord


# 中文注释：实现 ssim_simple 的核心流程，支撑指标计算层中的业务语义和异常边界。
def ssim_simple(x: np.ndarray, y: np.ndarray) -> float:
    ux, uy = float(x.mean()), float(y.mean())
    sx, sy = float(x.std()), float(y.std())
    cov = float(((x - ux) * (y - uy)).mean())
    c1, c2 = 1e-4, 9e-4
    num = (2 * ux * uy + c1) * (2 * cov + c2)
    den = (ux * ux + uy * uy + c1) * (sx * sx + sy * sy + c2)
    return float(num / (den + 1e-8))


# 中文注释：定义 SSIMMetric 的结构化职责，作为指标计算层中状态、配置或行为的边界。
class SSIMMetric(MetricPlugin):
    # 中文注释：实现 SSIMMetric.compute 的核心行为，维护指标计算层在该对象上的调用契约。
    def compute(self, record: EvalRecord) -> dict[str, float]:
        score = ssim_simple(record.sample.image, record.attacked.sample.image)
        return {"ssim": score}

