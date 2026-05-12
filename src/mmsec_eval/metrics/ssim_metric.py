# 文件说明：该文件属于指标计算层，集中实现 ssim metric 相关逻辑。
from __future__ import annotations

import numpy as np

from mmsec_eval.plugins.base import MetricPlugin
from mmsec_eval.types import EvalRecord


# 执行 `ssim simple` 辅助逻辑，保持指标计算层中的输入处理和结果输出一致。
def ssim_simple(x: np.ndarray, y: np.ndarray) -> float:
    ux, uy = float(x.mean()), float(y.mean())
    sx, sy = float(x.std()), float(y.std())
    cov = float(((x - ux) * (y - uy)).mean())
    c1, c2 = 1e-4, 9e-4
    num = (2 * ux * uy + c1) * (2 * cov + c2)
    den = (ux * ux + uy * uy + c1) * (sx * sx + sy * sy + c2)
    return float(num / (den + 1e-8))


# 实现 `SSIMMetric.compute` 的对象行为，维护该类在指标计算层中的调用契约。
class SSIMMetric(MetricPlugin):
    # 实现 SSIMMetric.compute 的核心行为，维护指标计算层在该对象上的调用契约。
    def compute(self, record: EvalRecord) -> dict[str, float]:
        score = ssim_simple(record.sample.image, record.attacked.sample.image)
        return {"ssim": score}

