# 文件说明：该文件属于攻击算法公共层，集中实现 base 相关逻辑。
from __future__ import annotations

import numpy as np


# 中文注释：实现 clip01 的核心流程，支撑攻击算法公共层中的业务语义和异常边界。
def clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


# 中文注释：实现 perturb_stats 的核心流程，支撑攻击算法公共层中的业务语义和异常边界。
def perturb_stats(clean: np.ndarray, adv: np.ndarray) -> tuple[int, float, float]:
    d = adv - clean
    l0 = int(np.count_nonzero(np.abs(d) > 1e-8))
    l2 = float(np.sqrt((d * d).sum()))
    linf = float(np.abs(d).max())
    return l0, l2, linf


# 中文注释：实现 total_variation 的核心流程，支撑攻击算法公共层中的业务语义和异常边界。
def total_variation(x: np.ndarray) -> float:
    dx = np.abs(np.diff(x, axis=1)).mean()
    dy = np.abs(np.diff(x, axis=0)).mean()
    return float(dx + dy)
