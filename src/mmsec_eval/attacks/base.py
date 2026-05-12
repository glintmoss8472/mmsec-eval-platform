# 文件说明：该文件属于攻击算法公共层，集中实现 base 相关逻辑。
from __future__ import annotations

import numpy as np


# 执行 `clip01` 辅助逻辑，保持攻击算法公共层中的输入处理和结果输出一致。
def clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


# 执行 `perturb stats` 辅助逻辑，保持攻击算法公共层中的输入处理和结果输出一致。
def perturb_stats(clean: np.ndarray, adv: np.ndarray) -> tuple[int, float, float]:
    d = adv - clean
    l0 = int(np.count_nonzero(np.abs(d) > 1e-8))
    l2 = float(np.sqrt((d * d).sum()))
    linf = float(np.abs(d).max())
    return l0, l2, linf


# 执行 `total variation` 辅助逻辑，保持攻击算法公共层中的输入处理和结果输出一致。
def total_variation(x: np.ndarray) -> float:
    dx = np.abs(np.diff(x, axis=1)).mean()
    dy = np.abs(np.diff(x, axis=0)).mean()
    return float(dx + dy)
