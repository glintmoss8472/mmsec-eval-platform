# 文件说明：该文件属于项目工程，集中实现 seed 相关逻辑。
from __future__ import annotations

import os
import random

import numpy as np


# 中文注释：实现 set_seed 的核心流程，支撑项目工程中的业务语义和异常边界。
def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except (ImportError, RuntimeError):
        # Torch is optional for CPU-only smoke paths.
        return
