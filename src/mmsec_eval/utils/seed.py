# 文件说明：该文件属于项目工程，集中实现 seed 相关逻辑。
from __future__ import annotations

import os
import random

import numpy as np


# 执行 `set seed` 辅助逻辑，保持项目工程中的输入处理和结果输出一致。
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
