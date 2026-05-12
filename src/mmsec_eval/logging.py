# 文件说明：该文件属于项目工程，集中实现 logging 相关逻辑。
from __future__ import annotations

import logging
from pathlib import Path


# 执行 `setup logging` 辅助逻辑，保持项目工程中的输入处理和结果输出一致。
def setup_logging(log_dir: str, level: str = "INFO") -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logfile = Path(log_dir) / "mmsec.log"

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(sh)

    fh = logging.FileHandler(logfile, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)

