# 文件说明：该文件属于项目工程，集中实现 main 相关逻辑。
from __future__ import annotations

import sys

from mmsec_eval.cli import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

