# 文件说明：该文件属于项目工程，集中实现 utils 相关逻辑。
from __future__ import annotations

from datetime import datetime, timezone


# 执行 `utc now iso` 辅助逻辑，保持项目工程中的输入处理和结果输出一致。
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
