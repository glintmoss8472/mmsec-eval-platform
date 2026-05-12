# 文件说明：该文件属于项目工程，集中实现 utils 相关逻辑。
from __future__ import annotations

from datetime import datetime, timezone


# 中文注释：实现 utc_now_iso 的核心流程，支撑项目工程中的业务语义和异常边界。
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
