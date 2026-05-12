# 文件说明：该文件属于后端接口路由，集中实现 reports 相关逻辑。
from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/api/v1/reports", tags=["reports"])


# 中文注释：处理 report_ping 对应的接口请求，并把后端接口路由结果整理为前端可消费的数据。
@router.get("/ping")
def report_ping() -> dict[str, str]:
    return {"status": "ok"}
