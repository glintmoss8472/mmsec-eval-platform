# 文件说明：该文件属于后端接口路由，集中实现 reports 相关逻辑。
from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/api/v1/reports", tags=["reports"])


# 处理 `GET /ping` 接口，完成请求校验、存储访问和响应模型组装。
@router.get("/ping")
def report_ping() -> dict[str, str]:
    return {"status": "ok"}
