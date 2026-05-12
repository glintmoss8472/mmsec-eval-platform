# 文件说明：该文件属于后端接口路由，集中实现 bootstrap 相关逻辑。
from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from mmsec_api.deps import get_bootstrap
from mmsec_api.schemas.models import BootstrapLogsResponse, BootstrapStatusResponse
from mmsec_api.services.bootstrap_orchestrator import BootstrapOrchestrator

router = APIRouter(prefix="/api/v1/bootstrap", tags=["bootstrap"])


# 中文注释：处理 bootstrap_status 对应的接口请求，并把后端接口路由结果整理为前端可消费的数据。
@router.get("/status", response_model=BootstrapStatusResponse)
def bootstrap_status(bootstrap: BootstrapOrchestrator = Depends(get_bootstrap)) -> BootstrapStatusResponse:
    return BootstrapStatusResponse(**bootstrap.get_status())


# 中文注释：处理 bootstrap_logs 对应的接口请求，并把后端接口路由结果整理为前端可消费的数据。
@router.get("/logs", response_model=BootstrapLogsResponse)
def bootstrap_logs(
    limit: int = Query(default=200, ge=1, le=1000),
    bootstrap: BootstrapOrchestrator = Depends(get_bootstrap),
) -> BootstrapLogsResponse:
    return BootstrapLogsResponse(items=bootstrap.get_logs(limit=limit))


# 中文注释：处理 bootstrap_retry 对应的接口请求，并把后端接口路由结果整理为前端可消费的数据。
@router.post("/retry", response_model=BootstrapStatusResponse)
def bootstrap_retry(bootstrap: BootstrapOrchestrator = Depends(get_bootstrap)) -> BootstrapStatusResponse:
    bootstrap.retry()
    return BootstrapStatusResponse(**bootstrap.get_status())
