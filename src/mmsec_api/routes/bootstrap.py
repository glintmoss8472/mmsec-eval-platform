# 文件说明：该文件属于后端接口路由，集中实现 bootstrap 相关逻辑。
from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from mmsec_api.deps import get_bootstrap
from mmsec_api.schemas.models import BootstrapLogsResponse, BootstrapStatusResponse
from mmsec_api.services.bootstrap_orchestrator import BootstrapOrchestrator

router = APIRouter(prefix="/api/v1/bootstrap", tags=["bootstrap"])


# 处理 `GET /status` 接口，完成请求校验、存储访问和响应模型组装。
@router.get("/status", response_model=BootstrapStatusResponse)
def bootstrap_status(bootstrap: BootstrapOrchestrator = Depends(get_bootstrap)) -> BootstrapStatusResponse:
    return BootstrapStatusResponse(**bootstrap.get_status())


# 处理 `GET /logs` 接口，完成请求校验、存储访问和响应模型组装。
@router.get("/logs", response_model=BootstrapLogsResponse)
def bootstrap_logs(
    limit: int = Query(default=200, ge=1, le=1000),
    bootstrap: BootstrapOrchestrator = Depends(get_bootstrap),
) -> BootstrapLogsResponse:
    return BootstrapLogsResponse(items=bootstrap.get_logs(limit=limit))


# 处理 `POST /retry` 接口，完成请求校验、存储访问和响应模型组装。
@router.post("/retry", response_model=BootstrapStatusResponse)
def bootstrap_retry(bootstrap: BootstrapOrchestrator = Depends(get_bootstrap)) -> BootstrapStatusResponse:
    bootstrap.retry()
    return BootstrapStatusResponse(**bootstrap.get_status())
