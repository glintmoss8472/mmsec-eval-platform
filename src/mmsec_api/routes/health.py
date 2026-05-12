# 文件说明：该文件属于后端接口路由，集中实现 health 相关逻辑。
from __future__ import annotations

from fastapi import APIRouter, Depends

from mmsec_api.deps import get_bootstrap
from mmsec_api.schemas.models import HealthResponse
from mmsec_api.services.bootstrap_orchestrator import BootstrapOrchestrator

router = APIRouter(prefix="/api/v1", tags=["health"])


# 处理 `GET /health` 接口，完成请求校验、存储访问和响应模型组装。
@router.get("/health", response_model=HealthResponse)
def health(bootstrap: BootstrapOrchestrator = Depends(get_bootstrap)) -> HealthResponse:
    status = bootstrap.get_status()
    return HealthResponse(
        status="ok",
        version="0.1.0",
        bootstrap_state=status.get("state", "pending"),
        degraded_reason=status.get("degraded_reason", ""),
    )
