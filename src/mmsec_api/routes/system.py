# 文件说明：该文件属于后端接口路由，集中实现 system 相关逻辑。
from __future__ import annotations

from fastapi import APIRouter, Request

from mmsec_api.schemas.models import SystemComplianceResponse, SystemOverviewResponse
from mmsec_api.services.system_overview import build_system_compliance, build_system_overview

router = APIRouter(prefix="/api/v1/system", tags=["system"])


# 处理 `GET /overview` 接口，完成请求校验、存储访问和响应模型组装。
@router.get("/overview", response_model=SystemOverviewResponse)
def system_overview(request: Request) -> SystemOverviewResponse:
    return SystemOverviewResponse(**build_system_overview(request))


# 处理 `GET /compliance` 接口，完成请求校验、存储访问和响应模型组装。
@router.get("/compliance", response_model=SystemComplianceResponse)
def system_compliance(request: Request) -> SystemComplianceResponse:
    return SystemComplianceResponse(**build_system_compliance(request))
