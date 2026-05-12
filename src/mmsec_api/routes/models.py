# 文件说明：该文件属于后端接口路由，集中实现 models 相关逻辑。
from __future__ import annotations

from fastapi import APIRouter, Request

from mmsec_api.schemas.models import ModelListResponse
from mmsec_api.services.system_overview import build_system_overview

router = APIRouter(prefix="/api/v1/models", tags=["models"])


# 中文注释：处理 list_models 对应的接口请求，并把后端接口路由结果整理为前端可消费的数据。
@router.get("", response_model=ModelListResponse)
def list_models(request: Request) -> ModelListResponse:
    overview = build_system_overview(request)
    models = overview.get("models", [])
    return ModelListResponse(total=len(models), items=models)
