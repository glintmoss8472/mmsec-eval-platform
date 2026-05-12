# 文件说明：该文件属于后端接口路由，集中实现 models 相关逻辑。
from __future__ import annotations

from fastapi import APIRouter, Request

from mmsec_api.schemas.models import ModelListResponse
from mmsec_api.services.system_overview import build_system_overview

router = APIRouter(prefix="/api/v1/models", tags=["models"])


# 处理 `GET /` 接口，完成请求校验、存储访问和响应模型组装。
@router.get("", response_model=ModelListResponse)
def list_models(request: Request) -> ModelListResponse:
    overview = build_system_overview(request)
    models = overview.get("models", [])
    return ModelListResponse(total=len(models), items=models)
