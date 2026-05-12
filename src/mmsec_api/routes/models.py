from __future__ import annotations

from fastapi import APIRouter, Request

from mmsec_api.schemas.models import ModelListResponse
from mmsec_api.services.system_overview import build_system_overview

router = APIRouter(prefix="/api/v1/models", tags=["models"])


@router.get("", response_model=ModelListResponse)
def list_models(request: Request) -> ModelListResponse:
    overview = build_system_overview(request)
    models = overview.get("models", [])
    return ModelListResponse(total=len(models), items=models)
