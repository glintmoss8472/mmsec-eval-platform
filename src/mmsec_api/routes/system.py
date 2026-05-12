from __future__ import annotations

from fastapi import APIRouter, Request

from mmsec_api.schemas.models import SystemComplianceResponse, SystemOverviewResponse
from mmsec_api.services.system_overview import build_system_compliance, build_system_overview

router = APIRouter(prefix="/api/v1/system", tags=["system"])


@router.get("/overview", response_model=SystemOverviewResponse)
def system_overview(request: Request) -> SystemOverviewResponse:
    return SystemOverviewResponse(**build_system_overview(request))


@router.get("/compliance", response_model=SystemComplianceResponse)
def system_compliance(request: Request) -> SystemComplianceResponse:
    return SystemComplianceResponse(**build_system_compliance(request))
