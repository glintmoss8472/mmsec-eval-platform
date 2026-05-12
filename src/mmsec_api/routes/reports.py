from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/api/v1/reports", tags=["reports"])


@router.get("/ping")
def report_ping() -> dict[str, str]:
    return {"status": "ok"}
