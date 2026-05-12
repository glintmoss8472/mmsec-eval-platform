from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, Request

from mmsec_api.deps import get_queue, get_store
from mmsec_api.schemas.models import DocsIngestRequest, DocsPayloadResponse, JobResponse
from mmsec_api.services.run_reader import read_json, read_jsonl
from mmsec_api.store.sqlite import SQLiteStore
from mmsec_api.worker.queue import JobQueue

router = APIRouter(prefix="/api/v1/docs", tags=["docs"])


def _artifacts_dir(request: Request) -> str:
    return str(getattr(request.app.state, "artifacts_dir", "artifacts"))


@router.post("/ingest", response_model=JobResponse)
def ingest_docs(
    req: DocsIngestRequest,
    store: SQLiteStore = Depends(get_store),
    q: JobQueue = Depends(get_queue),
) -> JobResponse:
    job = store.create_job(
        job_type="docs_ingest",
        config_path=req.config_path,
        override={},
        benchmark_mode=False,
        payload={},
    )
    q.enqueue(job["id"])
    return JobResponse(**job)


@router.get("/index", response_model=DocsPayloadResponse)
def docs_index(request: Request) -> DocsPayloadResponse:
    p = Path(_artifacts_dir(request)) / "docs_index.json"
    data = read_json(p, [])
    if not isinstance(data, list):
        data = []
    return DocsPayloadResponse(items=data)


@router.get("/snippets", response_model=DocsPayloadResponse)
def docs_snippets(request: Request) -> DocsPayloadResponse:
    p = Path(_artifacts_dir(request)) / "docs_snippets.jsonl"
    rows = read_jsonl(p)
    return DocsPayloadResponse(items=rows)
