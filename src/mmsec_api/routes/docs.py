# 文件说明：该文件属于后端接口路由，集中实现 docs 相关逻辑。
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, Request

from mmsec_api.deps import get_queue, get_store
from mmsec_api.schemas.models import DocsIngestRequest, DocsPayloadResponse, JobResponse
from mmsec_api.services.run_reader import read_json, read_jsonl
from mmsec_api.store.sqlite import SQLiteStore
from mmsec_api.worker.queue import JobQueue

router = APIRouter(prefix="/api/v1/docs", tags=["docs"])


# 中文注释：封装 _artifacts_dir 的内部步骤，让后端接口路由主流程保持清晰并隔离边界细节。
def _artifacts_dir(request: Request) -> str:
    return str(getattr(request.app.state, "artifacts_dir", "artifacts"))


# 中文注释：处理 ingest_docs 对应的接口请求，并把后端接口路由结果整理为前端可消费的数据。
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


# 中文注释：处理 docs_index 对应的接口请求，并把后端接口路由结果整理为前端可消费的数据。
@router.get("/index", response_model=DocsPayloadResponse)
def docs_index(request: Request) -> DocsPayloadResponse:
    p = Path(_artifacts_dir(request)) / "docs_index.json"
    data = read_json(p, [])
    if not isinstance(data, list):
        data = []
    return DocsPayloadResponse(items=data)


# 中文注释：处理 docs_snippets 对应的接口请求，并把后端接口路由结果整理为前端可消费的数据。
@router.get("/snippets", response_model=DocsPayloadResponse)
def docs_snippets(request: Request) -> DocsPayloadResponse:
    p = Path(_artifacts_dir(request)) / "docs_snippets.jsonl"
    rows = read_jsonl(p)
    return DocsPayloadResponse(items=rows)
