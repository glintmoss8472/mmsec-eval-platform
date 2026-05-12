# 文件说明：该文件属于后端接口路由，集中实现 datasets 相关逻辑。
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends

from mmsec_api.deps import get_queue, get_store
from mmsec_api.schemas.models import DatasetInfo, DatasetListResponse, DatasetPrepareRequest, JobResponse
from mmsec_api.services.dataset_status import enrich_dataset_registry_rows
from mmsec_api.store.sqlite import SQLiteStore
from mmsec_api.worker.queue import JobQueue

router = APIRouter(prefix="/api/v1/datasets", tags=["datasets"])


# 中文注释：处理 list_datasets 对应的接口请求，并把后端接口路由结果整理为前端可消费的数据。
@router.get("", response_model=DatasetListResponse)
def list_datasets(store: SQLiteStore = Depends(get_store)) -> DatasetListResponse:
    project_root = Path(__file__).resolve().parents[3]
    items = [DatasetInfo(**x) for x in enrich_dataset_registry_rows(store.list_datasets(), project_root)]
    return DatasetListResponse(items=items)


# 中文注释：处理 prepare_dataset 对应的接口请求，并把后端接口路由结果整理为前端可消费的数据。
@router.post("/prepare", response_model=JobResponse)
def prepare_dataset(
    payload: DatasetPrepareRequest,
    store: SQLiteStore = Depends(get_store),
    q: JobQueue = Depends(get_queue),
) -> JobResponse:
    job = store.create_job(
        job_type="dataset_prepare",
        config_path="configs/mvp.yaml",
        override={},
        benchmark_mode=False,
        payload=payload.model_dump(),
    )
    q.enqueue(job["id"])
    return JobResponse(**job)
