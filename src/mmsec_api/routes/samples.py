# 文件说明：该文件属于后端接口路由，集中实现 samples 相关逻辑。
from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Request

from mmsec_api.deps import get_store
from mmsec_api.routes.runs import _artifacts_dir
from mmsec_api.services.sample_assets import sync_sample_assets_from_runs
from mmsec_api.store.sqlite import SQLiteStore

router = APIRouter(prefix="/api/v1/samples", tags=["samples"])


# 中文注释：封装 _ensure_asset_index 的内部步骤，让后端接口路由主流程保持清晰并隔离边界细节。
def _ensure_asset_index(artifacts_dir: str, store: SQLiteStore) -> None:
    # First request after migration backfills the independent asset table from
    # existing report/case artifacts. Later task completion paths upsert only
    # the new run, so normal reads stay fast.
    if store.count_sample_assets() == 0:
        sync_sample_assets_from_runs(artifacts_dir, store)


# 中文注释：处理 list_sample_asset_batches 对应的接口请求，并把后端接口路由结果整理为前端可消费的数据。
@router.get("/batches")
def list_sample_asset_batches(
    request: Request,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=24, ge=1, le=500),
    task_kind: str = Query(default=""),
    attack: str = Query(default=""),
    scope: str = Query(default=""),
    reusable_status: str = Query(default=""),
    model: str = Query(default=""),
    dataset: str = Query(default=""),
    search: str = Query(default=""),
    sort_by: str = Query(default="created_at"),
    sort_dir: str = Query(default="desc"),
    include_asset_ids: bool = Query(default=True),
    store: SQLiteStore = Depends(get_store),
):
    artifacts_dir = _artifacts_dir(request)
    _ensure_asset_index(artifacts_dir, store)
    total, items, meta = store.list_sample_asset_batches(
        page=page,
        page_size=page_size,
        task_kind=task_kind,
        attack=attack,
        scope=scope,
        reusable_status=reusable_status,
        model=model,
        dataset=dataset,
        search=search,
        sort_by=sort_by,
        sort_dir=sort_dir,
        include_asset_ids=include_asset_ids,
    )
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "items": items,
        "summary": meta["summary"],
        "options": meta["options"],
    }


# 中文注释：处理 list_sample_assets 对应的接口请求，并把后端接口路由结果整理为前端可消费的数据。
@router.get("")
def list_sample_assets(
    request: Request,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=30, ge=1, le=200),
    task_kind: str = Query(default=""),
    attack: str = Query(default=""),
    scope: str = Query(default=""),
    reusable_status: str = Query(default=""),
    model: str = Query(default=""),
    dataset: str = Query(default=""),
    search: str = Query(default=""),
    store: SQLiteStore = Depends(get_store),
):
    artifacts_dir = _artifacts_dir(request)
    _ensure_asset_index(artifacts_dir, store)
    total, items, meta = store.list_sample_assets(
        page=page,
        page_size=page_size,
        task_kind=task_kind,
        attack=attack,
        scope=scope,
        reusable_status=reusable_status,
        model=model,
        dataset=dataset,
        search=search,
    )
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "items": items,
        "summary": meta["summary"],
        "options": meta["options"],
    }
