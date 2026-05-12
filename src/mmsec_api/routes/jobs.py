# 文件说明：该文件属于后端接口路由，集中实现 jobs 相关逻辑。
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from yaml import YAMLError

from mmsec_api.deps import get_queue, get_store
from mmsec_api.schemas.models import JobCreateRequest, JobListResponse, JobLogListResponse, JobLogResponse, JobProgressResponse, JobResponse, JobStageResponse
from mmsec_api.services.job_progress import (
    duration_seconds,
    estimate_eta_seconds,
    estimated_ready_at,
    pair_progress_percent,
    parse_pair_progress,
)
from mmsec_api.store.sqlite import SQLiteStore
from mmsec_api.services.model_runtime import model_supports_task
from mmsec_api.utils import utc_now_iso
from mmsec_api.worker.queue import JobQueue
from mmsec_eval.attacks.catalog import attack_surrogate_error
from mmsec_eval.config.loader import load_config
from mmsec_eval.config.sweep import apply_override

router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])


_GENERATION_JOB_TYPES = {"run_vqa", "run_caption"}


# 中文注释：封装 _generation_task_kind 的内部步骤，让后端接口路由主流程保持清晰并隔离边界细节。
def _generation_task_kind(job_type: str) -> str:
    return "vqa" if "vqa" in str(job_type) else "caption"


# 中文注释：封装 _request_override 的内部步骤，让后端接口路由主流程保持清晰并隔离边界细节。
def _request_override(req: JobCreateRequest) -> dict:
    return req.override if isinstance(req.override, dict) else {}


# 中文注释：封装 _load_request_config 的内部步骤，让后端接口路由主流程保持清晰并隔离边界细节。
def _load_request_config(req: JobCreateRequest):
    override = _request_override(req)
    try:
        cfg = load_config(req.config_path)
        return apply_override(cfg, override) if override else cfg
    except (OSError, TypeError, ValueError, YAMLError):
        # Full config validation still happens in the worker; these route checks
        # only reject known task/model/dataset mismatches early.
        return None


# 中文注释：封装 _validate_generation_dataset_compatibility 的内部步骤，让后端接口路由主流程保持清晰并隔离边界细节。
def _validate_generation_dataset_compatibility(req: JobCreateRequest) -> None:
    job_type = str(req.job_type)
    if job_type not in _GENERATION_JOB_TYPES:
        return
    task_kind = _generation_task_kind(job_type)
    cfg = _load_request_config(req)
    if cfg is None:
        return
    source = f"{cfg.task.cases_jsonl} {cfg.dataset.benchmark_tag}".lower()
    if task_kind == "vqa" and "coco_caption_object_val" in source:
        raise HTTPException(status_code=422, detail="VQA task requires a VQA JSONL dataset, not the COCO caption object JSONL")
    if task_kind == "caption" and ("vqa_v2_coco_val" in source or "coco_object_probe_val" in source):
        raise HTTPException(status_code=422, detail="Caption task requires the COCO caption object JSONL, not a VQA JSONL dataset")


# 中文注释：封装 _validate_vlr_attack_compatibility 的内部步骤，让后端接口路由主流程保持清晰并隔离边界细节。
def _validate_vlr_attack_compatibility(req: JobCreateRequest) -> None:
    job_type = str(req.job_type)
    if job_type not in {"run_vlr", *_GENERATION_JOB_TYPES}:
        return
    override = _request_override(req)
    plugins = dict(override.get("plugins", {})) if isinstance(override.get("plugins", {}), dict) else {}
    runner = dict(override.get("runner", {})) if isinstance(override.get("runner", {}), dict) else {}
    cfg = _load_request_config(req)
    attack = str(plugins.get("attack") or getattr(getattr(cfg, "plugins", None), "attack", "") or "").strip()
    cfg_runner = getattr(cfg, "runner", None)
    cfg_plugins = getattr(cfg, "plugins", None)
    if job_type in _GENERATION_JOB_TYPES:
        surrogate = str(runner.get("surrogate_model_adapter") or getattr(cfg_runner, "surrogate_model_adapter", "") or "clip_hf").strip()
    else:
        surrogate = str(
            runner.get("surrogate_model_adapter")
            or getattr(cfg_runner, "surrogate_model_adapter", "")
            or plugins.get("model_adapter")
            or getattr(cfg_plugins, "model_adapter", "")
            or ""
        ).strip()
    error = attack_surrogate_error(attack, surrogate)
    if error:
        raise HTTPException(status_code=422, detail=error)


# 中文注释：封装 _validate_model_task_compatibility 的内部步骤，让后端接口路由主流程保持清晰并隔离边界细节。
def _validate_model_task_compatibility(req: JobCreateRequest) -> None:
    job_type = str(req.job_type)
    if job_type not in {"run_vlr", *_GENERATION_JOB_TYPES}:
        return
    cfg = _load_request_config(req)
    if cfg is None:
        return

    if job_type in _GENERATION_JOB_TYPES:
        task_kind = "vqa" if "vqa" in job_type else "caption"
        adapter = str(getattr(getattr(cfg, "plugins", None), "model_adapter", "") or "").strip()
        if adapter and not model_supports_task(adapter, task_kind):
            raise HTTPException(
                status_code=422,
                detail=(
                    f"模型 {adapter} 不支持 {task_kind.upper()} 生成式真实测评；"
                    "请选择 OpenAI 兼容或 Gemini 视觉语言生成模型；内置演示模型、CLIP、BLIP、ViLT 不允许用于该任务。"
                ),
            )
        return

    cfg_runner = getattr(cfg, "runner", None)
    cfg_plugins = getattr(cfg, "plugins", None)
    victim_adapters = [str(item).strip() for item in list(getattr(cfg_runner, "victim_model_adapters", []) or []) if str(item).strip()]
    if not victim_adapters:
        fallback = str(getattr(cfg_plugins, "model_adapter", "") or "").strip()
        victim_adapters = [fallback] if fallback else []
    blocked = [adapter for adapter in victim_adapters if not model_supports_task(adapter, "vlr")]
    if blocked:
        raise HTTPException(
            status_code=422,
            detail=(
                "以下模型不支持 VLR 图文检索真实测评："
                + ", ".join(blocked)
                + "。内置演示模型不允许作为正式受测模型。"
            ),
        )


# 中文注释：处理 create_job 对应的接口请求，并把后端接口路由结果整理为前端可消费的数据。
@router.post("", response_model=JobResponse)
def create_job(
    req: JobCreateRequest,
    store: SQLiteStore = Depends(get_store),
    q: JobQueue = Depends(get_queue),
) -> JobResponse:
    _validate_generation_dataset_compatibility(req)
    _validate_vlr_attack_compatibility(req)
    _validate_model_task_compatibility(req)
    job = store.create_job(
        job_type=req.job_type,
        config_path=req.config_path,
        override=req.override,
        benchmark_mode=req.benchmark_mode,
        payload=req.payload,
    )
    store.init_job_progress(job["id"], req.job_type)
    q.enqueue(job["id"])
    return JobResponse(**job)


# 中文注释：处理 list_jobs 对应的接口请求，并把后端接口路由结果整理为前端可消费的数据。
@router.get("", response_model=JobListResponse)
def list_jobs(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=200),
    status: str = Query(default=""),
    store: SQLiteStore = Depends(get_store),
) -> JobListResponse:
    total, items = store.list_jobs(page=page, page_size=page_size, status=status)
    return JobListResponse(total=total, page=page, page_size=page_size, items=[JobResponse(**x) for x in items])


# 中文注释：处理 get_job 对应的接口请求，并把后端接口路由结果整理为前端可消费的数据。
@router.get("/{job_id}", response_model=JobResponse)
def get_job(job_id: str, store: SQLiteStore = Depends(get_store)) -> JobResponse:
    row = store.get_job(job_id)
    if not row:
        raise HTTPException(status_code=404, detail="job not found")
    return JobResponse(**row)


# 中文注释：处理 get_job_logs 对应的接口请求，并把后端接口路由结果整理为前端可消费的数据。
@router.get("/{job_id}/logs", response_model=JobLogListResponse)
def get_job_logs(
    job_id: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=100, ge=1, le=500),
    store: SQLiteStore = Depends(get_store),
) -> JobLogListResponse:
    if not store.get_job(job_id):
        raise HTTPException(status_code=404, detail="job not found")
    total, rows = store.list_job_logs(job_id=job_id, page=page, page_size=page_size)
    return JobLogListResponse(total=total, page=page, page_size=page_size, items=[JobLogResponse(**x) for x in rows])


# 中文注释：处理 get_job_progress 对应的接口请求，并把后端接口路由结果整理为前端可消费的数据。
@router.get("/{job_id}/progress", response_model=JobProgressResponse)
def get_job_progress(job_id: str, store: SQLiteStore = Depends(get_store), q: JobQueue = Depends(get_queue)) -> JobProgressResponse:
    row = store.get_job(job_id)
    if not row:
        raise HTTPException(status_code=404, detail="job not found")

    stages = store.list_job_progress(job_id)
    if not stages:
        store.init_job_progress(job_id, str(row.get("job_type", "")))
        stages = store.list_job_progress(job_id)

    queue_position = store.get_queue_position(job_id)
    elapsed_seconds = duration_seconds(row.get("started_at"), row.get("finished_at"))
    if str(row.get("status", "")) == "running" and not row.get("finished_at"):
        start = row.get("started_at")
        elapsed_seconds = duration_seconds(start, utc_now_iso())
    progress_percent = max([float(stage.get("progress_percent", 0.0) or 0.0) for stage in stages] or [0.0])
    job_status = str(row.get("status", ""))
    has_success_completed = any(stage.get("stage_key") == "completed" and stage.get("state") == "success" for stage in stages)
    if job_status == "success" and has_success_completed:
        stages = [
            {
                **stage,
                "state": "success" if stage.get("stage_key") != "completed" and stage.get("state") in {"pending", "running"} else stage.get("state"),
            }
            for stage in stages
        ]
    running_stage = next((stage for stage in stages if stage.get("state") == "running"), None)
    terminal_stage = next((stage for stage in reversed(stages) if stage.get("stage_key") == "completed" and stage.get("state") == "success"), None)
    selected_stage = terminal_stage if job_status == "success" and terminal_stage else (running_stage or (stages[-1] if stages else {}))
    current_stage = str((selected_stage or {}).get("stage_key", ""))
    current_stage_message = str((selected_stage or {}).get("message", ""))
    parsed_stage_progress = parse_pair_progress(current_stage_message)
    current_stage_units_done = int(parsed_stage_progress[0]) if parsed_stage_progress else 0
    current_stage_units_total = int(parsed_stage_progress[1]) if parsed_stage_progress else 0
    current_stage_progress_percent = pair_progress_percent(current_stage_message)
    recent_durations = [duration_seconds(item.get("started_at"), item.get("finished_at")) for item in store.list_success_durations(str(row.get("job_type", "")))]
    eta_seconds = estimate_eta_seconds(
        job_type=str(row.get("job_type", "")),
        status=str(row.get("status", "")),
        queue_position=queue_position,
        elapsed_seconds=elapsed_seconds,
        progress_percent=progress_percent,
        recent_durations=recent_durations,
        worker_count=getattr(q, "workers", 1),
        stage_message=current_stage_message,
    )
    last_log_row = store.get_latest_job_log(job_id)
    return JobProgressResponse(
        job_id=job_id,
        job_type=str(row.get("job_type", "")),
        status=str(row.get("status", "")),
        queue_position=queue_position,
        elapsed_seconds=float(round(elapsed_seconds, 2)),
        eta_seconds=float(round(eta_seconds, 2)),
        estimated_ready_at=estimated_ready_at(eta_seconds),
        current_stage=current_stage,
        progress_percent=float(round(progress_percent, 2)),
        progress_percent_semantics="overall pipeline completion percent across all job stages; use current_stage_* fields for stage-local hard counts",
        current_stage_message=current_stage_message,
        current_stage_units_done=current_stage_units_done,
        current_stage_units_total=current_stage_units_total,
        current_stage_progress_percent=float(round(current_stage_progress_percent, 2)),
        current_stage_updated_at=str((selected_stage or {}).get("updated_at", "")),
        stages=[JobStageResponse(**stage) for stage in stages],
        last_log=str((last_log_row or {}).get("message", "")),
        run_id=str(row.get("run_id") or ""),
    )


# 中文注释：处理 cancel_job 对应的接口请求，并把后端接口路由结果整理为前端可消费的数据。
@router.post("/{job_id}/cancel", response_model=JobResponse)
def cancel_job(job_id: str, store: SQLiteStore = Depends(get_store)) -> JobResponse:
    row = store.get_job(job_id)
    if not row:
        raise HTTPException(status_code=404, detail="job not found")
    if row.get("status") in {"success", "failed", "cancelled"}:
        return JobResponse(**row)
    note = "cancel requested"
    store.set_job_cancelled(job_id, note=note)
    row2 = store.get_job(job_id)
    assert row2
    return JobResponse(**row2)
