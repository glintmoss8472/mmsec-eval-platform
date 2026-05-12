from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

JobStatus = Literal["queued", "running", "success", "failed", "cancelled"]
JobStageState = Literal["pending", "queued", "running", "success", "failed", "cancelled"]
JobType = Literal[
    "run_eval",
    "run_benchmark",
    "run_vlr",
    "run_vqa",
    "run_caption",
    "train_advclip",
    "generate_sample_assets",
    "run_sweep",
    "docs_ingest",
    "dataset_prepare",
]
BootstrapState = Literal["pending", "seeding", "warming", "ready", "degraded"]
BootstrapStepState = Literal["pending", "running", "success", "failed", "skipped"]


class HealthResponse(BaseModel):
    status: str
    version: str
    bootstrap_state: BootstrapState = "pending"
    degraded_reason: str = ""


class PaginationQuery(BaseModel):
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=200)


class JobCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_type: JobType
    config_path: str = "configs/mvp.yaml"
    override: dict[str, Any] | None = None
    benchmark_mode: bool = False
    payload: dict[str, Any] | None = None


class JobResponse(BaseModel):
    id: str
    job_type: str
    status: JobStatus
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    config_path: str
    override_json: str = ""
    benchmark_mode: bool = False
    run_id: str = ""
    error_code: str = ""
    error_message: str = ""


class JobLogResponse(BaseModel):
    id: int
    job_id: str
    ts: str
    level: str
    message: str


class JobListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: list[JobResponse]


class JobLogListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: list[JobLogResponse]


class JobStageResponse(BaseModel):
    stage_key: str
    stage_label: str
    state: JobStageState
    progress_percent: float = 0.0
    message: str = ""
    updated_at: str = ""


class JobProgressResponse(BaseModel):
    job_id: str
    job_type: str
    status: JobStatus
    queue_position: int = 0
    elapsed_seconds: float = 0.0
    eta_seconds: float = 0.0
    estimated_ready_at: str = ""
    current_stage: str = ""
    progress_percent: float = 0.0
    progress_percent_semantics: str = ""
    current_stage_message: str = ""
    current_stage_units_done: int = 0
    current_stage_units_total: int = 0
    current_stage_progress_percent: float = 0.0
    current_stage_updated_at: str = ""
    stages: list[JobStageResponse] = Field(default_factory=list)
    last_log: str = ""
    run_id: str = ""


class DatasetPrepareRequest(BaseModel):
    name: Literal["flickr30k", "flickr1k", "coco_subset", "mini_flickr"]
    root_path: str = ""
    image_dir: str = ""
    split: str = ""
    max_items: int = 500
    download_annotations: bool = False
    download_images: bool = False
    captions_source: str = ""
    auto_download: bool = True


class DatasetInfo(BaseModel):
    name: str
    root_path: str
    prepared: bool
    ready: bool = False
    ready_reason: str = ""
    prepared_at: str = ""
    item_count: int = 0
    note: str = ""


class DatasetListResponse(BaseModel):
    items: list[DatasetInfo]


class RunSummary(BaseModel):
    run_id: str
    created_at: str = ""
    task_kind: str = ""
    dataset_name: str = ""
    benchmark_tag: str = ""
    attack: str = ""
    attack_modality: str = ""
    eval_scope: str = ""
    mode: str = ""
    experiment_id: str = ""
    suite: str = ""
    suite_label: str = ""
    evidence_group: str = ""
    experiment_label: str = ""
    model_adapter: str = ""
    surrogate_model_adapter: str = ""
    victim_model_adapters: list[str] = Field(default_factory=list)
    asr: float = 0.0
    asr_attack: float = 0.0
    metric_label: str = ""
    k_value: int = 0
    retrieval_direction_scope: str = ""
    victim_aggregation: str = ""
    sample_pair_count: int = 0
    metric_note: str = ""
    risk_score: float = 0.0
    risk_level: str = ""
    risk_scenario: str = ""
    avg_l2: float = 0.0
    avg_linf: float = 0.0
    clean_r1_mean: float | None = None
    attacked_r1_mean: float | None = None
    attack_drop_r1_mean: float | None = None
    clean_mean_rank: float | None = None
    attacked_mean_rank: float | None = None
    rank_delta_mean: float | None = None
    clean_accuracy: float | None = None
    attacked_accuracy: float | None = None
    answer_change_rate: float | None = None
    target_flip_rate: float | None = None
    semantic_preservation_rate: float | None = None
    caption_text_similarity: float | None = None
    object_jaccard: float | None = None
    semantic_preservation_score: float = 0.0
    case_count: int = 0
    evidence_sample_count: int = 0
    evidence_confidence: str = ""
    evidence_note: str = ""
    result_type: str = "formal"
    result_type_note: str = ""
    has_case_evidence: bool = False
    artifact_evidence_status: str = "unknown"
    path: str = ""
    evidence_row_id: str = ""
    image_branch_enabled: bool = False
    text_branch_enabled: bool = False
    text_edit_applied: bool = False
    text_changed_ratio: float | None = None
    joint_execution_declared: bool = False
    joint_execution_confirmed: bool = False
    joint_execution_evidence_source: str = ""
    joint_execution_basis: str = ""
    joint_execution_note: str = ""
    summary_path: str = ""
    report_path: str = ""
    archived_summary_path: str = ""
    archived_report_path: str = ""
    source_summary_path: str = ""
    source_report_data_path: str = ""
    source_report_path: str = ""
    portable_report_data_path: str = ""
    portable_report_path: str = ""
    artifact_index_path: str = ""


class RunListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: list[RunSummary]


class RunCompareResponse(BaseModel):
    run_ids: list[str]
    compare: dict[str, Any]


class RowsResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: list[dict[str, Any]]


class CaseDetailResponse(BaseModel):
    run_id: str
    sample_id: str
    case_bundle: dict[str, Any]
    attack_debug: dict[str, Any]


class DocsIngestRequest(BaseModel):
    config_path: str = "configs/mvp.yaml"


class DocsPayloadResponse(BaseModel):
    items: list[dict[str, Any]]


class BootstrapStep(BaseModel):
    name: str
    state: BootstrapStepState
    message: str = ""
    updated_at: str


class WarmupArtifactRef(BaseModel):
    docs_index: str = ""
    docs_snippets: str = ""
    seeded_runs: list[str] = Field(default_factory=list)
    seeded_data: list[str] = Field(default_factory=list)


class BootstrapStatusResponse(BaseModel):
    state: BootstrapState
    started_at: str = ""
    updated_at: str = ""
    degraded_reason: str = ""
    steps: list[BootstrapStep] = Field(default_factory=list)
    artifacts: WarmupArtifactRef = Field(default_factory=WarmupArtifactRef)


class BootstrapLogsResponse(BaseModel):
    items: list[dict[str, str]]


class ModelOverviewResponse(BaseModel):
    adapter: str
    display_name: str
    family: str
    launch_mode: str
    health_status: str
    health_detail: str = ""
    last_checked_at: str = ""
    endpoint_or_source: str
    model_name: str = ""
    role: str = ""
    task_capabilities: list[str] = Field(default_factory=list)
    formal_eval: bool = True
    capability_note: str = ""


class ModelListResponse(BaseModel):
    total: int
    items: list[ModelOverviewResponse] = Field(default_factory=list)


class DatasetOverviewItem(BaseModel):
    key: str
    name: str
    tier: str = ""
    prepared: bool = False
    ready: bool = False
    ready_reason: str = ""
    item_count: int = 0
    root_path: str = ""
    note: str = ""
    source: str = ""


class SystemOverviewResponse(BaseModel):
    generated_at: str
    project_root: str
    artifacts_dir: str
    app_db: str
    python_version: str
    platform: str
    torch: dict[str, Any]
    runtime: dict[str, Any]
    live_runtime_note: str = ""
    paper_result_environment_source_path: str = ""
    paper_result_environment_note: str = ""
    paper_result_environment: dict[str, Any] = Field(default_factory=dict)
    install_hint: str
    adapters: dict[str, dict[str, str]]
    models: list[ModelOverviewResponse] = Field(default_factory=list)
    supported_model_count: int = 0
    model_total_count: int = 0
    ready_model_count: int = 0
    online_models: list[str] = Field(default_factory=list)
    online_model_count: int = 0
    validated_models: list[str] = Field(default_factory=list)
    validated_model_count: int = 0
    scientific_quality_models: list[str] = Field(default_factory=list)
    scientific_quality_model_count: int = 0
    model_coverage: dict[str, Any] = Field(default_factory=dict)
    attacks: list[str]
    external_attack_status: dict[str, Any] = Field(default_factory=dict)
    datasets: list[DatasetOverviewItem] = Field(default_factory=list)
    formal_dataset_count: int = 0
    dataset_total_count: int = 0
    dataset_catalog: list[DatasetOverviewItem] = Field(default_factory=list)
    dataset_catalog_formal_count: int = 0
    dataset_catalog_total_count: int = 0
    source_documents: dict[str, Any]
    paper_repositories: list[dict[str, Any]]
    patch_registry: dict[str, Any]
    build_identity: dict[str, Any] = Field(default_factory=dict)
    latest_runs: list[RunSummary]
    latest_runs_note: str = ""
    latest_formal_runs: list[RunSummary] = Field(default_factory=list)
    latest_formal_runs_note: str = ""
    latest_primary_formal_runs: list[RunSummary] = Field(default_factory=list)
    latest_primary_formal_runs_note: str = ""
    primary_formal_runs_source_path: str = ""
    primary_formal_runs_source_kind: str = ""
    primary_formal_runs_artifact_index_path: str = ""
    latest_ablation_runs: list[RunSummary] = Field(default_factory=list)
    latest_ablation_runs_note: str = ""
    validation_snapshot: dict[str, Any] = Field(default_factory=dict)
    failing_primary_rows: list[dict[str, Any]] = Field(default_factory=list)
    validation_summary: dict[str, Any] = Field(default_factory=dict)


class ComplianceItem(BaseModel):
    id: str
    title: str
    status: Literal["ready", "partial", "missing"]
    evidence: str
    gap: str = ""


class PaperCoverageItem(BaseModel):
    paper: str
    repo: str
    repo_ready: bool
    impl_status: Literal["ready", "partial", "missing"]
    impl_evidence: str
    reproduction_fidelity: str = ""
    todo: str = ""


class UiPageItem(BaseModel):
    route: str
    page_file: str
    exists: bool


class ProjectStageResponse(BaseModel):
    stage: str
    final_stage: str
    criteria: dict[str, bool]


class ResultConformanceResponse(BaseModel):
    analysis_path: str = ""
    threshold_path: str = ""
    model_validation_path: str = ""
    available: bool = False
    passed: bool = False
    phase_count: int = 0
    row_count: int = 0
    e0_ok: bool = False
    e1_ok: bool = False
    e2_ok: bool = False
    e4_ok: bool = False
    defense_ok: bool = False
    adaptive_ok: bool = False
    fixation_ok: bool = False
    model_validation_ok: bool = False
    classic_conclusions: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)


class EngineeringViewsResponse(BaseModel):
    project_stage: ProjectStageResponse | None = None
    backend_interfaces: list[ComplianceItem] = Field(default_factory=list)
    ui_pages: list[UiPageItem] = Field(default_factory=list)


class SystemComplianceResponse(BaseModel):
    generated_at: str
    checklist_semantics: str = ""
    taskbook_items: list[ComplianceItem]
    paper_coverage: list[PaperCoverageItem]
    engineering_views: EngineeringViewsResponse | None = None
    result_conformance: ResultConformanceResponse = Field(default_factory=ResultConformanceResponse)
