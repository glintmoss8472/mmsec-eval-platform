// 文件说明：该文件属于前端业务工具，集中实现 api 相关逻辑。
import axios from "axios";

const API_BASE = import.meta.env.VITE_API_BASE || "/api/v1";

export const api = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
});

export type JobType =
  | "run_eval"
  | "run_benchmark"
  | "run_vlr"
  | "run_vqa"
  | "run_caption"
  | "train_advclip"
  | "generate_sample_assets"
  | "run_sweep"
  | "docs_ingest"
  | "dataset_prepare";

export type JobStatus = "queued" | "running" | "success" | "failed" | "cancelled";

export interface JobItem {
  id: string;
  job_type: string;
  status: JobStatus;
  created_at: string;
  started_at?: string;
  finished_at?: string;
  config_path: string;
  override_json?: string;
  run_id: string;
  error_code: string;
  error_message: string;
  benchmark_mode: boolean;
}

export interface JobLogResponse {
  id: number;
  level: string;
  message: string;
  ts: string;
}

export interface JobStageItem {
  stage_key: string;
  stage_label: string;
  state: "pending" | "queued" | "running" | "success" | "failed" | "cancelled";
  progress_percent: number;
  message: string;
  updated_at: string;
}

export interface JobProgressResponse {
  job_id: string;
  job_type: string;
  status: JobStatus;
  queue_position: number;
  elapsed_seconds: number;
  eta_seconds: number;
  estimated_ready_at: string;
  current_stage: string;
  progress_percent: number;
  progress_percent_semantics?: string;
  current_stage_message?: string;
  current_stage_units_done?: number;
  current_stage_units_total?: number;
  current_stage_progress_percent?: number;
  current_stage_updated_at?: string;
  stages: JobStageItem[];
  last_log: string;
  run_id: string;
}

export interface RunItem {
  run_id: string;
  created_at: string;
  task_kind: string;
  dataset_name: string;
  benchmark_tag: string;
  attack: string;
  attack_modality?: string;
  eval_scope?: string;
  mode: string;
  experiment_id: string;
  suite: string;
  suite_label: string;
  evidence_group: string;
  experiment_label: string;
  model_adapter: string;
  surrogate_model_adapter: string;
  victim_model_adapters: string[];
  asr: number;
  asr_attack: number;
  metric_label?: string;
  k_value?: number;
  retrieval_direction_scope?: string;
  victim_aggregation?: string;
  sample_pair_count?: number;
  metric_note?: string;
  risk_score: number;
  risk_level: string;
  risk_scenario: string;
  avg_l2: number;
  avg_linf?: number;
  clean_r1_mean?: number;
  attacked_r1_mean?: number;
  attack_drop_r1_mean?: number;
  clean_mean_rank?: number;
  attacked_mean_rank?: number;
  rank_delta_mean?: number;
  clean_accuracy?: number;
  attacked_accuracy?: number;
  answer_change_rate?: number;
  target_flip_rate?: number;
  semantic_preservation_rate?: number;
  caption_text_similarity?: number;
  object_jaccard?: number;
  semantic_preservation_score?: number;
  case_count?: number;
  evidence_sample_count?: number;
  evidence_confidence?: "low" | "medium" | "high" | string;
  evidence_note?: string;
  result_type?: "formal" | "debug" | string;
  result_type_note?: string;
  has_case_evidence?: boolean;
  artifact_evidence_status?: string;
  path: string;
  evidence_row_id?: string;
  image_branch_enabled?: boolean;
  text_branch_enabled?: boolean;
  text_edit_applied?: boolean;
  text_changed_ratio?: number | null;
  joint_execution_declared?: boolean;
  joint_execution_confirmed?: boolean;
  joint_execution_evidence_source?: string;
  joint_execution_basis?: string;
  joint_execution_note?: string;
  summary_path?: string;
  report_path?: string;
  archived_summary_path?: string;
  archived_report_path?: string;
  source_summary_path?: string;
  source_report_data_path?: string;
  source_report_path?: string;
  portable_report_data_path?: string;
  portable_report_path?: string;
  artifact_index_path?: string;
}


export interface RunAnalyticsResponse {
  total_runs: number;
  total_cases: number;
  avg_asr_attack?: number;
  formal_runs: number;
  debug_runs: number;
  high_risk_runs?: number;
  runs_with_case_evidence?: number;
  low_confidence_runs: number;
  task_groups: Array<Record<string, unknown>>;
  model_risk_groups?: Array<{
    model_adapter: string;
    count: number;
    avg_risk_score: number;
    max_risk_score: number;
    high_risk_count: number;
    medium_risk_count: number;
    low_risk_count: number;
    low_confidence_count: number;
    debug_count: number;
  }>;
  risk_distribution: Array<{ key: string; count: number }>;
  result_type_distribution: Array<{ key: string; count: number }>;
  confidence_distribution: Array<{ key: string; count: number }>;
  attack_matrix: Array<Record<string, unknown>>;
  latest_runs: RunItem[];
}

export interface RunOptionItem {
  key: string;
  value: string;
  count: number;
}

export interface RunOptionsResponse {
  task_kinds: RunOptionItem[];
  attacks: RunOptionItem[];
  risk_levels: RunOptionItem[];
  result_types: RunOptionItem[];
  confidences: RunOptionItem[];
}

export interface RunQueryParams {
  page?: number;
  page_size?: number;
  sort_by?: string;
  sort_dir?: string;
  task_kind?: string;
  dataset?: string;
  model?: string;
  attack?: string;
  result_type?: string;
  confidence?: string;
  search?: string;
  exclude_demo?: boolean;
}

export interface CaseIndexItem {
  run_id: string;
  sample_id: string;
  source_sample_id?: string;
  case_kind?: string;
  task_kind?: string;
  dataset_name?: string;
  benchmark_tag?: string;
  model_adapter?: string;
  attack?: string;
  risk_level?: string;
  risk_score?: number;
  judge_success?: boolean;
  retrieval_hit?: boolean;
  judge_reason?: string;
  text?: string;
  gt_image_id?: string;
  top5_image_ids?: string[];
  artifact_status?: string;
  created_at?: string;
  result_type?: string;
  evidence_confidence?: string;
  evidence_sample_count?: number;
  perturbation_l2?: number;
  perturbation_linf?: number;
}

export interface SampleAssetItem {
  asset_id: string;
  variant_id: string;
  run_id: string;
  sample_id: string;
  source_run_id?: string;
  source_case_id?: string;
  task_kind: string;
  dataset_name: string;
  benchmark_tag: string;
  model_adapter: string;
  attack: string;
  attack_scope: string;
  source_text: string;
  target_text: string;
  clean_image_ref: string;
  adv_image_ref: string;
  artifact_status: string;
  reusable_status: string;
  reusable_note: string;
  judge_success: boolean;
  risk_level: string;
  risk_score: number;
  perturbation_l2: number;
  perturbation_linf: number;
  semantic_score: number;
  created_at: string;
  linked_evaluation_count: number;
  used_count?: number;
  last_used_at?: string;
  metadata?: Record<string, unknown>;
  report_url: string;
  case_url: string;
}

export interface SampleAssetListResponse {
  total: number;
  page: number;
  page_size: number;
  items: SampleAssetItem[];
  summary: {
    total_assets: number;
    ready_assets: number;
    summary_only_assets: number;
    legacy_assets: number;
    task_distribution: Record<string, number>;
    attack_distribution: Record<string, number>;
    scope_distribution: Record<string, number>;
  };
  options: Record<string, Array<{ value: string; count: number }>>;
}

export interface SampleAssetBatchItem {
  batch_id: string;
  source_run_id: string;
  task_kind: string;
  dataset_name: string;
  benchmark_tag: string;
  model_adapter: string;
  attack: string;
  attack_scope: string;
  created_at: string;
  updated_at?: string;
  total_assets: number;
  ready_assets: number;
  callable_assets: number;
  summary_only_assets: number;
  legacy_assets: number;
  pending_evaluation_assets?: number;
  evidence_complete_count: number;
  evidence_integrity: number;
  successful_assets: number;
  avg_risk_score: number;
  avg_l2: number;
  avg_linf: number;
  used_count: number;
  batch_call_count?: number;
  sample_usage_count?: number;
  last_used_at?: string;
  asset_ids: string[];
  preview_assets: SampleAssetItem[];
  report_url: string;
  batch_status: "callable" | "partial" | "not_callable" | string;
}

export interface SampleAssetBatchListResponse {
  total: number;
  page: number;
  page_size: number;
  items: SampleAssetBatchItem[];
  summary: {
    total_batches: number;
    callable_batches: number;
    total_assets: number;
    ready_assets: number;
    callable_assets: number;
    pending_evaluation_assets?: number;
    summary_only_assets: number;
    legacy_assets: number;
    task_distribution: Record<string, number>;
    attack_distribution: Record<string, number>;
    scope_distribution: Record<string, number>;
  };
  options: Record<string, Array<{ value: string; count: number }>>;
}

export interface DatasetItem {
  name: string;
  root_path: string;
  prepared: boolean;
  ready?: boolean;
  ready_reason?: string;
  prepared_at: string;
  item_count: number;
  note: string;
}

export interface DatasetOverviewItem {
  key: string;
  name: string;
  tier?: string;
  prepared?: boolean;
  ready?: boolean;
  ready_reason?: string;
  item_count?: number;
  root_path?: string;
  note?: string;
  source?: string;
}

export type BootstrapState = "pending" | "seeding" | "warming" | "ready" | "degraded";

export interface BootstrapStep {
  name: string;
  state: "pending" | "running" | "success" | "failed" | "skipped";
  message: string;
  updated_at: string;
}

export interface BootstrapStatus {
  state: BootstrapState;
  started_at: string;
  updated_at: string;
  degraded_reason: string;
  steps: BootstrapStep[];
  artifacts: {
    docs_index: string;
    docs_snippets: string;
    seeded_runs: string[];
    seeded_data: string[];
  };
}

export interface ModelOverview {
  adapter: string;
  display_name: string;
  family: string;
  launch_mode: string;
  health_status: string;
  last_checked_at?: string;
  endpoint_or_source: string;
  model_name?: string;
  role?: string;
  task_capabilities?: string[];
  formal_eval?: boolean;
  capability_note?: string;
}

export type AttackRequirementState = "ready" | "missing" | "not_required" | "optional" | "unknown";

export interface AttackRequirementStatus {
  label: string;
  required: boolean;
  configured: boolean;
  exists: boolean;
  status: AttackRequirementState;
  path: string;
  note: string;
}

export interface ExternalAttackRuntimeStatus {
  attack_id: string;
  display_name: string;
  config_path: string;
  config_exists: boolean;
  command_template_configured: boolean;
  runnable: boolean;
  repo: AttackRequirementStatus;
  checkpoint: AttackRequirementStatus;
  target: AttackRequirementStatus;
  messages: string[];
}

export interface SystemOverview {
  generated_at: string;
  project_root: string;
  artifacts_dir: string;
  app_db: string;
  python_version: string;
  platform: string;
  torch: {
    installed: boolean;
    version?: string;
    cuda_version?: string | null;
    cuda_available?: boolean;
    device_count?: number;
    error?: string;
  };
  runtime: {
    current_device: string;
    cuda_required: boolean;
    strict_real: string;
  };
  live_runtime_note?: string;
  paper_result_environment_source_path?: string;
  paper_result_environment_note?: string;
  paper_result_environment?: Record<string, unknown> & {
    reference_kind?: string;
    captured_at?: string;
    source_artifact?: string;
    python_version?: string;
    platform?: string;
    torch?: {
      installed?: boolean;
      version?: string;
      cuda_version?: string | null;
      cuda_available?: boolean;
      device_count?: number;
    };
    runtime?: {
      current_device?: string;
      cuda_required?: boolean;
      strict_real?: string;
    };
  };
  install_hint: string;
  adapters: Record<string, { activation: string; model_name: string; role?: string; endpoint?: string }>;
  models?: ModelOverview[];
  supported_model_count?: number;
  model_total_count?: number;
  ready_model_count?: number;
  online_models?: string[];
  online_model_count?: number;
  validated_models?: string[];
  validated_model_count?: number;
  scientific_quality_models?: string[];
  scientific_quality_model_count?: number;
  model_coverage?: {
    integrated?: { count: number; models: string[]; semantics: string };
    online?: { count: number; models: string[]; semantics: string };
    engineering_validated?: { count: number; models: string[]; passed: boolean; semantics: string };
    scientific_quality?: { count: number; models: string[]; passed: boolean; semantics: string };
    validation_strategy?: {
      dataset_name: string;
      benchmark_attacks: string[];
      max_pairs: number;
      description: string;
    };
  };
  validation_summary?: Record<string, unknown> & {
    passed?: boolean;
  };
  attacks: string[];
  external_attack_status?: Record<string, ExternalAttackRuntimeStatus>;
  datasets: DatasetOverviewItem[];
  formal_dataset_count?: number;
  dataset_total_count?: number;
  dataset_catalog?: DatasetOverviewItem[];
  dataset_catalog_formal_count?: number;
  dataset_catalog_total_count?: number;
  source_documents: Record<string, unknown>;
  paper_repositories: Array<{
    name: string;
    path: string;
    exists: boolean;
    remote: string;
    commit: string;
  }>;
  patch_registry: Record<string, unknown>;
  build_identity?: {
    deployment_target: string;
    backend_version_stamp: string;
    version_source: string;
    backend_commit: string;
    runtime_transport?: string;
    runtime_context?: string;
    runtime_profile?: string;
    runtime_root?: string;
    frontend_index_exists: boolean;
    frontend_dist_fresh: boolean;
    frontend_index_built_at: string;
    frontend_inputs_latest_at: string;
    frontend_asset_refs: string[];
  };
  latest_runs: RunItem[];
  latest_runs_note?: string;
  latest_formal_runs?: RunItem[];
  latest_formal_runs_note?: string;
  latest_primary_formal_runs?: RunItem[];
  latest_primary_formal_runs_note?: string;
  primary_formal_runs_source_path?: string;
  primary_formal_runs_source_kind?: string;
  primary_formal_runs_artifact_index_path?: string;
  latest_ablation_runs?: RunItem[];
  latest_ablation_runs_note?: string;
  validation_snapshot?: {
    snapshot_id?: string;
    snapshot_generated_at?: string;
    summary_path?: string;
    stable_archive?: boolean;
    snapshot_passed?: boolean;
    live_job_in_progress?: boolean;
    stable_reference_note?: string;
  };
  failing_primary_rows?: Array<{
    model_adapter: string;
    attack: string;
    dataset_name: string;
    experiment_id: string;
    job_id: string;
    job_status: string;
    previous_failure_count: number;
    last_updated_at: string;
    error_message: string;
    blocking_reason: string;
    engineering_validated?: boolean;
    scientific_quality_ok?: boolean;
  }>;
}

export interface ComplianceItem {
  id: string;
  title: string;
  status: "ready" | "partial" | "missing";
  evidence: string;
  gap: string;
}

export interface PaperCoverageItem {
  paper: string;
  repo: string;
  repo_ready: boolean;
  impl_status: "ready" | "partial" | "missing";
  impl_evidence: string;
  reproduction_fidelity?: string;
  todo: string;
}

export interface UiPageItem {
  route: string;
  page_file: string;
  exists: boolean;
}

export interface SystemCompliance {
  generated_at: string;
  taskbook_items: ComplianceItem[];
  paper_coverage: PaperCoverageItem[];
  engineering_views?: {
    project_stage?: {
      stage: string;
      final_stage: string;
      criteria: Record<string, boolean>;
    };
    backend_interfaces: ComplianceItem[];
    ui_pages: UiPageItem[];
  };
  result_conformance: {
    analysis_path: string;
    threshold_path: string;
    model_validation_path: string;
    available: boolean;
    passed: boolean;
    phase_count: number;
    row_count: number;
    e0_ok: boolean;
    e1_ok: boolean;
    e2_ok: boolean;
    e4_ok: boolean;
    adaptive_ok: boolean;
    fixation_ok: boolean;
    model_validation_ok: boolean;
    classic_conclusions: string[];
    caveats: string[];
  };
}

/** 中文注释：实现 health 的核心流程，支撑前端业务工具中的业务语义和异常边界。 */
export async function health() {
  return (await api.get("/health")).data as {
    status: string;
    version: string;
    bootstrap_state: BootstrapState;
    degraded_reason: string;
  };
}

/** 中文注释：实现 listJobs 的核心流程，支撑前端业务工具中的业务语义和异常边界。 */
export async function listJobs(params: { page?: number; page_size?: number; status?: string } = {}) {
  return (await api.get("/jobs", { params })).data as { total: number; items: JobItem[] };
}

/** 中文注释：实现 createJob 的核心流程，支撑前端业务工具中的业务语义和异常边界。 */
export async function createJob(payload: {
  job_type: JobType;
  config_path: string;
  override?: Record<string, unknown>;
  benchmark_mode?: boolean;
  payload?: Record<string, unknown>;
}) {
  try {
    return (await api.post("/jobs", payload)).data as JobItem;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      const detail = typeof error.response?.data?.detail === "string" ? error.response?.data?.detail : "";
      if (detail) {
        throw new Error(detail);
      }
    }
    throw error;
  }
}

/** 中文注释：实现 listJobLogs 的核心流程，支撑前端业务工具中的业务语义和异常边界。 */
export async function listJobLogs(jobId: string, params: { page?: number; page_size?: number } = {}) {
  return (await api.get(`/jobs/${jobId}/logs`, { params })).data as { total: number; items: JobLogResponse[] };
}

/** 中文注释：实现 getJobProgress 的核心流程，支撑前端业务工具中的业务语义和异常边界。 */
export async function getJobProgress(jobId: string) {
  return (await api.get(`/jobs/${jobId}/progress`)).data as JobProgressResponse;
}

/** 中文注释：实现 cancelJob 的核心流程，支撑前端业务工具中的业务语义和异常边界。 */
export async function cancelJob(jobId: string) {
  return (await api.post(`/jobs/${jobId}/cancel`)).data as JobItem;
}

/** 中文注释：实现 listRuns 的核心流程，支撑前端业务工具中的业务语义和异常边界。 */
export async function listRuns(params: RunQueryParams = {}) {
  return (await api.get("/runs", { params })).data as { total: number; page: number; page_size: number; items: RunItem[] };
}

/** 中文注释：实现 getRunAnalytics 的核心流程，支撑前端业务工具中的业务语义和异常边界。 */
export async function getRunAnalytics(params: Omit<RunQueryParams, "page" | "page_size" | "sort_by" | "sort_dir"> = {}) {
  return (await api.get("/runs/analytics", { params })).data as RunAnalyticsResponse;
}

/** 中文注释：实现 getRunOptions 的核心流程，支撑前端业务工具中的业务语义和异常边界。 */
export async function getRunOptions(params: { exclude_demo?: boolean } = {}) {
  return (await api.get("/runs/options", { params })).data as RunOptionsResponse;
}

/** 中文注释：实现 listCaseIndex 的核心流程，支撑前端业务工具中的业务语义和异常边界。 */
export async function listCaseIndex(params: {
  page?: number;
  page_size?: number;
  sort_by?: string;
  sort_dir?: string;
  task_kind?: string;
  dataset?: string;
  model?: string;
  attack?: string;
  success?: string;
  risk_level?: string;
  result_type?: string;
  confidence?: string;
  artifact_status?: string;
  search?: string;
  exclude_demo?: boolean;
} = {}) {
  return (await api.get("/runs/cases", { params })).data as { total: number; items: CaseIndexItem[] };
}

/** 中文注释：实现 listSampleAssets 的核心流程，支撑前端业务工具中的业务语义和异常边界。 */
export async function listSampleAssets(params: {
  page?: number;
  page_size?: number;
  task_kind?: string;
  attack?: string;
  scope?: string;
  reusable_status?: string;
  model?: string;
  dataset?: string;
  search?: string;
} = {}) {
  return (await api.get("/samples", { params })).data as SampleAssetListResponse;
}


/** 中文注释：实现 listSampleAssetBatches 的核心流程，支撑前端业务工具中的业务语义和异常边界。 */
export async function listSampleAssetBatches(params: {
  page?: number;
  page_size?: number;
  task_kind?: string;
  attack?: string;
  scope?: string;
  reusable_status?: string;
  model?: string;
  dataset?: string;
  search?: string;
  sort_by?: string;
  sort_dir?: string;
  include_asset_ids?: boolean;
} = {}) {
  return (await api.get("/samples/batches", { params })).data as SampleAssetBatchListResponse;
}

/** 中文注释：实现 compareRuns 的核心流程，支撑前端业务工具中的业务语义和异常边界。 */
export async function compareRuns(runIds: string[]) {
  const joined = runIds.join(",");
  return (await api.get("/runs/compare", { params: { run_ids: joined } })).data as {
    run_ids: string[];
    compare: Record<string, unknown>;
  };
}

/** 中文注释：实现 getRunSummary 的核心流程，支撑前端业务工具中的业务语义和异常边界。 */
export async function getRunSummary(runId: string) {
  return (await api.get(`/runs/${runId}/summary`)).data as Record<string, unknown>;
}

/** 中文注释：实现 getRunReportData 的核心流程，支撑前端业务工具中的业务语义和异常边界。 */
export async function getRunReportData(runId: string) {
  return (await api.get(`/runs/${runId}/report-data`)).data as Record<string, unknown>;
}

/** 中文注释：实现 getRunCases 的核心流程，支撑前端业务工具中的业务语义和异常边界。 */
export async function getRunCases(runId: string, params: { page?: number; page_size?: number } = {}) {
  return (await api.get(`/runs/${runId}/cases`, { params })).data as { total: number; items: Record<string, unknown>[] };
}

/** 中文注释：实现 getCaseDetail 的核心流程，支撑前端业务工具中的业务语义和异常边界。 */
export async function getCaseDetail(runId: string, sampleId: string) {
  return (await api.get(`/runs/${runId}/cases/${sampleId}`)).data as {
    case_bundle: Record<string, unknown>;
    attack_debug: Record<string, unknown>;
  };
}

/** 中文注释：实现 listDatasets 的核心流程，支撑前端业务工具中的业务语义和异常边界。 */
export async function listDatasets() {
  return (await api.get("/datasets")).data as { items: DatasetItem[] };
}

/** 中文注释：实现 prepareDataset 的核心流程，支撑前端业务工具中的业务语义和异常边界。 */
export async function prepareDataset(payload: {
  name: "flickr30k" | "flickr1k" | "coco_subset";
  root_path?: string;
  image_dir?: string;
  split?: string;
  max_items?: number;
  download_annotations?: boolean;
  download_images?: boolean;
  captions_source?: string;
  auto_download?: boolean;
}) {
  return (await api.post("/datasets/prepare", payload)).data as JobItem;
}

/** 中文注释：实现 ingestDocs 的核心流程，支撑前端业务工具中的业务语义和异常边界。 */
export async function ingestDocs(configPath: string) {
  return (await api.post("/docs/ingest", { config_path: configPath })).data as JobItem;
}

/** 中文注释：实现 docsIndex 的核心流程，支撑前端业务工具中的业务语义和异常边界。 */
export async function docsIndex() {
  return (await api.get("/docs/index")).data as { items: Record<string, unknown>[] };
}

/** 中文注释：实现 docsSnippets 的核心流程，支撑前端业务工具中的业务语义和异常边界。 */
export async function docsSnippets() {
  return (await api.get("/docs/snippets")).data as { items: Record<string, unknown>[] };
}

/** 中文注释：实现 getBootstrapStatus 的核心流程，支撑前端业务工具中的业务语义和异常边界。 */
export async function getBootstrapStatus() {
  return (await api.get("/bootstrap/status")).data as BootstrapStatus;
}

/** 中文注释：实现 getBootstrapLogs 的核心流程，支撑前端业务工具中的业务语义和异常边界。 */
export async function getBootstrapLogs(limit = 200) {
  return (await api.get("/bootstrap/logs", { params: { limit } })).data as { items: { ts: string; level: string; message: string }[] };
}

/** 中文注释：实现 retryBootstrap 的核心流程，支撑前端业务工具中的业务语义和异常边界。 */
export async function retryBootstrap() {
  return (await api.post("/bootstrap/retry")).data as BootstrapStatus;
}

/** 中文注释：实现 getSystemOverview 的核心流程，支撑前端业务工具中的业务语义和异常边界。 */
export async function getSystemOverview() {
  return (await api.get("/system/overview")).data as SystemOverview;
}

/** 中文注释：实现 getSystemCompliance 的核心流程，支撑前端业务工具中的业务语义和异常边界。 */
export async function getSystemCompliance() {
  return (await api.get("/system/compliance")).data as SystemCompliance;
}

/** 中文注释：实现 runAssetUrl 的核心流程，支撑前端业务工具中的业务语义和异常边界。 */
export function runAssetUrl(runId: string, path: string) {
  const norm = path.replace(/\\/g, "/").replace(/^\/+/, "");
  return `${API_BASE}/runs/${runId}/assets/${norm}`;
}
