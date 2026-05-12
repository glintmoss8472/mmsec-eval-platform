// 文件说明：该文件属于前端业务工具，集中实现 validationEvidence 相关逻辑。
import type { JobItem, SystemOverview } from "./api";

export interface ValidationSnapshotView {
  snapshot_id?: string;
  snapshot_generated_at?: string;
  summary_path?: string;
  stable_archive?: boolean;
  snapshot_passed?: boolean;
  live_job_in_progress?: boolean;
  stable_reference_note?: string;
}

export interface ValidationBlockerRow {
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
}

/** 整理 `derive 验证 证据` 数据，把接口响应转换成页面可直接渲染的结构。 */
export function deriveValidationEvidence(
  overview: SystemOverview | undefined,
  _jobs: JobItem[] | undefined,
): { snapshot: ValidationSnapshotView | null; blockers: ValidationBlockerRow[] } {
  return {
    snapshot: overview?.validation_snapshot ?? null,
    blockers: overview?.failing_primary_rows ?? [],
  };
}
