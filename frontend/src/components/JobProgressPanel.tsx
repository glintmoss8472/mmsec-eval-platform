// 文件说明：该文件属于前端组件，集中实现 JobProgressPanel 相关逻辑。
import { useMemo } from "react";

import type { JobProgressResponse } from "../lib/api";
import { jobStageGlossaryIds, jobStatusGlossaryIds } from "../lib/glossaryRegistry";
import { formatJobStatus } from "../lib/uiLabels";
import { GlossaryLink } from "./GlossaryLink";
import { useDismissible } from "../hooks/useDismissible";

/** 格式化 `format duration`，统一页面展示文本和缺省值。 */
function formatDuration(seconds: number | undefined) {
  if (!Number.isFinite(seconds) || Number(seconds) <= 0) {
    return "正在估算";
  }
  const total = Math.round(Number(seconds));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const restSeconds = total % 60;
  if (hours > 0) return `${hours}小时${minutes}分${restSeconds}秒`;
  if (minutes > 0) return `${minutes}分${restSeconds}秒`;
  return `${restSeconds}秒`;
}

type JobProgressPanelProps = {
  progress?: JobProgressResponse;
  panelId?: string;
};

/** 渲染 `JobProgressPanel` 组件，组织该区域的数据读取、交互状态和可访问性标记。 */
export function JobProgressPanel({ progress, panelId }: JobProgressPanelProps) {
  const { visible, dismiss, restore } = useDismissible(panelId);
  const currentStage = useMemo(
    () => progress?.stages.find((item) => item.stage_key === progress.current_stage) ?? progress?.stages.find((item) => item.state === "running"),
    [progress],
  );

  if (panelId && !visible) {
    return (
      <section className="section-card p-5 dismissible-panel dismissible-panel-restore">
        <span>任务进度已隐藏</span>
        <button type="button" className="panel-restore-button" onClick={restore}>
          显示任务进度
        </button>
      </section>
    );
  }

  if (!progress) {
    return (
      <section className={`section-card p-5 ${panelId ? "dismissible-panel" : ""}`}>
        {panelId ? (
          <button type="button" className="panel-close-button" aria-label="关闭任务进度" title="关闭任务进度" onClick={dismiss}>
            ×
          </button>
        ) : null}
        <div className="panel-label">
          <GlossaryLink entryId="workflow-job-progress">当前任务进度</GlossaryLink>
        </div>
        <div className="mt-3 text-sm text-[var(--ink-soft)]">
          任务启动后，这里只显示答辩时真正需要讲的进度信息：阶段、耗时、预计完成时间和运行编号。
        </div>
      </section>
    );
  }

  const statusGlossaryId = jobStatusGlossaryIds[progress.status] ?? "status-running";
  return (
    <section className={`section-card p-5 ${panelId ? "dismissible-panel" : ""}`}>
      {panelId ? (
        <button type="button" className="panel-close-button" aria-label="关闭任务进度" title="关闭任务进度" onClick={dismiss}>
          ×
        </button>
      ) : null}
      <div className="workspace-header">
        <div>
          <div className="panel-label">
            <GlossaryLink entryId="workflow-job-progress">当前任务进度</GlossaryLink>
          </div>
          <h3 className="section-title mt-2">
            <GlossaryLink entryId="workflow-task-launch">任务状态与预计时间</GlossaryLink>
          </h3>
        </div>
        <span className="tag-chip">
          <GlossaryLink entryId={statusGlossaryId}>{formatJobStatus(progress.status)}</GlossaryLink>
        </span>
      </div>

      <div className="mt-4 progress-strip">
        <div style={{ width: `${Math.max(0, Math.min(100, progress.progress_percent))}%` }} />
      </div>
      <div className="mt-2 text-right text-xs font-semibold text-[var(--ink-dim)]">{progress.progress_percent}%</div>

      <div className="summary-list mt-4">
        <div className="summary-row">
          <span>
            <GlossaryLink entryId="workflow-current-stage">当前阶段</GlossaryLink>
          </span>
          <span className="summary-row-value">
            {currentStage ? <GlossaryLink entryId={jobStageGlossaryIds[currentStage.stage_key] ?? "workflow-job-progress"}>{currentStage.stage_label}</GlossaryLink> : "等待中"}
          </span>
        </div>
        <div className="summary-row">
          <span>
            <GlossaryLink entryId="workflow-queue-position">队列位置</GlossaryLink>
          </span>
          <span className="summary-row-value">{progress.queue_position > 0 ? `第 ${progress.queue_position} 位` : "已进入执行"}</span>
        </div>
        <div className="summary-row">
          <span>
            <GlossaryLink entryId="metric-elapsed-time">已耗时</GlossaryLink>
          </span>
          <span className="summary-row-value">{formatDuration(progress.elapsed_seconds)}</span>
        </div>
        <div className="summary-row">
          <span>
            <GlossaryLink entryId="metric-eta-time">预计剩余时间</GlossaryLink>
          </span>
          <span className="summary-row-value">{formatDuration(progress.eta_seconds)}</span>
        </div>
        <div className="summary-row">
          <span>
            <GlossaryLink entryId="metric-estimated-ready-at">预计完成时间</GlossaryLink>
          </span>
          <span className="summary-row-value">{progress.estimated_ready_at || "正在估算"}</span>
        </div>
        <div className="summary-row">
          <span>
            <GlossaryLink entryId="workflow-run-id">运行编号</GlossaryLink>
          </span>
          <span className="summary-row-value">{progress.run_id || "尚未生成"}</span>
        </div>
      </div>
      <div className="notice-banner mt-4">
        {currentStage
          ? `当前阶段为“${currentStage.stage_label}”，页面不再展示日志和调试步骤，只保留答辩需要的进度口径。`
          : "当前任务正在等待调度，页面不再展示日志和调试步骤，只保留答辩需要的进度口径。"}
      </div>
    </section>
  );
}
