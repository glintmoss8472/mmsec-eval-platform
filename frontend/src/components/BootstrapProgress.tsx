import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { getBootstrapLogs, getBootstrapStatus, retryBootstrap } from "../lib/api";
import { formatBootstrapState, formatLogLevel, formatLogMessage, formatWrapped } from "../lib/uiLabels";

const STEP_LABEL: Record<string, string> = {
  seed_sync: "种子同步",
  seed_docs: "文档索引",
  seed_runs: "演示运行",
  seed_data: "演示数据",
  queue_docs_ingest: "文档任务",
  queue_dataset_prepare: "数据任务",
  queue_benchmark_demo: "演示基准",
  queue_benchmark_public: "公开基准",
  model_warmup: "模型预热",
};

const STEP_STATE_LABEL: Record<string, string> = {
  pending: "等待中",
  running: "进行中",
  success: "已完成",
  failed: "失败",
  skipped: "已跳过",
};

function progressValue(steps: { state: string }[]) {
  if (!steps.length) return 0;
  const score = steps.reduce((acc, step) => {
    if (step.state === "success" || step.state === "skipped") return acc + 1;
    if (step.state === "running") return acc + 0.5;
    return acc;
  }, 0);
  return Math.round((score / steps.length) * 100);
}

export function BootstrapProgress() {
  const qc = useQueryClient();
  const statusQ = useQuery({
    queryKey: ["bootstrap-status"],
    queryFn: getBootstrapStatus,
    refetchInterval: 2500,
  });
  const logsQ = useQuery({
    queryKey: ["bootstrap-logs"],
    queryFn: () => getBootstrapLogs(8),
    refetchInterval: 2500,
  });

  const retry = useMutation({
    mutationFn: retryBootstrap,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["bootstrap-status"] });
      qc.invalidateQueries({ queryKey: ["bootstrap-logs"] });
      qc.invalidateQueries({ queryKey: ["jobs-overview"] });
    },
  });

  const status = statusQ.data;
  const steps = status?.steps ?? [];
  const progress = progressValue(steps);

  return (
    <section className="section-card">
      <div className="workspace-header">
        <div>
          <div className="panel-label">环境准备</div>
          <h3 className="section-title mt-2">开箱预热进度</h3>
          <div className="section-subtitle">
            {formatBootstrapState(status?.state ?? "pending")}
            {status?.degraded_reason ? ` / ${formatWrapped("降级原因", status.degraded_reason)}` : ""}
          </div>
        </div>
        <button className="action-button action-button-secondary" onClick={() => retry.mutate()}>
          重新尝试
        </button>
      </div>

      <div className="mt-4 progress-strip">
        <div style={{ width: `${progress}%` }} />
      </div>
      <div className="mt-2 text-right text-xs font-semibold text-[var(--ink-dim)]">{progress}%</div>

      <div className="progress-grid mt-4 md:grid-cols-2 xl:grid-cols-3">
        {steps.map((step) => (
          <div key={step.name} className="step-pill">
            <div className="text-sm font-semibold text-[var(--ink)]">{STEP_LABEL[step.name] ?? formatWrapped("步骤标识", step.name)}</div>
            <div className="step-pill-state">{STEP_STATE_LABEL[step.state] ?? formatWrapped("步骤状态", step.state)}</div>
            {step.message ? <div className="mt-2 text-xs text-[var(--danger)]">{formatLogMessage(step.message)}</div> : null}
          </div>
        ))}
      </div>

      <div className="mt-5">
        <div className="panel-label">预热日志</div>
        <div className="log-list mt-3">
          {(logsQ.data?.items ?? []).map((item, index) => (
            <div key={`${item.ts}-${index}`} className="log-row">
              <span>{item.ts.split("T")[1]?.split(".")[0] || item.ts}</span>
              <span className={item.level === "error" ? "log-level-error" : item.level === "warn" ? "log-level-warn" : "log-level-info"}>
                [{formatLogLevel(item.level)}]
              </span>
              <span className="break-words text-[var(--ink)]">{formatLogMessage(item.message)}</span>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
