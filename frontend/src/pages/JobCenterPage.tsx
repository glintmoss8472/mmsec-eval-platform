// 文件说明：该文件属于前端页面，集中实现 JobCenterPage 相关逻辑。
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";

import { CheckIcon, ClipboardIcon, ClockIcon, GovMetric, GovPanel } from "../components/GovCards";
import { cancelJob, getJobProgress, listJobLogs, listJobs } from "../lib/api";
import { formatAdapterName, formatAttackName, formatBackendMessage, formatJobStatus, formatJobType, formatLogMessage } from "../lib/uiLabels";

type StageView = {
  stage_key: string;
  stage_label: string;
  state: string;
  progress_percent: number;
  message: string;
  flow_message?: string;
};

/** 判断 `是否 running` 状态，支撑页面分支渲染或按钮可用性。 */
function isRunning(status?: string) {
  return status === "running" || status === "queued";
}

/** 生成 `eta label` 展示值，统一页面标签、颜色和缺省文案。 */
function etaLabel(status?: string, seconds?: number) {
  if (status === "success") return "已完成";
  if (status === "failed" || status === "cancelled") return "已结束";
  if (!seconds || seconds <= 0) return "等待估算";
  return `${Math.max(1, Math.round(seconds / 60))} 分钟`;
}

/** 转换 `parse override` 输入，保证接口数据能被页面安全使用。 */
function parseOverride(raw?: string): Record<string, unknown> {
  if (!raw) return {};
  try {
    const parsed = JSON.parse(raw);
    return typeof parsed === "object" && parsed !== null ? parsed as Record<string, unknown> : {};
  } catch {
    return {};
  }
}

/** 整理 `nested record` 前端辅助逻辑，保持数据转换和展示口径一致。 */
function nestedRecord(value: unknown): Record<string, unknown> {
  return typeof value === "object" && value !== null ? value as Record<string, unknown> : {};
}

/** 整理 `infer 攻击` 前端辅助逻辑，保持数据转换和展示口径一致。 */
function inferAttack(job?: { config_path?: string; override_json?: string }) {
  const override = parseOverride(job?.override_json);
  const attack = nestedRecord(override.plugins).attack || nestedRecord(override.attack).name;
  if (attack) return String(attack);
  const config = String(job?.config_path || "").toLowerCase();
  for (const key of ["advedm_plus", "advedm", "advclip", "mifgsm", "nifgsm", "difgsm", "tifgsm", "dtmifgsm", "vmifgsm", "vnifgsm", "fgsm", "bim", "pgd", "tmm", "cw"]) {
    if (config.includes(key)) return key;
  }
  return "";
}

/** 判断 `是否 样本 生成式评测 only 任务` 状态，支撑页面分支渲染或按钮可用性。 */
function isSampleGenerationOnlyJob(job?: { job_type?: string; override_json?: string }): boolean {
  return workflowType(job) === "sample_generation_only" || job?.job_type === "generate_sample_assets";
}

/** 整理 `infer 模型` 前端辅助逻辑，保持数据转换和展示口径一致。 */
function inferModel(job?: { config_path?: string; job_type?: string; override_json?: string }) {
  if (isSampleGenerationOnlyJob(job)) return "";
  const override = parseOverride(job?.override_json);
  const runner = nestedRecord(override.runner);
  const victims = Array.isArray(runner.victim_model_adapters) ? runner.victim_model_adapters : [];
  if (victims.length && victims[0]) return String(victims[0]);
  if (runner.victim_model_adapter) return String(runner.victim_model_adapter);
  if (runner.surrogate_model_adapter) return String(runner.surrogate_model_adapter);
  const config = String(job?.config_path || "").toLowerCase();
  if (config.includes("blip")) return "blip_itm";
  if (config.includes("vilt")) return "vilt_itm";
  if (config.includes("clip")) return "clip_hf";
  return "";
}

/** 整理 `infer current victim` 前端辅助逻辑，保持数据转换和展示口径一致。 */
function inferCurrentVictim(message?: string): string {
  const text = String(message || "");
  const match = text.match(/(openai_[a-z0-9_]+|clip_hf|blip_itm|vilt_itm)/i);
  return match ? match[1] : "";
}

/** 整理 `infer 样本 total` 前端辅助逻辑，保持数据转换和展示口径一致。 */
function inferSampleTotal(
  job?: { override_json?: string },
  stages?: Array<{ message?: string }>,
): number {
  for (const stage of stages ?? []) {
    const match = String(stage.message || "").match(/共纳入\s*(\d+)\s*条样本/);
    if (match) return Number(match[1]) || 0;
  }
  const override = parseOverride(job?.override_json);
  const dataset = nestedRecord(override.dataset);
  const runner = nestedRecord(override.runner);
  return Number(dataset.max_items || runner.max_samples || 0) || 0;
}

/** 整理 `界面 任务 名称` 前端辅助逻辑，保持数据转换和展示口径一致。 */
function uiTaskName(job?: { job_type?: string; override_json?: string }, model?: string, attack?: string): string {
  if (!job) return "暂无任务";
  const override = parseOverride(job.override_json);
  const extra = nestedRecord(override.extra);
  const configured = String(extra.ui_task_name || "").trim();
  if (configured) return configured;
  if (model && attack) return `${formatAdapterName(model)} ${formatAttackName(attack)}`;
  if (model) return `${formatAdapterName(model)}测评`;
  if (attack) return `${formatAttackName(attack)}测评`;
  return formatJobType(job.job_type || "run_eval");
}

/** 整理 `workflow 类型` 前端辅助逻辑，保持数据转换和展示口径一致。 */
function workflowType(job?: { override_json?: string }): string {
  const override = parseOverride(job?.override_json);
  const extra = nestedRecord(override.extra);
  return String(extra.workflow_type || "");
}

/** 整理 `infer workflow` 前端辅助逻辑，保持数据转换和展示口径一致。 */
function inferWorkflow(job?: { job_type?: string; override_json?: string }): string {
  if (isSampleGenerationOnlyJob(job)) return "待测评样本生成任务";
  const workflow = workflowType(job);
  if (workflow === "sample_generation") return "样本生成任务";
  if (workflow === "asset_evaluation") return "对抗样本集真实复测任务";
  if (workflow === "generate_and_evaluate") return "即时生成并测评";
  return "自动化测评任务";
}

/** 整理 `指标 任务 名称` 前端辅助逻辑，保持数据转换和展示口径一致。 */
function metricTaskName(value: string): string {
  const text = String(value || "").trim();
  const lower = text.toLowerCase();
  if (text === "多模态模型综合安全测评") return "多模态综合测评";
  if (lower.includes("caption") || text.includes("图像描述")) return text.includes("复核") ? "图像描述复核" : "图像描述测评";
  if (lower.includes("vqa") || text.includes("视觉问答")) return text.includes("复核") ? "视觉问答复核" : "视觉问答测评";
  if (lower.includes("vlr") || text.includes("图文检索")) return text.includes("复核") ? "图文检索复核" : "图文检索测评";
  if (text.length > 12) return `${text.slice(0, 8)}...`;
  return text || "暂无任务";
}

/** 生成 `状态 详情 label` 展示值，统一页面标签、颜色和缺省文案。 */
function statusDetailLabel(status?: string): string {
  if (status === "failed") return "失败原因：";
  if (status === "cancelled") return "取消原因：";
  if (status === "success") return "执行结果：";
  return "当前正在执行：";
}

/** 整理 `状态 详情 message` 前端辅助逻辑，保持数据转换和展示口径一致。 */
function statusDetailMessage(
  status: string | undefined,
  currentStageMessage: string | undefined,
  selectedJob?: { run_id?: string; error_message?: string },
  lastLog?: string,
): string {
  if (status === "success") {
    return selectedJob?.run_id ? `任务已完成，运行编号：${selectedJob.run_id}` : "任务已完成。";
  }
  if (status === "failed") {
    return formatBackendMessage(selectedJob?.error_message || lastLog || currentStageMessage || "任务失败，后端未返回具体错误。");
  }
  if (status === "cancelled") {
    return "任务已取消。";
  }
  return formatBackendMessage(currentStageMessage || selectedJob?.error_message || "等待后端状态");
}

/** 整理 `find stage state` 前端辅助逻辑，保持数据转换和展示口径一致。 */
function findStageState(stages: Array<{ stage_key?: string; state?: string; progress_percent?: number }>, key: string): string {
  return stages.find((stage) => stage.stage_key === key)?.state || "pending";
}

/** 整理 `phase state` 前端辅助逻辑，保持数据转换和展示口径一致。 */
function phaseState(
  stages: Array<{ stage_key?: string; state?: string; progress_percent?: number }>,
  keys: string[],
  fallback: string,
): string {
  const values = keys.map((key) => findStageState(stages, key));
  if (values.some((state) => state === "failed")) return "failed";
  if (values.some((state) => state === "cancelled")) return "cancelled";
  if (values.some((state) => state === "running")) return "running";
  if (values.every((state) => state === "success")) return "success";
  return fallback;
}

/** 整理 `phase 进度` 前端辅助逻辑，保持数据转换和展示口径一致。 */
function phaseProgress(stages: Array<{ stage_key?: string; progress_percent?: number }>, keys: string[], fallback = 0): number {
  const values = keys
    .map((key) => Number(stages.find((stage) => stage.stage_key === key)?.progress_percent))
    .filter((value) => Number.isFinite(value));
  if (!values.length) return fallback;
  return Math.round(values.reduce((sum, value) => sum + value, 0) / values.length);
}

/** 整理 `状态 by 任务` 前端辅助逻辑，保持数据转换和展示口径一致。 */
function statusByJob(status?: string): string {
  if (status === "success") return "success";
  if (status === "failed") return "failed";
  if (status === "cancelled") return "cancelled";
  return "pending";
}

/** 判断 `是否 生成式评测 任务` 状态，支撑页面分支渲染或按钮可用性。 */
function isGenerationJob(jobType?: string): boolean {
  return ["run_vqa", "run_caption"].includes(String(jobType || ""));
}

/** 整理 `phase 进度 所属 state` 前端辅助逻辑，保持数据转换和展示口径一致。 */
function phaseProgressForState(
  stages: Array<{ stage_key?: string; progress_percent?: number }>,
  keys: string[],
  state: string,
  fallback = 0,
): number {
  if (state === "success") return 100;
  return phaseProgress(stages, keys, fallback);
}

/** 整理 `phase state with failure` 前端辅助逻辑，保持数据转换和展示口径一致。 */
function phaseStateWithFailure(
  stages: Array<{ stage_key?: string; state?: string; progress_percent?: number }>,
  keys: string[],
  fallback: string,
  jobStatus: string | undefined,
  currentStage: string | undefined,
): string {
  if (jobStatus === "success") return "success";
  const state = phaseState(stages, keys, fallback);
  const terminalFailed = stages.some((stage) => stage.stage_key === "completed" && stage.state === "failed");
  if (jobStatus === "failed" && terminalFailed && currentStage && keys.includes(currentStage)) {
    return "failed";
  }
  return state;
}

/** 构建 `build display stages` 结构，供页面渲染或测试断言复用。 */
function buildDisplayStages(
  stages: Array<{ stage_key?: string; state?: string; progress_percent?: number }>,
  jobStatus: string | undefined,
  currentStage: string | undefined,
  jobType?: string,
  workflow?: string,
): StageView[] {
  const generationJob = isGenerationJob(jobType);
  const assetJob = workflow === "asset_evaluation";
  const sampleGenerationOnly = workflow === "sample_generation_only" || jobType === "generate_sample_assets";
  const dataKeys = ["queued", "model_preflight", "config_validation", "dataset_loading"];
  if (sampleGenerationOnly) {
    const sampleKeys = ["attack_execution"];
    const assetKeys = ["result_aggregation"];
    const doneKeys = ["completed"];
    const dataState = phaseStateWithFailure(stages, dataKeys, currentStage === "dataset_loading" ? "running" : "pending", jobStatus, currentStage);
    const sampleState = phaseStateWithFailure(stages, sampleKeys, currentStage === "attack_execution" ? "running" : "pending", jobStatus, currentStage);
    const assetState = phaseStateWithFailure(stages, assetKeys, currentStage === "result_aggregation" ? "running" : "pending", jobStatus, currentStage);
    const doneState = jobStatus === "success"
      ? "success"
      : phaseStateWithFailure(stages, doneKeys, currentStage === "completed" ? "running" : "pending", jobStatus, currentStage);

    return [
      {
        stage_key: "data_prepare",
        stage_label: "数据准备",
        state: dataState,
        progress_percent: phaseProgressForState(stages, dataKeys, dataState),
        message: "校验配置、来源数据集和攻击生成所需代理模型。",
        flow_message: "配置与数据",
      },
      {
        stage_key: "sample_generation",
        stage_label: "样本生成",
        state: sampleState,
        progress_percent: phaseProgressForState(stages, sampleKeys, sampleState),
        message: "生成原始图像、对抗图像、攻击参数和案例证据。",
        flow_message: "生成待测评样本",
      },
      {
        stage_key: "asset_ingestion",
        stage_label: "样本入库",
        state: assetState,
        progress_percent: phaseProgressForState(stages, assetKeys, assetState),
        message: "写入待测评样本资产，不计算模型风险。",
        flow_message: "写入样本资产",
      },
      {
        stage_key: "completed",
        stage_label: "完成",
        state: doneState,
        progress_percent: phaseProgressForState(stages, doneKeys, doneState),
        message: "生成批次已进入待测评状态，选择受测模型后再生成风险、报告和案例判断。",
        flow_message: "等待后续测评",
      },
    ];
  }
  const sampleKeys = assetJob ? ["dataset_loading"] : ["attack_execution"];
  const autoEvalKeys = generationJob ? ["attack_execution"] : ["victim_evaluation"];
  const metricKeys = ["result_aggregation"];
  const reportKeys = ["report_writing"];
  const dataState = phaseStateWithFailure(stages, dataKeys, currentStage === "dataset_loading" ? "running" : "pending", jobStatus, currentStage);
  const sampleState = phaseStateWithFailure(stages, sampleKeys, currentStage === "attack_execution" ? "running" : "pending", jobStatus, currentStage);
  const autoEvalState = phaseStateWithFailure(stages, autoEvalKeys, autoEvalKeys.includes(String(currentStage || "")) ? "running" : statusByJob(jobStatus), jobStatus, currentStage);
  const metricState = phaseStateWithFailure(stages, metricKeys, currentStage === "result_aggregation" ? "running" : "pending", jobStatus, currentStage);
  const reportState = jobStatus === "success"
    ? "success"
    : phaseStateWithFailure(stages, reportKeys, currentStage === "report_writing" ? "running" : "pending", jobStatus, currentStage);

  return [
    {
      stage_key: "data_prepare",
      stage_label: "数据准备",
      state: dataState,
      progress_percent: phaseProgressForState(stages, dataKeys, dataState),
      message: "校验模型、配置与数据集。",
      flow_message: "校验模型与数据",
    },
    {
      stage_key: "sample_generation",
      stage_label: assetJob ? "样本调取" : "样本生成",
      state: sampleState,
      progress_percent: phaseProgressForState(stages, sampleKeys, sampleState),
      message: assetJob ? "调取已入库原始图像与对抗图像，不重新执行攻击生成。" : generationJob ? "生成对抗图片与可视化证据。" : "生成并管理对抗样本。",
      flow_message: assetJob ? "调取样本资产" : "生成对抗样本",
    },
    {
      stage_key: "auto_evaluation",
      stage_label: "自动测评",
      state: autoEvalState,
      progress_percent: phaseProgressForState(stages, autoEvalKeys, autoEvalState),
      message: "分析模型在正常输入与对抗输入下的输出差异。",
      flow_message: "分析输出差异",
    },
    {
      stage_key: "metric_aggregation",
      stage_label: "指标统计",
      state: metricState,
      progress_percent: phaseProgressForState(stages, metricKeys, metricState),
      message: "统计攻击成功率与安全风险指标。",
      flow_message: "量化安全风险",
    },
    {
      stage_key: "report_writing",
      stage_label: "报告生成",
      state: reportState,
      progress_percent: phaseProgressForState(stages, reportKeys, reportState),
      message: "生成结果统计、可视化与证据报告。",
      flow_message: "生成可视化报告",
    },
  ];
}

/** 渲染 `JobCenterPage` 组件，组织该区域的数据读取、交互状态和可访问性标记。 */
export default function JobCenterPage() {
  const qc = useQueryClient();
  const [activeJob, setActiveJob] = useState("");
  const jobsQ = useQuery({ queryKey: ["jobs-monitor"], queryFn: () => listJobs({ page: 1, page_size: 100 }), refetchInterval: 1800 });
  const isInitialJobsLoading = jobsQ.isLoading && !jobsQ.data;
  const jobs = jobsQ.data?.items ?? [];
  const current = useMemo(() => jobs.find((item) => isRunning(item.status)) ?? jobs[0], [jobs]);
  const selectedJob = jobs.find((item) => item.id === activeJob) ?? current;

  useEffect(() => {
    if (!activeJob && current?.id) setActiveJob(current.id);
  }, [activeJob, current?.id]);

  const progressQ = useQuery({
    queryKey: ["job-progress", activeJob],
    queryFn: () => getJobProgress(activeJob),
    enabled: !!activeJob,
    refetchInterval: 1800,
  });
  const logsQ = useQuery({
    queryKey: ["job-logs", activeJob],
    queryFn: () => listJobLogs(activeJob, { page: 1, page_size: 80 }),
    enabled: !!activeJob,
    refetchInterval: 1800,
  });

  const cancel = useMutation({
    mutationFn: cancelJob,
    onSuccess: () => qc.invalidateQueries({ queryKey: ["jobs-monitor"] }),
  });

  const progress = progressQ.data?.progress_percent ?? (selectedJob?.status === "success" ? 100 : 0);
  const progressLabel = selectedJob?.status === "failed"
    ? "失败"
    : selectedJob?.status === "cancelled"
      ? "已取消"
      : `${Math.round(progress)}%`;
  const stages = progressQ.data?.stages?.length
    ? progressQ.data.stages
    : selectedJob
      ? [{ stage_key: selectedJob.status, stage_label: formatJobStatus(selectedJob.status), state: selectedJob.status, progress_percent: progress, message: selectedJob.error_message || "等待后端进度接口返回阶段明细。", updated_at: "" }]
      : [];
  const selectedWorkflowType = workflowType(selectedJob);
  const displayStages = buildDisplayStages(stages, selectedJob?.status, progressQ.data?.current_stage, selectedJob?.job_type, selectedWorkflowType);
  const inferredTotal = inferSampleTotal(selectedJob, stages);
  const totalCount = progressQ.data?.current_stage_units_total || inferredTotal;
  const completedCount = selectedJob?.status === "success"
    ? totalCount
    : progressQ.data?.current_stage_units_done ?? 0;
  const completedLabel = progressQ.data?.current_stage === "victim_evaluation" && totalCount > inferredTotal ? "已处理图文配对" : "已处理样本";
  const pairTotal = progressQ.data?.current_stage === "victim_evaluation" && totalCount > inferredTotal ? totalCount : 0;

  const hasTrendData = completedCount > 0;
  const trendScale = Math.max(totalCount, completedCount, 1);
  const trend = hasTrendData
    ? [0, Math.round(completedCount * 0.25), Math.round(completedCount * 0.5), Math.round(completedCount * 0.75), completedCount].map((value) => Math.min(trendScale, Number(value)))
    : [];
  const currentModel = inferCurrentVictim(progressQ.data?.current_stage_message) || inferModel(selectedJob);
  const currentAttack = inferAttack(selectedJob);
  const workflowLabel = inferWorkflow(selectedJob);
  const sampleGenerationOnly = isSampleGenerationOnlyJob(selectedJob);
  const flowPanelTitle = sampleGenerationOnly ? "样本生成流程进度" : "测评流程进度";
  const currentModelLabel = sampleGenerationOnly ? "不校验" : currentModel ? formatAdapterName(currentModel) : "由配置文件决定";
  const currentMessage = statusDetailMessage(selectedJob?.status, progressQ.data?.current_stage_message, selectedJob, progressQ.data?.last_log);
  const taskTitle = uiTaskName(selectedJob, currentModel, currentAttack);

  return (
    <div className="gov-stack">
      <div className="gov-metric-grid three">
        <GovMetric title={isRunning(selectedJob?.status) ? "当前任务" : "最近任务"} value={isInitialJobsLoading ? "加载中..." : metricTaskName(taskTitle)} tone="blue" icon={<ClipboardIcon />} />
        <GovMetric title="总体进度" value={progressLabel} tone={selectedJob?.status === "failed" ? "red" : "green"} icon={<CheckIcon />} />
        <GovMetric title="预计剩余时间" value={etaLabel(selectedJob?.status, progressQ.data?.eta_seconds)} tone="blue" icon={<ClockIcon />} />
      </div>

      <div className="gov-monitor-grid">
        <GovPanel title={flowPanelTitle} className="wide">
          <div className="gov-progress-flow">
            {displayStages.map((stage, idx) => (
              <div key={stage.stage_key || stage.stage_label} className={`gov-flow-step ${stage.state}`}>
                <span>{stage.state === "success" ? <CheckIcon /> : idx + 1}</span>
                <strong>{stage.stage_label}</strong>
                <em>{formatBackendMessage(stage.flow_message || stage.message || formatJobStatus(stage.state))}</em>
              </div>
            ))}
          </div>
        </GovPanel>

        <GovPanel title="当前状态说明">
          <div className="gov-status-list">
            <div className={selectedJob?.status === "failed" ? "danger" : ""}><span>{statusDetailLabel(selectedJob?.status)}</span><strong>{currentMessage}</strong></div>
            <div><span>任务类别</span><strong>{workflowLabel}</strong></div>
            <div><span>{completedLabel}</span><strong>{completedCount} / {totalCount}</strong></div>
            {pairTotal && inferredTotal ? <div><span>基准样本条数</span><strong>{inferredTotal} 条样本，形成 {pairTotal} 个图文配对</strong></div> : null}
            <div><span>{sampleGenerationOnly ? "校验模型" : "当前模型"}</span><strong>{currentModelLabel}</strong></div>
            <div><span>当前攻击方式</span><strong>{currentAttack ? formatAttackName(currentAttack) : "由配置文件决定"}</strong></div>
          </div>
        </GovPanel>
      </div>

      <div className="gov-monitor-bottom">
        <GovPanel title="样本处理进度趋势">
          {hasTrendData ? (
            <div className="gov-mini-line" style={{ gridTemplateColumns: `repeat(${trend.length}, minmax(0, 1fr))` }}>
              {trend.map((value, idx) => (
                <i key={`${value}-${idx}`} style={{ height: `${Math.max(6, (value / trendScale) * 100)}%` }}>
                  <span>{value}</span>
                </i>
              ))}
            </div>
          ) : (
            <div className="gov-empty-state-chart">
              当前任务还没有产生样本进度点；提交测评后这里会显示已处理样本或图文配对的增长趋势。
            </div>
          )}
        </GovPanel>

        <GovPanel title="阶段任务明细" className="wide">
          <div className="gov-table-wrap">
            <table className="gov-table att-stage-table">
              <thead>
                <tr><th>阶段名称</th><th>状态</th><th>进度</th><th>说明</th></tr>
              </thead>
              <tbody>
                {displayStages.map((stage) => (
                  <tr key={`row-${stage.stage_key || stage.stage_label}`}>
                    <td>{stage.stage_label}</td>
                    <td>{formatJobStatus(stage.state)}</td>
                    <td>{Math.round(stage.progress_percent || 0)}%</td>
                    <td>{formatBackendMessage(stage.message || "-")}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </GovPanel>

        <GovPanel title="实时运行记录">
          <div className="gov-log-list">
            {(logsQ.data?.items?.length ? logsQ.data.items : []).slice(0, 8).map((log) => (
              <div key={log.id}><time>{new Date(log.ts).toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" })}</time><i />{formatLogMessage(log.message)}</div>
            ))}
            {!logsQ.data?.items?.length ? (
              <div><time>未记录时间</time><i />暂无后端日志记录。</div>
            ) : null}
          </div>
          <div className="gov-job-actions">
            {jobs.slice(0, 4).map((job) => (
              <button key={job.id} type="button" className={job.id === activeJob ? "active" : ""} onClick={() => setActiveJob(job.id)}>
                {job.id.slice(0, 8)} · {formatJobStatus(job.status)}
              </button>
            ))}
            {activeJob && isRunning(selectedJob?.status) ? <button type="button" onClick={() => cancel.mutate(activeJob)}>取消当前任务</button> : null}
          </div>
        </GovPanel>
      </div>
    </div>
  );
}
