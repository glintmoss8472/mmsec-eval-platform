// 文件说明：该文件属于前端页面，集中实现 ReportDetailPage 相关逻辑。
import ReactECharts from "echarts-for-react";
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link, useParams } from "react-router-dom";

import { getRunCases, getRunReportData, getRunSummary } from "../lib/api";
import {
  formatAdapterName,
  formatAttackName,
  formatEvalScope,
  formatFeatureMethod,
  formatJobType,
  formatJudgeReason,
  formatModeName,
  formatNormLabel,
  formatPaperStatus,
  formatProjectionGroupName,
  formatRecallLabel,
  formatRiskDimension,
  formatRiskLevel,
  formatRunDatasetName,
  formatWrapped,
} from "../lib/uiLabels";

/** 中文注释：实现 asRecord 的核心流程，支撑前端页面中的业务语义和异常边界。 */
function asRecord(value: unknown): Record<string, unknown> {
  return typeof value === "object" && value !== null ? (value as Record<string, unknown>) : {};
}

/** 中文注释：实现 asRows 的核心流程，支撑前端页面中的业务语义和异常边界。 */
function asRows(value: unknown): Array<Record<string, unknown>> {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.map((x) => asRecord(x));
}

/** 中文注释：实现 asNum 的核心流程，支撑前端页面中的业务语义和异常边界。 */
function asNum(value: unknown): number {
  const n = Number(value);
  return Number.isFinite(n) ? n : 0;
}

/** 中文注释：实现 parseKs 的核心流程，支撑前端页面中的业务语义和异常边界。 */
function parseKs(value: unknown): number[] {
  if (!Array.isArray(value)) {
    return [1, 5, 10];
  }
  const out = value.map((x) => Number(x)).filter((x) => Number.isFinite(x) && x > 0);
  return out.length > 0 ? out : [1, 5, 10];
}

/** 中文注释：实现 formatOptionalMeta 的核心流程，支撑前端页面中的业务语义和异常边界。 */
function formatOptionalMeta(value: unknown): string {
  const text = String(value ?? "").trim();
  return text && text !== "-" && text !== "null" && text !== "undefined" ? text : "未记录 / 不适用";
}

/** 中文注释：实现 formatOptionalMetric 的核心流程，支撑前端页面中的业务语义和异常边界。 */
function formatOptionalMetric(value: unknown): string {
  const text = String(value ?? "").trim();
  return text && text !== "-" && text !== "null" && text !== "undefined" ? text : "未记录";
}

const REPORT_CASE_PAGE_SIZES = [50, 100, 200, 500];

const ATTACK_MODE_LABELS: Record<string, string> = {
  vqa_visual_corruption: "官方视觉退化函数",
  xtransfer_uap: "预训练通用扰动",
  foa_attack: "目标迁移优化",
  anyattack: "预训练生成器目标攻击",
  mpc_attack: "多范式协同迁移优化",
  m_attack: "局部语义匹配迁移优化",
};

/** 中文注释：实现 formatAttackModeName 的核心流程，支撑前端页面中的业务语义和异常边界。 */
function formatAttackModeName(attack: unknown, mode: unknown): string {
  const attackId = String(attack ?? "").trim().toLowerCase();
  return ATTACK_MODE_LABELS[attackId] ?? formatModeName(String(mode ?? "-"));
}

/** 中文注释：实现 confidenceLabel 的核心流程，支撑前端页面中的业务语义和异常边界。 */
function confidenceLabel(count: number): string {
  if (count >= 30) return "高置信证据";
  if (count >= 5) return "中置信证据";
  if (count > 0) return "低置信证据";
  return "仅运行摘要";
}

/** 中文注释：实现 resultTypeLabel 的核心流程，支撑前端页面中的业务语义和异常边界。 */
function resultTypeLabel(runId: string, summary: Record<string, unknown>): string {
  const text = [runId, summary.benchmark_tag, summary.dataset_name, summary.experiment_id].map((x) => String(x ?? "").toLowerCase()).join(" ");
  return /smoke|debug|vram_|quick|trial|toy|demo|staged_lifecycle/.test(text) ? "调试结果" : "正式结果";
}

/** 中文注释：实现 taskMetricExplanation 的核心流程，支撑前端页面中的业务语义和异常边界。 */
function taskMetricExplanation(taskKind: string, generationMetrics: Record<string, unknown>, summary: Record<string, unknown>): string {
  if (taskKind === "vqa") {
    return `视觉问答按“原始输入准确率、攻击后准确率、答案变化率”判断模型是否被误导：原始输入准确率=${asNum(generationMetrics.clean_accuracy).toFixed(4)}，攻击后准确率=${asNum(generationMetrics.attacked_accuracy).toFixed(4)}，答案变化率=${asNum(generationMetrics.answer_change_rate).toFixed(4)}。`;
  }
  if (taskKind === "caption") {
    return `图像描述按“对象集合重合度、描述文本相似度、目标翻转率”判断目标对象是否被错误删除或加入：对象集合重合度=${asNum(generationMetrics.object_jaccard).toFixed(4)}，描述文本相似度=${asNum(generationMetrics.caption_text_similarity).toFixed(4)}，目标翻转率=${asNum(generationMetrics.target_flip_rate).toFixed(4)}。`;
  }
  return `图文检索按“前 K 位召回率、条件攻击成功率、平均排名变化”判断图文匹配是否被破坏：攻击成功率=${asNum(summary.asr_attack ?? summary.asr).toFixed(4)}，平均排名变化=${formatOptionalMetric(summary.rank_delta_mean)}。`;
}

/** 中文注释：实现 supportConclusion 的核心流程，支撑前端页面中的业务语义和异常边界。 */
function supportConclusion(count: number, resultType: string): string {
  if (count >= 30) return `${resultType}，样本规模 n=${count}，可以作为稳定统计结论展示。`;
  if (count >= 5) return `${resultType}，样本规模 n=${count}，可以支撑趋势分析，正式论文中建议补充更大样本。`;
  if (count > 0) return `${resultType}，样本规模 n=${count}；本次运行仍保留在默认核心视图，但只能作为样本级证据或探索性结论。`;
  return `${resultType}，当前未登记可复盘样本，不能单独支撑样本级结论。`;
}

/** 中文注释：实现 conclusionBoundary 的核心流程，支撑前端页面中的业务语义和异常边界。 */
function conclusionBoundary(count: number, resultType: string): string {
  if (count >= 30) return "可用于答辩中的统计性结论，并可作为正式对比实验的主证据。";
  if (count >= 5) return "可用于展示趋势、任务覆盖和方法效果，正式论文结论需要结合更多运行交叉验证。";
  if (count > 0) return `${resultType}会继续展示，但讲解时应明确它是样本级复盘证据，不应扩展成总体性能结论。`;
  return "当前缺少可复盘样本，只能说明任务曾运行，不能说明攻击效果。";
}

/** 中文注释：实现 taskObjective 的核心流程，支撑前端页面中的业务语义和异常边界。 */
function taskObjective(taskKind: string): string {
  if (taskKind === "vqa") return "验证对抗图像是否改变视觉问答模型对同一问题的回答正确性。";
  if (taskKind === "caption") return "验证对抗扰动是否改变图像描述中的目标对象，同时尽量保留非目标语义。";
  return "验证图像或文本侧扰动是否破坏图文匹配和召回排序。";
}

/** 中文注释：实现 avgRecall 的核心流程，支撑前端页面中的业务语义和异常边界。 */
function avgRecall(metrics: Record<string, unknown>, k: number): number {
  const ir = asNum(metrics[`ir_r@${k}`]);
  const tr = asNum(metrics[`tr_r@${k}`]);
  return (ir + tr) / 2;
}

export default function ReportDetailPage() {
  const { runId = "" } = useParams();
  const [casePage, setCasePage] = useState(1);
  const [casePageSize, setCasePageSize] = useState(50);

  const summaryQ = useQuery({ queryKey: ["run-summary", runId], queryFn: () => getRunSummary(runId), enabled: !!runId, retry: false });
  const dataQ = useQuery({ queryKey: ["run-report-data", runId], queryFn: () => getRunReportData(runId), enabled: !!runId, retry: false });
  const casesQ = useQuery({ queryKey: ["run-cases", runId, casePage, casePageSize], queryFn: () => getRunCases(runId, { page: casePage, page_size: casePageSize }), enabled: !!runId, retry: false });

  if (!runId) {
    return (
      <div className="section-card p-6">
        <h2 className="section-title">报告详情</h2>
        <div className="mt-2 text-sm text-slate-600">缺少运行编号。</div>
      </div>
    );
  }

  if (summaryQ.isLoading || dataQ.isLoading || casesQ.isLoading) {
    return (
      <div className="section-card p-6">
        <h2 className="section-title">报告详情</h2>
        <div className="mt-2 text-sm text-slate-600">加载中...</div>
      </div>
    );
  }

  const err = summaryQ.error || dataQ.error || casesQ.error;
  if (err) {
    return (
      <div className="section-card p-6">
        <h2 className="section-title">报告不可用</h2>
        <div className="mt-2 text-sm text-red-700">加载失败：{String((err as Error)?.message ?? err)}</div>
        <div className="mt-2 text-sm text-slate-600">请确认后端已启动，并且相关运行摘要与报告数据接口可访问。</div>
      </div>
    );
  }

  const summary = asRecord(summaryQ.data);
  const hasReportSummary = Boolean(
    Object.keys(summary).length
    && (summary.run_id || summary.task_kind || summary.dataset_name || summary.benchmark_tag || summary.model_adapter || summary.attack)
  );
  if (!hasReportSummary) {
    return (
      <div className="section-card p-6">
        <h2 className="section-title">报告不可用</h2>
        <div className="mt-2 text-sm text-red-700">报告不存在或已移除。</div>
        <div className="mt-2 text-sm text-slate-600">该运行不再作为可查看测评结果展示，请从报告中心或案例库选择当前有效记录。</div>
        <div className="att-action-row mt-4">
          <Link className="gov-inline-link" to="/reports">返回报告中心</Link>
          <Link className="gov-inline-link" to="/cases">返回案例库</Link>
        </div>
      </div>
    );
  }
  const reportData = asRecord(dataQ.data);
  const taskKind = String(summary.task_kind ?? "").trim().toLowerCase();
  const isGenerationTask = taskKind === "vqa" || taskKind === "caption";
  const generationMetrics = asRecord(summary.generation_metrics ?? asRecord(reportData.generation).metrics);
  const caseCount = Number(casesQ.data?.total ?? 0);
  const caseTotalPages = Math.max(1, Math.ceil(caseCount / casePageSize));
  const safeCasePage = Math.min(casePage, caseTotalPages);
  const casePageStart = caseCount > 0 ? (safeCasePage - 1) * casePageSize + 1 : 0;
  const casePageEnd = Math.min(caseCount, safeCasePage * casePageSize);
  const evidenceCount = caseCount || asNum(summary.num_effective ?? summary.num_samples ?? summary.sample_pair_count ?? summary.num_images);
  const reportResultType = resultTypeLabel(runId, summary);
  const reportConfidence = confidenceLabel(evidenceCount);
  const metricExplanation = taskMetricExplanation(taskKind, generationMetrics, summary);
  const conclusionText = supportConclusion(evidenceCount, reportResultType);
  const boundaryText = conclusionBoundary(evidenceCount, reportResultType);

  const series = asRecord(reportData.metric_series);
  const l2 = Array.isArray(series.l2) ? series.l2.map((x) => asNum(x)) : [];
  const linf = Array.isArray(series.linf) ? series.linf.map((x) => asNum(x)) : [];
  const trendLength = Math.max(l2.length, linf.length);
  const hasLineSeries = trendLength > 0;

  const modeStats = asRecord(reportData.mode_stats);
  const modeRows = Object.entries(modeStats).map(([k, raw]) => {
    const v = asRecord(raw);
    const [attack, mode] = k.split(":");
    return { attack: attack || "未知", mode: mode || "A", count: asNum(v.count), asr: asNum(v.asr) };
  });

  const rowPreview = asRows(reportData.rows_preview);
  const failed = rowPreview.filter((x) => !Boolean(x.judge_success)).slice(0, 6);
  const fidelity = asRows(reportData.reproduction_fidelity);

  const ks = parseKs(summary.retrieval_k);
  const victimsRoot = asRecord(summary.victims);
  const victimRows = Object.entries(victimsRoot).map(([name, raw]) => {
    const node = asRecord(raw);
    const clean = asRecord(node.clean);
    const attacked = asRecord(node.attacked);
    return {
      name,
      clean,
      attacked,
    };
  });

  const failureCases = asRows(asRecord(reportData.vlr).failure_cases).slice(0, 10);
  const hasVictimMetrics = victimRows.length > 0;
  const hasModeStats = modeRows.length > 0;
  const attackDebug = asRecord(summary.attack_debug);
  const isAssetEvaluation = Boolean(summary.asset_evaluation_mode);
  const assetWorkflow = asRecord(summary.asset_workflow);
  const assetSourceIds = Array.isArray(assetWorkflow.source_asset_ids) ? assetWorkflow.source_asset_ids.map((item) => String(item)).filter(Boolean) : [];
  const assetSourceRunIds = Array.isArray(assetWorkflow.source_run_ids) ? assetWorkflow.source_run_ids.map((item) => String(item)).filter(Boolean) : [];
  const assetExecution = String(assetWorkflow.execution || "");
  const assetCallNote = assetExecution === "retest_existing_adversarial_images"
    ? "跳过攻击生成，直接调取已保存的原始图像与对抗图像，并重新调用当前受测模型计算输出和指标。"
    : "调用已入库资产，不重新生成对抗样本。";
  const victimAdapters = Array.isArray(summary.victim_model_adapters) ? summary.victim_model_adapters.map((item) => String(item)).filter(Boolean) : [];
  const primaryVictim = String(victimAdapters[0] ?? summary.model_adapter ?? "");
  const asrAttack = asNum(summary.asr_attack ?? summary.asr);
  const riskScore = asNum(summary.risk_score);
  const riskLevel = formatRiskLevel(String(summary.risk_level ?? "-"));
  const riskScenario = String(summary.risk_scenario ?? "-");
  const riskBreakdown = asRecord(summary.risk_breakdown);
  const riskWeights = asRecord(summary.risk_weights);
  const riskRecommendations = Array.isArray(summary.risk_recommendations) ? summary.risk_recommendations.map((x) => String(x)) : [];
  const riskDimensionOrder = ["task_damage", "output_instability", "semantic_disguise", "low_perturbation", "tail_case"];
  const riskItems = Object.entries(riskBreakdown)
    .filter(([k]) => riskDimensionOrder.includes(k))
    .map(([k, v]) => ({ key: k, value: asNum(v), weight: asNum(riskWeights[k]) }))
    .filter((x) => Number.isFinite(x.value))
    .sort((a, b) => riskDimensionOrder.indexOf(a.key) - riskDimensionOrder.indexOf(b.key));
  const metricQuality = asRecord(summary.metric_quality ?? reportData.metric_quality);
  const metricQualityFlags = asRows(metricQuality.flags).filter((flag) => String(flag.message ?? "").trim());
  const hasMetricQualityWarnings = !isGenerationTask && (Boolean(metricQuality.has_warnings) || metricQualityFlags.length > 0);
  const metricQualityNote = formatOptionalMeta(metricQuality.note);
  const metricQualityClaim =
    metricQuality.valid_for_attack_strength_claim === false
      ? "本次检索指标不适合直接作为攻击强度结论。"
      : "本次检索指标质量未报告阻断性问题。";
  const featureProjection = asRecord(reportData.feature_projection);
  const projPoints = asRows(featureProjection.points);
  const avgL2 = asNum(summary.avg_l2);
  const avgLinf = asNum(summary.avg_linf ?? summary.avg_l_inf);
  const maxL2 = l2.length ? Math.max(...l2) : avgL2;
  const maxLinf = linf.length ? Math.max(...linf) : avgLinf;
  const hasTrendChart = trendLength > 1;
  const evidenceCards = [
    { label: "结论类型", value: reportResultType, note: boundaryText },
    { label: "任务目标", value: formatJobType(String(summary.task_kind ?? "-")), note: taskObjective(taskKind) },
    { label: "样本与证据", value: `${evidenceCount} 条`, note: `${reportConfidence}，可复盘案例 ${caseCount} 条。` },
    { label: "核心风险", value: `${riskLevel} / ${riskScore.toFixed(4)}`, note: `攻击成功率 ${asrAttack.toFixed(4)}。` },
  ];
  const requirementRows = [
    { title: "对抗样本管理", body: isAssetEvaluation ? `本次调用 ${assetSourceIds.length || evidenceCount} 个已入库对抗样本，报告记录来源样本集、来源运行和来源案例。` : `已关联 ${caseCount} 条可复盘样本入口，展示原始样本和攻击后证据。` },
    { title: "自动化测评流程", body: `本次运行记录任务、模型、数据集、攻击方法、样本规模和风险结果。` },
    { title: "输出差异分析", body: metricExplanation },
    { title: "指标统计与可视化", body: `当前页面展示分任务指标、风险分解、扰动摘要和样本级复盘入口。` },
  ];

  const lineOpt = {
    tooltip: { trigger: "axis" },
    legend: { data: [formatNormLabel("L2"), formatNormLabel("Linf")] },
    xAxis: { type: "category", data: Array.from({ length: trendLength }, (_, i) => i + 1) },
    yAxis: { type: "value" },
    series: [
      { name: formatNormLabel("L2"), type: "line", smooth: true, data: l2 },
      { name: formatNormLabel("Linf"), type: "line", smooth: true, data: linf },
    ],
  };

  const modeOpt = {
    tooltip: { trigger: "axis" },
    xAxis: { type: "category", data: modeRows.map((x) => `${formatAttackName(x.attack)} / ${formatAttackModeName(x.attack, x.mode)}`) },
    yAxis: { type: "value", min: 0, max: 1 },
    series: [{ type: "bar", data: modeRows.map((x) => x.asr) }],
  };


  const riskRadarOpt = {
    tooltip: { trigger: "item" },
    radar: {
      indicator: riskItems.map((x) => ({ name: formatRiskDimension(x.key), max: 1 })),
      radius: "60%",
    },
    series: [
      {
        type: "radar",
        data: [{ value: riskItems.map((x) => x.value), name: "风险分解" }],
      },
    ],
  };

  const riskContribOpt = {
    tooltip: { trigger: "axis" },
    xAxis: { type: "category", data: riskItems.map((x) => formatRiskDimension(x.key)) },
    yAxis: { type: "value", min: 0, max: 1 },
    series: [
      {
        type: "bar",
        data: riskItems.map((x) => x.value * x.weight),
      },
    ],
  };

  const generationMetricRows = Object.entries(generationMetrics)
    .filter(([key, value]) => !/(defense|defended|recovery)/i.test(key) && value !== null && value !== undefined && String(value).trim() !== "")
    .map(([key, value]) => ({ key, value: asNum(value) }));

  const projGroups: Record<string, Array<[number, number, string]>> = {};
  for (const p of projPoints) {
    const stage = String(p.stage ?? "unknown");
    const modality = String(p.modality ?? "unknown");
    const key = formatProjectionGroupName(stage, modality);
    if (!projGroups[key]) projGroups[key] = [];
    projGroups[key].push([asNum(p.x), asNum(p.y), formatOptionalMetric(p.id)]);
  }
  const projOpt = {
    tooltip: {
      formatter: (params: any) => {
        const d = params?.data ?? [];
        return `${String(params?.seriesName ?? "")}<br/>编号=${formatOptionalMetric(d?.[2])}<br/>横轴=${asNum(d?.[0]).toFixed(3)} 纵轴=${asNum(d?.[1]).toFixed(3)}`;
      },
    },
    legend: { type: "scroll" },
    xAxis: { type: "value", name: "主成分 1" },
    yAxis: { type: "value", name: "主成分 2" },
    series: Object.entries(projGroups).map(([name, data]) => ({
      name,
      type: "scatter",
      symbolSize: 8,
      data,
    })),
  };

  return (
    <div className="space-y-6">
      <section className="section-card p-6">
        <div className="att-report-title-row">
          <div>
            <h2 className="section-title">报告详情 · {runId}</h2>
            <p className="section-subtitle">面向答辩展示本次运行的实验结论、证据边界、任务书对应关系和样本复盘入口。</p>
          </div>
          <div className={`att-risk-badge ${riskLevel.includes("高") ? "danger" : riskLevel.includes("中") ? "warn" : "ok"}`}>{riskLevel}</div>
        </div>
      </section>

      <section className="section-card p-5">
        <div className="att-summary-grid">
          {evidenceCards.map((item) => (
            <div key={item.label} className="att-summary-card">
              <span>{item.label}</span>
              <strong>{item.value}</strong>
              <p>{item.note}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="grid gap-4 xl:grid-cols-3">
        <div className="section-card p-5 xl:col-span-2">
          <div className="att-report-heading">
            <div>
              <span className="att-chip att-chip-ok">{reportResultType}</span>
              <span className={`att-chip ${evidenceCount >= 5 ? "att-chip-ok" : "att-chip-warn"}`}>{reportConfidence}</span>
              <span className="att-chip">样本规模 n={evidenceCount}</span>
            </div>
            <div className={`text-${riskLevel.includes("高") ? "red" : riskLevel.includes("中") ? "orange" : "green"}`}>风险等级：{riskLevel}</div>
          </div>
          <h3 className="mt-3 text-xl font-semibold">实验结论</h3>
          <p className="mt-2 text-base leading-7 text-slate-700">{conclusionText}</p>
          <p className="mt-2 text-base leading-7 text-slate-700">本次攻击方法为 {formatAttackName(String(summary.attack ?? "-"))}，受测模型为 {formatAdapterName(primaryVictim)}，攻击成功率={asrAttack.toFixed(4)}，风险分数={riskScore.toFixed(4)}。页面保留该运行的核心展示地位，同时把样本量和证据置信度作为结论边界显示。</p>
        </div>
        <div className="section-card p-5">
          <h3 className="text-xl font-semibold">分任务指标解释</h3>
          <p className="mt-2 text-base leading-7 text-slate-700">{metricExplanation}</p>
          <div className="mt-3 text-sm text-slate-600">案例入口：{caseCount} 条；任务类型：{formatJobType(String(summary.task_kind ?? "-"))}；数据集：{formatRunDatasetName(String(summary.dataset_name ?? ""), String(summary.benchmark_tag ?? ""), String(summary.task_kind ?? ""))}</div>
        </div>
      </section>

      <section className="section-card p-5">
        <h3 className="text-xl font-semibold">任务书要求对应关系</h3>
        <div className="att-requirement-grid mt-3">
          {requirementRows.map((item) => (
            <div key={item.title} className="att-requirement-item">
              <strong>{item.title}</strong>
              <p>{item.body}</p>
            </div>
          ))}
        </div>
      </section>

      {isAssetEvaluation ? (
        <section className="section-card p-5">
          <h3 className="text-xl font-semibold">对抗样本集来源</h3>
          <div className="att-requirement-grid mt-3">
            <div className="att-requirement-item"><strong>调用数量</strong><p>{asNum(assetWorkflow.asset_count || assetSourceIds.length || evidenceCount)} 个资产；{assetCallNote}</p></div>
            <div className="att-requirement-item"><strong>来源运行</strong><p>{assetSourceRunIds.slice(0, 6).join("；") || "未记录来源运行"}</p></div>
            <div className="att-requirement-item"><strong>资产编号</strong><p>{assetSourceIds.slice(0, 8).join("；") || "未记录资产编号"}{assetSourceIds.length > 8 ? ` 等 ${assetSourceIds.length} 个` : ""}</p></div>
          </div>
        </section>
      ) : null}

      <section className="grid items-start gap-4 xl:grid-cols-3">
        <div className="section-card p-5 xl:col-span-2">
          <h3 className="text-xl font-semibold">{hasTrendChart ? "扰动趋势" : "扰动摘要"}</h3>
          {hasTrendChart ? (
            <ReactECharts option={lineOpt} style={{ height: 300 }} />
          ) : hasLineSeries ? (
            <div className="att-compact-metric-grid mt-3">
              <div><span>记录点数</span><strong>{trendLength}</strong><p>当前运行只有单点或少量扰动记录，因此用摘要替代空旷折线图。</p></div>
              <div><span>平均二范数</span><strong>{avgL2.toFixed(4)}</strong><p>衡量整体扰动能量。</p></div>
              <div><span>最大二范数</span><strong>{maxL2.toFixed(4)}</strong><p>用于观察是否存在异常高扰动样本。</p></div>
              <div><span>最大无穷范数</span><strong>{maxLinf.toFixed(4)}</strong><p>衡量单像素最大改变量。</p></div>
            </div>
          ) : (
            <div className="gov-empty-state-chart">
              当前报告没有保存逐样本扰动曲线；可在关键摘要中查看平均二范数和攻击成功率。
            </div>
          )}
        </div>

        <div className="section-card p-5">
          <h3 className="text-xl font-semibold">关键摘要</h3>
          <div className="mt-2 space-y-2 text-sm text-[var(--ink-soft)]">
            <div>任务类型：{formatJobType(String(summary.task_kind ?? "-"))}</div>
            <div>数据集：{formatRunDatasetName(String(summary.dataset_name ?? ""), String(summary.benchmark_tag ?? ""), String(summary.task_kind ?? ""))}</div>
            <div>受测模型：{formatAdapterName(primaryVictim)}</div>
            <div>评测范围：{formatEvalScope(String(summary.eval_scope ?? "-"))}</div>
            <div>攻击成功率（ASR）：{asNum(summary.asr).toFixed(6)}</div>
            <div>平均二范数（L2）：{asNum(summary.avg_l2).toFixed(6)}</div>
            <div>
              攻击方法：{formatAttackName(String(summary.attack ?? "-"))} / {formatAttackModeName(summary.attack, summary.attack_mode)}
            </div>
            <div>攻击后攻击成功率（ASR）：{asrAttack.toFixed(6)}</div>
            <div>风险分数：{riskScore.toFixed(6)}</div>
            <div>风险等级：{riskLevel}</div>
            <div>风险场景：{formatEvalScope(riskScenario)}</div>
            <div>风险观察条数：{riskRecommendations.length}</div>
            <div>代理模型：{formatAdapterName(String(summary.surrogate_model_adapter ?? "-"))}</div>
            <div>补丁来源：{formatOptionalMeta(attackDebug.patch_source)}</div>
            <div>注册键：{formatOptionalMeta(attackDebug.registry_key)}</div>
          </div>
        </div>
      </section>

      {hasMetricQualityWarnings ? (
        <section className="section-card border border-amber-200 bg-amber-50/70 p-5">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <h3 className="text-xl font-semibold text-amber-950">指标质量提示</h3>
              <p className="mt-1 text-sm text-amber-900">{metricQualityClaim}</p>
            </div>
            <span className="rounded-full border border-amber-300 bg-white px-3 py-1 text-sm font-semibold text-amber-800">后端质量检查</span>
          </div>
          <div className="mt-3 grid gap-2 md:grid-cols-2">
            {metricQualityFlags.slice(0, 4).map((flag, idx) => {
              const diagnostics = asRecord(flag.diagnostics);
              const shape = Array.isArray(diagnostics.shape) ? diagnostics.shape.join(" × ") : "未记录";
              return (
                <div key={`metric-quality-${idx}`} className="rounded-xl border border-amber-200 bg-white/80 p-3 text-sm text-amber-950">
                  <div className="font-semibold">
                    {formatModeName(String(flag.stage ?? "-"))} / {formatAdapterName(String(flag.victim ?? "-"))}
                  </div>
                  <div className="mt-1">{String(flag.message ?? "")}</div>
                  <div className="mt-1 text-amber-800">矩阵规模：{shape}，唯一分数数：{formatOptionalMetric(diagnostics.unique_rounded)}</div>
                </div>
              );
            })}
          </div>
          {metricQualityNote !== "未记录 / 不适用" ? <div className="mt-3 text-sm text-amber-900">{metricQualityNote}</div> : null}
        </section>
      ) : null}

      {riskItems.length > 0 ? (
        <section className="grid gap-4 xl:grid-cols-2">
          <div className="section-card p-5">
            <h3 className="text-xl font-semibold">综合风险分解（雷达图）</h3>
            <ReactECharts option={riskRadarOpt} style={{ height: 320 }} />
          </div>
          <div className="section-card p-5">
            <h3 className="text-xl font-semibold">风险贡献（分值 × 权重）</h3>
            <ReactECharts option={riskContribOpt} style={{ height: 320 }} />
            {riskRecommendations.length > 0 ? (
              <div className="mt-3 text-sm text-[var(--ink-soft)]">
                {riskRecommendations.map((x, idx) => (
                  <div key={`risk-rec-${idx}`}>{formatWrapped("风险观察", x)}</div>
                ))}
              </div>
            ) : null}
          </div>
        </section>
      ) : null}

      {isGenerationTask ? (
        <section className="grid gap-4 xl:grid-cols-2">
          <div className="section-card p-5">
            <h3 className="text-xl font-semibold">生成式评测指标</h3>
            <div className="mt-3 grid gap-2 md:grid-cols-2">
              {generationMetricRows.length === 0 ? <div className="text-[var(--ink-soft)]">当前报告没有生成式指标。</div> : null}
              {generationMetricRows.map((item) => (
                <div key={item.key} className="rounded-xl border border-line bg-[var(--panel-strong)] p-3">
                  <div className="text-sm text-[var(--ink-soft)]">{formatRiskDimension(item.key)}</div>
                  <div className="mt-1 text-2xl font-semibold">{item.value.toFixed(4)}</div>
                </div>
              ))}
            </div>
            <div className="mt-3 text-sm text-[var(--ink-soft)]">
              {taskKind === "vqa"
                ? "视觉问答按同一张图的原始输入和攻击后输入答案是否正确、是否变化计算。"
                : "图像描述按目标对象是否被移除或加入、非目标对象是否保留和描述文本相似度计算。"}
            </div>
          </div>
          <div className="section-card p-5">
            <h3 className="text-xl font-semibold">生成式样本预览</h3>
            <div className="mt-2 space-y-2 text-base">
              {rowPreview.slice(0, 6).map((row, idx) => (
                <div key={`${String(row.sample_id ?? idx)}-generation-preview`} className="rounded-xl border border-line p-3">
                  <div className="font-mono text-sm">{formatWrapped("样本编号", row.sample_id ?? "-")}</div>
                  {taskKind === "vqa" ? <div className="text-sm text-slate-600">问题：{formatOptionalMeta(row.question)}</div> : null}
                  <div className="text-sm text-slate-600">原始输出：{formatOptionalMeta(row.clean_output)}</div>
                  <div className="text-sm text-slate-600">攻击后输出：{formatOptionalMeta(row.attacked_output)}</div>
                  <div className="text-sm text-slate-600">攻击成功：{Boolean(row.attack_success) ? "是" : "否"}</div>
                </div>
              ))}
            </div>
          </div>
        </section>
      ) : null}

      {!isGenerationTask ? <section className="section-card p-5">
        <h3 className="text-xl font-semibold">多模型检索指标（正常 / 攻击后）</h3>
        {hasVictimMetrics ? (
          <div className="mt-3 overflow-x-auto">
            <table className="w-full att-detail-table">
              <thead>
                <tr className="text-left text-[var(--ink-soft)]">
                  <th>受测模型</th>
                  {ks.map((k) => (
                    <th key={`r-clean-${k}`}>{formatRecallLabel(k, "clean")}</th>
                  ))}
                  {ks.map((k) => (
                    <th key={`r-adv-${k}`}>{formatRecallLabel(k, "attacked")}</th>
                  ))}
                  {ks.map((k) => (
                    <th key={`r-delta-${k}`}>{formatRecallLabel(k, "delta")}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {victimRows.map((row) => (
                  <tr key={row.name} className="border-t border-line">
                    <td>{formatAdapterName(row.name)}</td>
                    {ks.map((k) => (
                      <td key={`${row.name}-clean-${k}`}>{avgRecall(row.clean, k).toFixed(4)}</td>
                    ))}
                    {ks.map((k) => (
                      <td key={`${row.name}-att-${k}`}>{avgRecall(row.attacked, k).toFixed(4)}</td>
                    ))}
                    {ks.map((k) => (
                      <td key={`${row.name}-delta-${k}`}>{(avgRecall(row.attacked, k) - avgRecall(row.clean, k)).toFixed(4)}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="gov-empty-state-chart">
            当前报告没有登记受测模型检索指标；请检查本次运行是否完成受测模型评测阶段。
          </div>
        )}
      </section> : null}

      {!isGenerationTask ? <section className="grid gap-4 xl:grid-cols-3">
        <div className="section-card p-5">
          <h3 className="text-xl font-semibold">攻击模式统计</h3>
          {hasModeStats ? (
            <ReactECharts option={modeOpt} style={{ height: 280 }} />
          ) : (
            <div className="gov-empty-state-chart">
              当前报告没有分模式统计数据；单模式运行会在关键摘要中展示攻击方法和模式。
            </div>
          )}
          <div className="mt-2 text-sm text-[var(--ink-soft)]">按当前运行的模式统计生成。</div>
        </div>

        <div className="section-card p-5">
          <h3 className="text-xl font-semibold">失败样本解释</h3>
          <div className="mt-2 space-y-2 text-base">
            {failed.length === 0 ? <div className="text-[var(--ink-soft)]">暂无失败样本。</div> : null}
            {failed.map((row, idx) => {
              const sampleLabel = String(row.sample_id ?? row.text_id ?? row.image_id ?? `failure-${idx + 1}`);
              return (
              <div key={`${sampleLabel}-${idx}`} className="rounded-xl border border-line p-3">
                <div className="font-mono text-sm">{sampleLabel}</div>
                <div className="text-sm text-slate-600">受测模型：{formatAdapterName(String(row.victim ?? "-"))}</div>
                <div className="text-sm text-slate-600">原因：{formatJudgeReason(row.judge_reason)}</div>
                {row.text ? <div className="text-sm text-slate-600">查询文本：{String(row.text)}</div> : null}
                <div className="text-sm text-slate-600">文本差异分数：{formatOptionalMetric(asRecord(row.diagnostics).text_diff_score)}</div>
                <div className="text-sm text-slate-600">嵌入偏移：{formatOptionalMetric(asRecord(row.diagnostics).embedding_shift)}</div>
              </div>
              );
            })}
          </div>
        </div>

        <div className="section-card p-5">
          <h3 className="text-xl font-semibold">复现说明卡</h3>
          <div className="mt-2 space-y-2 text-base">
            {fidelity.length === 0 ? <div className="text-[var(--ink-soft)]">未提供复现说明信息。</div> : null}
            {fidelity.map((x, idx) => (
              <div key={idx} className="rounded-xl border border-line p-3">
                <div>{formatWrapped("论文条目", x.paper ?? "-")}</div>
                <div className="text-sm text-slate-600">状态：{formatPaperStatus(String(x.status ?? "-"))}</div>
                <div className="text-sm text-slate-600">来源：{formatWrapped("源码路径", x.source ?? "-")}</div>
              </div>
            ))}
          </div>
        </div>
      </section> : null}

      {!isGenerationTask ? <section className="section-card p-5">
        <h3 className="text-xl font-semibold">特征空间投影（主成分二维）</h3>
        {Boolean(featureProjection.available) && projPoints.length > 0 ? (
          <>
            <ReactECharts option={projOpt} style={{ height: 380 }} />
            <div className="mt-2 text-sm text-[var(--ink-soft)]">
              方法：{formatFeatureMethod(String(featureProjection.method ?? "pca"))} / 点数：{String(featureProjection.num_points ?? projPoints.length)}
            </div>
          </>
        ) : (
          <div className="mt-2 text-sm text-[var(--ink-soft)]">当前运行未提供可投影的双流嵌入数据。</div>
        )}
      </section> : null}

      {!isGenerationTask ? <section className="section-card p-5">
        <h3 className="text-xl font-semibold">典型失败案例（前五检索）</h3>
        {failureCases.length === 0 ? <div className="mt-2 text-sm text-[var(--ink-soft)]">当前运行未提供检索失败案例。</div> : null}
        <div className="mt-3 overflow-x-auto">
          <table className="w-full att-detail-table">
            <thead>
              <tr className="text-left text-[var(--ink-soft)]">
                <th>受测模型</th>
                <th>文本编号</th>
                <th>真实图像编号</th>
                <th>前五图像编号</th>
                <th>结果</th>
              </tr>
            </thead>
            <tbody>
              {failureCases.map((row, idx) => {
                const top5 = Array.isArray(row.top5_image_ids) && row.top5_image_ids.length ? row.top5_image_ids.join(", ") : "未记录前五候选";
                return (
                  <tr key={`${idx}-${String(row.victim ?? "")}`} className="border-t border-line">
                    <td>{formatAdapterName(String(row.victim ?? "-"))}</td>
                    <td className="font-mono text-sm">{formatOptionalMetric(row.text_id)}</td>
                    <td className="font-mono text-sm">{formatOptionalMetric(row.gt_image_id)}</td>
                    <td className="font-mono text-xs">{top5}</td>
                    <td>{Boolean(row.judge_success) ? "命中" : "未命中"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section> : null}

      <section className="section-card p-5">
        <h3 className="text-xl font-semibold">样本入口</h3>
        <div className="gov-table-toolbar mt-3">
          <span>共 {caseCount} 条可复盘样本，当前显示 {caseCount ? `${casePageStart} - ${casePageEnd}` : "0"}</span>
          <div className="gov-table-pagination">
            <label>每页行数
              <select value={casePageSize} onChange={(event) => { setCasePageSize(Number(event.target.value)); setCasePage(1); }}>
                {REPORT_CASE_PAGE_SIZES.map((size) => <option key={size} value={size}>{size}</option>)}
              </select>
            </label>
            <button type="button" disabled={safeCasePage <= 1} onClick={() => setCasePage(Math.max(1, safeCasePage - 1))}>上一页</button>
            <strong>{caseCount ? `${casePageStart} - ${casePageEnd} / ${caseCount}` : "0 / 0"}</strong>
            <button type="button" disabled={safeCasePage >= caseTotalPages} onClick={() => setCasePage(Math.min(caseTotalPages, safeCasePage + 1))}>下一页</button>
          </div>
        </div>
        <div className="mt-3 grid gap-2 md:grid-cols-2 xl:grid-cols-4">
          {(casesQ.data?.items ?? []).map((row) => {
            const sampleId = String(asRecord(row).sample_id ?? "");
            return (
              <Link
                key={sampleId}
                to={`/reports/${runId}/cases/${sampleId}`}
                className="rounded-xl border border-line bg-[var(--panel-strong)] px-3 py-2 text-base text-accent hover:bg-[var(--panel)]"
              >
                {formatWrapped("样本编号", sampleId)}
              </Link>
            );
          })}
          {!casesQ.isLoading && !(casesQ.data?.items ?? []).length ? <div className="gov-empty-state">当前页没有可复盘样本。</div> : null}
        </div>
      </section>
    </div>
  );
}
