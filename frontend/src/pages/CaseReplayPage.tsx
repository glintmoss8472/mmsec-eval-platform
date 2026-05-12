// 文件说明：该文件属于前端页面，集中实现 CaseReplayPage 相关逻辑。
import { useQuery } from "@tanstack/react-query";
import { Link, useParams } from "react-router-dom";

import { getCaseDetail, runAssetUrl } from "../lib/api";
import { caseInputLabel, caseInputText, caseOutputLabel, caseOutputText } from "../lib/caseBundleText";
import { formatAdapterName, formatAttackName, formatRunDatasetName } from "../lib/uiLabels";

/** 整理 `relative 运行记录 路径` 前端辅助逻辑，保持数据转换和展示口径一致。 */
function relativeRunPath(path: string) {
  return path.replace(/^.*runs[\\/][^\\/]+[\\/]/, "");
}

/** 整理 `as record` 前端辅助逻辑，保持数据转换和展示口径一致。 */
function asRecord(value: unknown): Record<string, unknown> {
  return typeof value === "object" && value !== null ? (value as Record<string, unknown>) : {};
}

/** 拼接 `asset URL`，把运行记录、资源路径和后端基址转换成可访问地址。 */
function assetUrl(runId: string, path: string): string {
  if (!path) return "";
  return runAssetUrl(runId, relativeRunPath(path));
}

/** 格式化 `format 可选 value`，统一页面展示文本和缺省值。 */
function formatOptionalValue(value: unknown, fallback = "未记录 / 不适用"): string {
  const text = String(value ?? "").trim();
  return text && text !== "-" && text !== "null" && text !== "undefined" ? text : fallback;
}

/** 格式化 `format 可选 指标`，统一页面展示文本和缺省值。 */
function formatOptionalMetric(value: unknown, fallback = "未记录 / 不适用"): string {
  const text = formatOptionalValue(value, fallback);
  if (text === fallback) return text;
  const numeric = typeof value === "number" ? value : Number(text);
  return Number.isFinite(numeric) ? numeric.toFixed(6) : text;
}

/** 格式化 `format boolean 指标`，统一页面展示文本和缺省值。 */
function formatBooleanMetric(value: unknown): string {
  if (value === true) return "是";
  if (value === false) return "否";
  const text = String(value ?? "").trim().toLowerCase();
  if (["true", "1", "yes", "是"].includes(text)) return "是";
  if (["false", "0", "no", "否"].includes(text)) return "否";
  return "未记录 / 不适用";
}

/** 格式化 `format 图像描述 goal`，统一页面展示文本和缺省值。 */
function formatCaptionGoal(value: unknown): string {
  const goal = String(value ?? "").trim().toLowerCase();
  if (["remove", "remove_object", "hide_object"].includes(goal)) return "移除目标对象（remove）";
  if (["add", "add_object", "insert_object"].includes(goal)) return "添加目标对象（add）";
  return formatOptionalValue(value);
}

/** 生成 `任务 类型 label` 展示值，统一页面标签、颜色和缺省文案。 */
function taskKindLabel(kind: string): string {
  if (kind === "vlr") return "图文检索";
  if (kind === "vqa") return "视觉问答";
  if (kind === "caption") return "图像描述";
  return "通用测评";
}

/** 生成 `verdict 文本` 展示值，统一页面标签、颜色和缺省文案。 */
function verdictText(taskKind: string, metrics: Record<string, unknown>, judge: Record<string, unknown>): string {
  const success = metrics.attack_success ?? judge.success;
  const ok = success === true || String(success).toLowerCase() === "true";
  if (taskKind === "vlr") return ok ? "攻击已造成检索失败" : "未形成检索失败";
  return ok ? "攻击成功" : "攻击未成功";
}

/** 整理 `verdict explain` 前端辅助逻辑，保持数据转换和展示口径一致。 */
function verdictExplain(taskKind: string, metrics: Record<string, unknown>): string {
  if (taskKind === "vqa") {
    return `标准答案为 ${formatOptionalValue(metrics.answer)}，原始答案正确=${formatBooleanMetric(metrics.clean_correct)}，攻击后答案正确=${formatBooleanMetric(metrics.attacked_correct)}，答案变化=${formatBooleanMetric(metrics.answer_changed)}。`;
  }
  if (taskKind === "caption") {
    return `目标对象为 ${formatOptionalValue(metrics.target_object)}，攻击目标为 ${formatCaptionGoal(metrics.attack_goal)}，对象集合重合度=${formatOptionalMetric(metrics.object_jaccard)}，描述文本相似度=${formatOptionalMetric(metrics.caption_text_similarity)}。`;
  }
  return `检索案例通过输入文本变化、检索分数和嵌入偏移判断是否破坏图文匹配。`;
}

/** 整理 `生成式评测 指标 rows` 前端辅助逻辑，保持数据转换和展示口径一致。 */
function generationMetricRows(taskKind: string, metrics: Record<string, unknown>) {
  if (taskKind === "vqa") {
    return [
      ["标准答案", formatOptionalValue(metrics.answer)],
      ["原始答案正确", formatBooleanMetric(metrics.clean_correct)],
      ["攻击后答案正确", formatBooleanMetric(metrics.attacked_correct)],
      ["答案发生变化", formatBooleanMetric(metrics.answer_changed)],
      ["攻击成功", formatBooleanMetric(metrics.attack_success)],
    ];
  }
  if (taskKind === "caption") {
    return [
      ["目标对象", formatOptionalValue(metrics.target_object)],
      ["攻击目标", formatCaptionGoal(metrics.attack_goal)],
      ["原始阶段目标出现", formatBooleanMetric(metrics.target_present_clean)],
      ["攻击后目标出现", formatBooleanMetric(metrics.target_present_attacked)],
      ["语义保持率", formatOptionalMetric(metrics.semantic_preservation_rate)],
      ["对象集合重合度", formatOptionalMetric(metrics.object_jaccard)],
      ["描述文本相似度", formatOptionalMetric(metrics.caption_text_similarity)],
      ["攻击成功", formatBooleanMetric(metrics.attack_success)],
    ];
  }
  return [];
}

/** 渲染 `CaseReplayPage` 组件，组织该区域的数据读取、交互状态和可访问性标记。 */
export default function CaseReplayPage() {
  const { runId = "", sampleId = "" } = useParams();

  const caseQ = useQuery({
    queryKey: ["case-detail", runId, sampleId],
    queryFn: () => getCaseDetail(runId, sampleId),
    enabled: !!runId && !!sampleId,
  });

  if (!caseQ.data && (caseQ.isLoading || caseQ.isFetching)) {
    return (
      <div className="space-y-6">
        <section className="section-card p-6">
          <h2 className="section-title">案例复盘</h2>
          <p className="section-subtitle">正在从后端读取真实样本、攻击前后图片、生成输出和调试可视化，请稍候。</p>
        </section>
        <section className="section-card p-8">
          <div className="gov-empty-state gov-empty-state-chart">正在加载样本详情，不会用空白占位冒充缺失数据。</div>
        </section>
      </div>
    );
  }

  if (caseQ.isError) {
    const errorMessage = caseQ.error instanceof Error ? caseQ.error.message : "后端没有返回可读取的样本详情";
    return (
      <div className="space-y-6">
        <section className="section-card p-6">
          <h2 className="section-title">案例不可用</h2>
          <p className="section-subtitle">样本详情读取失败，请检查运行编号、样本编号和后端日志。</p>
        </section>
        <section className="section-card p-8">
          <div className="gov-empty-state gov-empty-state-chart">读取失败：{errorMessage}</div>
        </section>
      </div>
    );
  }

  const bundle = (caseQ.data?.case_bundle as Record<string, any>) ?? {};
  if (!Object.keys(bundle).length) {
    return (
      <div className="space-y-6">
        <section className="section-card p-6">
          <h2 className="section-title">案例不可用</h2>
          <p className="section-subtitle">后端返回了空样本，当前无法展示真实复盘内容。</p>
        </section>
        <section className="section-card p-8">
          <div className="gov-empty-state gov-empty-state-chart">未找到该样本的 case_bundle，请重新选择报告中的样本入口。</div>
        </section>
      </div>
    );
  }
  const refs = (bundle.artifact_refs as Record<string, string>) ?? {};
  const visualLabels = asRecord(bundle.visual_labels);
  const taskKind = String(bundle.task_kind || "").trim().toLowerCase();
  const isGenerationTask = taskKind === "vqa" || taskKind === "caption";
  const artifactCapabilityRaw = Array.isArray(bundle.artifact_capability) ? bundle.artifact_capability.map(asRecord) : [];
  const evidenceCapability = artifactCapabilityRaw.filter((item) => String(item.key) !== "defended_image");
  const advMeta = asRecord(asRecord(bundle.adversarial).metadata);
  const diagnostics = asRecord(bundle.diagnostics);
  const metrics = asRecord(bundle.metrics);
  const judge = asRecord(bundle.judge);
  const sampleMeta = asRecord(asRecord(bundle.sample).metadata);
  const datasetTag = String(bundle.dataset_tag || sampleMeta.source_dataset || "");
  const modelTag = String(bundle.model_tag || "");
  const attackTag = String(advMeta.attack || advMeta.attack_name || bundle.attack || "");
  const assetLineage = asRecord(bundle.asset_lineage);

  const clean = refs.clean_image ? assetUrl(runId, refs.clean_image) : "";
  const adv = refs.adv_image ? assetUrl(runId, refs.adv_image) : assetUrl(runId, String(refs.attack_visualization ?? ""));
  const cleanLabel = String(visualLabels.clean || "原始图像");
  const advLabel = String(visualLabels.adv || (refs.adv_image ? "对抗图像" : "攻击证据图（真实注意力/掩码）"));
  const cleanInputText = caseInputText(bundle, "clean");
  const advInputText = caseInputText(bundle, "adv");
  const cleanOutputText = caseOutputText(bundle, "clean");
  const advOutputText = caseOutputText(bundle, "adv");
  const textDiffScore = diagnostics.text_diff_score ?? metrics.text_diff_score;
  const embeddingShift = diagnostics.embedding_shift ?? metrics.score_drop;
  const cotShiftScore = diagnostics.cot_shift_score ?? metrics.cot_shift_score;
  const hasCotShiftScore = formatOptionalMetric(cotShiftScore) !== "未记录 / 不适用";
  const outputFallback = isGenerationTask ? "当前案例未记录该阶段生成输出" : "当前案例未记录该阶段检索分数";
  const generationRows = isGenerationTask ? generationMetricRows(taskKind, metrics) : [];
  /** 整理 `stage output heading` 前端辅助逻辑，保持数据转换和展示口径一致。 */
  const stageInputHeading = (stageKey: "clean" | "adv", value: unknown) => caseInputLabel(stageKey, taskKind, value);
  /** 处理 `stage output heading` 逻辑，保持前端页面的数据转换和展示口径一致。 */
  const stageOutputHeading = (stageKey: "clean" | "adv", value: unknown) => caseOutputLabel(stageKey, taskKind, value);
  const visualStages = [
    {
      key: "clean",
      label: cleanLabel,
      image: clean,
      missing: "无原始图像",
      input: cleanInputText,
      score: cleanOutputText,
      tone: "clean",
    },
    {
      key: "adv",
      label: advLabel,
      image: adv,
      missing: "无攻击图像或证据图",
      input: advInputText,
      score: advOutputText,
      tone: "attack",
    },
  ];
  const availableArtifacts = evidenceCapability.filter((item) => String(item.status) === "available").length;
  const artifactTotal = evidenceCapability.length;
  const attackVerdict = verdictText(taskKind, metrics, judge);
  const attackVerdictTone = attackVerdict.includes("未") ? "warn" : "danger";
  const contextRows = [
    ["任务类型", taskKindLabel(taskKind)],
    ["数据集", formatRunDatasetName(datasetTag, datasetTag, taskKind)],
    ["受测模型", formatAdapterName(modelTag || "-")],
    ["攻击方法", formatAttackName(attackTag || "-")],
    ["运行编号", runId],
    ["样本编号", sampleId],
    ...(String(assetLineage.asset_id || "").trim()
      ? [["来源资产", String(assetLineage.asset_id)], ["来源运行", String(assetLineage.source_run_id || "-")], ["来源案例", String(assetLineage.source_case_id || "-")]]
      : []),
  ];
  const judgementRows = [
    { label: "攻击判定", value: attackVerdict, note: verdictExplain(taskKind, metrics) },
    { label: "输出变化", value: formatOptionalMetric(textDiffScore), note: isGenerationTask ? "生成式任务使用攻击前后输出差异辅助判定。" : "检索任务使用文本差异和检索分数辅助判定。" },
    { label: "扰动强度", value: formatOptionalMetric(embeddingShift), note: "结合二范数、无穷范数和嵌入偏移观察攻击代价。" },
    { label: "证据完整性", value: `${availableArtifacts}/${artifactTotal || 0}`, note: artifactTotal ? "可查看项越多，复盘证据越完整。" : "当前后端未返回证据能力清单。" },
  ];

  return (
    <div className="space-y-6">
      <section className="section-card p-6">
        <div className="att-report-title-row">
          <div>
            <h2 className="section-title">案例复盘 · {sampleId}</h2>
            <p className="section-subtitle">按任务背景、攻击判定、攻击前后证据和输出差异复盘单个对抗样本。</p>
          </div>
          <span className={`att-risk-badge ${attackVerdictTone}`}>{attackVerdict}</span>
        </div>
        <div className="att-action-row mt-3">
          <Link className="gov-inline-link" to={`/reports/${runId}`}>返回报告</Link>
          <Link className="gov-inline-link" to="/cases">返回案例库</Link>
        </div>
      </section>

      <section className="grid gap-4 xl:grid-cols-3">
        <div className="section-card p-5 xl:col-span-2">
          <h3 className="text-xl font-semibold">复盘结论</h3>
          <div className="att-case-judgement-grid mt-3">
            {judgementRows.map((item) => (
              <div key={item.label} className="att-case-judgement-card">
                <span>{item.label}</span>
                <strong>{item.value}</strong>
                <p>{item.note}</p>
              </div>
            ))}
          </div>
        </div>
        <div className="section-card p-5">
          <h3 className="text-xl font-semibold">任务上下文</h3>
          <dl className="att-case-context-list mt-3">
            {contextRows.map(([label, value]) => (
              <div key={label}>
                <dt>{label}</dt>
                <dd>{value}</dd>
              </div>
            ))}
          </dl>
        </div>
      </section>

      <section className="grid gap-4 xl:grid-cols-3">
        <div className="section-card p-5 xl:col-span-3">
          <h3 className="text-xl font-semibold">攻击前后对照</h3>
          <div className="case-stage-grid">
            {visualStages.map((stage) => (
              <div key={stage.key} className="case-stage-card">
                <div className="case-stage-title">{stage.label}</div>
                {stage.image ? <img src={stage.image} className="case-stage-media" alt={stage.label} /> : <div className="gov-empty-state gov-empty-state-compact case-stage-media">{stage.missing}</div>}
                <div className={`case-stage-evidence ${stage.tone}`}>
                  <strong>{stageInputHeading(stage.key as "clean" | "adv", stage.input)}</strong>
                  <p>{formatOptionalValue(stage.input, "当前案例未记录该阶段输入文本")}</p>
                  <strong>{stageOutputHeading(stage.key as "clean" | "adv", stage.score)}</strong>
                  <p>{formatOptionalValue(stage.score, outputFallback)}</p>
                </div>
              </div>
            ))}
          </div>
          <div className="case-diagnostics-grid mt-4 grid gap-2 text-base text-slate-600 md:grid-cols-3">
            <div>{isGenerationTask ? "输出差异分数" : "文本差异分数"}：{formatOptionalMetric(textDiffScore)}</div>
            <div>嵌入偏移：{formatOptionalMetric(embeddingShift)}</div>
            {hasCotShiftScore ? <div>思维链偏移分数：{formatOptionalMetric(cotShiftScore)}</div> : null}
          </div>
          {generationRows.length ? (
            <div className="mt-4 rounded-xl border border-line bg-slate-50 p-3">
              <h3 className="text-base font-semibold">生成式判定指标</h3>
              <dl className="mt-2 grid gap-2 text-sm text-slate-700 md:grid-cols-2">
                {generationRows.map(([label, value]) => (
                  <div key={label} className="flex gap-2">
                    <dt className="font-medium text-slate-500">{label}：</dt>
                    <dd>{value}</dd>
                  </div>
                ))}
              </dl>
            </div>
          ) : null}
        </div>
      </section>
    </div>
  );
}
