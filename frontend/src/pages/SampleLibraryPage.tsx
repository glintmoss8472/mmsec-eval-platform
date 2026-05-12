// 文件说明：该文件属于前端页面，集中实现 SampleLibraryPage 相关逻辑。
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Fragment, useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";

import { AlertIcon, CheckIcon, ClipboardIcon, DatabaseIcon, GovMetric, GovPanel } from "../components/GovCards";
import { attackCatalog } from "../lib/attackCatalog";
import { createJob, getSystemOverview, listSampleAssetBatches, type SampleAssetBatchItem } from "../lib/api";
import { recommendedModels } from "../lib/modelCatalog";
import { formatAdapterName, formatAttackName, formatAttackScopeName, formatDatasetName, formatHealthStatus, formatRiskLevel } from "../lib/uiLabels";
import {
  ATTACK_MODE_OPTIONS,
  GENERATION_DATASET_META,
  STANDARD_EPSILON,
  asDatasetOverride,
  attackUsesBudget,
  attackUsesStepSize,
  attackUsesSteps,
  boundedNumber,
  budgetControlForAttack,
  configPathFor,
  defaultStepCount,
  defaultStepSize,
  defaultStrengthForAttack,
  generationDatasetSupportsTask,
  inferredTaskCapabilities,
  modelSupportsTask,
  surrogateSelectableForRun,
  usesOfficialExternalAlignmentRecipe,
  victimSelectableForLaunch,
  type AttackParamMode,
  type TaskKind,
} from "./experiment/experimentStudioConfig";

type FilterState = {
  task_kind: string;
  attack: string;
  scope: string;
  reusable_status: string;
  search: string;
};
type SortDirection = "asc" | "desc";
type BatchSortKey = "created_at" | "batch_id" | "task_dataset" | "attack" | "sample_count" | "avg_l2" | "risk";
type BatchSortState = { key: BatchSortKey; direction: SortDirection };

const TASK_OPTIONS: Array<[TaskKind, string]> = [["vlr", "图文检索"], ["vqa", "视觉问答"], ["caption", "图像描述"]];
const NO_CHECK_MODEL = "__no_check__";
const FALLBACK_DATASETS = [
  { key: "coco_subset", name: "coco_subset", tier: "benchmark", ready: true, item_count: 5000 },
  { key: "vqa_v2_coco_val", name: "vqa_v2_coco_val", tier: "generation", ready: true, item_count: 300 },
  { key: "coco_caption_object_val", name: "coco_caption_object_val", tier: "generation", ready: true, item_count: 100 },
];
const BATCH_PAGE_SIZES = [10, 20, 50, 100];
const DEFAULT_BATCH_SORT: BatchSortState = { key: "created_at", direction: "desc" };

/** 中文注释：实现 ariaSort 的核心流程，支撑前端页面中的业务语义和异常边界。 */
function ariaSort(sort: BatchSortState, key: BatchSortKey): "none" | "ascending" | "descending" {
  if (sort.key !== key) return "none";
  return sort.direction === "asc" ? "ascending" : "descending";
}

/** 中文注释：实现 BatchSortHeader 的核心流程，支撑前端页面中的业务语义和异常边界。 */
function BatchSortHeader({ sort, sortKey, label, onSort }: { sort: BatchSortState; sortKey: BatchSortKey; label: string; onSort: (key: BatchSortKey) => void }) {
  const active = sort.key === sortKey;
  const icon = active ? (sort.direction === "asc" ? "↑" : "↓") : "↕";
  return (
    <button type="button" className={`gov-sort-button ${active ? "active" : ""}`} onClick={() => onSort(sortKey)} aria-label={`${label}排序`}>
      <span>{label}</span>
      <i aria-hidden="true">{icon}</i>
    </button>
  );
}

/** 中文注释：实现 taskLabel 的核心流程，支撑前端页面中的业务语义和异常边界。 */
function taskLabel(value: string) {
  if (value === "mixed") return "混合任务";
  return TASK_OPTIONS.find(([key]) => key === value)?.[1] || value || "未记录";
}

/** 中文注释：实现 reuseLabel 的核心流程，支撑前端页面中的业务语义和异常边界。 */
function reuseLabel(value: string) {
  if (value === "ready") return "可复用";
  if (value === "pending_evaluation") return "待测评";
  if (value === "summary_only") return "仅摘要";
  if (value === "legacy") return "历史线索";
  return value || "未记录";
}

/** 中文注释：实现 statusClass 的核心流程，支撑前端页面中的业务语义和异常边界。 */
function statusClass(value: string) {
  if (value === "ready" || value === "callable") return "ok";
  if (value === "pending_evaluation" || value === "summary_only" || value === "partial") return "warn";
  return "muted";
}

/** 中文注释：实现 batchStatusLabel 的核心流程，支撑前端页面中的业务语义和异常边界。 */
function batchStatusLabel(batch: SampleAssetBatchItem) {
  if ((batch.callable_assets || 0) >= 1) return "可用于测评";
  if (batch.batch_status === "pending_evaluation" || (batch.pending_evaluation_assets || 0) >= 1) return "仅生成 / 待测评";
  return "仅可复盘";
}

/** 中文注释：实现 firstReadyModel 的核心流程，支撑前端页面中的业务语义和异常边界。 */
function firstReadyModel(models: any[], taskKind: TaskKind) {
  return models.find((model) => victimSelectableForLaunch(model.health_status) && modelSupportsTask(model, taskKind)) ?? models.find((model) => victimSelectableForLaunch(model.health_status)) ?? models[0];
}

/** 中文注释：实现 firstDataset 的核心流程，支撑前端页面中的业务语义和异常边界。 */
function firstDataset(datasets: any[], taskKind: TaskKind) {
  return compatibleDatasetsForTask(datasets, taskKind)[0];
}

/** 中文注释：实现 compatibleDatasetsForTask 的核心流程，支撑前端页面中的业务语义和异常边界。 */
function compatibleDatasetsForTask(datasets: any[], taskKind: TaskKind) {
  if (taskKind === "vlr") {
    return datasets.filter((item) => String(item.tier || "").trim() !== "generation" && String(item.tier || "").trim() !== "demo");
  }
  return datasets.filter((item) => String(item.tier || "").trim() === "generation" && generationDatasetSupportsTask(item.key, taskKind));
}

/** 中文注释：实现 percentLabel 的核心流程，支撑前端页面中的业务语义和异常边界。 */
function percentLabel(value: number) {
  return `${Math.round((Number(value) || 0) * 100)}%`;
}

/** 中文注释：实现 compactId 的核心流程，支撑前端页面中的业务语义和异常边界。 */
function compactId(value: string) {
  return value ? value.replace(/^20/, "20") : "未记录";
}

/** 中文注释：实现 displayAttack 的核心流程，支撑前端页面中的业务语义和异常边界。 */
function displayAttack(value: string) {
  return value === "mixed" ? "混合攻击" : formatAttackName(value);
}

/** 中文注释：实现 displayDataset 的核心流程，支撑前端页面中的业务语义和异常边界。 */
function displayDataset(batch: SampleAssetBatchItem) {
  const dataset = batch.benchmark_tag || batch.dataset_name;
  return dataset === "mixed" ? "混合数据集" : formatDatasetName(dataset);
}

export default function SampleLibraryPage() {
  const qc = useQueryClient();
  const [filters, setFilters] = useState<FilterState>({ task_kind: "", attack: "", scope: "", reusable_status: "", search: "" });
  const [expandedBatchId, setExpandedBatchId] = useState("");
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);
  const [sort, setSort] = useState<BatchSortState>(DEFAULT_BATCH_SORT);
  const [taskKind, setTaskKind] = useState<TaskKind>("vlr");
  const [attack, setAttack] = useState("fgsm");
  const [sampleCount, setSampleCount] = useState("8");
  const [paramMode, setParamMode] = useState<AttackParamMode>("standard");
  const [strength, setStrength] = useState(STANDARD_EPSILON);
  const [steps, setSteps] = useState("10");
  const [stepSize, setStepSize] = useState("0.008");
  const [textBudget, setTextBudget] = useState("1");
  const [patchSize, setPatchSize] = useState("32");
  const [attackMode, setAttackMode] = useState("A");
  const [targetText, setTargetText] = useState("");
  const [showGenerator, setShowGenerator] = useState(false);
  const [selectedDatasetKey, setSelectedDatasetKey] = useState("");
  const [selectedModelAdapter, setSelectedModelAdapter] = useState("");

  const overviewQ = useQuery({ queryKey: ["sample-library-overview"], queryFn: getSystemOverview, refetchInterval: 10000 });
  const batchQ = useQuery({ queryKey: ["sample-asset-batches", filters, page, pageSize, sort], queryFn: () => listSampleAssetBatches({ page, page_size: pageSize, sort_by: sort.key, sort_dir: sort.direction, include_asset_ids: false, ...filters }), staleTime: 5000 });
  const models = overviewQ.data?.models?.length
    ? overviewQ.data.models
    : recommendedModels.map((item) => ({
        adapter: item.adapter,
        display_name: item.name,
        family: item.family,
        launch_mode: item.family,
        health_status: "ready",
        endpoint_or_source: item.summary,
        model_name: item.name,
        role: item.adapter === "clip_hf" ? "surrogate/local" : "victim/api_or_self_hosted",
        formal_eval: true,
        task_capabilities: inferredTaskCapabilities(item.adapter),
      }));
  const formalModels = models.filter((model) => model.formal_eval !== false);
  const datasets = overviewQ.data?.datasets?.length ? overviewQ.data.datasets.filter((item) => item.ready !== false) : FALLBACK_DATASETS;
  const compatibleDatasets = useMemo(() => compatibleDatasetsForTask(datasets, taskKind), [datasets, taskKind]);
  const compatibleValidationModels = useMemo(() => formalModels.filter((model) => victimSelectableForLaunch(model.health_status) && modelSupportsTask(model, taskKind)), [formalModels, taskKind]);
  const selectedDataset = compatibleDatasets.find((item) => item.key === selectedDatasetKey) ?? firstDataset(datasets, taskKind);
  const selectedModel = selectedModelAdapter === NO_CHECK_MODEL
    ? undefined
    : compatibleValidationModels.find((model) => model.adapter === selectedModelAdapter) ?? firstReadyModel(compatibleValidationModels, taskKind);
  const selectedValidationValue = selectedModelAdapter === NO_CHECK_MODEL ? NO_CHECK_MODEL : selectedModel?.adapter || "";
  const skipValidation = selectedValidationValue === NO_CHECK_MODEL;
  const readyModels = formalModels.filter((model) => victimSelectableForLaunch(model.health_status));
  const batches = batchQ.data?.items ?? [];
  const summary = batchQ.data?.summary;
  const totalBatches = batchQ.data?.total ?? 0;
  const totalPages = Math.max(1, Math.ceil(totalBatches / pageSize));
  const safePage = Math.min(page, totalPages);
  const attackInfo = attackCatalog.find((item) => item.id === attack);
  const budgetControl = budgetControlForAttack(attack);
  const isGenerationTask = taskKind === "vqa" || taskKind === "caption";
  const selectedDatasetItemCount = Math.max(0, Number(selectedDataset?.item_count) || 0);
  const requestedSampleCountNumber = Math.max(1, Number(sampleCount) || 1);
  const sampleCountNumber = selectedDatasetItemCount > 0 ? Math.min(requestedSampleCountNumber, selectedDatasetItemCount) : requestedSampleCountNumber;
  const strengthNumber = Number(strength) || Number(defaultStrengthForAttack(attack));
  /** 中文注释：实现 toggleSort 的核心流程，支撑前端页面中的业务语义和异常边界。 */
  const toggleSort = (key: BatchSortKey) => {
    setSort((current) => ({ key, direction: current.key === key && current.direction === "desc" ? "asc" : "desc" }));
    setPage(1);
  };
  const effectiveSteps = paramMode === "advanced" ? Math.round(boundedNumber(steps, defaultStepCount(attack), 1, 3000)) : defaultStepCount(attack);
  const effectiveStepSize = paramMode === "advanced" ? boundedNumber(stepSize, defaultStepSize(attack, strengthNumber), 0.0001, 0.1) : defaultStepSize(attack, strengthNumber);
  const effectiveTextBudget = paramMode === "advanced" ? Math.round(boundedNumber(textBudget, 0, 0, 5)) : 0;
  const compatibleSurrogate = useMemo(() => {
    return models.find((model) => surrogateSelectableForRun(attack, model.adapter) && modelSupportsTask(model, "vlr")) ?? models.find((model) => model.adapter === "clip_hf") ?? models[0];
  }, [attack, models]);

  useEffect(() => {
    const next = selectedDataset?.key || "";
    if (next !== selectedDatasetKey) setSelectedDatasetKey(next);
  }, [selectedDataset?.key, selectedDatasetKey]);

  useEffect(() => {
    if (!selectedModelAdapter || selectedModelAdapter === NO_CHECK_MODEL) return;
    if (!compatibleValidationModels.some((model) => model.adapter === selectedModelAdapter)) setSelectedModelAdapter("");
  }, [compatibleValidationModels, selectedModelAdapter]);

  const generationDatasetMeta = selectedDataset?.key ? GENERATION_DATASET_META[selectedDataset.key] : undefined;
  const override = useMemo(() => {
    const baseAttack: Record<string, unknown> = {};
    if (attackUsesBudget(attack)) baseAttack.epsilon = strengthNumber;
    if (attackUsesSteps(attack)) baseAttack.steps = effectiveSteps;
    if (attackUsesStepSize(attack)) baseAttack.step_size = effectiveStepSize;
    if (paramMode === "advanced") {
      if (["advclip", "advedm", "advedm_plus", "tmm"].includes(attack)) baseAttack.mode = attackMode;
      if (["advclip", "advedm", "advedm_plus"].includes(attack)) baseAttack.patch_size = Math.round(boundedNumber(patchSize, 32, 4, 96));
      if (["tmm", "advedm_plus"].includes(attack)) {
        baseAttack.eps_t = effectiveTextBudget;
        baseAttack.text_candidates_k = 12;
      }
      if (targetText.trim()) baseAttack.target_text = targetText.trim();
    }
    if (isGenerationTask) {
      return {
        task: {
          kind: taskKind,
          eval_scope: "image",
          cases_jsonl: generationDatasetMeta?.casesJsonl || (taskKind === "vqa" ? "data/coco2014/generation/vqa_v2_coco_val.jsonl" : "data/coco2014/generation/coco_caption_object_val.jsonl"),
        },
        dataset: {
          kind: "generation_jsonl",
          max_items: sampleCountNumber,
          benchmark_tag: generationDatasetMeta?.benchmarkTag || (taskKind === "vqa" ? "vqa_v2_coco_val_real" : "coco_caption_object_val_real"),
        },
        plugins: { attack, model_adapter: skipValidation ? compatibleSurrogate?.adapter || "clip_hf" : selectedModel?.adapter || "" },
        attack: baseAttack,
        report: { save_heatmaps: true, save_patch_preview: true, top_k_cases: Math.max(1, sampleCountNumber) },
        sample_store: { enabled: true, save_images: true, save_traces: true },
        runner: {
          max_samples: sampleCountNumber,
          surrogate_model_adapter: compatibleSurrogate?.adapter || "clip_hf",
          victim_model_adapters: skipValidation ? [] : [selectedModel?.adapter || ""].filter(Boolean),
          staged_model_lifecycle: true,
          stop_local_vlm_before_attack: true,
          restart_local_vlm_for_evaluation: true,
        },
        extra: { workflow_type: skipValidation ? "sample_generation_only" : "sample_generation", ui_task_name: "对抗样本集生成", ui_note: skipValidation ? "生成后写入待测评批次，选择受测模型后再计算风险与报告。" : "生成后自动写入对抗样本库，首轮模型输出只用于校验证据完整性。" },
      };
    }
    return {
      task: { kind: "vlr", eval_scope: attack === "tmm" || attack === "advedm_plus" ? "joint" : "image" },
      dataset: { ...asDatasetOverride(selectedDataset?.key || "coco_subset"), max_items: sampleCountNumber },
      plugins: { attack, model_adapter: compatibleSurrogate?.adapter || "clip_hf" },
      attack: baseAttack,
      report: { save_heatmaps: true, save_patch_preview: true, top_k_cases: Math.max(1, sampleCountNumber) },
      sample_store: { enabled: true, save_images: true, save_traces: true },
      runner: {
        max_samples: sampleCountNumber,
        max_pairs: Math.max(sampleCountNumber * sampleCountNumber, sampleCountNumber),
        surrogate_model_adapter: compatibleSurrogate?.adapter || "clip_hf",
        victim_model_adapters: skipValidation ? [] : [selectedModel?.adapter || "clip_hf"],
        staged_model_lifecycle: true,
        stop_local_vlm_before_attack: true,
        restart_local_vlm_for_evaluation: true,
      },
      extra: { workflow_type: skipValidation ? "sample_generation_only" : "sample_generation", ui_task_name: "对抗样本集生成", ui_note: skipValidation ? "生成后写入待测评批次，选择受测模型后再计算风险与报告。" : "生成后自动写入对抗样本库，首轮模型输出只用于校验证据完整性。" },
    };
  }, [attack, attackMode, compatibleSurrogate?.adapter, effectiveStepSize, effectiveSteps, effectiveTextBudget, generationDatasetMeta, isGenerationTask, paramMode, patchSize, sampleCountNumber, selectedDataset?.key, selectedModel?.adapter, skipValidation, strengthNumber, targetText, taskKind]);

  const generate = useMutation({
    mutationFn: () => createJob({
      job_type: skipValidation ? "generate_sample_assets" : taskKind === "vqa" ? "run_vqa" : taskKind === "caption" ? "run_caption" : "run_vlr",
      config_path: configPathFor(attack, "standard", taskKind),
      benchmark_mode: false,
      override,
    }),
    onSuccess: () => {
      if (skipValidation) {
        setFilters((prev) => ({ ...prev, reusable_status: "pending_evaluation" }));
        setPage(1);
      }
      qc.invalidateQueries({ queryKey: ["sample-asset-batches"] });
      qc.invalidateQueries({ queryKey: ["sample-assets"] });
      qc.invalidateQueries({ queryKey: ["jobs-monitor"] });
    },
  });

  /** 中文注释：实现 updateFilter 的核心流程，支撑前端页面中的业务语义和异常边界。 */
  function updateFilter(key: keyof FilterState, value: string) {
    setFilters((prev) => ({ ...prev, [key]: value }));
    setPage(1);
  }

  /** 中文注释：实现 chooseAttack 的核心流程，支撑前端页面中的业务语义和异常边界。 */
  function chooseAttack(nextAttack: string) {
    setAttack(nextAttack);
    setStrength(defaultStrengthForAttack(nextAttack));
    setSteps(String(defaultStepCount(nextAttack)));
    setStepSize(String(defaultStepSize(nextAttack, Number(defaultStrengthForAttack(nextAttack)))));
    setTextBudget("1");
    setPatchSize(nextAttack === "advclip" ? "32" : "16");
    setAttackMode("A");
  }

  return (
    <div className="gov-stack sample-library-page">
      <div className="sample-library-hero">
        <div>
          <h2>对抗样本库</h2>
          <p>按生成批次管理可复用的对抗样本集；每个样本集保留任务、数据集、攻击参数、证据完整性和复用记录，新建测评只调用样本集。</p>
        </div>
        <div className="sample-hero-actions">
          <Link className="gov-button primary" to="/testing?mode=assets">调用样本集新建测评</Link>
          <button type="button" className="gov-button ghost" onClick={() => setShowGenerator((value) => !value)}>{showGenerator ? "收起生成配置" : "生成新样本集"}</button>
        </div>
      </div>

      <div className="gov-metric-grid four">
        <GovMetric title="样本集批次" value={summary?.total_batches ?? batchQ.data?.total ?? 0} tone="blue" icon={<DatabaseIcon />} />
        <GovMetric title="可测评批次" value={summary?.callable_batches ?? 0} tone="green" icon={<CheckIcon />} />
        <GovMetric title="可调用样本" value={summary?.callable_assets ?? 0} tone="orange" icon={<AlertIcon />} />
        <GovMetric title="可提交模型" value={readyModels.length} tone="purple" icon={<ClipboardIcon />} />
      </div>

      <div className="sample-workflow-grid">
        {showGenerator ? <GovPanel title="生成新对抗样本集" className="sample-generator-panel">
          <div className="sample-generator-form">
            <div className="gov-radio-row">
              <span>任务类型</span>
              {TASK_OPTIONS.map(([value, label]) => (
                <button key={value} type="button" className={taskKind === value ? "selected" : ""} onClick={() => setTaskKind(value)}>{label}</button>
              ))}
            </div>
            <label>
              <span>来源数据集</span>
              <select value={selectedDataset?.key || ""} disabled={!compatibleDatasets.length} onChange={(event) => setSelectedDatasetKey(event.target.value)}>
                {compatibleDatasets.map((dataset) => (
                  <option key={dataset.key} value={dataset.key}>{formatDatasetName(dataset.key || dataset.name)} · {dataset.item_count ?? 0} 条</option>
                ))}
                {!compatibleDatasets.length ? <option value="">当前没有可用于该任务的数据集</option> : null}
              </select>
              <small className="gov-field-help">这里决定样本集来源；生成完成后会保存原始图像、对抗图像、攻击参数和案例证据。</small>
            </label>
            <label>
              <span>校验模型</span>
              <select value={selectedValidationValue || NO_CHECK_MODEL} onChange={(event) => setSelectedModelAdapter(event.target.value)}>
                <option value={NO_CHECK_MODEL}>不校验</option>
                {compatibleValidationModels.map((model) => (
                  <option key={model.adapter} value={model.adapter}>{formatAdapterName(model.adapter)} · {formatHealthStatus(model.health_status)}</option>
                ))}
              </select>
              <small className="gov-field-help">{skipValidation ? "只生成原始图像和对抗图像，批次会标记为待测评，风险和报告需后续选择受测模型后产生。" : "生成阶段跑一次轻量模型校验，确保该批次能进入报告和案例复盘。"}</small>
            </label>
            <label>
              <span>样本条数</span>
              <input type="number" min="1" value={sampleCount} onChange={(event) => setSampleCount(event.target.value)} />
            </label>
            <label>
              <span>攻击方式</span>
              <select value={attack} onChange={(event) => chooseAttack(event.target.value)}>
                {attackCatalog.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}
              </select>
              <small className="gov-field-help">{attackInfo?.summary || "请选择攻击方法。"}</small>
            </label>
            <div className="gov-param-mode">
              <span>参数模式</span>
              <div>
                <button type="button" className={paramMode === "standard" ? "selected" : ""} onClick={() => setParamMode("standard")}>标准参数</button>
                <button type="button" className={paramMode === "advanced" ? "selected" : ""} onClick={() => setParamMode("advanced")}>高级参数</button>
              </div>
            </div>
            {budgetControl ? (
              <div className="gov-strength-row compact">
                <span>{budgetControl.label}</span>
                <div className="gov-strength-control">
                  <input type="range" min="0.005" max="0.1" step="0.005" value={strength} onChange={(event) => setStrength(event.target.value)} />
                  <input type="number" min="0.005" max="0.1" step="0.005" value={strength} onChange={(event) => setStrength(event.target.value)} />
                </div>
                <small>{budgetControl.help}</small>
              </div>
            ) : <div className="gov-param-hint"><strong>该攻击不使用通用扰动预算</strong><p>强度由方法专用参数或外部实现控制。</p></div>}
            {paramMode === "advanced" ? (
              <div className="gov-advanced-grid sample-advanced-grid">
                {(attackUsesSteps(attack) || attack === "advclip") ? <label><span>{attack === "advclip" ? "补丁训练步数" : "优化步数"}</span><input type="number" min="1" max="3000" value={steps} onChange={(event) => setSteps(event.target.value)} /></label> : null}
                {attackUsesStepSize(attack) ? <label><span>单步步长</span><input type="number" min="0.0001" max="0.1" step="0.0005" value={stepSize} onChange={(event) => setStepSize(event.target.value)} /></label> : null}
                {["advclip", "advedm", "advedm_plus", "tmm"].includes(attack) ? <label><span>攻击模式</span><select value={attackMode} onChange={(event) => setAttackMode(event.target.value)}>{ATTACK_MODE_OPTIONS.map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label> : null}
                {["advclip", "advedm", "advedm_plus"].includes(attack) ? <label><span>补丁尺寸</span><input type="number" min="4" max="96" value={patchSize} onChange={(event) => setPatchSize(event.target.value)} /></label> : null}
                {["tmm", "advedm_plus"].includes(attack) ? <label><span>文本替换预算</span><input type="number" min="0" max="5" value={textBudget} onChange={(event) => setTextBudget(event.target.value)} /></label> : null}
                <label><span>目标文本</span><input value={targetText} onChange={(event) => setTargetText(event.target.value)} placeholder="留空使用服务器默认目标" /></label>
              </div>
            ) : (
              <div className="gov-param-hint"><strong>标准参数</strong><p>{usesOfficialExternalAlignmentRecipe(attack) ? "外部论文方法按服务器官方配置执行。" : "按当前攻击方法的真实默认参数生成样本集。"}</p></div>
            )}
            <div className="sample-submit-row">
              <button type="button" className="gov-button primary" disabled={generate.isPending || !selectedDataset || (!skipValidation && !selectedModel)} onClick={() => generate.mutate()}>{generate.isPending ? "正在提交" : "生成并保存样本集"}</button>
              {generate.data ? <Link className="gov-inline-link" to="/jobs">已提交任务：{generate.data.id.slice(0, 8)}，去任务监控</Link> : null}
              {generate.isError ? <span className="gov-error">提交失败：{String(generate.error instanceof Error ? generate.error.message : generate.error)}</span> : null}
            </div>
          </div>
        </GovPanel> : null}

        <GovPanel title="样本集筛选">
          <div className="sample-filter-grid">
            <label><span>任务类型</span><select value={filters.task_kind} onChange={(event) => updateFilter("task_kind", event.target.value)}><option value="">全部</option>{TASK_OPTIONS.map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
            <label><span>攻击方法</span><select value={filters.attack} onChange={(event) => updateFilter("attack", event.target.value)}><option value="">全部</option>{attackCatalog.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}</select></label>
            <label><span>扰动类型</span><select value={filters.scope} onChange={(event) => updateFilter("scope", event.target.value)}><option value="">全部</option><option value="图像">图像扰动</option><option value="文本">文本扰动</option><option value="图文联合">图文联合扰动</option></select></label>
            <label><span>批次状态</span><select value={filters.reusable_status} onChange={(event) => updateFilter("reusable_status", event.target.value)}><option value="">全部</option><option value="ready">可复用</option><option value="pending_evaluation">待测评</option><option value="summary_only">仅摘要</option><option value="legacy">历史线索</option></select></label>
            <label className="wide"><span>搜索</span><input value={filters.search} onChange={(event) => updateFilter("search", event.target.value)} placeholder="批次/运行编号、攻击、数据集或文本" /></label>
          </div>
        </GovPanel>
      </div>

      <GovPanel title="可管理样本集">
        <div className="gov-table-note">对抗样本集按批次管理，表格不展示图像；单条证据可通过“样本明细”进入案例复盘。</div>
        <div className="gov-table-toolbar">
          <div className="gov-table-note">共 {totalBatches} 个样本集，当前筛选后可调用样本 {summary?.callable_assets ?? 0} 条，待测评样本 {summary?.pending_evaluation_assets ?? 0} 条。</div>
          <div className="gov-table-pagination">
            <label>每页行数<select value={pageSize} onChange={(event) => { setPageSize(Number(event.target.value)); setPage(1); }}>{BATCH_PAGE_SIZES.map((size) => <option key={size} value={size}>{size}</option>)}</select></label>
            <button type="button" disabled={safePage <= 1} onClick={() => setPage(Math.max(1, safePage - 1))}>上一页</button>
            <span>{totalBatches ? `${(safePage - 1) * pageSize + 1} - ${Math.min(totalBatches, safePage * pageSize)} / ${totalBatches}` : "0 / 0"}</span>
            <button type="button" disabled={safePage >= totalPages} onClick={() => setPage(Math.min(totalPages, safePage + 1))}>下一页</button>
          </div>
        </div>
        <div className="gov-table-wrap">
          <table className="gov-table gov-table-roomy att-sample-batch-table">
            <thead>
              <tr>
                <th aria-sort={ariaSort(sort, "batch_id")}><BatchSortHeader sort={sort} sortKey="batch_id" label="资产批次" onSort={toggleSort} /></th>
                <th aria-sort={ariaSort(sort, "task_dataset")}><BatchSortHeader sort={sort} sortKey="task_dataset" label="任务 / 数据集" onSort={toggleSort} /></th>
                <th aria-sort={ariaSort(sort, "attack")}><BatchSortHeader sort={sort} sortKey="attack" label="攻击 / 扰动" onSort={toggleSort} /></th>
                <th aria-sort={ariaSort(sort, "sample_count")}><BatchSortHeader sort={sort} sortKey="sample_count" label="样本规模 / 证据" onSort={toggleSort} /></th>
                <th aria-sort={ariaSort(sort, "avg_l2")}><BatchSortHeader sort={sort} sortKey="avg_l2" label="平均扰动" onSort={toggleSort} /></th>
                <th aria-sort={ariaSort(sort, "risk")}><BatchSortHeader sort={sort} sortKey="risk" label="风险 / 调用" onSort={toggleSort} /></th>
                <th>状态</th>
                <th>操作</th>
              </tr>
            </thead>
            <tbody>
              {batchQ.isLoading ? <tr><td colSpan={8}>正在读取样本集批次。</td></tr> : null}
              {batches.map((batch) => {
                const canUse = (batch.callable_assets || 0) >= 1;
                const pending = batch.batch_status === "pending_evaluation" || (batch.pending_evaluation_assets || 0) >= 1;
                const expanded = expandedBatchId === batch.batch_id;
                const previews = batch.preview_assets || [];
                return (
                  <Fragment key={batch.batch_id}>
                    <tr className={expanded ? "is-selected" : ""}>
                      <td><strong>{compactId(batch.batch_id)}</strong><span>{batch.created_at ? new Date(batch.created_at).toLocaleString("zh-CN", { hour12: false }) : "未记录时间"}</span></td>
                      <td><strong>{taskLabel(batch.task_kind)}</strong><span>{displayDataset(batch)}</span></td>
                      <td><strong>{displayAttack(batch.attack)}</strong><span>{formatAttackScopeName(batch.attack_scope || "")}</span></td>
                      <td><strong>{batch.callable_assets} / {batch.total_assets}</strong><span>证据完整 {percentLabel(batch.evidence_integrity)}</span></td>
                      <td><strong>L2 {batch.avg_l2.toFixed(4)}</strong><span>Linf {batch.avg_linf.toFixed(4)}</span></td>
                      <td>{pending ? <><strong>待测评</strong><span>暂不参与风险排序</span></> : <><strong>{formatRiskLevel(batch.avg_risk_score >= 0.66 ? "high" : batch.avg_risk_score >= 0.33 ? "medium" : "low")} · {batch.avg_risk_score.toFixed(2)}</strong><span>批次调用 {batch.batch_call_count ?? batch.used_count ?? 0} 次</span></>}</td>
                      <td><span className={`sample-status ${statusClass(batch.batch_status)}`}>{batchStatusLabel(batch)}</span></td>
                      <td><div className={`sample-table-actions ${pending ? "sample-table-actions-inline" : ""}`}>{canUse ? <Link className="gov-inline-link" to={`/testing?mode=assets&batch=${encodeURIComponent(batch.batch_id)}`}>调用测评</Link> : pending ? <Link className="gov-inline-link" to={`/testing?mode=assets&batch=${encodeURIComponent(batch.batch_id)}&asset_status=pending_evaluation`}>测评此批次</Link> : <span className="sample-action-muted">暂无证据</span>}{!pending && batch.report_url ? <Link className="gov-inline-link" to={batch.report_url}>来源报告</Link> : null}<button type="button" className="sample-link-button" onClick={() => setExpandedBatchId(expanded ? "" : batch.batch_id)}>{expanded ? "收起明细" : "样本明细"}</button></div></td>
                    </tr>
                    {expanded ? (
                      <tr className="sample-batch-detail-row">
                        <td colSpan={8}>
                          <div className="sample-batch-detail">
                            <div><strong>样本明细预览</strong><p>这里只列出文本和证据入口，不在样本集表格中展示图像。</p></div>
                            <div className="sample-batch-case-list">
                              {previews.map((asset) => (
                                <div key={asset.asset_id} className="sample-batch-case-row">
                                  <span>{asset.sample_id}</span>
                                  <em>{asset.source_text || "未记录文本摘要"}</em>
                                  {pending || asset.reusable_status === "pending_evaluation" || !asset.case_url ? <span className="sample-action-muted">待测评样本</span> : <Link className="gov-inline-link" to={asset.case_url}>证据</Link>}
                                </div>
                              ))}
                              {!previews.length ? <div className="gov-empty-state gov-empty-state-compact">该批次暂无可预览样本明细。</div> : null}
                            </div>
                          </div>
                        </td>
                      </tr>
                    ) : null}
                  </Fragment>
                );
              })}
              {!batchQ.isLoading && !batches.length ? <tr><td colSpan={8}>{filters.reusable_status === "pending_evaluation" ? "当前没有待测评样本集；可在上方选择不校验后生成待测评批次。" : "当前筛选条件下没有样本集；可以点击上方“生成新样本集”创建可复用批次。"}</td></tr> : null}
            </tbody>
          </table>
        </div>
      </GovPanel>
    </div>
  );
}
