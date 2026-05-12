import { useState } from "react";

import { CheckIcon, GovPanel } from "../../components/GovCards";
import { RISK_THRESHOLDS } from "../../lib/riskPolicy";
import { formatAdapterName, formatAttackName, formatAttackScopeName, formatDatasetName, formatHealthStatus, formatRunDatasetName } from "../../lib/uiLabels";

type ExperimentStudioViewProps = Record<string, any>;
type SortDirection = "asc" | "desc";
type AssetBatchSortKey = "created_at" | "batch_id" | "task_dataset" | "attack" | "sample_count" | "avg_l2" | "risk";
type AssetBatchSortState = { key: AssetBatchSortKey; direction: SortDirection };

const DEFAULT_ASSET_BATCH_SORT: AssetBatchSortState = { key: "created_at", direction: "desc" };

function percentLabel(value: number) {
  return `${Math.round((Number(value) || 0) * 100)}%`;
}

function assetBatchSortValue(batch: any, key: AssetBatchSortKey): string | number {
  if (key === "created_at") return Date.parse(batch.created_at || "") || 0;
  if (key === "batch_id") return String(batch.batch_id || "");
  if (key === "task_dataset") return `${String(batch.task_kind || "")} ${formatRunDatasetName(batch.benchmark_tag || batch.dataset_name, batch.dataset_name, batch.task_kind)}`;
  if (key === "attack") return `${formatAttackName(batch.attack)} ${formatAttackScopeName(batch.attack_scope || "")}`;
  if (key === "sample_count") return Number(batch.selectable_assets || batch.callable_assets || 0);
  if (key === "avg_l2") return Number(batch.avg_l2 || 0);
  return Number(batch.avg_risk_score || 0);
}

function compareAssetBatchValues(left: string | number, right: string | number) {
  if (typeof left === "number" && typeof right === "number") return left - right;
  return String(left).localeCompare(String(right), "zh-CN", { numeric: true, sensitivity: "base" });
}

function sortAssetBatches(items: any[], sort: AssetBatchSortState) {
  return [...items].sort((left, right) => {
    const primary = compareAssetBatchValues(assetBatchSortValue(left, sort.key), assetBatchSortValue(right, sort.key));
    const created = compareAssetBatchValues(Date.parse(left.created_at || "") || 0, Date.parse(right.created_at || "") || 0);
    const batchId = String(left.batch_id || "").localeCompare(String(right.batch_id || ""), "zh-CN", { numeric: true, sensitivity: "base" });
    const result = primary || created || batchId;
    return sort.direction === "asc" ? result : -result;
  });
}

function ariaSort(sort: AssetBatchSortState, key: AssetBatchSortKey): "none" | "ascending" | "descending" {
  if (sort.key !== key) return "none";
  return sort.direction === "asc" ? "ascending" : "descending";
}

function AssetBatchSortHeader({ sort, sortKey, label, onSort }: { sort: AssetBatchSortState; sortKey: AssetBatchSortKey; label: string; onSort: (key: AssetBatchSortKey) => void }) {
  const active = sort.key === sortKey;
  const icon = active ? (sort.direction === "asc" ? "↑" : "↓") : "↕";
  return (
    <button type="button" className={`gov-sort-button ${active ? "active" : ""}`} onClick={() => onSort(sortKey)} aria-label={`${label}排序`}>
      <span>{label}</span>
      <i aria-hidden="true">{icon}</i>
    </button>
  );
}

export function ExperimentStudioView(props: ExperimentStudioViewProps) {
  const {
    WIZARD_STEPS,
    evaluationMode,
    setEvaluationMode,
    sampleBatches,
    selectedSampleBatch,
    selectedBatchId,
    setSelectedBatchId,
    assetSelectionLimit,
    batchCallableMax,
    sampleAssets,
    selectedSampleAssets,
    selectedAssetIds,
    setSelectedAssetIds,
    sampleAssetLoading,
    assetBatchStatus,
    sampleAssetMinCount,
    selectedAssetReady,
    currentStep,
    setCurrentStep,
    taskName,
    setTaskName,
    taskKind,
    setTaskKind,
    selectedVictim,
    victimAdapter,
    setVictimAdapter,
    taskCompatibleVictims,
    formatModelDisplayText,
    isGenerationTask,
    selectedVictimCanSubmit,
    selectedModelSupportsTask,
    setVictimScope,
    victimScope,
    taskGenerationDatasets,
    selectedGenerationDataset,
    setDatasetId,
    vlrDatasets,
    datasetId,
    sampleCount,
    setSampleCount,
    effectiveSampleCount,
    generationDatasetName,
    generationCaseCount,
    datasetItemCount,
    effectivePairBudget,
    groupedAttacks,
    ATTACK_GROUPS,
    groupIdForAttack,
    EXTERNAL_RUNTIME_ATTACKS,
    externalStatuses,
    attack,
    setAttack,
    applyAttackDefaults,
    ExternalStatusPills,
    CLIP_AUXILIARY_SURROGATE_ATTACKS,
    selectedSurrogate,
    surrogate,
    setSurrogate,
    compatibleSurrogates,
    surrogateHelpForAttack,
    externalAttackSelected,
    selectedExternalRunnable,
    selectedExternalStatus,
    attackParamMode,
    setAttackParamMode,
    budgetControl,
    strength,
    setStrength,
    strengthNumber,
    strength255,
    usesOfficialExternalAlignmentRecipe,
    attackUsesSteps,
    attackUsesStepSize,
    advancedSteps,
    setAdvancedSteps,
    stepHelpForAttack,
    advancedStepSize,
    setAdvancedStepSize,
    stepSizeHelpForAttack,
    ATTACK_MODE_OPTIONS,
    attackModeOverride,
    setAttackModeOverride,
    patchSize,
    setPatchSize,
    lambdaAt,
    setLambdaAt,
    lambdaTpd,
    setLambdaTpd,
    tauPatch,
    setTauPatch,
    topK,
    setTopK,
    regionThreshold,
    setRegionThreshold,
    alphaWeight,
    setAlphaWeight,
    betaWeight,
    setBetaWeight,
    gammaWeight,
    setGammaWeight,
    lambdaAtt,
    setLambdaAtt,
    ratioR,
    setRatioR,
    VQA_CORRUPTION_OPTIONS,
    corruptionType,
    setCorruptionType,
    corruptionSeverity,
    setCorruptionSeverity,
    corruptionSeed,
    setCorruptionSeed,
    uapName,
    setUapName,
    THREAT_MODEL_OPTIONS,
    threatModel,
    setThreatModel,
    textBudget,
    setTextBudget,
    textCandidatesK,
    setTextCandidatesK,
    targetImageOverride,
    setTargetImageOverride,
    targetTextOverride,
    setTargetTextOverride,
    surrogateModelsOverride,
    setSurrogateModelsOverride,
    cropScale,
    setCropScale,
    cropRatio,
    setCropRatio,
    clipBackbonesOverride,
    setClipBackbonesOverride,
    mpcLam,
    setMpcLam,
    mpcTau,
    setMpcTau,
    mpcOmega,
    setMpcOmega,
    canSubmit,
    selectedDataset,
    attackInfo,
    effectiveSteps,
    effectivePatchSize,
    effectiveTopK,
    effectiveTextBudget,
    effectiveTextCandidatesK,
    effectiveCorruptionSeverity,
    boundedNumber,
    note,
    setNote,
    sidePanelTitle,
    taskKindLabel,
    submittedVictims,
    readyVictims,
    formalModels,
    launchableVictims,
    sampleCountNumber,
    requestedPairBudget,
    sidePanelNote,
    surrogateRequirementNote,
    runningJob,
    saveDraft,
    goPrev,
    canAdvance,
    goNext,
    submitJob,
    draftStatus,
  } = props;

  const assetNeedCount = Math.max(1, Number(sampleAssetMinCount || 1));
  const batchItems = Array.isArray(sampleBatches) ? sampleBatches : [];
  const pendingAssetMode = assetBatchStatus === "pending_evaluation";
  const selectedBatch = selectedSampleBatch || null;
  const chosenAssetCount = Array.isArray(selectedAssetIds) ? selectedAssetIds.length : 0;
  const [assetBatchPage, setAssetBatchPage] = useState(1);
  const [assetBatchPageSize, setAssetBatchPageSize] = useState(10);
  const [assetBatchSort, setAssetBatchSort] = useState<AssetBatchSortState>(DEFAULT_ASSET_BATCH_SORT);
  const assetBatchTotal = batchItems.length;
  const assetBatchTotalPages = Math.max(1, Math.ceil(assetBatchTotal / assetBatchPageSize));
  const safeAssetBatchPage = Math.min(assetBatchPage, assetBatchTotalPages);
  const sortedBatchItems = sortAssetBatches(batchItems, assetBatchSort);
  const visibleBatchItems = sortedBatchItems.slice((safeAssetBatchPage - 1) * assetBatchPageSize, safeAssetBatchPage * assetBatchPageSize);
  const effectiveBatchMax = Math.max(assetNeedCount, Number(batchCallableMax || selectedBatch?.callable_assets || 0));

  const toggleAssetBatchSort = (key: AssetBatchSortKey) => {
    setAssetBatchSort((current) => ({ key, direction: current.key === key && current.direction === "desc" ? "asc" : "desc" }));
    setAssetBatchPage(1);
  };

  const chooseBatch = (batch: any) => {
    setSelectedBatchId(batch.batch_id);
    const selectableCount = Number(batch.selectable_assets || batch.callable_assets || 0);
    const maxCount = Math.min(selectableCount, Array.isArray(batch.asset_ids) ? batch.asset_ids.length : selectableCount);
    const nextCount = Math.max(assetNeedCount, maxCount || assetNeedCount);
    setSampleCount(String(nextCount));
    setSelectedAssetIds((batch.asset_ids || []).slice(0, nextCount));
  };

  return (
    <div className="gov-stack">
      <GovPanel className="gov-step-panel">
        {(WIZARD_STEPS as Array<{ step: any; label: any }>).map(({ step, label }: { step: any; label: any }) => (
          <button key={label} type="button" className={`gov-step ${currentStep === step ? "active" : ""} ${currentStep > step ? "done" : ""}`} onClick={() => setCurrentStep(step)}>
            <span>{step}</span>
            <strong>{label}</strong>
          </button>
        ))}
      </GovPanel>

      <div className="gov-form-layout">
        <GovPanel title={(WIZARD_STEPS as any[]).find((item: any) => item.step === currentStep)?.label}>
          <div className="gov-form">
            {currentStep === 1 ? (
              <>
                <label htmlFor="task-name">
                  <span>任务名称</span>
                  <input id="task-name" name="taskName" value={taskName} onChange={(event) => setTaskName(event.target.value)} />
                </label>
                <div className="gov-radio-row">
                  <span>任务类型</span>
                  <button type="button" className={taskKind === "vlr" ? "selected" : ""} onClick={() => setTaskKind("vlr")}>
                    图文检索
                  </button>
                  <button type="button" className={taskKind === "vqa" ? "selected" : ""} onClick={() => setTaskKind("vqa")}>
                    视觉问答
                  </button>
                  <button type="button" className={taskKind === "caption" ? "selected" : ""} onClick={() => setTaskKind("caption")}>
                    图像描述
                  </button>
                </div>
                <div className="gov-mode-choice-grid" aria-label="测评方式">
                  <button type="button" className={`gov-mode-choice ${evaluationMode === "assets" ? "selected" : ""}`} onClick={() => setEvaluationMode("assets")}>
                    <strong>从对抗样本库调用</strong>
                    <span>先选择已入库、可复用的对抗样本集，再对当前模型做自动化测评。</span>
                  </button>
                  <button type="button" className={`gov-mode-choice ${evaluationMode === "generate" ? "selected" : ""}`} onClick={() => setEvaluationMode("generate")}>
                    <strong>即时生成并测评</strong>
                    <span>本次从数据集抽样生成新对抗样本，并把样本、参数和证据沉淀到样本库。</span>
                  </button>
                </div>
                <label htmlFor="victim-model">
                  <span>测评对象</span>
                  <select id="victim-model" name="victimModel" value={selectedVictim?.adapter || victimAdapter} onChange={(event) => setVictimAdapter(event.target.value)}>
                    {(taskCompatibleVictims as any[]).map((model: any) => {
                      const label = `${formatAdapterName(model.adapter)} · ${formatModelDisplayText(model)} · ${formatHealthStatus(model.health_status)}`;
                      return <option key={model.adapter} value={model.adapter} title={label}>{label}</option>;
                    })}
                    {!taskCompatibleVictims.length ? <option value="">当前没有可用于该任务的模型</option> : null}
                  </select>
                  <small className="gov-field-help">
                    {isGenerationTask
                      ? "生成式任务必须选择可接收图片并生成答案或描述的真实视觉语言模型。"
                      : "受测模型负责最终评分或生成结果；攻击代理模型会在第 3 步随攻击方法一起配置。"}
                  </small>
                </label>

                <div className="gov-model-detail">
                  <span>模型状态</span>
                  <div id="victim-model-detail" className={selectedVictimCanSubmit ? "ok" : "warn"}>
                    <strong>{formatAdapterName(selectedVictim?.adapter || "-")}</strong>
                    <p>{formatModelDisplayText(selectedVictim)}</p>
                    <em>
                      {selectedVictimCanSubmit
                        ? "可提交测评"
                        : isGenerationTask && !selectedModelSupportsTask
                          ? "当前模型不支持该任务；视觉问答和图像描述不能选择演示模型、CLIP、BLIP 或 ViLT"
                          : "当前状态不可提交，需先补齐模型入口或启动服务"}
                    </em>
                  </div>
                </div>

                {isGenerationTask ? (
                  <div className="gov-model-detail">
                    <span>受测范围</span>
                    <div className="ok">
                      <strong>当前真实生成模型</strong>
                      <p>视觉问答和图像描述按同一图片和同一问题或描述指令逐条生成输出，一次只提交当前受测模型。</p>
                      <em>如需横向比较多个生成模型，请分别创建独立任务。</em>
                    </div>
                  </div>
                ) : (
                  <div className="gov-radio-row">
                    <span>受测范围</span>
                    <button type="button" className={victimScope === "selected" ? "selected" : ""} onClick={() => setVictimScope("selected")}>
                      仅当前选择模型
                    </button>
                    <button type="button" className={victimScope === "all" ? "selected" : ""} onClick={() => setVictimScope("all")}>
                      全部可提交模型
                    </button>
                  </div>
                )}
              </>
            ) : null}

            {currentStep === 2 ? (
              evaluationMode === "assets" ? (
                <>
                  <div className="gov-asset-intro">
                    <strong>选择要调用的对抗样本集</strong>
                    <p>这里按生成批次调用资产，不再逐条勾选样本。攻击参数、原始图像、对抗图像和文本输入都来自该批次的生成记录。</p>
                    <a className="gov-inline-link" href="/samples">打开对抗样本库</a>
                  </div>
                  <div className="gov-asset-toolbar">
                    <button type="button" className="gov-button ghost" disabled={!selectedBatch} onClick={() => selectedBatch ? chooseBatch(selectedBatch) : undefined}>使用当前样本集</button>
                    <button type="button" className="gov-button ghost" disabled={!selectedBatch} onClick={() => selectedBatch ? chooseBatch(selectedBatch) : undefined}>恢复最大数量</button>
                    <span className={selectedAssetReady ? "ok" : "warn"}>{selectedBatch ? `已选样本集，调用 ${chosenAssetCount} / ${effectiveBatchMax}` : "等待选择样本集"}</span>
                  </div>
                  <label className="short" htmlFor="asset-batch-count">
                    <span>本次调用样本数</span>
                    <input id="asset-batch-count" type="number" min={assetNeedCount} max={effectiveBatchMax} value={sampleCount} disabled={!selectedBatch} onChange={(event) => setSampleCount(event.target.value)} />
                    <small className="gov-field-help">默认调用所选样本集内全部证据完整样本；样本集只有 1 条也可提交，报告会标注样本规模边界。</small>
                  </label>
                  <div className="gov-table-toolbar">
                    <div className="gov-table-note">共 {assetBatchTotal} 个{pendingAssetMode ? "待测评" : "可调用"}样本集；点击表格行或左侧方格即可调用该批次。</div>
                    <div className="gov-table-pagination">
                      <label>每页行数<select value={assetBatchPageSize} onChange={(event) => { setAssetBatchPageSize(Number(event.target.value)); setAssetBatchPage(1); }}><option value={10}>10</option><option value={20}>20</option><option value={50}>50</option></select></label>
                      <button type="button" disabled={safeAssetBatchPage <= 1} onClick={() => setAssetBatchPage(Math.max(1, safeAssetBatchPage - 1))}>上一页</button>
                      <span>{assetBatchTotal ? `${(safeAssetBatchPage - 1) * assetBatchPageSize + 1} - ${Math.min(assetBatchTotal, safeAssetBatchPage * assetBatchPageSize)} / ${assetBatchTotal}` : "0 / 0"}</span>
                      <button type="button" disabled={safeAssetBatchPage >= assetBatchTotalPages} onClick={() => setAssetBatchPage(Math.min(assetBatchTotalPages, safeAssetBatchPage + 1))}>下一页</button>
                    </div>
                  </div>
                  <div className="gov-table-wrap">
                    <table className="gov-table gov-table-roomy att-asset-select-table">
                      <thead>
                        <tr>
                          <th className="sample-selector-cell" aria-label="选择"></th>
                          <th aria-sort={ariaSort(assetBatchSort, "batch_id")}><AssetBatchSortHeader sort={assetBatchSort} sortKey="batch_id" label="资产批次" onSort={toggleAssetBatchSort} /></th>
                          <th aria-sort={ariaSort(assetBatchSort, "task_dataset")}><AssetBatchSortHeader sort={assetBatchSort} sortKey="task_dataset" label="任务 / 数据集" onSort={toggleAssetBatchSort} /></th>
                          <th aria-sort={ariaSort(assetBatchSort, "attack")}><AssetBatchSortHeader sort={assetBatchSort} sortKey="attack" label="攻击 / 扰动" onSort={toggleAssetBatchSort} /></th>
                          <th aria-sort={ariaSort(assetBatchSort, "sample_count")}><AssetBatchSortHeader sort={assetBatchSort} sortKey="sample_count" label="样本规模 / 证据" onSort={toggleAssetBatchSort} /></th>
                          <th aria-sort={ariaSort(assetBatchSort, "avg_l2")}><AssetBatchSortHeader sort={assetBatchSort} sortKey="avg_l2" label="平均扰动" onSort={toggleAssetBatchSort} /></th>
                          <th aria-sort={ariaSort(assetBatchSort, "risk")}><AssetBatchSortHeader sort={assetBatchSort} sortKey="risk" label="风险 / 调用" onSort={toggleAssetBatchSort} /></th>
                        </tr>
                      </thead>
                      <tbody>
                        {visibleBatchItems.map((batch: any) => {
                          const selected = batch.batch_id === (selectedBatch?.batch_id || selectedBatchId);
                          return (
                            <tr key={batch.batch_id} className={selected ? "is-selected" : ""} tabIndex={0} onClick={() => chooseBatch(batch)} onKeyDown={(event: any) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); chooseBatch(batch); } }}>
                              <td className="sample-selector-cell"><button type="button" className={`sample-select-control ${selected ? "selected" : ""}`} aria-label={selected ? `当前选择 ${batch.batch_id}` : `选择 ${batch.batch_id}`} title={selected ? "当前选择" : "选择该样本集"} onClick={(event: any) => { event.stopPropagation(); chooseBatch(batch); }}><span aria-hidden="true" /></button></td>
                              <td><strong>{batch.batch_id}</strong><span className={`sample-status ${selected ? "ok" : pendingAssetMode ? "warn" : "muted"}`}>{selected ? "当前调用" : pendingAssetMode ? "待测评" : "可选择"}</span></td>
                              <td><strong>{formatRunDatasetName(batch.benchmark_tag || batch.dataset_name, batch.dataset_name, batch.task_kind)}</strong><span>{batch.task_kind === "vlr" ? "图文检索" : batch.task_kind === "vqa" ? "视觉问答" : batch.task_kind === "caption" ? "图像描述" : "混合任务"}</span></td>
                              <td><strong>{formatAttackName(batch.attack)}</strong><span>{formatAttackScopeName(batch.attack_scope || "")}</span></td>
                              <td><strong>{Number(batch.selectable_assets || batch.callable_assets || 0)} / {batch.total_assets}</strong><span>证据完整 {percentLabel(batch.evidence_integrity)}</span></td>
                              <td><strong>L2 {Number(batch.avg_l2 || 0).toFixed(4)}</strong><span>Linf {Number(batch.avg_linf || 0).toFixed(4)}</span></td>
                              <td>{pendingAssetMode ? <><strong>待测评</strong><span>提交后生成风险</span></> : <><strong>{Number(batch.avg_risk_score || 0).toFixed(2)}</strong><span>批次调用 {batch.batch_call_count ?? batch.used_count ?? 0} 次</span></>}</td>
                            </tr>
                          );
                        })}
                        {!sampleAssetLoading && !batchItems.length ? <tr><td colSpan={7}>{pendingAssetMode ? "当前没有待测评样本集。" : "当前没有可直接调用的样本集，请先在对抗样本库生成至少 1 条证据完整样本。"}</td></tr> : null}
                        {sampleAssetLoading ? <tr><td colSpan={7}>正在读取样本集。</td></tr> : null}
                      </tbody>
                    </table>
                  </div>
                  <div className="gov-model-detail">
                    <span>已选样本集</span>
                    <div className={selectedAssetReady ? "ok" : "warn"}>
                      <strong>{selectedBatch?.batch_id || "未选择样本集"}</strong>
                      <p>{selectedBatch ? `${formatAttackName(selectedBatch.attack)} · ${formatRunDatasetName(selectedBatch.benchmark_tag || selectedBatch.dataset_name, selectedBatch.dataset_name, selectedBatch.task_kind)} · 本次调用 ${chosenAssetCount} 条` : pendingAssetMode ? `请先选择待测评样本批次。` : `请先选择有可调用样本的批次。`}</p>
                      <em>报告会记录样本集批次和被调用的样本编号，单条证据仍可回到来源案例查看。</em>
                    </div>
                  </div>
                  {selectedBatch ? (
                    <dl className="gov-confirm-list gov-asset-selected-summary">
                      <dt>{pendingAssetMode ? "待测评样本" : "可调用样本"}</dt><dd>{chosenAssetCount} / {effectiveBatchMax}</dd>
                      <dt>证据完整度</dt><dd>{percentLabel(selectedBatch.evidence_integrity)}</dd>
                      <dt>平均风险</dt><dd>{pendingAssetMode ? "待测评" : Number(selectedBatch.avg_risk_score || 0).toFixed(2)}</dd>
                    </dl>
                  ) : null}
                </>
              ) : (
              <>
                <label htmlFor="dataset-id">
                  <span>数据集</span>
                  {isGenerationTask ? (
                    <select id="dataset-id" name="datasetId" value={selectedGenerationDataset?.key || datasetId} onChange={(event) => setDatasetId(event.target.value)} disabled={!taskGenerationDatasets.length}>
                      {(taskGenerationDatasets as any[]).map((dataset: any) => (
                        <option key={dataset.key} value={dataset.key} title={`${formatDatasetName(dataset.key || dataset.name)} · ${dataset.item_count ?? 0} 条`}>
                          {formatDatasetName(dataset.key || dataset.name)} · {dataset.item_count ?? 0} 条
                        </option>
                      ))}
                      {!taskGenerationDatasets.length ? <option value="">当前没有可用于该任务的真实数据集</option> : null}
                    </select>
                  ) : (
                    <select id="dataset-id" name="datasetId" value={datasetId} onChange={(event) => setDatasetId(event.target.value)}>
                      {(vlrDatasets as any[]).map((dataset: any) => (
                        <option key={dataset.key} value={dataset.key} title={`${formatDatasetName(dataset.key || dataset.name)} · ${dataset.item_count ?? 0} 条`}>
                          {formatDatasetName(dataset.key || dataset.name)} · {dataset.item_count ?? 0} 条
                        </option>
                      ))}
                    </select>
                  )}
                  <small className="gov-field-help">
                    {isGenerationTask
                      ? "生成式任务使用真实 COCO/VQA 清单文件（JSONL），逐条读取图片、问题或描述目标和标准答案，不再构造 N×N 图文矩阵。"
                      : "检索任务使用真实检索数据集或正式验证子集，报告会记录图文配对数和攻击前后检索指标。"}
                  </small>
                </label>

                <label className="short" htmlFor="sample-count">
                  <span>样本条数</span>
                  <input id="sample-count" name="sampleCount" type="number" min="1" value={sampleCount} onChange={(event) => setSampleCount(event.target.value)} />
                  <small className="gov-field-help">样本条数不再设置人为上限；实际纳入数量只受数据集可用样本数影响。当前预计实际样本数为 {effectiveSampleCount}。</small>
                </label>

                <div className="gov-model-detail">
                  <span>数据规模</span>
                  <div className="ok">
                    <strong>{isGenerationTask ? generationDatasetName : formatDatasetName(selectedDataset?.key || datasetId || "未选择")}</strong>
                    <p>{isGenerationTask ? `生成式样本 ${generationCaseCount} 条；本次提交 ${effectiveSampleCount} 条。` : `数据集条目 ${datasetItemCount || "未记录"}；预计图文配对 ${effectivePairBudget}。`}</p>
                    <em>样本数会写入任务配置，后续报告、结果明细、样本回放都按同一入口追踪。</em>
                  </div>
                </div>
              </>
              )
            ) : null}

            {currentStep === 3 ? (
              evaluationMode === "assets" ? (
                <>
                  <div className={`gov-submit-readiness ${selectedAssetReady ? "ready" : "missing"}`}>
                    <strong>{selectedAssetReady ? "样本集参数已固定" : `还需要选择一个可调用样本集`}</strong>
                    <p>攻击方法、扰动预算和高级参数来自样本集生成记录；测评阶段只更换受测模型并生成新的报告与案例入口。</p>
                  </div>
                  <dl className="gov-confirm-list">
                    <dt>调用样本集</dt><dd>{selectedBatch?.batch_id || "未选择"}</dd>
                    <dt>本次样本</dt><dd>{chosenAssetCount} 个</dd>
                    <dt>参数来源</dt><dd>对抗样本库生成记录，只读</dd>
                    <dt>报告输出</dt><dd>保留运行指标、来源批次、样本编号和案例复盘入口</dd>
                  </dl>
                </>
              ) : (
              <>
                <div className="gov-attack-groups" aria-label="攻击方式分组">
                  {(groupedAttacks as any[]).map((group: any) => (
                    <section key={group.id} className="gov-attack-group">
                      <header>
                        <div>
                          <h3>{group.title}</h3>
                          <p>{group.subtitle}</p>
                        </div>
                        <span>{group.items.length} 项</span>
                      </header>
                      <div className="gov-attack-grid">
                        {(group.items as any[]).map((item: any) => {
                          const isExternal = EXTERNAL_RUNTIME_ATTACKS.has(item.id);
                          const runtime = externalStatuses[item.id];
                          const notReady = isExternal && runtime && runtime.runnable === false;
                          return (
                            <button key={item.id} type="button" className={`gov-attack-card ${attack === item.id ? "selected" : ""} ${notReady ? "not-ready" : ""}`} onClick={() => { setAttack(item.id); applyAttackDefaults(item.id); }}>
                              <span className="gov-attack-card-title">
                                <i>{attack === item.id ? <CheckIcon /> : null}</i>
                                <strong>{formatAttackName(item.id)}</strong>
                              </span>
                              <span className="gov-attack-card-meta">{item.perturbation} · {item.modality}</span>
                              <span className="gov-attack-card-summary">{item.id === "vqa_visual_corruption" ? "视觉退化鲁棒性基线，调用官方退化生成函数。" : item.summary}</span>
                              {isExternal ? <ExternalStatusPills status={runtime} /> : null}
                            </button>
                          );
                        })}
                      </div>
                    </section>
                  ))}
                </div>

                <label htmlFor="attack-surrogate-model">
                  <span>{CLIP_AUXILIARY_SURROGATE_ATTACKS.has(attack) ? "辅助代理模型" : "攻击代理模型"}</span>
                  <select id="attack-surrogate-model" name="surrogateModel" value={selectedSurrogate?.adapter || surrogate} onChange={(event) => setSurrogate(event.target.value)} disabled={CLIP_AUXILIARY_SURROGATE_ATTACKS.has(attack)}>
                    {(compatibleSurrogates as any[]).map((model: any) => {
                      const label = `${formatAdapterName(model.adapter)} · ${formatModelDisplayText(model)} · ${formatHealthStatus(model.health_status)}`;
                      return <option key={model.adapter} value={model.adapter} title={label}>{label}</option>;
                    })}
                    {!compatibleSurrogates.length ? <option value="">当前攻击方法没有可用代理模型</option> : null}
                  </select>
                  <small className="gov-field-help">{surrogateHelpForAttack(attack, isGenerationTask)}</small>
                </label>

                {externalAttackSelected ? (
                  <div className={`gov-external-runtime ${selectedExternalRunnable ? "ready" : "missing"}`}>
                    <strong>{selectedExternalRunnable ? "外部攻击配置完整" : "外部攻击配置未完整"}</strong>
                    <p>{selectedExternalStatus?.config_path ? `状态来源：${selectedExternalStatus.config_path}` : "后端暂未返回配置状态；提交时仍会由真实攻击执行链路检查。"}</p>
                    {selectedExternalStatus?.messages?.length ? (
                      <ul>
                        {(selectedExternalStatus.messages as string[]).map((message: string) => <li key={message}>{message}</li>)}
                      </ul>
                    ) : null}
                  </div>
                ) : null}

                <div className="gov-param-mode">
                  <span>参数模式</span>
                  <div>
                    <button type="button" className={attackParamMode === "standard" ? "selected" : ""} onClick={() => setAttackParamMode("standard")}>标准参数</button>
                    <button type="button" className={attackParamMode === "advanced" ? "selected" : ""} onClick={() => setAttackParamMode("advanced")}>高级参数</button>
                  </div>
                </div>

                <div className="gov-attack-params">

                  {budgetControl ? (
                    <div className="gov-strength-row">
                      <span>{budgetControl.label}</span>
                      <div className="gov-strength-control">
                        <div>
                          <input
                            id="attack-strength"
                            name="attackStrength"
                            aria-label={`${budgetControl.label}滑块`}
                            type="range"
                            min="0.005"
                            max="0.1"
                            step="0.005"
                            value={strength}
                            onChange={(event) => setStrength(event.target.value)}
                          />
                          <small>{budgetControl.help} 当前扰动上限 ε={strengthNumber.toFixed(3)}，约 {strength255}/255。</small>
                        </div>
                        <input
                          id="attack-strength-number"
                          name="attackStrengthNumber"
                          aria-label={`${budgetControl.label}数值`}
                          type="number"
                          min="0.005"
                          max="0.1"
                          step="0.005"
                          value={strength}
                          onChange={(event) => setStrength(event.target.value)}
                        />
                      </div>
                    </div>
                  ) : (
                    <div className="gov-param-hint">
                      <strong>该攻击不使用扰动预算</strong>
                      <p>{attack === "vqa_visual_corruption" ? "视觉退化攻击由退化类型、严重度和随机种子控制，不向后端提交扰动上限 ε。" : "当前攻击的强度由方法专用参数控制，不显示无效预算项。"}</p>
                    </div>
                  )}

                  {attackParamMode === "standard" ? (
                    <div className="gov-param-hint">
                      <strong>标准参数</strong>
                      <p>{usesOfficialExternalAlignmentRecipe(attack) ? "特征最优对齐迁移攻击（FOA-Attack）、局部语义匹配迁移攻击（M-Attack）和多范式协同迁移攻击（MPCAttack）按官方配置执行：CUDA、300 步、B16+B32+Laion 集成、16/255 预算。" : budgetControl ? "当前页面显示并提交该攻击真实使用的标准参数；样本数只控制抽样规模，不改变优化步数。" : "该攻击没有通用扰动预算；标准参数沿用服务器方法配置中的专用字段，只调整样本数。"}</p>
                    </div>
                  ) : (
                    <div className="gov-advanced-panel">
                      <div className="gov-advanced-grid">
                        {attackUsesSteps(attack) || attack === "advclip" ? (
                          <>
                            <label htmlFor="advanced-steps">
                              <span>{attack === "advclip" ? "补丁训练步数" : "优化步数"}</span>
                              <input id="advanced-steps" type="number" min="1" max="3000" value={advancedSteps} onChange={(event) => setAdvancedSteps(event.target.value)} />
                              <small className="gov-field-help">{stepHelpForAttack(attack)}</small>
                            </label>
                          </>
                        ) : null}

                        {attackUsesStepSize(attack) ? (
                          <>
                            <label htmlFor="advanced-step-size">
                              <span>{attack === "advclip" ? "训练步长" : "单步步长"}</span>
                              <input id="advanced-step-size" type="number" min="0.0001" max="0.1" step="0.0005" value={advancedStepSize} onChange={(event) => setAdvancedStepSize(event.target.value)} />
                              <small className="gov-field-help">{stepSizeHelpForAttack(attack)}</small>
                            </label>
                          </>
                        ) : null}

                        {["advclip", "advedm", "advedm_plus", "tmm"].includes(attack) ? (
                          <label htmlFor="attack-mode-override">
                            <span>攻击模式</span>
                            <select id="attack-mode-override" value={attackModeOverride} onChange={(event) => setAttackModeOverride(event.target.value)}>
                              {(ATTACK_MODE_OPTIONS as Array<[string, string]>).map(([value, label]: [string, string]) => <option key={value} value={value}>{label}</option>)}
                            </select>
                          </label>
                        ) : null}

                        {["advclip", "advedm", "advedm_plus"].includes(attack) ? (
                          <label htmlFor="patch-size">
                            <span>补丁尺寸</span>
                            <input id="patch-size" type="number" min="4" max="96" step="1" value={patchSize} onChange={(event) => setPatchSize(event.target.value)} />
                          </label>
                        ) : null}

                        {attack === "advclip" ? (
                          <>
                            <label htmlFor="lambda-at">
                              <span>注意拓扑权重</span>
                              <input id="lambda-at" type="number" min="0" max="20" step="0.1" value={lambdaAt} onChange={(event) => setLambdaAt(event.target.value)} />
                            </label>
                            <label htmlFor="lambda-tpd">
                              <span>局部差异权重</span>
                              <input id="lambda-tpd" type="number" min="0" max="20" step="0.1" value={lambdaTpd} onChange={(event) => setLambdaTpd(event.target.value)} />
                            </label>
                            <label htmlFor="tau-patch">
                              <span>补丁温度</span>
                              <input id="tau-patch" type="number" min="0.001" max="2" step="0.001" value={tauPatch} onChange={(event) => setTauPatch(event.target.value)} />
                            </label>
                          </>
                        ) : null}

                        {(attack === "advedm" || attack === "advedm_plus") ? (
                          <>
                            <label htmlFor="region-topk">
                              <span>语义区域前 k 项</span>
                              <input id="region-topk" type="number" min="1" max="64" step="1" value={topK} onChange={(event) => setTopK(event.target.value)} />
                            </label>
                            <label htmlFor="region-threshold">
                              <span>区域阈值</span>
                              <input id="region-threshold" type="number" min="0" max="1" step="0.05" value={regionThreshold} onChange={(event) => setRegionThreshold(event.target.value)} />
                            </label>
                            <label htmlFor="alpha-weight">
                              <span>目标权重 α（alpha）</span>
                              <input id="alpha-weight" type="number" min="0" max="100" step="0.5" value={alphaWeight} onChange={(event) => setAlphaWeight(event.target.value)} />
                            </label>
                            <label htmlFor="beta-weight">
                              <span>补丁权重 β（beta）</span>
                              <input id="beta-weight" type="number" min="0" max="100" step="0.5" value={betaWeight} onChange={(event) => setBetaWeight(event.target.value)} />
                            </label>
                            <label htmlFor="gamma-weight">
                              <span>保真权重 γ（gamma）</span>
                              <input id="gamma-weight" type="number" min="0" max="100" step="0.1" value={gammaWeight} onChange={(event) => setGammaWeight(event.target.value)} />
                            </label>
                          </>
                        ) : null}

                        {attack === "tmm" ? (
                          <>
                            <label htmlFor="lambda-att">
                              <span>注意阈值</span>
                              <input id="lambda-att" type="number" min="0" max="1" step="0.05" value={lambdaAtt} onChange={(event) => setLambdaAtt(event.target.value)} />
                            </label>
                            <label htmlFor="ratio-r">
                              <span>非关键预算比例</span>
                              <input id="ratio-r" type="number" min="0" max="1" step="0.05" value={ratioR} onChange={(event) => setRatioR(event.target.value)} />
                            </label>
                          </>
                        ) : null}

                        {attack === "vqa_visual_corruption" ? (
                          <>
                            <label htmlFor="corruption-type">
                              <span>退化类型</span>
                              <select id="corruption-type" value={corruptionType} onChange={(event) => setCorruptionType(event.target.value)}>
                                {(VQA_CORRUPTION_OPTIONS as Array<[string, string]>).map(([value, label]: [string, string]) => <option key={value} value={value}>{label}</option>)}
                              </select>
                            </label>
                            <label htmlFor="corruption-severity">
                              <span>严重度</span>
                              <input id="corruption-severity" type="number" min="1" max="5" value={corruptionSeverity} onChange={(event) => setCorruptionSeverity(event.target.value)} />
                            </label>
                            <label htmlFor="corruption-seed">
                              <span>随机种子</span>
                              <input id="corruption-seed" type="number" min="0" value={corruptionSeed} onChange={(event) => setCorruptionSeed(event.target.value)} />
                            </label>
                          </>
                        ) : null}

                        {attack === "xtransfer_uap" ? (
                          <>
                            <label htmlFor="uap-name">
                              <span>通用扰动名称（UAP）</span>
                              <input id="uap-name" value={uapName} onChange={(event) => setUapName(event.target.value)} />
                            </label>
                            <label htmlFor="threat-model">
                              <span>威胁模型</span>
                              <select id="threat-model" value={threatModel} onChange={(event) => setThreatModel(event.target.value)}>
                                {(THREAT_MODEL_OPTIONS as Array<[string, string]>).map(([value, label]: [string, string]) => <option key={value} value={value}>{label}</option>)}
                              </select>
                            </label>
                          </>
                        ) : null}

                        {(attack === "tmm" || attack === "advedm_plus") ? (
                          <>
                            <label htmlFor="text-budget">
                              <span>文本替换预算</span>
                              <input id="text-budget" type="number" min="0" max="5" step="1" value={textBudget} onChange={(event) => setTextBudget(event.target.value)} />
                            </label>
                            <label htmlFor="text-candidates-k">
                              <span>候选词数</span>
                              <input id="text-candidates-k" type="number" min="1" max="64" step="1" value={textCandidatesK} onChange={(event) => setTextCandidatesK(event.target.value)} />
                            </label>
                          </>
                        ) : null}

                        {["foa_attack", "anyattack", "mpc_attack", "m_attack"].includes(attack) ? (
                          <>
                            <label htmlFor="target-image-override">
                              <span>目标图</span>
                              <input id="target-image-override" value={targetImageOverride} onChange={(event) => setTargetImageOverride(event.target.value)} placeholder="留空使用服务器默认目标图" />
                            </label>
                          </>
                        ) : null}

                        {attack === "mpc_attack" ? (
                          <>
                            <label htmlFor="target-text-override">
                              <span>目标文本</span>
                              <input id="target-text-override" value={targetTextOverride} onChange={(event) => setTargetTextOverride(event.target.value)} placeholder="留空使用服务器默认文本" />
                            </label>
                          </>
                        ) : null}

                        {["foa_attack", "mpc_attack", "m_attack"].includes(attack) ? (
                          <>
                            <label htmlFor="crop-scale">
                              <span>裁剪尺度</span>
                              <input id="crop-scale" type="number" min="0.05" max="1" step="0.05" value={cropScale} onChange={(event) => setCropScale(event.target.value)} />
                            </label>
                            <label htmlFor="crop-ratio">
                              <span>裁剪比例</span>
                              <input id="crop-ratio" type="number" min="0.05" max="1" step="0.05" value={cropRatio} onChange={(event) => setCropRatio(event.target.value)} />
                            </label>
                          </>
                        ) : null}

                        {(attack === "foa_attack" || attack === "m_attack") ? (
                          <label htmlFor="surrogate-models-override">
                            <span>{attack === "m_attack" ? "集成模型" : "代理模型组"}</span>
                            <input id="surrogate-models-override" value={surrogateModelsOverride} onChange={(event) => setSurrogateModelsOverride(event.target.value)} placeholder="例如 B16,B32,Laion" />
                            <small className="gov-field-help">特征最优对齐迁移攻击（FOA-Attack）和局部语义匹配迁移攻击（M-Attack）官方单图入口默认使用 B16、B32、Laion 三模型集成。</small>
                          </label>
                        ) : null}

                        {attack === "mpc_attack" ? (
                          <>
                            <label htmlFor="clip-backbones-override">
                              <span>CLIP 骨干</span>
                              <input id="clip-backbones-override" value={clipBackbonesOverride} onChange={(event) => setClipBackbonesOverride(event.target.value)} placeholder="例如 B16,B32,Laion" />
                              <small className="gov-field-help">MPCAttack 官方入口默认使用 B16、B32、Laion 三个 CLIP 骨干，并叠加 InternVL/DINO 特征。</small>
                            </label>
                            <label htmlFor="mpc-lam">
                              <span>协同权重 λ（lam）</span>
                              <input id="mpc-lam" type="number" min="0" max="10" step="0.1" value={mpcLam} onChange={(event) => setMpcLam(event.target.value)} />
                            </label>
                            <label htmlFor="mpc-tau">
                              <span>温度系数 τ（tau）</span>
                              <input id="mpc-tau" type="number" min="0" max="10" step="0.1" value={mpcTau} onChange={(event) => setMpcTau(event.target.value)} />
                            </label>
                            <label htmlFor="mpc-omega">
                              <span>平衡项 ω（omega）</span>
                              <input id="mpc-omega" type="number" min="0" max="20" step="0.1" value={mpcOmega} onChange={(event) => setMpcOmega(event.target.value)} />
                            </label>
                          </>
                        ) : null}
                      </div>
                      <p className="gov-field-help standalone">高级参数会写入本次任务的临时覆盖配置；留空的外部路径字段继续使用服务器默认值，不会覆盖已有可运行配置。</p>
                    </div>
                  )}
                </div>
              </>
              )
            ) : null}

            {currentStep === 4 ? (
              <>
                <div className={`gov-submit-readiness ${canSubmit ? "ready" : "missing"}`}>
                  <strong>{canSubmit ? "配置完整，可以提交" : "仍有必填配置未就绪"}</strong>
                  <p>{canSubmit ? "提交后会进入任务队列，报告会保留攻击图、追踪记录、样本入口和样本回放。" : evaluationMode === "assets" ? `请先选择包含至少 ${assetNeedCount} 条可调用样本的样本集。` : "请返回前面的步骤补齐模型、数据集或外部攻击配置。"}</p>
                </div>

                <dl className="gov-confirm-list">
                  <dt>任务名称</dt><dd>{taskName}</dd>
                  <dt>任务类型</dt><dd>{taskKind === "vqa" ? "视觉问答" : taskKind === "caption" ? "图像描述" : "图文检索"}</dd>
                  <dt>测评方式</dt><dd>{evaluationMode === "assets" ? `调用样本集（${chosenAssetCount} 个样本）` : "即时生成并测评"}</dd>
                  <dt>测评对象</dt><dd>{formatAdapterName(selectedVictim?.adapter || victimAdapter || "-")}</dd>
                  <dt>{CLIP_AUXILIARY_SURROGATE_ATTACKS.has(attack) ? "辅助代理" : "攻击代理"}</dt><dd>{formatAdapterName(selectedSurrogate?.adapter || surrogate)}</dd>
                  <dt>数据集</dt><dd>{isGenerationTask ? generationDatasetName : formatDatasetName(selectedDataset?.key || datasetId || "未选择")}</dd>
                  <dt>攻击方式</dt><dd>{attackInfo?.name || formatAttackName(attack)}</dd>
                  <dt>参数模式</dt><dd>{evaluationMode === "assets" ? "来自样本集，只读" : attackParamMode === "advanced" ? "高级参数" : "标准参数"}</dd>
                  <dt>样本条数</dt><dd>{evaluationMode === "assets" ? chosenAssetCount : effectiveSampleCount}</dd>
                  <dt>{budgetControl?.label || "强度参数"}</dt><dd>{budgetControl ? `${strengthNumber.toFixed(3)}（约 ${strength255}/255）` : "不适用"}</dd>
                  <dt>{attack === "advclip" ? "补丁训练步数" : "优化步数"}</dt><dd>{attackUsesSteps(attack) || (attack === "advclip" && attackParamMode === "advanced") ? effectiveSteps : "不适用"}</dd>
                  {attackParamMode === "advanced" && ["advclip", "advedm", "advedm_plus", "tmm"].includes(attack) ? (
                    <>
                      <dt>攻击模式</dt><dd>{(ATTACK_MODE_OPTIONS as Array<[string, string]>).find(([value]: [string, string]) => value === attackModeOverride)?.[1] || attackModeOverride}</dd>
                    </>
                  ) : null}
                  {attackParamMode === "advanced" && ["advclip", "advedm", "advedm_plus"].includes(attack) ? (
                    <>
                      <dt>补丁/区域</dt><dd>补丁尺寸 {effectivePatchSize}{attack === "advedm" || attack === "advedm_plus" ? `，语义区域前 k 项 ${effectiveTopK}` : ""}</dd>
                    </>
                  ) : null}
                  {attackParamMode === "advanced" && (attack === "tmm" || attack === "advedm_plus") ? (
                    <>
                      <dt>文本分支</dt><dd>替换预算 {Math.round(effectiveTextBudget)}，候选词 {effectiveTextCandidatesK}</dd>
                    </>
                  ) : null}
                  {attackParamMode === "advanced" && attack === "vqa_visual_corruption" ? (
                    <>
                      <dt>视觉退化</dt><dd>{(VQA_CORRUPTION_OPTIONS as Array<[string, string]>).find(([value]: [string, string]) => value === corruptionType)?.[1] || corruptionType} · 严重度 {effectiveCorruptionSeverity}</dd>
                    </>
                  ) : null}
                  {attackParamMode === "advanced" && attack === "xtransfer_uap" ? (
                    <>
                      <dt>UAP 配置</dt><dd>{uapName} · {(THREAT_MODEL_OPTIONS as Array<[string, string]>).find(([value]: [string, string]) => value === threatModel)?.[1] || threatModel}</dd>
                    </>
                  ) : null}
                  {attackParamMode === "advanced" && ["foa_attack", "anyattack", "mpc_attack", "m_attack"].includes(attack) ? (
                    <>
                      <dt>目标覆盖</dt><dd>{targetImageOverride.trim() || (attack === "mpc_attack" && targetTextOverride.trim()) ? "已填写本次覆盖目标" : "使用服务器默认目标"}</dd>
                    </>
                  ) : null}
                  {attackParamMode === "advanced" && attack === "mpc_attack" ? (
                    <>
                      <dt>协同参数</dt><dd>lam={boundedNumber(mpcLam, 0.6, 0, 10).toFixed(2)}，tau={boundedNumber(mpcTau, 0.2, 0, 10).toFixed(2)}，omega={boundedNumber(mpcOmega, 2.0, 0, 20).toFixed(2)}</dd>
                    </>
                  ) : null}
                  {externalAttackSelected ? (
                    <>
                      <dt>外部配置</dt><dd><ExternalStatusPills status={selectedExternalStatus} /></dd>
                    </>
                  ) : null}
                </dl>

                <label htmlFor="task-note">
                  <span>备注说明</span>
                  <textarea id="task-note" name="taskNote" maxLength={200} value={note} onChange={(event) => setNote(event.target.value)} placeholder="请输入备注说明（选填）" />
                  <em>{note.length} / 200</em>
                </label>
              </>
            ) : null}
          </div>
        </GovPanel>

        <GovPanel title={sidePanelTitle} className="gov-side-context">
          <dl className="gov-preview-list compact">
            {currentStep === 1 ? (
              <>
                <dt>任务类型</dt><dd>{taskKindLabel}</dd>
                <dt>受测模型</dt><dd>{formatAdapterName(selectedVictim?.adapter || victimAdapter || "-")}</dd>
                <dt>受测范围</dt><dd>{isGenerationTask ? "当前真实生成模型" : victimScope === "all" ? "全部可提交模型" : "仅当前选择模型"}（{submittedVictims.length || 0} 个）</dd>
                <dt>实时就绪</dt><dd>{readyVictims.length} / {formalModels.length}</dd>
                <dt>可提交</dt><dd>{launchableVictims.length} / {formalModels.length}</dd>
              </>
            ) : null}

            {currentStep === 2 ? (
              <>
                {evaluationMode === "assets" ? (
                  <>
                    <dt>已选样本集</dt><dd>{selectedBatch?.batch_id || "等待选择"}</dd>
                    <dt>本次调用</dt><dd>{chosenAssetCount} / {assetNeedCount} 个</dd>
                    <dt>证据完整</dt><dd>{selectedBatch ? percentLabel(selectedBatch.evidence_integrity) : "等待选择"}</dd>
                    <dt>参数归属</dt><dd>来自对抗样本库，当前步骤只选择批次</dd>
                  </>
                ) : (
                  <>
                    <dt>数据集</dt><dd>{isGenerationTask ? generationDatasetName : formatDatasetName(selectedDataset?.key || datasetId || "未选择")}</dd>
                    <dt>提交样本</dt><dd>{sampleCountNumber}</dd>
                    {datasetItemCount > 0 && effectiveSampleCount !== sampleCountNumber ? (
                      <>
                        <dt>实际样本</dt><dd>{effectiveSampleCount} / {datasetItemCount}</dd>
                      </>
                    ) : null}
                    {isGenerationTask ? (
                      <>
                        <dt>生成式查询</dt><dd>{effectiveSampleCount * 2}</dd>
                      </>
                    ) : (
                      <>
                        <dt>图文配对</dt><dd>{effectivePairBudget}</dd>
                        {requestedPairBudget !== effectivePairBudget ? (
                          <>
                            <dt>输入估算</dt><dd>{requestedPairBudget}</dd>
                          </>
                        ) : null}
                      </>
                    )}
                  </>
                )}
              </>
            ) : null}

            {currentStep === 3 ? (
              <>
                {evaluationMode === "assets" ? (
                  <>
                    <dt>测评设置</dt><dd>调用样本集，攻击参数只读</dd>
                    <dt>报告输出</dt><dd>记录批次编号、样本编号、模型输出和案例入口</dd>
                  </>
                ) : (
                  <>
                    <dt>攻击方式</dt><dd>{attackInfo?.name || formatAttackName(attack)}</dd>
                    <dt>攻击分组</dt><dd>{(ATTACK_GROUPS as any[]).find((group: any) => group.id === groupIdForAttack(attack))?.title || "未分组"}</dd>
                    <dt>{CLIP_AUXILIARY_SURROGATE_ATTACKS.has(attack) ? "辅助代理" : "攻击代理"}</dt><dd>{formatAdapterName(selectedSurrogate?.adapter || surrogate)}</dd>
                    <dt>参数模式</dt><dd>{attackParamMode === "advanced" ? "高级参数" : "标准参数"}</dd>
                    <dt>{budgetControl?.label || "强度参数"}</dt><dd>{budgetControl ? `${strengthNumber.toFixed(3)}（约 ${strength255}/255）` : "不适用"}</dd>
                    <dt>{attack === "advclip" ? "补丁训练步数" : "优化步数"}</dt><dd>{attackUsesSteps(attack) || (attack === "advclip" && attackParamMode === "advanced") ? effectiveSteps : "不适用"}</dd>
                    {attackParamMode === "advanced" && ["advclip", "advedm", "advedm_plus", "tmm"].includes(attack) ? (
                      <>
                        <dt>攻击模式</dt><dd>{(ATTACK_MODE_OPTIONS as Array<[string, string]>).find(([value]: [string, string]) => value === attackModeOverride)?.[1] || attackModeOverride}</dd>
                      </>
                    ) : null}
                    {attackParamMode === "advanced" && ["advclip", "advedm", "advedm_plus"].includes(attack) ? (
                      <>
                        <dt>补丁尺寸</dt><dd>{effectivePatchSize}</dd>
                      </>
                    ) : null}
                    <dt>外部配置</dt><dd>{externalAttackSelected ? <ExternalStatusPills status={selectedExternalStatus} /> : "项目内执行"}</dd>
                  </>
                )}
              </>
            ) : null}

            {currentStep === 4 ? (
              <>
                <dt>任务名称</dt><dd>{taskName}</dd>
                <dt>任务类型</dt><dd>{taskKindLabel}</dd>
                <dt>数据来源</dt><dd>{evaluationMode === "assets" ? "对抗样本集" : isGenerationTask ? generationDatasetName : formatDatasetName(selectedDataset?.key || datasetId || "未选择")}</dd>
                <dt>攻击方式</dt><dd>{attackInfo?.name || formatAttackName(attack)}</dd>
                <dt>样本条数</dt><dd>{evaluationMode === "assets" ? chosenAssetCount : effectiveSampleCount}</dd>
                <dt>提交状态</dt><dd>{canSubmit ? "可以提交" : "仍需补齐配置"}</dd>
              </>
            ) : null}
          </dl>

          <div className="gov-side-note">
            <span className="info"><CheckIcon /></span>
            <p>{sidePanelNote}</p>
          </div>
          {currentStep === 3 ? (
            <div className="gov-side-note">
              <span className="warn"><CheckIcon /></span>
              <p>{surrogateRequirementNote(attack) || "当前攻击方法可按所选代理模型执行。"}</p>
            </div>
          ) : null}
          {currentStep === 4 ? (
            <div className="gov-side-note">
              <span className="ok"><CheckIcon /></span>
              <p>风险等级按分数分段：{RISK_THRESHOLDS.map((item) => `${item.level} ${item.range}`).join("；")}。</p>
            </div>
          ) : null}
          {runningJob ? (
            <div className="gov-side-note">
              <span className="info"><CheckIcon /></span>
              <p>当前已有任务排队或运行：{runningJob.id.slice(0, 8)}</p>
            </div>
          ) : null}
        </GovPanel>
      </div>

      <div className="gov-actions">
        <button type="button" className="gov-button ghost" onClick={saveDraft}>保存草稿</button>
        {currentStep > 1 ? <button type="button" className="gov-button ghost" onClick={goPrev}>上一步</button> : null}
        {currentStep < 4 ? (
          <button type="button" className="gov-button primary" disabled={!canAdvance} onClick={goNext}>下一步</button>
        ) : (
          <button type="button" className="gov-button primary" disabled={submitJob.isPending || !canSubmit} onClick={() => submitJob.mutate()}>
            {submitJob.isPending ? "正在提交" : "开始测评"}
          </button>
        )}
      </div>
      {draftStatus ? <div className="gov-success">{draftStatus}</div> : null}
      {submitJob.isError ? <div className="gov-error">提交失败：{String(submitJob.error.message || submitJob.error)}</div> : null}
      {submitJob.data ? <div className="gov-success">任务已提交：{submitJob.data.id}</div> : null}
    </div>
  );
}
