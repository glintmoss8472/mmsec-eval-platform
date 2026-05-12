import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";

import { attackCatalog, attackCatalogMap } from "../lib/attackCatalog";
import { createJob, getSystemOverview, listJobs, listSampleAssetBatches } from "../lib/api";
import { recommendedModels } from "../lib/modelCatalog";
import { formatDatasetName } from "../lib/uiLabels";
import { ExperimentStudioView } from "./experiment/ExperimentStudioView";
import type { AttackParamMode, LaunchMode, TaskKind, VictimScope, WizardStep } from "./experiment/experimentStudioConfig";
import {
  ATTACK_GROUPS,
  ATTACK_MODE_OPTIONS,
  CLIP_AUXILIARY_SURROGATE_ATTACKS,
  DRAFT_STORAGE_KEY,
  EXTERNAL_RUNTIME_ATTACKS,
  ExternalStatusPills,
  GENERATION_DATASET_META,
  OFFICIAL_ALIGNMENT_BACKBONES,
  OFFICIAL_ALIGNMENT_EPSILON,
  OFFICIAL_ALIGNMENT_STEP_SIZE,
  OFFICIAL_ALIGNMENT_STEPS,
  STANDARD_EPSILON,
  THREAT_MODEL_OPTIONS,
  VQA_CORRUPTION_OPTIONS,
  WIZARD_STEPS,
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
  formatModelDisplayText,
  generationDatasetSupportsTask,
  groupIdForAttack,
  inferredTaskCapabilities,
  modelSupportsTask,
  splitList,
  stepHelpForAttack,
  stepSizeHelpForAttack,
  surrogateHelpForAttack,
  surrogateRequirementNote,
  surrogateSelectableForRun,
  usesOfficialExternalAlignmentRecipe,
  victimSelectableForLaunch,
} from "./experiment/experimentStudioConfig";
export { generationCapableAdapter, modelTaskCapabilities, modelSupportsTask, surrogateRequirementNote, surrogateSupportedForAttack, victimSelectableForLaunch } from "./experiment/experimentStudioConfig";

const ASSET_MIN_SAMPLE_COUNT = 1;

export default function ExperimentStudioPage() {
  const qc = useQueryClient();
  const [searchParams] = useSearchParams();
  const querySelectionAppliedRef = useRef(false);
  const attackMap = useMemo(() => attackCatalogMap(), []);
  const overviewQ = useQuery({
    queryKey: ["new-eval-overview"],
    queryFn: getSystemOverview,
    refetchInterval: 5000,
  });
  const jobsQ = useQuery({
    queryKey: ["new-eval-jobs"],
    queryFn: () => listJobs({ page: 1, page_size: 20 }),
    refetchInterval: 5000,
  });
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
        role: item.adapter === "clip_hf" ? "surrogate/local" : item.family === "local" ? "victim/local" : "victim/api_or_self_hosted",
        formal_eval: true,
        task_capabilities: inferredTaskCapabilities(item.adapter),
      }));
  const formalModels = models.filter((model) => model.formal_eval !== false);
  const datasets = (overviewQ.data?.datasets ?? []).filter((item) => item.ready !== false);
  const vlrDatasets = datasets.filter((item) => {
    const tier = String(item.tier || "").trim();
    return tier !== "generation" && tier !== "demo";
  });
  const generationDatasets = datasets.filter((item) => String(item.tier || "").trim() === "generation");
  const attacks = overviewQ.data?.attacks?.length ? attackCatalog.filter((item) => overviewQ.data?.attacks.includes(item.id)) : attackCatalog;

  const [taskName, setTaskName] = useState("多模态模型综合安全测评");
  const [taskKind, setTaskKind] = useState<TaskKind>("vlr");
  const [surrogate, setSurrogate] = useState("clip_hf");
  const [victimAdapter, setVictimAdapter] = useState("");
  const [datasetId, setDatasetId] = useState("");
  const [attack, setAttack] = useState("advedm_plus");
  const [victimScope, setVictimScope] = useState<VictimScope>("selected");
  const [strength, setStrength] = useState(STANDARD_EPSILON);
  const [sampleCount, setSampleCount] = useState("8");
  const [note, setNote] = useState("");
  const [draftStatus, setDraftStatus] = useState("");
  const [launchMode] = useState<LaunchMode>("standard");
  const [currentStep, setCurrentStep] = useState<WizardStep>(1);
  const [evaluationMode, setEvaluationMode] = useState<"assets" | "generate">(() => (searchParams.get("mode") === "assets" || searchParams.get("asset") || searchParams.get("batch") ? "assets" : "generate"));
  const assetBatchStatus = searchParams.get("asset_status") === "pending_evaluation" ? "pending_evaluation" : "ready";
  const [selectedBatchId, setSelectedBatchId] = useState(() => searchParams.get("batch") || "");
  const [selectedAssetIds, setSelectedAssetIds] = useState<string[]>(() => {
    const asset = searchParams.get("asset");
    return asset ? [asset] : [];
  });
  const [attackParamMode, setAttackParamMode] = useState<AttackParamMode>("standard");
  const [attackModeOverride, setAttackModeOverride] = useState("A");
  const [advancedSteps, setAdvancedSteps] = useState("12");
  const [advancedStepSize, setAdvancedStepSize] = useState("0.008");
  const [textBudget, setTextBudget] = useState("1");
  const [patchSize, setPatchSize] = useState("32");
  const [topK, setTopK] = useState("6");
  const [regionThreshold, setRegionThreshold] = useState("0.5");
  const [alphaWeight, setAlphaWeight] = useState("10");
  const [betaWeight, setBetaWeight] = useState("5");
  const [gammaWeight, setGammaWeight] = useState("1");
  const [lambdaAtt, setLambdaAtt] = useState("0.5");
  const [ratioR, setRatioR] = useState("0.4");
  const [textCandidatesK, setTextCandidatesK] = useState("12");
  const [lambdaAt, setLambdaAt] = useState("1");
  const [lambdaTpd, setLambdaTpd] = useState("1");
  const [tauPatch, setTauPatch] = useState("0.07");
  const [corruptionType, setCorruptionType] = useState("gaussian_noise");
  const [corruptionSeverity, setCorruptionSeverity] = useState("2");
  const [corruptionSeed, setCorruptionSeed] = useState("7");
  const [uapName, setUapName] = useState("xtransfer_large_linf_eps12_non_targeted");
  const [threatModel, setThreatModel] = useState("linf_non_targeted");
  const [targetImageOverride, setTargetImageOverride] = useState("");
  const [targetTextOverride, setTargetTextOverride] = useState("");
  const [surrogateModelsOverride, setSurrogateModelsOverride] = useState(OFFICIAL_ALIGNMENT_BACKBONES);
  const [clipBackbonesOverride, setClipBackbonesOverride] = useState(OFFICIAL_ALIGNMENT_BACKBONES);
  const [cropScale, setCropScale] = useState("0.5");
  const [cropRatio, setCropRatio] = useState("0.9");
  const [mpcLam, setMpcLam] = useState("0.6");
  const [mpcTau, setMpcTau] = useState("0.2");
  const [mpcOmega, setMpcOmega] = useState("2.0");
  const sampleBatchQ = useQuery({
    queryKey: ["new-eval-sample-asset-batches", taskKind, assetBatchStatus],
    queryFn: () => listSampleAssetBatches({ page: 1, page_size: 200, reusable_status: assetBatchStatus, task_kind: taskKind, include_asset_ids: true }),
    staleTime: 5000,
  });

  const externalAttackSelected = EXTERNAL_RUNTIME_ATTACKS.has(attack);
  const externalStatuses = overviewQ.data?.external_attack_status ?? {};
  const selectedExternalStatus = externalStatuses[attack];
  const selectedExternalRunnable = !externalAttackSelected || !selectedExternalStatus || selectedExternalStatus.runnable !== false;
  const isGenerationTask = taskKind === "vqa" || taskKind === "caption";
  const taskCompatibleVictims = models.filter((model) => victimSelectableForLaunch(model.health_status) && modelSupportsTask(model, taskKind));
  const compatibleSurrogates = models.filter((model) => model.formal_eval !== false && modelSupportsTask(model, "vlr") && surrogateSelectableForRun(attack, model.adapter));
  const selectedSurrogate = compatibleSurrogates.find((item) => item.adapter === surrogate) ?? compatibleSurrogates[0] ?? models.find((item) => item.adapter === "clip_hf") ?? models[0];
  const selectedVictim = taskCompatibleVictims.find((item) => item.adapter === victimAdapter) ?? taskCompatibleVictims.find((item) => item.role?.startsWith("victim")) ?? taskCompatibleVictims[0];
  const taskGenerationDatasets = generationDatasets.filter((item) => generationDatasetSupportsTask(item.key, taskKind));
  const selectedDataset = vlrDatasets.find((item) => item.key === datasetId) ?? vlrDatasets[0];
  const selectedGenerationDataset = taskGenerationDatasets.find((item) => item.key === datasetId) ?? taskGenerationDatasets[0];
  const selectedGenerationDatasetKey = selectedGenerationDataset?.key || "";
  const selectedGenerationDatasetMeta = selectedGenerationDatasetKey ? GENERATION_DATASET_META[selectedGenerationDatasetKey] : undefined;
  const selectedTaskDataset = isGenerationTask ? selectedGenerationDataset : selectedDataset;
  const generationCaseCount = Math.max(0, Number(selectedGenerationDataset?.item_count) || selectedGenerationDatasetMeta?.fallbackCount || (taskKind === "vqa" ? 300 : 100));
  const generationDatasetName = selectedGenerationDataset
    ? `${formatDatasetName(selectedGenerationDataset.key || selectedGenerationDataset.name)} · ${generationCaseCount} 条`
    : "当前没有可用于该任务的真实生成式数据集";
  const activeAttacks = attacks;
  const rawSampleBatches = sampleBatchQ.data?.items ?? [];
  const selectableAssetCount = (batch: any) => {
    if (assetBatchStatus === "pending_evaluation") {
      const ids = Array.isArray(batch.asset_ids) ? batch.asset_ids.length : 0;
      return Math.max(ids, Number(batch.pending_evaluation_assets || batch.evidence_complete_count || 0));
    }
    return Number(batch.callable_assets || 0);
  };
  const sampleBatches = useMemo(() => rawSampleBatches.map((item: any) => ({ ...item, selectable_assets: selectableAssetCount(item) })).filter((item: any) => {
    const batchTask = String(item.task_kind || "").trim();
    return (!batchTask || batchTask === taskKind || batchTask === "mixed") && Number(item.selectable_assets || 0) >= ASSET_MIN_SAMPLE_COUNT;
  }), [assetBatchStatus, rawSampleBatches, taskKind]);
  const selectedSampleBatch = selectedBatchId ? sampleBatches.find((item: any) => item.batch_id === selectedBatchId) : sampleBatches[0];
  const selectedSampleAssets = selectedSampleBatch?.preview_assets ?? [];
  const batchCallableMax = selectedSampleBatch ? Math.min(Number(selectedSampleBatch.selectable_assets || 0), selectedSampleBatch.asset_ids?.length || Number(selectedSampleBatch.selectable_assets || 0)) : 0;
  const assetSelectionLimit = selectedSampleBatch ? Math.max(ASSET_MIN_SAMPLE_COUNT, Math.min(Number(sampleCount) || batchCallableMax || ASSET_MIN_SAMPLE_COUNT, batchCallableMax || ASSET_MIN_SAMPLE_COUNT)) : ASSET_MIN_SAMPLE_COUNT;
  const selectedAssetCount = selectedAssetIds.length;
  const selectedAssetReady = Boolean(selectedSampleBatch) && selectedAssetCount >= ASSET_MIN_SAMPLE_COUNT;
  const selectedAssetKey = selectedAssetIds.join("|");
  const selectedBatchAssetKey = selectedSampleBatch ? `${selectedSampleBatch.batch_id}:${selectedSampleBatch.asset_ids?.length || 0}` : "";
  const attackInfo = attackMap.get(attack);
  const launchableVictims = taskCompatibleVictims;
  const readyVictims = taskCompatibleVictims.filter((item) => String(item.health_status || "").trim() === "ready");
  const selectedVictims = victimScope === "all" ? launchableVictims : selectedVictim ? [selectedVictim] : readyVictims;
  const submittedVictimsBase = selectedVictims.length ? selectedVictims : launchableVictims;
  const submittedVictims = isGenerationTask && selectedVictim ? [selectedVictim] : submittedVictimsBase;
  const requestedSampleCountRaw = Math.max(1, Number(sampleCount) || (isGenerationTask ? generationCaseCount || 1 : 16));
  const sampleCountNumber = requestedSampleCountRaw;
  const datasetItemCount = isGenerationTask ? generationCaseCount : Math.max(0, Number(selectedDataset?.item_count) || 0);
  const effectiveSampleCount = datasetItemCount > 0 ? Math.min(sampleCountNumber, datasetItemCount) : sampleCountNumber;
  const requestedPairBudget = Math.max(sampleCountNumber * sampleCountNumber, sampleCountNumber);
  const effectivePairBudget = Math.max(effectiveSampleCount * effectiveSampleCount, effectiveSampleCount);
  const strengthNumber = Number(strength) || 0.05;
  const strength255 = Math.round(strengthNumber * 255 * 10) / 10;
  const standardSteps = defaultStepCount(attack);
  const defaultStepSizeValue = defaultStepSize(attack, strengthNumber);
  const effectiveSteps = attackParamMode === "advanced" ? Math.round(boundedNumber(advancedSteps, standardSteps, 1, 3000)) : standardSteps;
  const effectiveStepSize = attackParamMode === "advanced" ? boundedNumber(advancedStepSize, defaultStepSizeValue, 0.0001, 0.1) : defaultStepSizeValue;
  const effectiveTextBudget = attackParamMode === "advanced" ? boundedNumber(textBudget, 0, 0, 5) : 0;
  const effectivePatchSize = Math.round(boundedNumber(patchSize, attack === "advclip" ? 32 : 16, 4, 96));
  const effectiveTopK = Math.round(boundedNumber(topK, 6, 1, 64));
  const effectiveThreshold = boundedNumber(regionThreshold, 0.5, 0, 1);
  const effectiveAlpha = boundedNumber(alphaWeight, 10, 0, 100);
  const effectiveBeta = boundedNumber(betaWeight, 5, 0, 100);
  const effectiveGamma = boundedNumber(gammaWeight, 1, 0, 100);
  const effectiveLambdaAtt = boundedNumber(lambdaAtt, 0.5, 0, 1);
  const effectiveRatioR = boundedNumber(ratioR, 0.4, 0, 1);
  const effectiveTextCandidatesK = Math.round(boundedNumber(textCandidatesK, 12, 1, 64));
  const effectiveLambdaAt = boundedNumber(lambdaAt, 1, 0, 20);
  const effectiveLambdaTpd = boundedNumber(lambdaTpd, 1, 0, 20);
  const effectiveTauPatch = boundedNumber(tauPatch, 0.07, 0.001, 2);
  const effectiveCorruptionSeverity = Math.round(boundedNumber(corruptionSeverity, 2, 1, 5));
  const effectiveCorruptionSeed = Math.round(boundedNumber(corruptionSeed, 7, 0, 1000000));
  const effectiveCropScale = boundedNumber(cropScale, 0.5, 0.05, 1);
  const effectiveCropRatio = boundedNumber(cropRatio, 0.9, 0.05, 1);
  const selectedModelSupportsTask = selectedVictim ? modelSupportsTask(selectedVictim, taskKind) : false;
  const selectedVictimCanSubmit = selectedVictim ? victimSelectableForLaunch(selectedVictim.health_status) && selectedModelSupportsTask : false;
  const generationModelAdapter = isGenerationTask && selectedVictimCanSubmit ? String(selectedVictim?.adapter || "") : "";
  const attackLabels = activeAttacks.length ? activeAttacks : attackCatalog;
  const groupedAttacks = ATTACK_GROUPS.map((group) => ({
    ...group,
    items: attackLabels.filter((item) => groupIdForAttack(item.id) === group.id),
  })).filter((group) => group.items.length > 0);
  const budgetControl = budgetControlForAttack(attack);
  const submittedSurrogateAdapter = selectedSurrogate?.adapter || surrogate;
  const canSubmit = Boolean((evaluationMode === "assets" || selectedTaskDataset) && submittedSurrogateAdapter && selectedVictimCanSubmit && selectedExternalRunnable && (evaluationMode === "generate" || selectedAssetReady));
  const canAdvance = currentStep === 1
    ? Boolean(selectedVictimCanSubmit)
    : currentStep === 2
      ? evaluationMode === "assets"
        ? selectedAssetReady
        : Boolean(selectedTaskDataset)
      : currentStep === 3
        ? evaluationMode === "assets"
          ? selectedVictimCanSubmit && selectedAssetReady
          : Boolean(attack && selectedSurrogate && selectedExternalRunnable)
        : canSubmit;
  const baseAttackOverride: Record<string, unknown> = {};
  if (attackUsesBudget(attack)) {
    baseAttackOverride.epsilon = strengthNumber;
  }
  if (attackUsesStepSize(attack)) {
    baseAttackOverride.step_size = effectiveStepSize;
  }
  if (attackUsesSteps(attack)) {
    baseAttackOverride.steps = effectiveSteps;
  }
  if (attack === "advclip" && attackParamMode === "advanced") {
    baseAttackOverride.patch_train_steps = effectiveSteps;
  }
  const advancedAttackOverride: Record<string, unknown> = {};
  if (attackParamMode === "advanced") {
    if (["advclip", "advedm", "advedm_plus", "tmm"].includes(attack)) {
      advancedAttackOverride.mode = attackModeOverride;
    }
    if (["advclip", "advedm", "advedm_plus"].includes(attack)) {
      advancedAttackOverride.patch_size = effectivePatchSize;
    }
    if (attack === "advclip") {
      advancedAttackOverride.lambda_at = effectiveLambdaAt;
      advancedAttackOverride.lambda_tpd = effectiveLambdaTpd;
      advancedAttackOverride.tau_patch = effectiveTauPatch;
    }
    if (attack === "advedm" || attack === "advedm_plus") {
      advancedAttackOverride.topk = effectiveTopK;
      advancedAttackOverride.threshold = effectiveThreshold;
      advancedAttackOverride.alpha = effectiveAlpha;
      advancedAttackOverride.beta = effectiveBeta;
      advancedAttackOverride.gamma = effectiveGamma;
    }
    if (attack === "tmm") {
      advancedAttackOverride.lambda_att = effectiveLambdaAtt;
      advancedAttackOverride.ratio_r = effectiveRatioR;
      advancedAttackOverride.eps_t = Math.round(effectiveTextBudget);
      advancedAttackOverride.text_candidates_k = effectiveTextCandidatesK;
    }
    if (attack === "vqa_visual_corruption") {
      advancedAttackOverride.corruption_type = corruptionType;
      advancedAttackOverride.severity = effectiveCorruptionSeverity;
      advancedAttackOverride.corruption_seed = effectiveCorruptionSeed;
    }
    if (attack === "xtransfer_uap") {
      advancedAttackOverride.uap_name = uapName;
      advancedAttackOverride.threat_model = threatModel;
    }
    if (attack === "tmm" || attack === "advedm_plus") {
      advancedAttackOverride.eps_t = Math.round(effectiveTextBudget);
      advancedAttackOverride.text_candidates_k = effectiveTextCandidatesK;
    }
    if (["foa_attack", "anyattack", "mpc_attack", "m_attack"].includes(attack)) {
      if (targetImageOverride.trim()) advancedAttackOverride.target_image = targetImageOverride.trim();
    }
    if (attack === "mpc_attack") {
      if (targetTextOverride.trim()) advancedAttackOverride.target_text = targetTextOverride.trim();
    }
    if (["foa_attack", "mpc_attack", "m_attack"].includes(attack)) {
      advancedAttackOverride.crop_scale = effectiveCropScale;
      advancedAttackOverride.crop_ratio = effectiveCropRatio;
    }
    if (attack === "foa_attack") {
      advancedAttackOverride.surrogate_models = splitList(surrogateModelsOverride);
    }
    if (attack === "m_attack") {
      advancedAttackOverride.ensemble_models = splitList(surrogateModelsOverride);
      advancedAttackOverride.disable_wandb = true;
    }
    if (attack === "mpc_attack") {
      advancedAttackOverride.clip_backbones = splitList(clipBackbonesOverride);
      advancedAttackOverride.lam = boundedNumber(mpcLam, 0.6, 0, 10);
      advancedAttackOverride.tau = boundedNumber(mpcTau, 0.2, 0, 10);
      advancedAttackOverride.omega = boundedNumber(mpcOmega, 2.0, 0, 20);
    }
  }
  const submittedAttackOverride = {
    ...baseAttackOverride,
    ...(isGenerationTask && (attack === "tmm" || attack === "advedm_plus") && !("eps_t" in advancedAttackOverride) ? { eps_t: 0 } : {}),
    ...advancedAttackOverride,
  };

  useEffect(() => {
    if (querySelectionAppliedRef.current) return;
    const requestedBatch = searchParams.get("batch");
    const requestedAsset = searchParams.get("asset");
    if (requestedBatch) {
      setEvaluationMode("assets");
      setSelectedBatchId(requestedBatch);
      querySelectionAppliedRef.current = true;
    } else if (requestedAsset) {
      setEvaluationMode("assets");
      const owner = sampleBatches.find((batch: any) => Array.isArray(batch.asset_ids) && batch.asset_ids.includes(requestedAsset));
      if (owner) {
        setSelectedBatchId(owner.batch_id);
        querySelectionAppliedRef.current = true;
      } else if (sampleBatches.length) {
        querySelectionAppliedRef.current = true;
      }
    }
  }, [sampleBatches, searchParams]);

  useEffect(() => {
    if (evaluationMode !== "assets") return;
    if (!selectedSampleBatch && sampleBatches[0]) {
      if (!selectedBatchId || !searchParams.get("batch")) setSelectedBatchId(sampleBatches[0].batch_id);
      return;
    }
    if (!selectedSampleBatch) return;
    const batchAssetIds = selectedSampleBatch.asset_ids || [];
    const batchAssetIdSet = new Set(batchAssetIds);
    const currentIdsBelongToBatch = selectedAssetIds.length > 0 && selectedAssetIds.every((id) => batchAssetIdSet.has(id));
    const targetCount = currentIdsBelongToBatch ? assetSelectionLimit : batchCallableMax;
    const next = batchAssetIds.slice(0, targetCount);
    if (!currentIdsBelongToBatch && targetCount >= ASSET_MIN_SAMPLE_COUNT && sampleCount !== String(targetCount)) setSampleCount(String(targetCount));
    if (next.length >= ASSET_MIN_SAMPLE_COUNT && next.join("|") !== selectedAssetKey) setSelectedAssetIds(next);
  }, [assetSelectionLimit, batchCallableMax, evaluationMode, sampleBatches, sampleCount, searchParams, selectedAssetIds, selectedAssetKey, selectedBatchAssetKey, selectedBatchId, selectedSampleBatch]);

  useEffect(() => {
    if (evaluationMode !== "assets" || !selectedSampleBatch) return;
    const batchAttack = String(selectedSampleBatch.attack || "").trim();
    if (batchAttack && batchAttack !== "mixed" && batchAttack !== attack) setAttack(batchAttack);
    const currentAssetCount = Number(sampleCount) || 0;
    const normalizedAssetCount = String(assetSelectionLimit);
    if ((currentAssetCount < ASSET_MIN_SAMPLE_COUNT || currentAssetCount > batchCallableMax) && sampleCount !== normalizedAssetCount) setSampleCount(normalizedAssetCount);
  }, [assetSelectionLimit, attack, batchCallableMax, evaluationMode, sampleCount, selectedSampleBatch]);

  useEffect(() => {
    const candidateDatasets = isGenerationTask ? taskGenerationDatasets : vlrDatasets;
    const hasDataset = candidateDatasets.some((item) => item.key === datasetId);
    if ((!datasetId || !hasDataset) && candidateDatasets[0]) setDatasetId(candidateDatasets[0].key);
  }, [datasetId, isGenerationTask, taskGenerationDatasets, vlrDatasets]);

  useEffect(() => {
    try {
      const raw = window.localStorage.getItem(DRAFT_STORAGE_KEY);
      if (!raw) return;
      const draft = JSON.parse(raw) as Partial<{
        taskName: string;
        taskKind: TaskKind;
        surrogate: string;
        victimAdapter: string;
        datasetId: string;
        attack: string;
        victimScope: VictimScope;
        strength: string;
        sampleCount: string;
        note: string;
        attackParamMode: AttackParamMode | "quick";
        advancedSteps: string;
        advancedStepSize: string;
        textBudget: string;
        patchSize: string;
        topK: string;
        regionThreshold: string;
        alphaWeight: string;
        betaWeight: string;
        gammaWeight: string;
        lambdaAtt: string;
        ratioR: string;
        textCandidatesK: string;
        lambdaAt: string;
        lambdaTpd: string;
        tauPatch: string;
        attackModeOverride: string;
        corruptionType: string;
        corruptionSeverity: string;
        corruptionSeed: string;
        uapName: string;
        threatModel: string;
        targetImageOverride: string;
        targetTextOverride: string;
        surrogateModelsOverride: string;
        clipBackbonesOverride: string;
        cropScale: string;
        cropRatio: string;
        mpcLam: string;
        mpcTau: string;
        mpcOmega: string;
      }>;
      if (draft.taskName) setTaskName(String(draft.taskName));
      if (draft.taskKind === "vlr" || draft.taskKind === "vqa" || draft.taskKind === "caption") setTaskKind(draft.taskKind);
      if (draft.surrogate) setSurrogate(String(draft.surrogate));
      if (draft.victimAdapter) setVictimAdapter(String(draft.victimAdapter));
      if (draft.datasetId) setDatasetId(String(draft.datasetId));
      if (draft.attack) setAttack(String(draft.attack));
      if (draft.victimScope === "selected" || draft.victimScope === "all") setVictimScope(draft.victimScope);
      if (draft.strength) setStrength(String(draft.strength));
      if (draft.sampleCount) setSampleCount(String(draft.sampleCount));
      if (typeof draft.note === "string") setNote(draft.note);
      if (draft.attackParamMode === "standard" || draft.attackParamMode === "advanced") setAttackParamMode(draft.attackParamMode);
      if (draft.attackParamMode === "quick") setAttackParamMode("standard");
      if (draft.advancedSteps) setAdvancedSteps(String(draft.advancedSteps));
      if (draft.advancedStepSize) setAdvancedStepSize(String(draft.advancedStepSize));
      if (draft.textBudget) setTextBudget(String(draft.textBudget));
      if (draft.patchSize) setPatchSize(String(draft.patchSize));
      if (draft.topK) setTopK(String(draft.topK));
      if (draft.regionThreshold) setRegionThreshold(String(draft.regionThreshold));
      if (draft.alphaWeight) setAlphaWeight(String(draft.alphaWeight));
      if (draft.betaWeight) setBetaWeight(String(draft.betaWeight));
      if (draft.gammaWeight) setGammaWeight(String(draft.gammaWeight));
      if (draft.lambdaAtt) setLambdaAtt(String(draft.lambdaAtt));
      if (draft.ratioR) setRatioR(String(draft.ratioR));
      if (draft.textCandidatesK) setTextCandidatesK(String(draft.textCandidatesK));
      if (draft.lambdaAt) setLambdaAt(String(draft.lambdaAt));
      if (draft.lambdaTpd) setLambdaTpd(String(draft.lambdaTpd));
      if (draft.tauPatch) setTauPatch(String(draft.tauPatch));
      if (draft.attackModeOverride) setAttackModeOverride(String(draft.attackModeOverride).toUpperCase() === "B" ? "B" : "A");
      if (draft.corruptionType) setCorruptionType(String(draft.corruptionType));
      if (draft.corruptionSeverity) setCorruptionSeverity(String(draft.corruptionSeverity));
      if (draft.corruptionSeed) setCorruptionSeed(String(draft.corruptionSeed));
      if (draft.uapName) setUapName(String(draft.uapName));
      if (draft.threatModel) setThreatModel(String(draft.threatModel));
      if (typeof draft.targetImageOverride === "string") setTargetImageOverride(draft.targetImageOverride);
      if (typeof draft.targetTextOverride === "string") setTargetTextOverride(draft.targetTextOverride);
      if (draft.surrogateModelsOverride) setSurrogateModelsOverride(String(draft.surrogateModelsOverride));
      if (draft.clipBackbonesOverride) setClipBackbonesOverride(String(draft.clipBackbonesOverride));
      if (draft.cropScale) setCropScale(String(draft.cropScale));
      if (draft.cropRatio) setCropRatio(String(draft.cropRatio));
      if (draft.mpcLam) setMpcLam(String(draft.mpcLam));
      if (draft.mpcTau) setMpcTau(String(draft.mpcTau));
      if (draft.mpcOmega) setMpcOmega(String(draft.mpcOmega));
      setDraftStatus("已恢复上次保存的草稿。");
    } catch {
      setDraftStatus("草稿读取失败，已使用默认配置。");
    }
  }, []);

  useEffect(() => {
    if (selectedVictim?.adapter && victimAdapter !== selectedVictim.adapter) setVictimAdapter(selectedVictim.adapter);
  }, [victimAdapter, selectedVictim?.adapter]);

  useEffect(() => {
    const methodStrength = defaultStrengthForAttack(attack);
    if (["0.05", STANDARD_EPSILON, OFFICIAL_ALIGNMENT_EPSILON].includes(strength)) setStrength(methodStrength);
    const methodStepSize = String(defaultStepSize(attack, Number(methodStrength)));
    if (["0.01", "0.008", OFFICIAL_ALIGNMENT_STEP_SIZE].includes(advancedStepSize)) setAdvancedStepSize(methodStepSize);
    if (["8", "5", "12", "16", String(OFFICIAL_ALIGNMENT_STEPS)].includes(advancedSteps)) setAdvancedSteps(String(defaultStepCount(attack)));
    if (usesOfficialExternalAlignmentRecipe(attack)) {
      if (surrogateModelsOverride === "B32") setSurrogateModelsOverride(OFFICIAL_ALIGNMENT_BACKBONES);
      if (clipBackbonesOverride === "B32") setClipBackbonesOverride(OFFICIAL_ALIGNMENT_BACKBONES);
    }
  }, [advancedStepSize, advancedSteps, attack, clipBackbonesOverride, strength, surrogateModelsOverride]);


  useEffect(() => {
    if (selectedSurrogate && selectedSurrogate.adapter !== surrogate && !compatibleSurrogates.some((item) => item.adapter === surrogate)) {
      setSurrogate(selectedSurrogate.adapter);
    }
  }, [attack, compatibleSurrogates, selectedSurrogate, surrogate]);

  function applyAttackDefaults(nextAttack: string) {
    setAdvancedSteps(String(defaultStepCount(nextAttack)));
    setAdvancedStepSize(String(defaultStepSize(nextAttack, Number(defaultStrengthForAttack(nextAttack)))));
    setTextBudget(isGenerationTask ? "0" : "1");
    setPatchSize(nextAttack === "advclip" ? "32" : "16");
    setTopK("6");
    setRegionThreshold("0.5");
    setAlphaWeight("10");
    setBetaWeight("5");
    setGammaWeight("1");
    setLambdaAtt("0.5");
    setRatioR("0.4");
    setTextCandidatesK("12");
    setLambdaAt("1");
    setLambdaTpd("1");
    setTauPatch("0.07");
    setAttackModeOverride("A");
  }

  function saveDraft() {
    try {
      window.localStorage.setItem(
        DRAFT_STORAGE_KEY,
        JSON.stringify({
          taskName,
          taskKind,
          surrogate,
          victimAdapter: selectedVictim?.adapter || victimAdapter,
          datasetId,
          attack,
          victimScope,
          strength,
          sampleCount,
          note,
          attackParamMode,
          attackModeOverride,
          advancedSteps,
          advancedStepSize,
          textBudget,
          patchSize,
          topK,
          regionThreshold,
          alphaWeight,
          betaWeight,
          gammaWeight,
          lambdaAtt,
          ratioR,
          textCandidatesK,
          lambdaAt,
          lambdaTpd,
          tauPatch,
          corruptionType,
          corruptionSeverity,
          corruptionSeed,
          uapName,
          threatModel,
          targetImageOverride,
          targetTextOverride,
          surrogateModelsOverride,
          clipBackbonesOverride,
          cropScale,
          cropRatio,
          mpcLam,
          mpcTau,
          mpcOmega,
          savedAt: new Date().toISOString(),
        }),
      );
      setDraftStatus("草稿已保存到当前浏览器。");
    } catch {
      setDraftStatus("草稿保存失败，请检查浏览器本地存储权限。");
    }
  }

  const submitJob = useMutation({
    mutationFn: () =>
      createJob({
        job_type: taskKind === "vqa" ? "run_vqa" : taskKind === "caption" ? "run_caption" : "run_vlr",
        config_path: configPathFor(attack, launchMode, taskKind),
        benchmark_mode: false,
        override: isGenerationTask
          ? {
              task: {
                kind: taskKind,
                eval_scope: "image",
                cases_jsonl: selectedGenerationDatasetMeta?.casesJsonl || (taskKind === "vqa" ? "data/coco2014/generation/vqa_v2_coco_val.jsonl" : "data/coco2014/generation/coco_caption_object_val.jsonl"),
              },
              dataset: {
                kind: "generation_jsonl",
                max_items: effectiveSampleCount,
                benchmark_tag: selectedGenerationDatasetMeta?.benchmarkTag || (taskKind === "vqa" ? "vqa_v2_coco_val_real" : "coco_caption_object_val_real"),
              },
              plugins: { attack, model_adapter: generationModelAdapter },
              attack: submittedAttackOverride,
              report: {
                save_heatmaps: true,
                save_patch_preview: false,
                top_k_cases: Math.max(1, effectiveSampleCount),
              },
              sample_store: {
                enabled: true,
                save_images: true,
                save_traces: true,
              },
              runner: {
                max_samples: evaluationMode === "assets" ? selectedAssetCount : effectiveSampleCount,
                surrogate_model_adapter: submittedSurrogateAdapter,
                victim_model_adapters: [generationModelAdapter],
                staged_model_lifecycle: true,
                stop_local_vlm_before_attack: true,
                restart_local_vlm_for_evaluation: true,
              },
              extra: { ui_task_name: taskName, ui_note: note, workflow_type: evaluationMode === "assets" ? "asset_evaluation" : "generate_and_evaluate", sample_asset_batch_id: selectedSampleBatch?.batch_id || selectedBatchId, sample_asset_ids: selectedAssetIds, sample_asset_count: selectedAssetCount },
            }
          : {
              task: {
                kind: "vlr",
                eval_scope: attack === "tmm" || attack === "advedm_plus" ? "joint" : "image",
              },
              dataset: {
                ...asDatasetOverride(datasetId),
                max_items: effectiveSampleCount,
              },
              plugins: { attack, model_adapter: submittedSurrogateAdapter },
              attack: submittedAttackOverride,
              report: {
                save_heatmaps: true,
                save_patch_preview: true,
                top_k_cases: Math.max(1, effectiveSampleCount),
              },
              sample_store: {
                enabled: true,
                save_images: true,
                save_traces: true,
              },
              runner: {
                max_samples: evaluationMode === "assets" ? selectedAssetCount : effectiveSampleCount,
                max_pairs: effectivePairBudget,
                surrogate_model_adapter: submittedSurrogateAdapter,
                victim_model_adapters: submittedVictims.map((item) => item.adapter),
                staged_model_lifecycle: true,
                stop_local_vlm_before_attack: true,
                restart_local_vlm_for_evaluation: true,
              },
              extra: { ui_task_name: taskName, ui_note: note, workflow_type: evaluationMode === "assets" ? "asset_evaluation" : "generate_and_evaluate", sample_asset_batch_id: selectedSampleBatch?.batch_id || selectedBatchId, sample_asset_ids: selectedAssetIds, sample_asset_count: selectedAssetCount },
            },
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["new-eval-jobs"] });
      qc.invalidateQueries({ queryKey: ["jobs-monitor"] });
    },
  });

  const runningJob = jobsQ.data?.items?.find((item) => item.status === "running" || item.status === "queued");
  const goNext = () => setCurrentStep((step) => Math.min(4, step + 1) as WizardStep);
  const goPrev = () => setCurrentStep((step) => Math.max(1, step - 1) as WizardStep);
  const taskKindLabel = taskKind === "vqa" ? "视觉问答" : taskKind === "caption" ? "图像描述" : "图文检索";
  const sidePanelTitle = currentStep === 1 ? "对象状态" : currentStep === 2 ? evaluationMode === "assets" ? "样本集" : "数据规模" : currentStep === 3 ? evaluationMode === "assets" ? "复用测评" : "样本生成参数" : "提交检查";
  const sidePanelNote = currentStep === 1
    ? "先确认受测模型能执行当前任务；最终指标会按这里选择的测评对象统计。"
    : currentStep === 2
      ? evaluationMode === "assets"
        ? "这里选择已经入库的对抗样本集；攻击参数保持只读，由样本集生成记录决定。"
        : isGenerationTask
          ? "生成式任务按真实样本逐条评测，生成出的对抗样本会进入样本集库。"
          : "检索任务会记录样本条数和图文配对预算，生成出的攻击证据会沉淀为样本集。"
      : currentStep === 3
        ? evaluationMode === "assets"
          ? "测评阶段只配置模型输出和报告，不再修改已生成样本集的攻击参数。"
          : attackParamMode === "advanced"
            ? "高级参数会写入样本生成配置，只影响这一次生成与测评，不改服务器标准配置。"
            : "标准参数按当前方法的真实执行参数生成样本；样本条数会作为本次任务的提交规模。"
        : "提交前只保留必要检查，完整追踪记录、攻击图和案例复盘会在任务报告中查看。";

  return (
    <ExperimentStudioView
      WIZARD_STEPS={WIZARD_STEPS.map((item) => item.step === 2 ? { ...item, label: evaluationMode === "assets" ? "选择样本集" : "选择数据集" } : item.step === 3 ? { ...item, label: evaluationMode === "assets" ? "配置复用测评" : "生成样本配置" } : item)}
      evaluationMode={evaluationMode}
      setEvaluationMode={setEvaluationMode}
      sampleAssets={selectedSampleAssets}
      sampleBatches={sampleBatches}
      selectedSampleBatch={selectedSampleBatch}
      selectedBatchId={selectedSampleBatch?.batch_id || selectedBatchId}
      setSelectedBatchId={setSelectedBatchId}
      selectedSampleAssets={selectedSampleAssets}
      selectedAssetIds={selectedAssetIds}
      setSelectedAssetIds={setSelectedAssetIds}
      assetSelectionLimit={assetSelectionLimit}
      batchCallableMax={batchCallableMax}
      sampleAssetLoading={sampleBatchQ.isLoading}
      assetBatchStatus={assetBatchStatus}
      sampleAssetMinCount={ASSET_MIN_SAMPLE_COUNT}
      selectedAssetReady={selectedAssetReady}
      currentStep={currentStep}
      setCurrentStep={setCurrentStep}
      taskName={taskName}
      setTaskName={setTaskName}
      taskKind={taskKind}
      setTaskKind={setTaskKind}
      selectedVictim={selectedVictim}
      victimAdapter={victimAdapter}
      setVictimAdapter={setVictimAdapter}
      taskCompatibleVictims={taskCompatibleVictims}
      formatModelDisplayText={formatModelDisplayText}
      isGenerationTask={isGenerationTask}
      selectedVictimCanSubmit={selectedVictimCanSubmit}
      selectedModelSupportsTask={selectedModelSupportsTask}
      setVictimScope={setVictimScope}
      victimScope={victimScope}
      taskGenerationDatasets={taskGenerationDatasets}
      selectedGenerationDataset={selectedGenerationDataset}
      setDatasetId={setDatasetId}
      vlrDatasets={vlrDatasets}
      datasetId={datasetId}
      sampleCount={sampleCount}
      setSampleCount={setSampleCount}
      effectiveSampleCount={effectiveSampleCount}
      generationDatasetName={generationDatasetName}
      generationCaseCount={generationCaseCount}
      datasetItemCount={datasetItemCount}
      effectivePairBudget={effectivePairBudget}
      groupedAttacks={groupedAttacks}
      ATTACK_GROUPS={ATTACK_GROUPS}
      groupIdForAttack={groupIdForAttack}
      EXTERNAL_RUNTIME_ATTACKS={EXTERNAL_RUNTIME_ATTACKS}
      externalStatuses={externalStatuses}
      attack={attack}
      setAttack={setAttack}
      applyAttackDefaults={applyAttackDefaults}
      ExternalStatusPills={ExternalStatusPills}
      CLIP_AUXILIARY_SURROGATE_ATTACKS={CLIP_AUXILIARY_SURROGATE_ATTACKS}
      selectedSurrogate={selectedSurrogate}
      surrogate={surrogate}
      setSurrogate={setSurrogate}
      compatibleSurrogates={compatibleSurrogates}
      surrogateHelpForAttack={surrogateHelpForAttack}
      externalAttackSelected={externalAttackSelected}
      selectedExternalRunnable={selectedExternalRunnable}
      selectedExternalStatus={selectedExternalStatus}
      attackParamMode={attackParamMode}
      setAttackParamMode={setAttackParamMode}
      budgetControl={budgetControl}
      strength={strength}
      setStrength={setStrength}
      strengthNumber={strengthNumber}
      strength255={strength255}
      usesOfficialExternalAlignmentRecipe={usesOfficialExternalAlignmentRecipe}
      attackUsesSteps={attackUsesSteps}
      attackUsesStepSize={attackUsesStepSize}
      advancedSteps={advancedSteps}
      setAdvancedSteps={setAdvancedSteps}
      stepHelpForAttack={stepHelpForAttack}
      advancedStepSize={advancedStepSize}
      setAdvancedStepSize={setAdvancedStepSize}
      stepSizeHelpForAttack={stepSizeHelpForAttack}
      ATTACK_MODE_OPTIONS={ATTACK_MODE_OPTIONS}
      attackModeOverride={attackModeOverride}
      setAttackModeOverride={setAttackModeOverride}
      patchSize={patchSize}
      setPatchSize={setPatchSize}
      lambdaAt={lambdaAt}
      setLambdaAt={setLambdaAt}
      lambdaTpd={lambdaTpd}
      setLambdaTpd={setLambdaTpd}
      tauPatch={tauPatch}
      setTauPatch={setTauPatch}
      topK={topK}
      setTopK={setTopK}
      regionThreshold={regionThreshold}
      setRegionThreshold={setRegionThreshold}
      alphaWeight={alphaWeight}
      setAlphaWeight={setAlphaWeight}
      betaWeight={betaWeight}
      setBetaWeight={setBetaWeight}
      gammaWeight={gammaWeight}
      setGammaWeight={setGammaWeight}
      lambdaAtt={lambdaAtt}
      setLambdaAtt={setLambdaAtt}
      ratioR={ratioR}
      setRatioR={setRatioR}
      VQA_CORRUPTION_OPTIONS={VQA_CORRUPTION_OPTIONS}
      corruptionType={corruptionType}
      setCorruptionType={setCorruptionType}
      corruptionSeverity={corruptionSeverity}
      setCorruptionSeverity={setCorruptionSeverity}
      corruptionSeed={corruptionSeed}
      setCorruptionSeed={setCorruptionSeed}
      uapName={uapName}
      setUapName={setUapName}
      THREAT_MODEL_OPTIONS={THREAT_MODEL_OPTIONS}
      threatModel={threatModel}
      setThreatModel={setThreatModel}
      textBudget={textBudget}
      setTextBudget={setTextBudget}
      textCandidatesK={textCandidatesK}
      setTextCandidatesK={setTextCandidatesK}
      targetImageOverride={targetImageOverride}
      setTargetImageOverride={setTargetImageOverride}
      targetTextOverride={targetTextOverride}
      setTargetTextOverride={setTargetTextOverride}
      surrogateModelsOverride={surrogateModelsOverride}
      setSurrogateModelsOverride={setSurrogateModelsOverride}
      cropScale={cropScale}
      setCropScale={setCropScale}
      cropRatio={cropRatio}
      setCropRatio={setCropRatio}
      clipBackbonesOverride={clipBackbonesOverride}
      setClipBackbonesOverride={setClipBackbonesOverride}
      mpcLam={mpcLam}
      setMpcLam={setMpcLam}
      mpcTau={mpcTau}
      setMpcTau={setMpcTau}
      mpcOmega={mpcOmega}
      setMpcOmega={setMpcOmega}
      canSubmit={canSubmit}
      selectedDataset={selectedDataset}
      attackInfo={attackInfo}
      effectiveSteps={effectiveSteps}
      effectivePatchSize={effectivePatchSize}
      effectiveTopK={effectiveTopK}
      effectiveTextBudget={effectiveTextBudget}
      effectiveTextCandidatesK={effectiveTextCandidatesK}
      effectiveCorruptionSeverity={effectiveCorruptionSeverity}
      boundedNumber={boundedNumber}
      note={note}
      setNote={setNote}
      sidePanelTitle={sidePanelTitle}
      taskKindLabel={taskKindLabel}
      submittedVictims={submittedVictims}
      readyVictims={readyVictims}
      formalModels={formalModels}
      launchableVictims={launchableVictims}
      sampleCountNumber={sampleCountNumber}
      requestedPairBudget={requestedPairBudget}
      sidePanelNote={sidePanelNote}
      surrogateRequirementNote={surrogateRequirementNote}
      runningJob={runningJob}
      saveDraft={saveDraft}
      goPrev={goPrev}
      canAdvance={canAdvance}
      goNext={goNext}
      submitJob={submitJob}
      draftStatus={draftStatus}
    />
  );
}
