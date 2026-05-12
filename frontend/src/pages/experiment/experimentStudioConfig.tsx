// 文件说明：该文件属于前端页面，集中实现 experimentStudioConfig 相关逻辑。
import { clipOnlySurrogateAdapters, localTorchSurrogateAdapters, surrogatePolicyForAttack } from "../../lib/attackCatalog";
import { datasetCatalogMap } from "../../lib/datasetCatalog";
import type { AttackRequirementStatus, ExternalAttackRuntimeStatus, ModelOverview } from "../../lib/api";

export type LaunchMode = "standard";
export type VictimScope = "selected" | "all";
export type TaskKind = "vlr" | "vqa" | "caption";
export type WizardStep = 1 | 2 | 3 | 4;
export type AttackParamMode = "standard" | "advanced";
export const RUNNABLE_VICTIM_STATUSES = new Set(["ready", "launchable"]);
export const DRAFT_STORAGE_KEY = "att-project.experiment-draft";
export const EXTERNAL_RUNTIME_ATTACKS = new Set(["vqa_visual_corruption", "xtransfer_uap", "foa_attack", "anyattack", "mpc_attack", "m_attack"]);
export const WIZARD_STEPS: Array<{ step: WizardStep; label: string }> = [
  { step: 1, label: "选择测评对象" },
  { step: 2, label: "选择数据集" },
  { step: 3, label: "配置攻击方式" },
  { step: 4, label: "确认并提交" },
];
export const INTERNAL_PAPER_ATTACKS = new Set(["advclip", "tmm", "advedm", "advedm_plus"]);
export const CLASSIC_GRADIENT_ATTACKS = new Set(["fgsm", "bim", "pgd", "mifgsm", "nifgsm", "difgsm", "tifgsm", "dtmifgsm", "vmifgsm", "vnifgsm", "cw"]);
export const CLIP_AUXILIARY_SURROGATE_ATTACKS = new Set(["advclip", "vqa_visual_corruption", "xtransfer_uap", "foa_attack", "anyattack", "mpc_attack", "m_attack"]);
export const IMAGE_BUDGET_ATTACKS = new Set([
  "fgsm",
  "bim",
  "pgd",
  "mifgsm",
  "nifgsm",
  "difgsm",
  "tifgsm",
  "dtmifgsm",
  "vmifgsm",
  "vnifgsm",
  "cw",
  "tmm",
  "advedm",
  "advedm_plus",
  "foa_attack",
  "anyattack",
  "mpc_attack",
  "m_attack",
]);
export const UAP_BUDGET_ATTACKS = new Set(["xtransfer_uap"]);
export const STEP_PARAM_ATTACKS = new Set([
  "bim",
  "pgd",
  "mifgsm",
  "nifgsm",
  "difgsm",
  "tifgsm",
  "dtmifgsm",
  "vmifgsm",
  "vnifgsm",
  "cw",
  "tmm",
  "advedm",
  "advedm_plus",
  "foa_attack",
  "mpc_attack",
  "m_attack",
]);
export const STEP_SIZE_PARAM_ATTACKS = new Set([...STEP_PARAM_ATTACKS, "advclip"]);
export const ATTACK_GROUPS = [
  {
    id: "internal-paper",
    title: "项目内置论文方法",
    subtitle: "平台代码内直接执行的论文方法，适合展示项目核心实现。",
  },
  {
    id: "classic-gradient",
    title: "经典梯度攻击基线",
    subtitle: "快速梯度符号法（FGSM）、投影梯度下降法（PGD）、CW 攻击等白盒或迁移梯度基线，用于横向对照。",
  },
  {
    id: "external-paper",
    title: "外部论文官方实现接入",
    subtitle: "调用官方仓库、官方软件包或本地权重，状态会按服务器真实配置检查。",
  },
] as const;
export const GENERATION_DATASET_META: Record<string, { task: "vqa" | "caption"; casesJsonl: string; benchmarkTag: string; fallbackCount: number }> = {
  vqa_v2_coco_val: {
    task: "vqa",
    casesJsonl: "data/coco2014/generation/vqa_v2_coco_val.jsonl",
    benchmarkTag: "vqa_v2_coco_val_real",
    fallbackCount: 300,
  },
  coco_object_probe_val: {
    task: "vqa",
    casesJsonl: "data/coco2014/generation/coco_object_probe_val.jsonl",
    benchmarkTag: "coco_object_probe_val_real",
    fallbackCount: 200,
  },
  coco_caption_object_val: {
    task: "caption",
    casesJsonl: "data/coco2014/generation/coco_caption_object_val.jsonl",
    benchmarkTag: "coco_caption_object_val_real",
    fallbackCount: 100,
  },
};
export const VQA_CORRUPTION_OPTIONS = [
  ["gaussian_noise", "高斯噪声"],
  ["gaussian_blur", "高斯/失焦模糊"],
  ["motion_blur", "运动/缩放模糊"],
  ["jpeg_compression", "JPEG 压缩"],
  ["brightness", "亮度变化"],
  ["contrast", "对比度变化"],
  ["occlusion", "遮挡/污损"],
  ["resize_compress", "降采样压缩"],
] as const;
export const THREAT_MODEL_OPTIONS = [
  ["linf_non_targeted", "L∞ 非目标"],
  ["linf_targeted", "L∞ 目标式"],
  ["l2_non_targeted", "L2 非目标"],
] as const;
export const ATTACK_MODE_OPTIONS = [
  ["A", "A / 语义去除"],
  ["B", "B / 语义添加"],
] as const;
export const OFFICIAL_ALIGNMENT_BACKBONES = "B16,B32,Laion";
export const OFFICIAL_ALIGNMENT_EPSILON = "0.062745";
export const OFFICIAL_ALIGNMENT_STEP_SIZE = "0.003922";
export const OFFICIAL_ALIGNMENT_STEPS = 300;
export const STANDARD_EPSILON = "0.0470588";
export const STANDARD_ATTACK_STRENGTH: Record<string, string> = {
  foa_attack: OFFICIAL_ALIGNMENT_EPSILON,
  m_attack: OFFICIAL_ALIGNMENT_EPSILON,
  mpc_attack: OFFICIAL_ALIGNMENT_EPSILON,
  anyattack: OFFICIAL_ALIGNMENT_EPSILON,
};
export const STANDARD_STEP_SIZE: Record<string, number> = {
  tmm: 0.006,
  advedm: 0.008,
  advedm_plus: 0.008,
  advclip: 0.008,
  foa_attack: Number(OFFICIAL_ALIGNMENT_STEP_SIZE),
  m_attack: Number(OFFICIAL_ALIGNMENT_STEP_SIZE),
  mpc_attack: Number(OFFICIAL_ALIGNMENT_STEP_SIZE),
};
export const STANDARD_ATTACK_STEPS: Record<string, number> = {
  fgsm: 1,
  bim: 10,
  pgd: 10,
  mifgsm: 10,
  nifgsm: 10,
  difgsm: 10,
  tifgsm: 10,
  dtmifgsm: 10,
  vmifgsm: 10,
  vnifgsm: 10,
  cw: 50,
  tmm: 16,
  advedm: 12,
  advedm_plus: 12,
  vqa_visual_corruption: 1,
  xtransfer_uap: 1,
  anyattack: 1,
};

/** 封装 `usesOfficialExternalAlignmentRecipe` Hook，把页面状态、副作用和持久化逻辑集中管理。 */
export function usesOfficialExternalAlignmentRecipe(attack: string): boolean {
  return attack === "foa_attack" || attack === "m_attack" || attack === "mpc_attack";
}

/** 整理 `surrogate requirement note` 前端辅助逻辑，保持数据转换和展示口径一致。 */
export function surrogateRequirementNote(attack: string): string {
  const policy = surrogatePolicyForAttack(attack);
  if (attack === "tmm") {
    return "可迁移多模态攻击（TMM）只允许具备注意力图、PyTorch 后端打分和投影特征的本地代理模型：CLIP、BLIP、ViLT。其余模型当前只能作为受测模型。";
  }
  if (policy === "clip_only") {
    return "细粒度具身决策攻击（AdvEDM）和增强细粒度具身决策攻击（AdvEDM+）当前只允许 CLIP 作为代理模型，因为图像优化链路需要补丁语义相似度、注意力图和 PyTorch 后端梯度打分。其余模型当前只能作为受测模型。";
  }
  if (policy === "local_torch") {
    return "经典梯度攻击当前只允许 CLIP、BLIP、ViLT 这三类本地代理模型，因为攻击生成阶段必须直接调用 PyTorch 后端梯度打分。Qwen2.5-VL 这类兼容接口视觉语言模型当前只能作为受测模型。";
  }
  if (["vqa_visual_corruption", "xtransfer_uap", "foa_attack", "anyattack", "mpc_attack", "m_attack"].includes(attack)) {
    return "外部论文方法按真实仓库入口执行：官方视觉退化攻击调用视觉鲁棒性仓库的退化生成函数；X-Transfer 调用官方扰动库或本地通用扰动权重；特征最优对齐迁移攻击（FOA-Attack）、局部语义匹配迁移攻击（M-Attack）、任意图像目标生成攻击（AnyAttack）和多范式协同迁移攻击（MPCAttack）调用已克隆仓库生成攻击图，再由平台统一评测。";
  }
  return "";
}

/** 整理 `surrogate supported 所属 攻击` 前端辅助逻辑，保持数据转换和展示口径一致。 */
export function surrogateSupportedForAttack(attack: string, adapter: string): boolean {
  const policy = surrogatePolicyForAttack(attack);
  if (policy === "clip_only") return clipOnlySurrogateAdapters.has(adapter);
  if (policy === "local_torch") return localTorchSurrogateAdapters.has(adapter);
  return true;
}

/** 整理 `surrogate selectable 所属 运行记录` 前端辅助逻辑，保持数据转换和展示口径一致。 */
export function surrogateSelectableForRun(attack: string, adapter: string): boolean {
  if (CLIP_AUXILIARY_SURROGATE_ATTACKS.has(attack)) return adapter === "clip_hf";
  return surrogateSupportedForAttack(attack, adapter);
}

/** 整理 `victim selectable 所属 launch` 前端辅助逻辑，保持数据转换和展示口径一致。 */
export function victimSelectableForLaunch(healthStatus: string): boolean {
  return RUNNABLE_VICTIM_STATUSES.has(String(healthStatus || "").trim());
}

/** 整理 `配置 路径 所属` 前端辅助逻辑，保持数据转换和展示口径一致。 */
export function configPathFor(attack: string, launchMode: LaunchMode, taskKind: TaskKind): string {
  const externalStandardConfigs: Record<string, { generation: string; vlr: string }> = {
    xtransfer_uap: {
      generation: "configs/bench/bootstrap_standard_caption_xtransfer_uap_cuda.yaml",
      vlr: "configs/bench/bootstrap_standard_vlr_xtransfer_uap_cuda.yaml",
    },
    foa_attack: {
      generation: "configs/bench/bootstrap_standard_caption_foa_attack_cuda.yaml",
      vlr: "configs/bench/bootstrap_standard_vlr_foa_attack_cuda.yaml",
    },
    anyattack: {
      generation: "configs/bench/bootstrap_standard_caption_anyattack_cuda.yaml",
      vlr: "configs/bench/bootstrap_standard_vlr_anyattack_cuda.yaml",
    },
    mpc_attack: {
      generation: "configs/bench/bootstrap_standard_caption_mpc_attack_cuda.yaml",
      vlr: "configs/bench/bootstrap_standard_vlr_mpc_attack_cuda.yaml",
    },
    m_attack: {
      generation: "configs/bench/bootstrap_standard_caption_m_attack_cuda.yaml",
      vlr: "configs/bench/bootstrap_standard_vlr_m_attack_cuda.yaml",
    },
  };
  if (attack === "vqa_visual_corruption") return "configs/bench/bootstrap_standard_vqa_visual_corruption_cuda.yaml";
  const externalConfig = externalStandardConfigs[attack];
  if (externalConfig) return taskKind === "vlr" ? externalConfig.vlr : externalConfig.generation;
  if (taskKind === "vqa") return "configs/bench/bootstrap_quick_vqa.yaml";
  if (taskKind === "caption") return "configs/bench/bootstrap_quick_caption.yaml";
  if (attack === "tmm") return "configs/bench/bootstrap_full_vlr_tmm_cuda.yaml";
  if (attack === "advedm") return "configs/bench/bootstrap_full_vlr_advedm_cuda.yaml";
  if (attack === "advedm_plus") return "configs/bench/bootstrap_full_vlr_advedm_plus_cuda.yaml";
  return "configs/bench/bootstrap_full_vlr_cuda.yaml";
}

/** 整理 `inferred 任务 capabilities` 前端辅助逻辑，保持数据转换和展示口径一致。 */
export function inferredTaskCapabilities(adapter: string): string[] {
  const id = String(adapter || "").trim();
  if (!id || id === "fixture_vlm") return [];
  if (id === "clip_hf" || id === "blip_itm" || id === "vilt_itm") return ["vlr"];
  if (id === "openai_compat" || id === "openai_gpt4o" || id === "gemini_vision" || id === "http" || id.startsWith("openai_")) {
    return ["vlr", "vqa", "caption"];
  }
  return [];
}

/** 整理 `生成式评测 capable adapter` 前端辅助逻辑，保持数据转换和展示口径一致。 */
export function generationCapableAdapter(adapter: string): boolean {
  const capabilities = inferredTaskCapabilities(adapter);
  return capabilities.includes("vqa") && capabilities.includes("caption");
}

/** 整理 `模型 任务 capabilities` 前端辅助逻辑，保持数据转换和展示口径一致。 */
export function modelTaskCapabilities(model: Pick<ModelOverview, "adapter" | "task_capabilities" | "formal_eval"> | undefined): string[] {
  if (!model) return [];
  if (model.formal_eval === false) return [];
  const explicit = Array.isArray(model.task_capabilities) ? model.task_capabilities.map((item) => String(item)) : [];
  return explicit.length ? explicit : inferredTaskCapabilities(model.adapter);
}

/** 整理 `模型 supports 任务` 前端辅助逻辑，保持数据转换和展示口径一致。 */
export function modelSupportsTask(model: Pick<ModelOverview, "adapter" | "task_capabilities" | "formal_eval"> | undefined, taskKind: TaskKind): boolean {
  return modelTaskCapabilities(model).includes(taskKind);
}

/** 整理 `group id 所属 攻击` 数据，把接口响应转换成页面可直接渲染的结构。 */
export function groupIdForAttack(attackId: string) {
  if (EXTERNAL_RUNTIME_ATTACKS.has(attackId)) return "external-paper";
  if (CLASSIC_GRADIENT_ATTACKS.has(attackId)) return "classic-gradient";
  if (INTERNAL_PAPER_ATTACKS.has(attackId)) return "internal-paper";
  return "classic-gradient";
}

/** 生成 `requirement 文本` 展示值，统一页面标签、颜色和缺省文案。 */
export function requirementText(item: AttackRequirementStatus | undefined): string {
  if (!item) return "未知";
  if (item.status === "ready") return "已配置";
  if (item.status === "not_required") return "不需要";
  if (item.status === "missing") return item.required ? "未配置" : "未找到";
  return "未知";
}

/** 生成 `requirement tone` 展示值，统一页面标签、颜色和缺省文案。 */
export function requirementTone(item: AttackRequirementStatus | undefined): string {
  if (!item) return "unknown";
  if (item.status === "ready") return "ready";
  if (item.status === "not_required") return "muted";
  if (item.status === "missing") return item.required ? "missing" : "warn";
  return "unknown";
}

/** 生成 `状态 title` 展示值，统一页面标签、颜色和缺省文案。 */
export function statusTitle(item: AttackRequirementStatus | undefined): string {
  if (!item) return "当前后端未返回该配置项状态";
  const path = item.path ? `；路径：${item.path}` : "";
  return `${item.note || ""}${path}`;
}

/** 格式化 `format 模型 display 文本`，统一页面展示文本和缺省值。 */
export function formatModelDisplayText(model?: ModelOverview): string {
  const raw = String(model?.display_name || model?.model_name || model?.endpoint_or_source || model?.adapter || "").trim();
  if (!raw) return "未选择模型";
  return raw.replace(/视觉语言\s*Transformer\s*匹配模型/g, "视觉语言变换器匹配模型");
}

/** 渲染 `ExternalStatusPills` 组件，组织该区域的数据读取、交互状态和可访问性标记。 */
export function ExternalStatusPills({ status }: { status: ExternalAttackRuntimeStatus | undefined }) {
  if (!status) {
    return <span className="gov-attack-status-missing">等待后端状态</span>;
  }
  const entries: Array<[string, AttackRequirementStatus]> = [
    ["仓库", status.repo],
    ["权重", status.checkpoint],
    ["目标", status.target],
  ];
  return (
    <span className="gov-attack-status-row" aria-label={`${status.display_name} 外部配置状态`}>
      {entries.map(([label, item]) => (
        <span key={label} className={`gov-attack-status ${requirementTone(item)}`} title={statusTitle(item)}>
          {label} {requirementText(item)}
        </span>
      ))}
    </span>
  );
}

/** 整理 `as 数据集 override` 前端辅助逻辑，保持数据转换和展示口径一致。 */
export function asDatasetOverride(datasetId: string) {
  const catalogItem = datasetCatalogMap.get(datasetId);
  return catalogItem?.override ?? { kind: datasetId, benchmark_tag: datasetId };
}

/** 整理 `生成式评测 数据集 supports 任务` 前端辅助逻辑，保持数据转换和展示口径一致。 */
export function generationDatasetSupportsTask(datasetKey: string, taskKind: TaskKind): boolean {
  const meta = GENERATION_DATASET_META[String(datasetKey || "").trim()];
  return Boolean(meta && meta.task === taskKind);
}

/** 整理 `split list` 前端辅助逻辑，保持数据转换和展示口径一致。 */
export function splitList(value: string): string[] {
  return String(value || "")
    .split(/[,\s]+/)
    .map((item) => item.trim())
    .filter(Boolean);
}

/** 整理 `bounded number` 前端辅助逻辑，保持数据转换和展示口径一致。 */
export function boundedNumber(value: string, fallback: number, min: number, max: number): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.max(min, Math.min(max, parsed));
}

/** 整理 `default step count` 前端辅助逻辑，保持数据转换和展示口径一致。 */
export function defaultStepCount(attack: string): number {
  if (usesOfficialExternalAlignmentRecipe(attack)) return OFFICIAL_ALIGNMENT_STEPS;
  return STANDARD_ATTACK_STEPS[attack] ?? 16;
}

/** 整理 `default strength 所属 攻击` 前端辅助逻辑，保持数据转换和展示口径一致。 */
export function defaultStrengthForAttack(attack: string): string {
  return STANDARD_ATTACK_STRENGTH[attack] ?? STANDARD_EPSILON;
}

/** 整理 `default step size` 前端辅助逻辑，保持数据转换和展示口径一致。 */
export function defaultStepSize(attack: string, strength: number): number {
  return STANDARD_STEP_SIZE[attack] ?? Math.min(strength, 0.01);
}

/** 整理 `budget control 所属 攻击` 前端辅助逻辑，保持数据转换和展示口径一致。 */
export function budgetControlForAttack(attack: string): { label: string; help: string } | null {
  if (UAP_BUDGET_ATTACKS.has(attack)) {
    return {
      label: "UAP 缩放预算",
      help: "该值会缩放或裁剪预训练通用扰动；攻击形态仍主要由所选 UAP 权重决定。",
    };
  }
  if (IMAGE_BUDGET_ATTACKS.has(attack)) {
    if (attack === "tmm" || attack === "advedm_plus") {
      return {
        label: "图像扰动预算",
        help: "该值控制图像分支扰动上限；文本分支由文本替换预算和候选词数控制。",
      };
    }
    if (["foa_attack", "mpc_attack", "m_attack"].includes(attack)) {
      return {
        label: "外部优化预算",
        help: "该值会作为扰动上限 ε 传入外部攻击脚本，同时配合单步步长和优化步数生效。",
      };
    }
    if (attack === "anyattack") {
      return {
        label: "生成扰动预算",
        help: "该值会传入 AnyAttack 官方演示入口；目标图和解码器仍是主要控制项。",
      };
    }
    return {
      label: "图像扰动预算",
      help: "该值控制图像像素扰动上限，是当前攻击的主要强度参数。",
    };
  }
  return null;
}

/** 整理 `step help 所属 攻击` 前端辅助逻辑，保持数据转换和展示口径一致。 */
export function stepHelpForAttack(attack: string): string {
  if (attack === "advclip") return "仅在需要自动训练通用补丁时使用；已有补丁时评测阶段不会重新优化。";
  if (["foa_attack", "mpc_attack", "m_attack"].includes(attack)) return "会作为迭代步数传入外部攻击脚本。";
  return "会写入攻击优化过程，控制梯度或局部优化迭代次数。";
}

/** 整理 `step size help 所属 攻击` 前端辅助逻辑，保持数据转换和展示口径一致。 */
export function stepSizeHelpForAttack(attack: string): string {
  if (attack === "advclip") return "仅作为补丁训练学习率使用；贴加补丁评测阶段不使用。";
  if (["foa_attack", "mpc_attack", "m_attack"].includes(attack)) return "会转换为外部脚本的步长参数 α（alpha）。";
  return "控制每次优化更新的步长。";
}

/** 整理 `surrogate help 所属 攻击` 前端辅助逻辑，保持数据转换和展示口径一致。 */
export function surrogateHelpForAttack(attack: string, isGenerationTask: boolean): string {
  if (CLIP_AUXILIARY_SURROGATE_ATTACKS.has(attack)) {
    if (attack === "advclip") return "通用对抗补丁（AdvCLIP）在评测阶段加载并贴加通用补丁；若缺少补丁会自动训练，当前固定使用 CLIP 代理。";
    if (isGenerationTask) return "该外部攻击由官方仓库或权重生成图像扰动；这里固定使用 CLIP 作为预检查与辅助语义评估模型。";
    return "该外部攻击不直接使用顶部代理模型生成扰动；平台固定使用 CLIP 做预检查、记录和辅助语义评估。";
  }
  if (attack === "tmm") return "可迁移多模态攻击（TMM）会直接调用该代理模型的注意力、PyTorch 后端打分和投影特征来生成图文联合扰动。";
  if (attack === "advedm" || attack === "advedm_plus") return "AdvEDM 系列会直接调用 CLIP 的补丁-文本相似度、注意力图和梯度打分。";
  if (CLASSIC_GRADIENT_ATTACKS.has(attack)) return "经典梯度攻击会直接调用该代理模型的 PyTorch 后端打分生成图像扰动。";
  return "该代理模型用于攻击生成阶段。";
}

/** 整理 `攻击 uses budget` 前端辅助逻辑，保持数据转换和展示口径一致。 */
export function attackUsesBudget(attack: string): boolean {
  return IMAGE_BUDGET_ATTACKS.has(attack) || UAP_BUDGET_ATTACKS.has(attack);
}

/** 整理 `攻击 uses steps` 前端辅助逻辑，保持数据转换和展示口径一致。 */
export function attackUsesSteps(attack: string): boolean {
  return STEP_PARAM_ATTACKS.has(attack);
}

/** 整理 `攻击 uses step size` 前端辅助逻辑，保持数据转换和展示口径一致。 */
export function attackUsesStepSize(attack: string): boolean {
  return STEP_SIZE_PARAM_ATTACKS.has(attack);
}

