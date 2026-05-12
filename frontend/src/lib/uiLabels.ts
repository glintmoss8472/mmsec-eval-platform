// 文件说明：该文件属于前端业务工具，集中实现 uiLabels 相关逻辑。
/** 中文注释：规范化外部输入的文本形态，避免标签格式化逻辑反复处理空值和首尾空白。 */
function cleanText(value: unknown): string {
  return String(value ?? "").trim();
}

/** 中文注释：实现 isMissingText 的核心流程，支撑前端业务工具中的业务语义和异常边界。 */
function isMissingText(value: unknown): boolean {
  const text = cleanText(value);
  const normalized = text.toLowerCase();
  return !text || text === "-" || normalized === "null" || normalized === "undefined";
}

/** 中文注释：实现 keyText 的核心流程，支撑前端业务工具中的业务语义和异常边界。 */
function keyText(value: unknown): string {
  return cleanText(value).toLowerCase();
}

/** 中文注释：实现 formatWrapped 的核心流程，支撑前端业务工具中的业务语义和异常边界。 */
export function formatWrapped(label: string, value: unknown): string {
  const text = cleanText(value);
  if (isMissingText(text)) {
    return "未记录";
  }
  return `${label}（${text}）`;
}

const JOB_TYPE_LABELS: Record<string, string> = {
  vlr: "图文检索",
  vqa: "视觉问答",
  caption: "图像描述",
  run_eval: "评测任务",
  run_benchmark: "正式基准评测",
  run_vlr: "交互式试跑",
  run_vqa: "视觉问答试跑",
  run_caption: "图像描述试跑",
  train_advclip: "AdvCLIP 通用对抗补丁训练",
  run_sweep: "参数扫描",
  docs_ingest: "文档摄取",
  dataset_prepare: "数据准备",
};

const JOB_STATUS_LABELS: Record<string, string> = {
  queued: "排队中",
  running: "运行中",
  pending: "等待中",
  success: "成功",
  failed: "失败",
  cancelled: "已取消",
};

const HEALTH_STATUS_LABELS: Record<string, string> = {
  ok: "正常",
  healthy: "正常",
  ready: "就绪",
  launchable: "已发现启动脚本",
  missing_assets: "本地资产缺失",
  unavailable: "未就绪",
  degraded: "降级可用",
  frontend_not_built: "前端未构建",
  loading: "加载中",
};

const BOOTSTRAP_STATE_LABELS: Record<string, string> = {
  pending: "等待启动",
  seeding: "导入资产",
  warming: "后台预热",
  ready: "已就绪",
  degraded: "降级可用",
};

const ADAPTER_ROLE_LABELS: Record<string, string> = {
  "surrogate/local": "本地代理模型",
  "victim/local": "本地受测模型",
  "victim/demo": "演示受测模型",
  "victim/api": "接口受测模型",
  "victim/api_or_self_hosted": "接口或自托管受测模型",
  always_on: "始终启用",
  env_configured: "按环境启用",
};

const RISK_LEVEL_LABELS: Record<string, string> = {
  minimal: "极低",
  low: "低",
  medium: "中",
  moderate: "中",
  high: "高",
  critical: "极高",
  very_low: "极低",
};

const LOG_LEVEL_LABELS: Record<string, string> = {
  info: "信息",
  warn: "警告",
  warning: "警告",
  error: "错误",
};

const ATTACK_LABELS: Record<string, string> = {
  advclip: "通用对抗补丁（AdvCLIP）",
  tmm: "可迁移多模态攻击（TMM）",
  advedm: "细粒度具身决策攻击（AdvEDM）",
  advedm_plus: "增强细粒度具身决策攻击（AdvEDM+）",
  fgsm: "快速梯度符号法（FGSM）",
  bim: "基础迭代法（BIM）",
  pgd: "投影梯度下降法（PGD）",
  mifgsm: "动量迭代快速梯度符号法（MI-FGSM）",
  nifgsm: "Nesterov 迭代快速梯度符号法（NI-FGSM）",
  difgsm: "输入多样化迭代快速梯度符号法（DI-FGSM）",
  tifgsm: "平移不变迭代快速梯度符号法（TI-FGSM）",
  dtmifgsm: "输入多样化平移不变动量迭代快速梯度符号法（DTMI-FGSM）",
  vmifgsm: "方差调节动量迭代快速梯度符号法（VMI-FGSM）",
  vnifgsm: "方差调节 Nesterov 迭代快速梯度符号法（VNI-FGSM）",
  cw: "卡里尼瓦格纳攻击（CW）",
  vqa_visual_corruption: "官方视觉退化攻击（VQA Visual Robustness）",
  xtransfer_uap: "跨任务通用扰动（X-Transfer UAP）",
  foa_attack: "特征最优对齐迁移攻击（FOA-Attack）",
  anyattack: "任意图像目标生成攻击（AnyAttack）",
  mpc_attack: "多范式协同迁移攻击（MPCAttack）",
  m_attack: "局部语义匹配迁移攻击（M-Attack）",
};

const MODE_LABELS: Record<string, string> = {
  a: "语义移除模式（A 模式，AdvEDM-R）",
  b: "语义添加模式（B 模式，AdvEDM-A）",
  asset_fixed_retest: "样本库固定复测",
  clean: "正常样本",
  attacked: "受攻击样本",
  attack: "攻击阶段",
};

const DATASET_LABELS: Record<string, string> = {
  generation_jsonl: "生成式评测清单数据集（JSONL）",
  vqa_v2_coco_val: "VQA v2 COCO 验证真实子集",
  coco_object_probe_val: "COCO 对象存在性探测真实子集",
  coco_caption_object_val: "COCO 图像描述对象级真实子集",
  vqa_v2_val2014: "VQA v2 COCO 验证集",
  staged_lifecycle_smoke: "阶段流程调试数据集",
  flickr30k: "Flickr30k 测试集（1000 图 × 5 文本）",
  flickr1k: "Flickr1k 单文本切片",
  coco_subset: "COCO val2017 完整验证子集",
  mini_flickr: "Mini Flickr 演示集",
  toy_shapes: "几何形状演示数据集（Toy Shapes）",
};

const ADAPTER_NAME_LABELS: Record<string, string> = {
  clip_hf: "CLIP 检索模型",
  blip_itm: "BLIP 匹配模型",
  vilt_itm: "ViLT 匹配模型",
  openai_compat: "OpenAI 兼容接口",
  openai_gpt4o: "ChatGPT-4o 模型",
  gemini_vision: "Gemini 视觉接口",
  openai_qwen35_9b: "Qwen3.5-9B",
  openai_qwen3_vl: "Qwen3-VL-8B",
  openai_qwen25_vl: "Qwen2.5-VL-7B",
  openai_internvl35: "InternVL3.5-8B",
  openai_minicpm_v: "MiniCPM-V 4.5",
  openai_ovis25: "Ovis2.5-9B",
  openai_gemma3_12b: "Gemma 3-12B",
  "qwen35-9b": "Qwen3.5-9B",
  "qwen3-vl": "Qwen3-VL-8B",
  "qwen25-vl": "Qwen2.5-VL-7B",
  internvl35: "InternVL3.5-8B",
  "minicpm-v": "MiniCPM-V 4.5",
  ovis25: "Ovis2.5-9B",
  "gemma3-12b": "Gemma 3-12B",
};

const MODEL_SOURCE_LABELS: Record<string, string> = {
  "openai/clip-vit-base-patch32": "对比语言图像预训练基础模型（openai/clip-vit-base-patch32）",
  "salesforce/blip-itm-base-coco": "自举语言图像预训练匹配基础模型（Salesforce/blip-itm-base-coco）",
  "dandelin/vilt-b32-finetuned-coco": "视觉语言变换器微调模型（dandelin/vilt-b32-finetuned-coco）",
  "chatgpt-4o-latest": "聊天生成四零最新模型（chatgpt-4o-latest）",
  "gpt-4o": "聊天生成四零模型（gpt-4o）",
  "gemini-2.5-pro": "双子星二点五专业模型（gemini-2.5-pro）",
  "qwen/qwen3.5-9b": "通义千问三点五九十亿参数模型（Qwen/Qwen3.5-9B）",
  "qwen/qwen3-vl-8b-instruct": "通义千问三视觉语言模型八十亿参数版（Qwen/Qwen3-VL-8B-Instruct）",
  "qwen/qwen2.5-vl-7b-instruct": "通义千问二点五视觉语言模型七十亿参数版（Qwen/Qwen2.5-VL-7B-Instruct）",
  "opengvlab/internvl3_5-8b-hf": "书生万象三点五八十亿参数模型（OpenGVLab/InternVL3_5-8B-HF）",
  "openbmb/minicpm-v-4_5": "迷你通用处理模型视觉版四点五（openbmb/MiniCPM-V-4_5）",
  "aidc-ai/ovis2.5-9b": "奥维斯二点五九十亿参数模型（AIDC-AI/Ovis2.5-9B）",
  "google/gemma-3-12b-it": "谷歌 Gemma 三代一百二十亿参数模型（google/gemma-3-12b-it）",
};

const EVAL_SCOPE_LABELS: Record<string, string> = {
  qa: "视觉问答",
  vqa: "视觉问答",
  caption: "图像描述",
  clean: "正常基线",
  image: "图像侧攻击",
  text: "文本侧攻击",
  joint: "图文联合攻击",
  retrieval: "检索评测",
  vlr: "图文检索",
  multimodal: "多模态评测",
  general: "通用场景",
  asset_retest: "样本库复测",
};

const ATTACK_SCOPE_LABELS: Record<string, string> = {
  image: "图像扰动",
  vision: "图像扰动",
  text: "文本扰动",
  joint: "图文联合扰动",
  multimodal: "图文联合扰动",
  mixed: "混合扰动",
  图像: "图像扰动",
  文本: "文本扰动",
  图文联合: "图文联合扰动",
};

const JUDGE_REASON_LABELS: Record<string, string> = {
  asset_retest: "样本库复测判定",
  asset_retest_top1_drop: "复测后前一位召回下降",
  asset_retest_top1_not_dropped: "复测后前一位召回未下降",
  asset_retest_generation_success: "复测后生成结果被攻击改变",
  asset_retest_generation_not_successful: "复测后生成结果未达到攻击成功条件",
};

const FEATURE_METHOD_LABELS: Record<string, string> = {
  pca: "主成分分析（PCA）",
  tsne: "随机邻域嵌入（t-SNE）",
  umap: "统一流形近似投影（UMAP）",
};

const STAGE_LABELS: Record<string, string> = {
  clean: "正常输入阶段",
  attacked: "攻击后阶段",
  attack: "攻击阶段",
  unknown: "未知阶段",
};

const MODALITY_LABELS: Record<string, string> = {
  image: "图像模态",
  vision: "图像模态",
  text: "文本模态",
  multimodal: "多模态",
  joint: "联合模态",
  unknown: "未知模态",
};

const RISK_DIMENSION_LABELS: Record<string, string> = {
  task_damage: "任务破坏风险",
  output_instability: "输出失稳风险",
  semantic_disguise: "语义伪装风险",
  low_perturbation: "低扰动可达风险",
  tail_case: "尾部案例风险",
  effectiveness: "任务破坏风险",
  semantic: "语义伪装风险",
  cost: "低扰动可达风险",
  stability: "尾部案例风险",
  asr: "攻击成功率（ASR）",
  avg_l2: "平均二范数（L2）",
  retrieval_drop: "检索降幅",
  clean_accuracy: "原始输入准确率",
  attacked_accuracy: "攻击后准确率",
  answer_change_rate: "答案变化率",
  target_flip_rate: "目标翻转率",
  semantic_preservation_rate: "语义保持率",
  caption_text_similarity: "图像描述文本相似度",
  object_jaccard: "对象集合重合度",
  text_diff_score: "文本差异分数",
  embedding_shift: "嵌入偏移",
  cot_shift_score: "思维链偏移分数",
};

const PAPER_STATUS_LABELS: Record<string, string> = {
  approx: "工程化近似复现",
  ready: "已就绪",
  partial: "部分完成",
  missing: "缺失",
  official: "官方实现",
  reproduced: "已复现",
};

const BACKEND_MESSAGE_LABELS: Array<[string, string]> = [
  [
    "API worker restarted while the job was running; in-process jobs cannot be resumed automatically. Please resubmit the task.",
    "后端工作进程重启时该任务仍在运行，进程内任务无法自动续跑，请重新提交任务。",
  ],
  [
    "dataset.captions_file not found",
    "数据集标注文件不存在，请先完成数据准备或切换到已就绪数据集。",
  ],
  ["job started", "任务已启动。"],
  ["job finished", "任务已完成。"],
  ["job recovered after API worker startup", "任务已在后端启动后恢复排队。"],
];

/** 中文注释：实现 formatInlineIdentifiers 的核心流程，支撑前端业务工具中的业务语义和异常边界。 */
function formatInlineIdentifiers(value: string): string {
  return value
    .replace(/\bcase_index\b/g, "案例索引")
    .replace(/\battack_debug\b/g, "攻击调试产物")
    .replace(/\bdebug\b/gi, "调试")
    .replace(/\brun_id\b/g, "运行编号")
    .replace(/\brun\b/gi, "运行")
    .replace(/\bClean Acc\b/g, "原始输入准确率")
    .replace(/\bAttacked Acc\b/g, "攻击后准确率")
    .replace(/\bAnswer Change\b/g, "答案变化率")
    .replace(/\bObject Jaccard\b/g, "对象集合重合度")
    .replace(/\bCaption Similarity\b/g, "图像描述文本相似度")
    .replace(/\bCaption Sim\b/g, "图像描述文本相似度")
    .replace(/\bText Sim\b/g, "文本相似度")
    .replace(/\bTarget Flip\b/g, "目标翻转率")
    .replace(/\bRank Δ\b/g, "排名变化")
    .replace(/\bRecall@(\d+)\b/g, "前 $1 位召回率")
    .replace(
      /\b(clip_hf|blip_itm|vilt_itm|openai_qwen35_9b|openai_qwen3_vl|openai_qwen25_vl|openai_internvl35|openai_minicpm_v|openai_ovis25|openai_gemma3_12b)\b/g,
      (match) => formatAdapterName(match),
    )
    .replace(
      /\b(advclip|tmm|advedm_plus|advedm|vqa_visual_corruption|xtransfer_uap|foa_attack|anyattack|mpc_attack|m_attack|fgsm|bim|pgd|mifgsm|nifgsm|difgsm|tifgsm|dtmifgsm|vmifgsm|vnifgsm|cw)\b/g,
      (match) => formatAttackName(match),
    );
}

/** 中文注释：实现 formatJobType 的核心流程，支撑前端业务工具中的业务语义和异常边界。 */
export function formatJobType(value: string): string {
  return JOB_TYPE_LABELS[keyText(value)] ?? formatWrapped("任务类型", value);
}

/** 中文注释：实现 formatJobStatus 的核心流程，支撑前端业务工具中的业务语义和异常边界。 */
export function formatJobStatus(value: string): string {
  return JOB_STATUS_LABELS[keyText(value)] ?? formatWrapped("任务状态", value);
}

/** 中文注释：实现 formatHealthStatus 的核心流程，支撑前端业务工具中的业务语义和异常边界。 */
export function formatHealthStatus(value: string): string {
  if (keyText(value) === "launch_blocked") {
    return "当前环境不可启动";
  }
  return HEALTH_STATUS_LABELS[keyText(value)] ?? formatWrapped("健康状态", value);
}

/** 中文注释：实现 formatBootstrapState 的核心流程，支撑前端业务工具中的业务语义和异常边界。 */
export function formatBootstrapState(value: string): string {
  return BOOTSTRAP_STATE_LABELS[keyText(value)] ?? formatWrapped("预热状态", value);
}

/** 中文注释：实现 formatAdapterRole 的核心流程，支撑前端业务工具中的业务语义和异常边界。 */
export function formatAdapterRole(value: string): string {
  return ADAPTER_ROLE_LABELS[value] ?? ADAPTER_ROLE_LABELS[keyText(value)] ?? formatWrapped("模型角色", value);
}

/** 中文注释：实现 formatRiskLevel 的核心流程，支撑前端业务工具中的业务语义和异常边界。 */
export function formatRiskLevel(value: string): string {
  return RISK_LEVEL_LABELS[keyText(value)] ?? formatWrapped("风险等级", value);
}

/** 中文注释：实现 formatLogLevel 的核心流程，支撑前端业务工具中的业务语义和异常边界。 */
export function formatLogLevel(value: string): string {
  return LOG_LEVEL_LABELS[keyText(value)] ?? formatWrapped("日志级别", value);
}

/** 中文注释：实现 formatAttackName 的核心流程，支撑前端业务工具中的业务语义和异常边界。 */
export function formatAttackName(value: string): string {
  const text = cleanText(value);
  if (isMissingText(text)) {
    return "未记录";
  }
  return ATTACK_LABELS[keyText(text)] ?? formatWrapped("攻击标识", text);
}

/** 中文注释：实现 formatModeName 的核心流程，支撑前端业务工具中的业务语义和异常边界。 */
export function formatModeName(value: string): string {
  const text = cleanText(value);
  if (isMissingText(text)) {
    return "未记录";
  }
  return MODE_LABELS[keyText(text)] ?? formatWrapped("模式标识", text);
}

/** 中文注释：实现 formatDatasetName 的核心流程，支撑前端业务工具中的业务语义和异常边界。 */
export function formatDatasetName(value: string): string {
  const text = cleanText(value);
  if (isMissingText(text)) {
    return "未记录";
  }
  return DATASET_LABELS[keyText(text)] ?? formatWrapped("数据集标识", text);
}

/** 中文注释：实现 formatRunDatasetName 的核心流程，支撑前端业务工具中的业务语义和异常边界。 */
export function formatRunDatasetName(datasetName: string, benchmarkTag = "", taskKind = ""): string {
  const datasetKey = keyText(cleanText(datasetName));
  const benchmarkKey = keyText(cleanText(benchmarkTag));
  const taskKey = keyText(cleanText(taskKind));
  if (datasetKey === "generation_jsonl") {
    if (benchmarkKey.includes("coco_object_probe_val")) return DATASET_LABELS.coco_object_probe_val;
    if (benchmarkKey.includes("coco_caption_object_val")) return DATASET_LABELS.coco_caption_object_val;
    if (benchmarkKey.includes("vqa_v2_coco_val")) return DATASET_LABELS.vqa_v2_coco_val;
    if (taskKey === "caption") return DATASET_LABELS.coco_caption_object_val;
    if (taskKey === "vqa") return DATASET_LABELS.vqa_v2_coco_val;
  }
  return formatDatasetName(datasetName || benchmarkTag);
}

/** 中文注释：实现 formatAdapterName 的核心流程，支撑前端业务工具中的业务语义和异常边界。 */
export function formatAdapterName(value: string): string {
  const text = cleanText(value);
  if (isMissingText(text)) {
    return "未记录";
  }
  return ADAPTER_NAME_LABELS[keyText(text)] ?? formatWrapped("适配器标识", text);
}

/** 中文注释：实现 formatModelSource 的核心流程，支撑前端业务工具中的业务语义和异常边界。 */
export function formatModelSource(value: string): string {
  const text = cleanText(value);
  if (isMissingText(text)) {
    return "未记录";
  }
  return MODEL_SOURCE_LABELS[keyText(text)] ?? formatWrapped("模型标识", text);
}

/** 中文注释：实现 formatEvalScope 的核心流程，支撑前端业务工具中的业务语义和异常边界。 */
export function formatEvalScope(value: string): string {
  const text = cleanText(value);
  if (isMissingText(text)) {
    return "未记录";
  }
  return EVAL_SCOPE_LABELS[keyText(text)] ?? formatWrapped("评测范围", text);
}

/** 中文注释：实现 formatAttackScopeName 的核心流程，支撑前端业务工具中的业务语义和异常边界。 */
export function formatAttackScopeName(value: string): string {
  const text = cleanText(value);
  if (isMissingText(text)) {
    return "未记录扰动类型";
  }
  return ATTACK_SCOPE_LABELS[keyText(text)] ?? formatWrapped("扰动类型", text);
}

/** 中文注释：实现 formatJudgeReason 的核心流程，支撑前端业务工具中的业务语义和异常边界。 */
export function formatJudgeReason(value: unknown): string {
  const text = cleanText(value);
  if (isMissingText(text)) {
    return "未记录原因";
  }
  const key = keyText(text);
  if (JUDGE_REASON_LABELS[key]) return JUDGE_REASON_LABELS[key];
  if (key.includes("asset_retest")) return "样本库复测判定";
  return formatWrapped("原因说明", text);
}

/** 中文注释：实现 formatFeatureMethod 的核心流程，支撑前端业务工具中的业务语义和异常边界。 */
export function formatFeatureMethod(value: string): string {
  const text = cleanText(value);
  if (isMissingText(text)) {
    return "未记录";
  }
  return FEATURE_METHOD_LABELS[keyText(text)] ?? formatWrapped("投影方法", text);
}

/** 中文注释：实现 formatStageName 的核心流程，支撑前端业务工具中的业务语义和异常边界。 */
export function formatStageName(value: string): string {
  const text = cleanText(value);
  if (isMissingText(text)) {
    return "未记录";
  }
  return STAGE_LABELS[keyText(text)] ?? formatWrapped("阶段标识", text);
}

/** 中文注释：实现 formatModalityName 的核心流程，支撑前端业务工具中的业务语义和异常边界。 */
export function formatModalityName(value: string): string {
  const text = cleanText(value);
  if (isMissingText(text)) {
    return "未记录";
  }
  return MODALITY_LABELS[keyText(text)] ?? formatWrapped("模态标识", text);
}

/** 中文注释：实现 formatProjectionGroupName 的核心流程，支撑前端业务工具中的业务语义和异常边界。 */
export function formatProjectionGroupName(stage: string, modality: string): string {
  return `${formatStageName(stage)} / ${formatModalityName(modality)}`;
}

/** 中文注释：实现 formatRiskDimension 的核心流程，支撑前端业务工具中的业务语义和异常边界。 */
export function formatRiskDimension(value: string): string {
  const text = cleanText(value);
  if (isMissingText(text)) {
    return "未记录";
  }
  return RISK_DIMENSION_LABELS[keyText(text)] ?? "未命名风险维度";
}

/** 中文注释：实现 formatPaperStatus 的核心流程，支撑前端业务工具中的业务语义和异常边界。 */
export function formatPaperStatus(value: string): string {
  const text = cleanText(value);
  if (isMissingText(text)) {
    return "未记录";
  }
  return PAPER_STATUS_LABELS[keyText(text)] ?? formatWrapped("复现状态", text);
}

/** 中文注释：实现 formatPlatformLabel 的核心流程，支撑前端业务工具中的业务语义和异常边界。 */
export function formatPlatformLabel(value: string): string {
  const text = cleanText(value);
  if (isMissingText(text)) {
    return "未记录";
  }
  if (keyText(text).includes("linux")) {
    return `Linux 系统（${text}）`;
  }
  if (keyText(text).includes("windows")) {
    return `Windows 系统（${text}）`;
  }
  return formatWrapped("运行平台", text);
}

/** 中文注释：实现 formatNormLabel 的核心流程，支撑前端业务工具中的业务语义和异常边界。 */
export function formatNormLabel(value: string): string {
  const text = cleanText(value);
  if (keyText(text) === "l2") {
    return "二范数（L2）";
  }
  if (keyText(text) === "linf") {
    return "无穷范数（Linf）";
  }
  return formatWrapped("范数标识", text);
}

/** 中文注释：实现 formatRecallLabel 的核心流程，支撑前端业务工具中的业务语义和异常边界。 */
export function formatRecallLabel(k: number, stage: "clean" | "attacked" | "delta"): string {
  const stageLabel = stage === "clean" ? "正常输入" : stage === "attacked" ? "受攻击输入" : "相对变化";
  return `${stageLabel}前 ${k} 位召回率`;
}

/** 中文注释：实现 formatLogMessage 的核心流程，支撑前端业务工具中的业务语义和异常边界。 */
export function formatLogMessage(value: string): string {
  const text = cleanText(value);
  if (isMissingText(text)) {
    return "未记录日志内容";
  }
  return `日志内容：${formatBackendMessage(text)}`;
}

/** 中文注释：实现 formatBackendMessage 的核心流程，支撑前端业务工具中的业务语义和异常边界。 */
export function formatBackendMessage(value: string): string {
  const text = cleanText(value);
  if (isMissingText(text)) {
    return "未记录后端说明";
  }
  const vlrStart = text.match(/^run-vlr start:\s*dataset=([^\s]+)\s+attack=([^\s]+)\s+scope=([^\s]+)$/i);
  if (vlrStart) {
    return `图文检索测评开始：数据集 ${formatDatasetName(vlrStart[1])}，攻击方法 ${formatAttackName(vlrStart[2])}，评测范围 ${formatEvalScope(vlrStart[3])}。`;
  }
  const vlrSuccess = text.match(/^run-vlr success:\s*run_id=([^\s]+)$/i);
  if (vlrSuccess) {
    return `图文检索测评完成，运行编号：${vlrSuccess[1]}。`;
  }
  const deferredLocalVlm = text.match(/^defer local VLM startup until post-attack evaluation:\s*adapter=(.+)$/i);
  if (deferredLocalVlm) {
    return `攻击图生成完成后再启动本地受测模型：${formatAdapterName(deferredLocalVlm[1])}。`;
  }
  const generationStart = text.match(/^run-generation start:\s*task=([^\s]+)\s+dataset=([^\s]+)(?:\s+benchmark=([^\s]+))?\s+attack=([^\s]+)(?:\s+[^\s=]+=[^\s]+)?$/i);
  if (generationStart) {
    const task = generationStart[1];
    const dataset = generationStart[2];
    const benchmark = generationStart[3] || "";
    const attack = generationStart[4];
    return `生成式测评开始：任务 ${formatJobType(task)}，数据集 ${formatRunDatasetName(dataset, benchmark, task)}，攻击方法 ${formatAttackName(attack)}。`;
  }
  const generationSuccess = text.match(/^run-generation success:\s*run_id=([^\s]+)$/i);
  if (generationSuccess) {
    return `生成式测评完成，运行编号：${generationSuccess[1]}。`;
  }
  const jobFailed = text.match(/^job failed:\s*(.+)$/i);
  if (jobFailed) {
    return `任务失败：${formatBackendMessage(jobFailed[1])}`;
  }
  if (text.includes("TaskConfig.__init__() got an unexpected keyword argument 'max_pairs'")) {
    return "检索配对数应放在运行器配置中，请重新提交任务。";
  }
  if (text.includes("ReportConfig.__init__() got an unexpected keyword argument 'task_name'")) {
    return "任务名称应放在扩展元数据中，请重新提交任务。";
  }
  const normalized = keyText(text);
  const hit = BACKEND_MESSAGE_LABELS.find(([needle]) => normalized.includes(needle.toLowerCase()));
  if (hit) {
    return hit[1];
  }
  return formatInlineIdentifiers(text);
}
