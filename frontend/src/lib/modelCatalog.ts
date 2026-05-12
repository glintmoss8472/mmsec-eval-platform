// 文件说明：该文件属于前端业务工具，集中实现 modelCatalog 相关逻辑。
export interface RecommendedModel {
  id: string;
  glossaryId: string;
  name: string;
  family: "local" | "service";
  adapter: string;
  summary: string;
  deployment: string;
}

export const recommendedModels: RecommendedModel[] = [
  {
    id: "clip-hf",
    glossaryId: "model-clip",
    name: "对比语言图像预训练模型（CLIP）",
    family: "local",
    adapter: "clip_hf",
    summary: "作为统一代理模型，承接基础攻击与经典检索评测。",
    deployment: "本地离线模型",
  },
  {
    id: "blip-itm",
    glossaryId: "model-blip",
    name: "自举语言图像预训练匹配模型（BLIP）",
    family: "local",
    adapter: "blip_itm",
    summary: "用于补齐经典图文匹配模型，对比 CLIP 和 ViLT。",
    deployment: "本地离线模型",
  },
  {
    id: "vilt-itm",
    glossaryId: "model-vilt",
    name: "视觉语言变换器匹配模型（ViLT）",
    family: "local",
    adapter: "vilt_itm",
    summary: "用于补齐轻量级视觉语言变换器路线。",
    deployment: "本地离线模型",
  },
  {
    id: "qwen35-9b",
    glossaryId: "model-qwen35-9b",
    name: "通义千问三点五九十亿参数模型（Qwen3.5-9B）",
    family: "service",
    adapter: "openai_qwen35_9b",
    summary: "新接入的千问三点五九十亿参数模型，用于验证最新本地模型服务链路。",
    deployment: "本地自托管服务",
  },
  {
    id: "qwen3-vl",
    glossaryId: "model-qwen3-vl",
    name: "通义千问三视觉语言模型八十亿参数版（Qwen3-VL-8B）",
    family: "service",
    adapter: "openai_qwen3_vl",
    summary: "替换旧四十亿参数版，作为当前千问视觉语言主力受测模型。",
    deployment: "本地自托管服务",
  },
  {
    id: "qwen25-vl",
    glossaryId: "model-qwen25-vl",
    name: "通义千问二点五视觉语言模型七十亿参数版（Qwen2.5-VL-7B）",
    family: "service",
    adapter: "openai_qwen25_vl",
    summary: "替换旧三十亿参数版，用于补齐千问二点五视觉语言路线。",
    deployment: "本地自托管服务",
  },
  {
    id: "internvl35",
    glossaryId: "model-internvl35",
    name: "书生万象三点五八十亿参数模型（InternVL3.5-8B）",
    family: "service",
    adapter: "openai_internvl35",
    summary: "替换旧一十亿参数版，用于补齐跨架构视觉语言模型矩阵。",
    deployment: "本地自托管服务",
  },
  {
    id: "minicpm-v",
    glossaryId: "model-minicpm-v",
    name: "迷你通用处理模型视觉版四点五（MiniCPM-V 4.5）",
    family: "service",
    adapter: "openai_minicpm_v",
    summary: "替换旧二代 MiniCPM-V，用于补齐轻量级视觉语言模型接入能力。",
    deployment: "本地自托管服务",
  },
  {
    id: "ovis25",
    glossaryId: "model-ovis25",
    name: "奥维斯二点五九十亿参数模型（Ovis2.5-9B）",
    family: "service",
    adapter: "openai_ovis25",
    summary: "用于补齐 Ovis 系列视觉语言模型路线，验证不同架构的攻击迁移表现。",
    deployment: "本地自托管服务",
  },
  {
    id: "gemma3-12b",
    glossaryId: "model-gemma3-12b",
    name: "谷歌 Gemma 三代一百二十亿参数模型（Gemma 3-12B）",
    family: "service",
    adapter: "openai_gemma3_12b",
    summary: "用于补齐 Gemma 系列多模态模型路线，并测试单卡部署边界。",
    deployment: "本地自托管服务",
  },
];
