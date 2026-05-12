// 文件说明：该文件属于前端业务工具，集中实现 datasetCatalog 相关逻辑。
export interface DatasetCatalogItem {
  id: string;
  glossaryId: string;
  title: string;
  shortName: string;
  summary: string;
  benchmarkTag: string;
  tier: "benchmark" | "generation" | "demo";
  override: Record<string, unknown>;
}

export const datasetCatalog: DatasetCatalogItem[] = [
  {
    id: "coco_subset",
    glossaryId: "dataset-coco-subset",
    title: "COCO val2017 完整验证子集（5000 图文对）",
    shortName: "COCO val2017 完整验证集",
    summary: "公共图文检索主实验数据集，当前使用完整 COCO val2017 验证子集，共 5000 个一图一文样本。",
    benchmarkTag: "coco_subset",
    tier: "benchmark",
    override: {
      kind: "coco_subset",
      root: "data/coco",
      image_dir: "val2017",
      captions_file: "annotations/captions_val2017_subset.json",
      split: "val",
      benchmark_tag: "coco_subset",
    },
  },
  {
    id: "flickr30k",
    glossaryId: "dataset-flickr30k",
    title: "Flickr30k 测试集（1000 图 × 5 文本）",
    shortName: "Flickr30k 五描述测试集",
    summary: "经典图文检索测试集，当前保留 1000 张测试图片及每图 5 条描述，共 5000 条图文记录。",
    benchmarkTag: "flickr30k",
    tier: "benchmark",
    override: {
      kind: "flickr30k",
      root: "data/flickr30k",
      image_dir: "images",
      captions_file: "captions_index.jsonl",
      split: "test",
      benchmark_tag: "flickr30k",
    },
  },
  {
    id: "flickr1k",
    glossaryId: "dataset-flickr1k",
    title: "Flickr1k 单文本测试切片",
    shortName: "Flickr1k 单描述切片",
    summary: "从同一 Flickr30k 测试来源中为每张图片保留 1 条描述，形成 1000 个一图一文样本，便于构造一对一检索矩阵。",
    benchmarkTag: "flickr1k",
    tier: "benchmark",
    override: {
      kind: "flickr1k",
      root: "data/flickr30k",
      image_dir: "images",
      captions_file: "captions_index_single.jsonl",
      split: "test",
      benchmark_tag: "flickr1k",
    },
  },
  {
    id: "vqa_v2_coco_val",
    glossaryId: "dataset-vqa-v2-coco-val",
    title: "VQA v2 COCO 验证真实子集（300 问答样本）",
    shortName: "VQA v2 COCO 验证子集",
    summary: "生成式视觉问答真实数据集，图片来自 MS-COCO 2014 验证集，问题和答案来自 VQA v2 标注；运行时模型必须根据图片和问题真实生成答案。",
    benchmarkTag: "vqa_v2_coco_val_real",
    tier: "generation",
    override: {
      kind: "generation_jsonl",
      root: "data/coco2014",
      cases_jsonl: "data/coco2014/generation/vqa_v2_coco_val.jsonl",
      split: "val",
      benchmark_tag: "vqa_v2_coco_val_real",
      task_kind: "vqa",
    },
  },
  {
    id: "coco_object_probe_val",
    glossaryId: "dataset-coco-object-probe-val",
    title: "COCO 对象存在性探测真实子集（200 问答样本）",
    shortName: "COCO 对象存在性探测",
    summary: "基于 MS-COCO 实例标注自动构造是/否型对象存在性问题，用来稳定判断目标对象是否被攻击隐藏或错误引入。",
    benchmarkTag: "coco_object_probe_val_real",
    tier: "generation",
    override: {
      kind: "generation_jsonl",
      root: "data/coco2014",
      cases_jsonl: "data/coco2014/generation/coco_object_probe_val.jsonl",
      split: "val",
      benchmark_tag: "coco_object_probe_val_real",
      task_kind: "vqa",
    },
  },
  {
    id: "coco_caption_object_val",
    glossaryId: "dataset-coco-caption-object-val",
    title: "COCO 图像描述对象级真实子集（100 图像样本）",
    shortName: "COCO 图像描述对象级子集",
    summary: "图像描述生成式真实数据集，参考描述来自 COCO 图像描述标注，目标对象和非目标对象来自 COCO 实例标注，用于对象级语义变化与语义保持评测。",
    benchmarkTag: "coco_caption_object_val_real",
    tier: "generation",
    override: {
      kind: "generation_jsonl",
      root: "data/coco2014",
      cases_jsonl: "data/coco2014/generation/coco_caption_object_val.jsonl",
      split: "val",
      benchmark_tag: "coco_caption_object_val_real",
      task_kind: "caption",
    },
  },
  {
    id: "mini_flickr",
    glossaryId: "dataset-mini-flickr",
    title: "迷你图片描述数据集（Mini Flickr）",
    shortName: "Mini Flickr",
    summary: "轻量演示与回归数据集，用于快速验证平台链路、界面演示和真实冒烟任务。",
    benchmarkTag: "mini_flickr",
    tier: "demo",
    override: {
      kind: "mini_flickr",
      root: "",
      image_dir: "images",
      captions_file: "captions_index.jsonl",
      split: "test",
      benchmark_tag: "mini_flickr",
    },
  },
];

export const datasetCatalogMap = new Map(datasetCatalog.map((item) => [item.id, item]));
export const formalDatasetCatalog = datasetCatalog.filter((item) => item.tier === "benchmark");
