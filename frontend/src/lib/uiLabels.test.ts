import { describe, expect, it } from "vitest";

import {
  formatAdapterName,
  formatAttackName,
  formatDatasetName,
  formatRunDatasetName,
  formatEvalScope,
  formatLogMessage,
  formatModeName,
  formatWrapped,
} from "./uiLabels";

describe("ui label missing values", () => {
  it("uses explicit missing-state labels instead of bare dashes", () => {
    expect(formatWrapped("实验编号", "-")).toBe("未记录");
    expect(formatAttackName("-")).toBe("未记录");
    expect(formatAdapterName("")).toBe("未记录");
    expect(formatDatasetName("undefined")).toBe("未记录");
    expect(formatEvalScope("null")).toBe("未记录");
    expect(formatModeName("-")).toBe("未记录");
    expect(formatLogMessage("-")).toBe("未记录日志内容");
  });

  it("explains ADVEDM mode letters with semantic objectives", () => {
    expect(formatModeName("A")).toBe("语义移除模式（A 模式，AdvEDM-R）");
    expect(formatModeName("B")).toBe("语义添加模式（B 模式，AdvEDM-A）");
  });

  it("translates generation job logs with concrete dataset names", () => {
    expect(formatLogMessage("run-generation start: task=vqa dataset=generation_jsonl benchmark=coco_object_probe_val_real attack=advedm_plus")).toContain("COCO 对象存在性探测真实子集");
    expect(formatLogMessage("run-generation success: run_id=20260504_030128_565966")).toContain("生成式测评完成");
  });

  it("expands generation JSONL run labels to their real dataset names", () => {
    expect(formatRunDatasetName("generation_jsonl", "vqa_v2_coco_val_real", "vqa")).toBe("VQA v2 COCO 验证真实子集");
    expect(formatRunDatasetName("generation_jsonl", "coco_caption_object_val_real", "caption")).toBe("COCO 图像描述对象级真实子集");
    expect(formatRunDatasetName("generation_jsonl", "coco_object_probe_val_real", "vqa")).toBe("COCO 对象存在性探测真实子集");
  });
});
