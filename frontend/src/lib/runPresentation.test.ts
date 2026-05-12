// 文件说明：该文件属于前端业务工具，集中实现 runPresentation.test 相关逻辑。
import { describe, expect, it } from "vitest";

import { isDemoRun, isHighRisk, riskBucket, riskText, riskTone } from "./runPresentation";

describe("runPresentation", () => {
  it("filters demo and toy runs out of formal pages", () => {
    expect(isDemoRun({ run_id: "archived_demo", model_adapter: "clip_hf" })).toBe(true);
    expect(isDemoRun({ run_id: "formal", model_adapter: "dummy" })).toBe(true);
    expect(isDemoRun({ run_id: "formal", model_adapter: "fixture_vlm" })).toBe(true);
    expect(isDemoRun({ run_id: "formal", victim_model_adapters: ["fixture_vlm"] })).toBe(true);
    expect(isDemoRun({ run_id: "formal", dataset_name: "toy_shapes" })).toBe(true);
    expect(isDemoRun({ run_id: "formal", benchmark_tag: "flickr1k_validation", model_adapter: "clip_hf" })).toBe(false);
  });

  it("keeps critical risk as the highest risk level", () => {
  expect(riskText("critical")).toBe("极高风险");
  expect(riskText("中")).toBe("中风险");
  expect(riskTone("critical")).toBe("red");
    expect(riskBucket("critical")).toBe("high");
    expect(isHighRisk("critical")).toBe(true);
  });
});
