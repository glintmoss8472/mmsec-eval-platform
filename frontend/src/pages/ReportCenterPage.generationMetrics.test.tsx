// 文件说明：该文件属于前端页面，集中实现 ReportCenterPage.generationMetrics.test 相关逻辑。
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";

import ReportCenterPage from "./ReportCenterPage";

vi.mock("../lib/api", () => ({
  getRunAnalytics: vi.fn(async () => ({ total_runs: 1, total_cases: 1, task_groups: [], risk_distribution: [], result_type_distribution: [], confidence_distribution: [], attack_matrix: [], latest_runs: [] })),
  getRunSummary: vi.fn(async () => ({ task_kind: "caption", generation_metrics: { caption_text_similarity: 1, semantic_preservation_rate: 1, object_jaccard: 1, target_flip_rate: 0 } })),
  listRuns: vi.fn(async () => ({
    total: 1,
    items: [{
      run_id: "caption_real",
      created_at: "2026-05-04T00:00:00Z",
      task_kind: "caption",
      dataset_name: "generation_jsonl",
      benchmark_tag: "coco_caption_object_val_real",
      attack: "advedm_plus",
      model_adapter: "openai_qwen35_9b",
      surrogate_model_adapter: "clip_hf",
      victim_model_adapters: ["openai_qwen35_9b"],
      asr: 0,
      asr_attack: 0,
      risk_score: 0.1,
      risk_level: "low",
      risk_scenario: "caption",
      avg_l2: 0.5,
      caption_text_similarity: 1,
      semantic_preservation_rate: 1,
      object_jaccard: 1,
      case_count: 1,
      evidence_sample_count: 1,
      evidence_confidence: "low",
      result_type: "formal",
      path: "",
    }],
  })),
}));

/** 构建 `create client` 结构，供页面渲染或测试断言复用。 */
function createClient() {
  return new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
}

describe("ReportCenterPage generation metrics", () => {
  it("uses caption similarity as the caption baseline instead of showing missing data", async () => {
    render(
      <QueryClientProvider client={createClient()}>
        <MemoryRouter initialEntries={["/analysis"]} future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
          <ReportCenterPage />
        </MemoryRouter>
      </QueryClientProvider>,
    );
    expect(await screen.findByText("COCO 图像描述对象级真实子集")).toBeInTheDocument();
    expect((await screen.findAllByText(/100%/)).length).toBeGreaterThan(0);
    expect(screen.queryByRole("cell", { name: "暂无数据" })).toBeNull();
  });
});
