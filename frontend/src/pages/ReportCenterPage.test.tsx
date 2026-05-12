// 文件说明：该文件属于前端页面，集中实现 ReportCenterPage.test 相关逻辑。
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";

import ReportCenterPage from "./ReportCenterPage";

afterEach(() => cleanup());

vi.mock("../lib/api", () => ({
  getRunAnalytics: vi.fn(async () => ({ total_runs: 12, total_cases: 12, task_groups: [], risk_distribution: [], result_type_distribution: [], confidence_distribution: [], attack_matrix: [], latest_runs: [] })),
  getRunSummary: vi.fn(async () => ({
    risk_component_audit: [
      { key: "effectiveness", label_zh: "攻击有效性", value: 0.25, weight: 0.35, contribution: 0.0875 },
      { key: "semantic", label_zh: "语义保持风险", value: 0.75, weight: 0.15, contribution: 0.1125 },
    ],
  })),
  listRuns: vi.fn(async () => ({
    total: 12,
    items: Array.from({ length: 12 }, (_, idx) => ({
      run_id: `formal_qwen3_${idx}`,
      created_at: "2026-05-02T00:00:00Z",
      task_kind: "vlr",
      dataset_name: idx === 11 ? "coco_subset" : "flickr1k",
      benchmark_tag: "flickr1k_validation",
      attack: idx % 2 === 0 ? "advedm_plus" : "fgsm",
      model_adapter: "clip_hf",
      surrogate_model_adapter: "clip_hf",
      victim_model_adapters: ["openai_qwen3_vl"],
      asr: idx / 20,
      asr_attack: idx / 20,
      risk_score: 0.7,
      risk_level: "critical",
      risk_scenario: "retrieval",
      avg_l2: 1.2,
      clean_r1_mean: 0,
      attacked_r1_mean: 0,
      evidence_sample_count: idx < 2 ? 1 : 8,
      evidence_confidence: idx < 2 ? "low" : "medium",
      result_type: "formal",
      case_count: 1,
      path: "",
    })),
  })),
}));

/** 中文注释：实现 createClient 的核心流程，支撑前端页面中的业务语义和异常边界。 */
function createClient() {
  return new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
}

/** 中文注释：实现 renderPage 的核心流程，支撑前端页面中的业务语义和异常边界。 */
function renderPage(path = "/analysis") {
  render(
    <QueryClientProvider client={createClient()}>
      <MemoryRouter initialEntries={[path]} future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
        <ReportCenterPage />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe("ReportCenterPage", () => {
  it("renders cross-run analysis with victim labels, confidence, and zero metrics", async () => {
    renderPage();
    expect(await screen.findByText("分任务指标明细")).toBeInTheDocument();
    expect((await screen.findAllByText(/攻击前基线/)).length).toBeGreaterThan(0);
    expect((await screen.findAllByText(/Qwen3-VL-8B/)).length).toBeGreaterThan(0);
    expect((await screen.findAllByText("极高风险")).length).toBeGreaterThanOrEqual(1);
    expect(await screen.findByText(/样本数为 1 或 2/)).toBeInTheDocument();
  });

  it("paginates the result table", async () => {
    renderPage();
    expect(await screen.findByText("1 - 10 / 12")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "下一页" }));
    expect(await screen.findByText("11 - 12 / 12")).toBeInTheDocument();
  });

  it("switches the selected core record and keeps risk audit data-driven", async () => {
    renderPage();
    await screen.findByTestId("detail-row-formal_qwen3_0");
    expect(screen.getByTestId("analysis-context")).toHaveTextContent("formal_qwen3_0");
    fireEvent.click(await screen.findByTestId("detail-row-formal_qwen3_4"));
    expect(screen.getByTestId("analysis-context")).toHaveTextContent("formal_qwen3_4");
    expect(await screen.findByText("攻击有效性 25%")).toBeInTheDocument();
  });

  it("uses /reports as report center mode", async () => {
    renderPage("/reports");
    expect(await screen.findByText("报告中心")).toBeInTheDocument();
    expect(await screen.findByText("报告总数")).toBeInTheDocument();
  });
});
