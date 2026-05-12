import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";

import DashboardPage from "./pages/DashboardPage";

vi.mock("./lib/api", () => ({
  getRunAnalytics: vi.fn(async () => ({ total_runs: 1, total_cases: 1, avg_asr_attack: 0.5, formal_runs: 1, debug_runs: 0, high_risk_runs: 1, runs_with_case_evidence: 1, low_confidence_runs: 0, task_groups: [{ task_kind: "vlr", count: 1 }], risk_distribution: [{ key: "high", count: 1 }], result_type_distribution: [], confidence_distribution: [], attack_matrix: [], latest_runs: [] })),
  getSystemOverview: vi.fn(async () => ({
    supported_model_count: 2,
    dataset_total_count: 1,
    models: [
      {
        adapter: "clip_hf",
        display_name: "对比语言-图像预训练模型（CLIP）",
        family: "本地模型",
        launch_mode: "本地加载",
        health_status: "ready",
        endpoint_or_source: "openai/clip-vit-base-patch32",
      },
    ],
    datasets: [{ key: "coco_subset", name: "COCO val2017 完整验证子集", ready: true }],
    attacks: ["advclip", "advedm_plus"],
    latest_runs: [],
    torch: { cuda_available: true },
    source_documents: {},
    paper_repositories: [],
    patch_registry: {},
  })),
  listRuns: vi.fn(async () => ({
    total: 1,
    items: [
      {
        run_id: "formal_e1",
        created_at: "2026-05-02T00:00:00Z",
        task_kind: "vlr",
        dataset_name: "coco_subset",
        benchmark_tag: "paper_e1_advedm_plus_coco64",
        attack: "advedm_plus",
        mode: "",
        experiment_id: "paper_e1_advedm_plus_coco",
        suite: "E1_classic_coco",
        suite_label: "",
        evidence_group: "",
        experiment_label: "",
        model_adapter: "clip_hf",
        surrogate_model_adapter: "clip_hf",
        victim_model_adapters: ["openai_qwen3_vl"],
        asr: 0.5,
        asr_attack: 0.5,
        risk_score: 0.6,
        risk_level: "high",
        risk_scenario: "",
        avg_l2: 1.0,
        path: "",
      },
    ],
  })),
  listJobs: vi.fn(async () => ({ total: 0, items: [] })),
}));

function createClient() {
  return new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        staleTime: Infinity,
      },
    },
  });
}

describe("DashboardPage visual overview", () => {
  it("renders the new five-page dashboard summary from backend data", async () => {
    render(
      <QueryClientProvider client={createClient()}>
        <MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
          <DashboardPage />
        </MemoryRouter>
      </QueryClientProvider>,
    );

    expect(await screen.findByText("已接入模型数量")).toBeInTheDocument();
    expect(await screen.findByText("已接入正式数据集")).toBeInTheDocument();
    expect(await screen.findByText("最近七天测评任务趋势")).toBeInTheDocument();
    expect(await screen.findByText("风险等级分布")).toBeInTheDocument();
    expect(await screen.findByText("最近完成的测评任务")).toBeInTheDocument();
    expect(await screen.findByText(/增强细粒度具身决策攻击/)).toBeInTheDocument();
    expect((await screen.findAllByText("高风险")).length).toBeGreaterThan(0);
  });
});
