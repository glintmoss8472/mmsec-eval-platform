// 文件说明：该文件属于前端工程配置，集中实现 DashboardPage.validationCount.test 相关逻辑。
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";

import DashboardPage from "./pages/DashboardPage";

vi.mock("./lib/api", () => ({
  getRunAnalytics: vi.fn(async () => ({ total_runs: 0, total_cases: 0, avg_asr_attack: 0, formal_runs: 0, debug_runs: 0, high_risk_runs: 0, runs_with_case_evidence: 0, low_confidence_runs: 0, task_groups: [], risk_distribution: [], result_type_distribution: [], confidence_distribution: [], attack_matrix: [], latest_runs: [] })),
  getSystemOverview: vi.fn(async () => ({
    supported_model_count: 10,
    dataset_total_count: 6,
    models: [
      { adapter: "clip_hf", display_name: "CLIP 检索模型", formal_eval: true, health_status: "ready" },
    ],
    datasets: [{ key: "coco_subset", name: "COCO val2017 完整验证子集", ready: true }],
    attacks: ["fgsm", "pgd", "advclip", "tmm", "advedm", "advedm_plus"],
    latest_runs: [],
    torch: { cuda_available: true },
    source_documents: {},
    paper_repositories: [],
    patch_registry: {},
  })),
  listRuns: vi.fn(async () => ({ total: 0, items: [] })),
  listJobs: vi.fn(async () => ({
    total: 2,
    items: [
      {
        id: "job1",
        job_type: "run_vlr",
        status: "success",
        created_at: "",
        config_path: "",
        run_id: "",
        error_code: "",
        error_message: "",
        benchmark_mode: false,
      },
      {
        id: "job2",
        job_type: "run_vlr",
        status: "success",
        created_at: "",
        config_path: "",
        run_id: "",
        error_code: "",
        error_message: "",
        benchmark_mode: false,
      },
    ],
  })),
}));

/** 中文注释：实现 createClient 的核心流程，支撑前端工程配置中的业务语义和异常边界。 */
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

describe("DashboardPage counts", () => {
  it("renders backend-connected platform counts in the redesigned dashboard", async () => {
    render(
      <QueryClientProvider client={createClient()}>
        <MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
          <DashboardPage />
        </MemoryRouter>
      </QueryClientProvider>,
    );

    expect(await screen.findByText("已接入模型数量")).toBeInTheDocument();
    expect(await screen.findByText("已完成测评任务")).toBeInTheDocument();
    expect(await screen.findByText("系统提示")).toBeInTheDocument();
    expect(await screen.findByText(/系统累计已有 0 个任务完成报告生成/)).toBeInTheDocument();
    expect(await screen.findByText("各模型风险对比")).toBeInTheDocument();
    expect(await screen.findByText("CLIP 检索")).toBeInTheDocument();
    expect(screen.queryByText("内置生成式演示")).toBeNull();
  });
});
