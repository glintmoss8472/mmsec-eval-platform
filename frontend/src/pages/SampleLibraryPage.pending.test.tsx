import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";

import SampleLibraryPage from "./SampleLibraryPage";

const apiState = vi.hoisted(() => ({
  batchCalls: [] as any[],
}));

vi.mock("../lib/api", () => ({
  createJob: vi.fn(),
  getSystemOverview: vi.fn(async () => ({
    models: [
      {
        adapter: "clip_hf",
        display_name: "CLIP",
        formal_eval: true,
        health_status: "ready",
        task_capabilities: ["vlr"],
      },
    ],
    datasets: [
      { key: "coco_subset", name: "coco_subset", tier: "benchmark", ready: true, item_count: 5000 },
    ],
  })),
  listSampleAssetBatches: vi.fn(async (params) => {
    apiState.batchCalls.push(params);
    return {
      total: 1,
      page: 1,
      page_size: 10,
      items: [
        {
          batch_id: "20260507_153610_133285",
          source_run_id: "20260507_153610_133285",
          task_kind: "vlr",
          dataset_name: "coco_subset",
          benchmark_tag: "coco_subset",
          model_adapter: "",
          attack: "fgsm",
          attack_scope: "图像扰动",
          created_at: "2026-05-07T15:36:13+08:00",
          updated_at: "2026-05-07T15:36:14+08:00",
          total_assets: 1,
          ready_assets: 0,
          callable_assets: 0,
          pending_evaluation_assets: 1,
          summary_only_assets: 0,
          legacy_assets: 0,
          evidence_complete_count: 1,
          evidence_integrity: 1,
          successful_assets: 0,
          avg_risk_score: 0,
          avg_l2: 5.979,
          avg_linf: 0.008,
          used_count: 0,
          batch_call_count: 0,
          sample_usage_count: 0,
          asset_ids: ["20260507_153610_133285::38"],
          preview_assets: [
            {
              asset_id: "20260507_153610_133285::38",
              sample_id: "38",
              source_text: "A black Honda motorcycle parked in front of a garage.",
              reusable_status: "pending_evaluation",
              case_url: "",
            },
          ],
          report_url: "",
          batch_status: "pending_evaluation",
        },
      ],
      summary: {
        total_batches: 1,
        callable_batches: 0,
        total_assets: 1,
        ready_assets: 0,
        callable_assets: 0,
        pending_evaluation_assets: 1,
        summary_only_assets: 0,
        legacy_assets: 0,
        task_distribution: { vlr: 1 },
        attack_distribution: { fgsm: 1 },
        scope_distribution: { "图像扰动": 1 },
      },
      options: {},
    };
  }),
}));

function renderPage() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter>
        <SampleLibraryPage />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe("SampleLibraryPage pending batches", () => {
  beforeEach(() => {
    cleanup();
    apiState.batchCalls = [];
  });

  it("shows pending evaluation batches in the default managed list without report links", async () => {
    renderPage();

    await screen.findByText("20260507_153610_133285");
    expect(screen.getByText("仅生成 / 待测评")).toBeInTheDocument();
    expect(screen.getByText("待测评样本 1 条", { exact: false })).toBeInTheDocument();
    expect(screen.getByText("测评此批次")).toBeInTheDocument();
    expect(screen.queryByText("生成任务")).not.toBeInTheDocument();
    expect(screen.queryByText("来源报告")).not.toBeInTheDocument();
    await waitFor(() => expect(apiState.batchCalls.some((params) => params.reusable_status === "")).toBe(true));
  });
});
