import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";

import ExperimentComparePage from "./ExperimentComparePage";

const { listRunsMock, compareRunsMock } = vi.hoisted(() => ({
  listRunsMock: vi.fn(async () => ({
    total: 2,
    items: [
      {
        run_id: "r1",
        created_at: "2026-02-16T00:00:00Z",
        task_kind: "vlr",
        dataset_name: "flickr30k",
        benchmark_tag: "b1",
        attack: "advclip",
        mode: "A",
        experiment_id: "exp_1",
        model_adapter: "clip_hf",
        asr: 0.5,
        asr_attack: 0.5,
        risk_score: 0.62,
        risk_level: "high",
        risk_scenario: "retrieval",
        avg_l2: 1.2,
        path: "artifacts/runs/r1",
      },
      {
        run_id: "r2",
        created_at: "2026-02-16T00:01:00Z",
        task_kind: "vlr",
        dataset_name: "flickr30k",
        benchmark_tag: "b2",
        attack: "tmm",
        mode: "A",
        experiment_id: "exp_1",
        model_adapter: "blip_itm",
        asr: 0.6,
        asr_attack: 0.6,
        risk_score: 0.68,
        risk_level: "high",
        risk_scenario: "retrieval",
        avg_l2: 1.5,
        path: "artifacts/runs/r2",
      },
    ],
  })),
  compareRunsMock: vi.fn(async () => ({
    run_ids: ["r1", "r2"],
    compare: {
      victims: {
        clip_hf: {
          runs: {
            r1: { attack_drop: 0.2, attacked_recall: 0.1, clean_recall: 0.05 },
            r2: { attack_drop: 0.3, attacked_recall: 0.12, clean_recall: 0.08 },
          },
        },
        blip_itm: {
          runs: {
            r1: { attack_drop: 0.18, attacked_recall: 0.09, clean_recall: 0.06 },
            r2: { attack_drop: 0.22, attacked_recall: 0.11, clean_recall: 0.07 },
          },
        },
      },
    },
  })),
}));

vi.mock("../lib/api", () => {
  return {
    listRuns: listRunsMock,
    compareRuns: compareRunsMock,
  };
});

describe("ExperimentComparePage charts", () => {
  it("renders chart sections instead of only raw json", async () => {
    const qc = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
    render(
      <QueryClientProvider client={qc}>
        <MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
          <ExperimentComparePage />
        </MemoryRouter>
      </QueryClientProvider>
    );

    const cb1 = await screen.findByLabelText("r1");
    const cb2 = await screen.findByLabelText("r2");
    fireEvent.click(cb1);
    fireEvent.click(cb2);
    fireEvent.click(screen.getByText("生成对比"));

    await waitFor(() => {
      expect(compareRunsMock).toHaveBeenCalled();
    });

    expect(await screen.findByText("运行级攻击对照（受测模型均值）")).toBeTruthy();
    expect(await screen.findByText("运行综合风险评分")).toBeTruthy();
    expect(await screen.findByText("受测模型 × 运行热力图")).toBeTruthy();
    expect(await screen.findByText("单模型趋势")).toBeTruthy();
    expect(await screen.findByText("受测模型指标表")).toBeTruthy();
  });
});
