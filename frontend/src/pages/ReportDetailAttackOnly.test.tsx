// 文件说明：该文件属于前端页面，集中实现 ReportDetailAttackOnly.test 相关逻辑。
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";

import ReportDetailPage from "./ReportDetailPage";

vi.mock("../lib/api", () => {
  return {
    getRunSummary: vi.fn(async () => ({
      run_id: "r1",
      task_kind: "vlr",
      dataset_name: "flickr1k",
      benchmark_tag: "flickr1k_validation",
      retrieval_k: [1, 5, 10],
      model_adapter: "clip_hf",
      victim_model_adapters: ["clip_hf"],
      attack: "advclip",
      attack_mode: "A",
      asr: 0.6,
      asr_attack: 0.6,
      risk_score: 0.66,
      risk_level: "high",
      risk_scenario: "retrieval",
      victims: {
        clip_hf: {
          clean: { "ir_r@1": 0.8, "tr_r@1": 0.8 },
          attacked: { "ir_r@1": 0.3, "tr_r@1": 0.3 },
        },
      },
      metric_quality: {
        valid_for_attack_strength_claim: false,
        has_warnings: true,
        note: "攻击后错误率仅作诊断。",
        flags: [
          {
            stage: "attacked",
            victim: "clip_hf",
            message: "该阶段打分几乎为常数，不应解释为有效攻击成功率。",
            diagnostics: { shape: [1, 1], unique_rounded: 1 },
          },
        ],
      },
    })),
    getRunReportData: vi.fn(async () => ({
      mode_stats: { "advclip:A": { count: 4, asr: 0.6 } },
      metric_series: { l2: [1, 2], linf: [0.1, 0.2] },
      rows_preview: [],
      vlr: { failure_cases: [] },
      reproduction_fidelity: [],
    })),
    getRunCases: vi.fn(async () => ({ total: 0, items: [] })),
  };
});

describe("ReportDetailPage attack-only view", () => {
  it("renders attack metrics and omits retired comparison fields", async () => {
    const qc = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
    render(
      <QueryClientProvider client={qc}>
        <MemoryRouter initialEntries={["/reports/r1"]}>
          <Routes>
            <Route path="/reports/:runId" element={<ReportDetailPage />} />
          </Routes>
        </MemoryRouter>
      </QueryClientProvider>,
    );

    expect(await screen.findByText("攻击模式统计")).toBeTruthy();
    expect(await screen.findByText("数据集：Flickr1k 单文本切片")).toBeTruthy();
    expect(await screen.findByText("受测模型：CLIP 检索模型")).toBeTruthy();
    expect((await screen.findAllByText(/攻击成功率/)).length).toBeGreaterThan(0);
    expect(await screen.findByText("风险分数：0.660000")).toBeTruthy();
    expect(await screen.findByText("补丁来源：未记录 / 不适用")).toBeTruthy();
    expect(await screen.findByText("注册键：未记录 / 不适用")).toBeTruthy();
    expect(await screen.findByText("多模型检索指标（正常 / 攻击后）")).toBeTruthy();
    expect(await screen.findByText("受攻击输入前 1 位召回率")).toBeTruthy();
    expect(await screen.findByText("指标质量提示")).toBeTruthy();
    expect(await screen.findByText("本次检索指标不适合直接作为攻击强度结论。")).toBeTruthy();
    expect(await screen.findByText("该阶段打分几乎为常数，不应解释为有效攻击成功率。")).toBeTruthy();
    expect(await screen.findByText("矩阵规模：1 × 1，唯一分数数：1")).toBeTruthy();
    expect(screen.queryByText(/\u9632\u5fa1|\u653b\u9632/)).toBeNull();
  });
});
