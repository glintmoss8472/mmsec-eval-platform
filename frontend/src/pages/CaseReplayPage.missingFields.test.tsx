// 文件说明：该文件属于前端页面，集中实现 CaseReplayPage.missingFields.test 相关逻辑。
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";

import CaseReplayPage from "./CaseReplayPage";

const mockGetCaseDetail = vi.hoisted(() => vi.fn());

vi.mock("../lib/api", () => ({
  getCaseDetail: mockGetCaseDetail,
  runAssetUrl: vi.fn((_runId: string, path: string) => `/assets/${path}`),
}));

/** 中文注释：实现 createClient 的核心流程，支撑前端页面中的业务语义和异常边界。 */
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

/** 中文注释：实现 renderPage 的核心流程，支撑前端页面中的业务语义和异常边界。 */
function renderPage() {
  return render(
    <QueryClientProvider client={createClient()}>
      <MemoryRouter initialEntries={["/reports/run-1/cases/sample-1"]}>
        <Routes>
          <Route path="/reports/:runId/cases/:sampleId" element={<CaseReplayPage />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

afterEach(() => {
  cleanup();
  mockGetCaseDetail.mockReset();
});

describe("CaseReplayPage missing fields", () => {
  it("uses an explicit loading state before the real case bundle arrives", () => {
    mockGetCaseDetail.mockImplementation(() => new Promise(() => undefined));
    renderPage();

    expect(screen.getByText("正在从后端读取真实样本、攻击前后图片、生成输出和调试可视化，请稍候。")).toBeTruthy();
    expect(screen.getByText("正在加载样本详情，不会用空白占位冒充缺失数据。")).toBeTruthy();
  });

  it("uses explicit empty-state text instead of bare dash placeholders", async () => {
    mockGetCaseDetail.mockResolvedValue({
      case_bundle: {
        artifact_refs: {},
        visual_labels: {},
        adversarial: { metadata: {} },
        diagnostics: {},
        stages: {},
      },
      attack_debug: { debug: {}, files: [] },
    });

    renderPage();

    expect((await screen.findAllByText("当前案例未记录该阶段输入文本")).length).toBeGreaterThan(0);
    expect((await screen.findAllByText("未记录输出说明")).length).toBeGreaterThan(0);
    expect(screen.queryByText("思维链偏移分数：未记录 / 不适用")).toBeNull();
    expect(screen.queryByText("调试文件：")).toBeNull();
    expect(screen.queryByText("未记录调试文件")).toBeNull();
    expect(screen.queryByText("注意力与掩码可视化")).toBeNull();
    expect(screen.queryByText("思维链轨迹查看")).toBeNull();
    expect(screen.queryByText("当前检索型案例没有生成式思维链轨迹；请查看输入文本变化和检索分数说明。")).toBeNull();
  });

  it("renders real generation task inputs and outputs without optional debug-only panels", async () => {
    mockGetCaseDetail.mockResolvedValue({
      case_bundle: {
        task_kind: "vqa",
        artifact_refs: {
          clean_image: "cases/sample-1/clean.png",
          adv_image: "cases/sample-1/adv.png",
          attention_map: "attack_debug/sample-1/advedm_plus_attention.png",
          mask_map: "attack_debug/sample-1/advedm_plus_mask.png",
        },
        visual_labels: { clean: "原始图片", adv: "攻击后图片" },
        inputs: {
          clean: { text: "What color are the gym shoes?" },
          adv: { text: "What color are the gym shoes?" },
        },
        outputs: {
          clean: { text: "white" },
          adv: { text: "blue" },
        },
        adversarial: { metadata: {} },
        diagnostics: { text_diff_score: 0.25, embedding_shift: 0.12 },
        metrics: { answer: "white", clean_correct: true, attacked_correct: false, answer_changed: true, attack_success: true },
      },
      attack_debug: { debug: { trace_steps: 1, trace_tail: [{ step: 1, loss_total: 0.5 }] }, files: ["sample-1/debug.json"] },
    });

    renderPage();

    expect(await screen.findByText("原始英文问题")).toBeTruthy();
    expect(await screen.findByText("攻击后英文问题")).toBeTruthy();
    expect(await screen.findByText("原始英文输出")).toBeTruthy();
    expect(await screen.findByText("攻击后英文输出")).toBeTruthy();
    expect((await screen.findAllByText("What color are the gym shoes?")).length).toBe(2);
    expect(await screen.findByText("blue")).toBeTruthy();
    expect(await screen.findByRole("img", { name: "原始图片" })).toBeTruthy();
    expect(screen.queryByRole("img", { name: "注意力热图" })).toBeNull();
    expect(screen.queryByRole("img", { name: "攻击掩码" })).toBeNull();
    expect(await screen.findByText("输出差异分数：0.250000")).toBeTruthy();
    expect(await screen.findByText("生成式判定指标")).toBeTruthy();
    expect(await screen.findByText("标准答案：")).toBeTruthy();
    expect(await screen.findByText("攻击成功：")).toBeTruthy();
    expect((await screen.findAllByText("是")).length).toBeGreaterThan(0);
    expect(screen.queryByText("文件（样本调试记录）")).toBeNull();
  });
});
