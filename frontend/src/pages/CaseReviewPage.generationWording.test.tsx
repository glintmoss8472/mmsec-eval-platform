// 文件说明：该文件属于前端页面，集中实现 CaseReviewPage.generationWording.test 相关逻辑。
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import CaseReviewPage from "./CaseReviewPage";

const mockListRuns = vi.hoisted(() => vi.fn());
const mockListCaseIndex = vi.hoisted(() => vi.fn());
const mockGetCaseDetail = vi.hoisted(() => vi.fn());
const mockGetRunOptions = vi.hoisted(() => vi.fn());

vi.mock("../lib/api", () => ({
  listRuns: mockListRuns,
  listCaseIndex: mockListCaseIndex,
  getCaseDetail: mockGetCaseDetail,
  getRunOptions: mockGetRunOptions,
  runAssetUrl: vi.fn((_runId: string, path: string) => `/assets/${path}`),
}));

/** 构建 `create client` 结构，供页面渲染或测试断言复用。 */
function createClient() {
  return new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
}

/** 整理 `render page` 前端辅助逻辑，保持数据转换和展示口径一致。 */
function renderPage() {
  return render(
    <QueryClientProvider client={createClient()}>
      <MemoryRouter>
        <CaseReviewPage />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

/** 整理 `案例 item` 前端辅助逻辑，保持数据转换和展示口径一致。 */
function caseItem(index: number) {
  return {
    run_id: `run-${index}`,
    sample_id: `case-${index}`,
    task_kind: "vlr",
    dataset_name: "coco_subset",
    benchmark_tag: "coco_subset",
    attack: "fgsm",
    model_adapter: "clip_hf",
    judge_success: true,
    risk_level: "low",
    artifact_status: "complete",
    evidence_confidence: "low",
    created_at: "2026-05-07T01:48:09+08:00",
    text: `case text ${index}`,
  };
}

/** 整理 `page items` 前端辅助逻辑，保持数据转换和展示口径一致。 */
function pageItems(start: number, count = 50) {
  return Array.from({ length: count }, (_, offset) => caseItem(start + offset));
}

beforeEach(() => {
  mockGetRunOptions.mockResolvedValue({ task_kinds: [], attacks: [], risk_levels: [] });
});

afterEach(() => {
  cleanup();
  mockListRuns.mockReset();
  mockListCaseIndex.mockReset();
  mockGetCaseDetail.mockReset();
  mockGetRunOptions.mockReset();
});

describe("CaseReviewPage case library", () => {
  it("uses question/output wording and preview evidence for VQA cases", async () => {
    mockListRuns.mockResolvedValue({ items: [{ run_id: "run-vqa", task_kind: "vqa", attack: "advedm_plus", model_adapter: "openai_qwen35_9b", risk_level: "medium" }] });
    mockListCaseIndex.mockResolvedValue({ total: 1, items: [{ run_id: "run-vqa", sample_id: "sample-vqa", task_kind: "vqa", dataset_name: "generation_jsonl", benchmark_tag: "vqa_v2_coco_val", attack: "advedm_plus", model_adapter: "openai_qwen35_9b", judge_success: true, risk_level: "medium", artifact_status: "complete", evidence_confidence: "low" }] });
    mockGetCaseDetail.mockResolvedValue({
      case_bundle: {
        task_kind: "vqa",
        sample: { sample_id: "sample-vqa" },
        artifact_refs: {},
        inputs: { clean: { text: "What color are the gym shoes?" }, adv: { text: "What color are the gym shoes?" } },
        outputs: { clean: { text: "white" }, adv: { text: "blue" } },
        metrics: { perturbation_l2: 1.2, perturbation_linf: 0.02 },
        artifact_capability: [
          { key: "output_diff", label: "输出差异", status: "available", reason: "可查看" },
          { key: "debug_files", label: "调试文件", status: "missing", reason: "未记录调试文件" },
          { key: "attention_map", label: "注意力热图", status: "not_applicable", reason: "当前方法不适用" },
          { key: "mask_map", label: "攻击掩码", status: "not_applicable", reason: "当前方法不适用" },
          { key: "patch_preview", label: "补丁预览", status: "not_applicable", reason: "当前方法不适用" },
          { key: "cot_trace", label: "CoT 轨迹", status: "not_applicable", reason: "当前方法不适用" },
        ],
      },
      attack_debug: { files: [] },
    });

    renderPage();

    expect(await screen.findByText("案例证据预览")).toBeTruthy();
    expect((await screen.findAllByText("What color are the gym shoes?")).length).toBeGreaterThanOrEqual(2);
    expect(await screen.findByText("blue")).toBeTruthy();
    expect(await screen.findByText("输出差异：可查看")).toBeTruthy();
    expect(screen.queryByText("调试文件：未记录调试文件")).toBeNull();
    expect(screen.queryByText("注意力热图：当前方法不适用")).toBeNull();
    expect(screen.queryByText("攻击掩码：当前方法不适用")).toBeNull();
    expect(screen.queryByText("补丁预览：当前方法不适用")).toBeNull();
    expect(screen.queryByText("CoT 轨迹：当前方法不适用")).toBeNull();
  });

  it("defaults to cross-run case index instead of a pinned historical run", async () => {
    mockListRuns.mockResolvedValue({ items: [{ run_id: "20260504_050112_171793", task_kind: "vlr", dataset_name: "coco_subset", attack: "advedm_plus", model_adapter: "clip_hf", case_count: 1, risk_level: "low" }] });
    mockListCaseIndex.mockResolvedValue({ total: 1, items: [{ run_id: "20260504_050112_171793", sample_id: "vlr-clip_hf-t2i-38", task_kind: "vlr", dataset_name: "coco_subset", attack: "advedm_plus", model_adapter: "clip_hf", judge_success: false, risk_level: "low", artifact_status: "complete", evidence_confidence: "low", text: "A black Honda motorcycle parked in front of a garage." }] });
    mockGetCaseDetail.mockResolvedValue({ case_bundle: { task_kind: "vlr", artifact_refs: {}, inputs: { clean: { text: "A black Honda motorcycle parked in front of a garage." }, adv: { text: "A black Honda motorcycle parked in front of a garage." } }, outputs: { clean: { text: "目标图像" }, adv: { text: "Top-5" } }, metrics: {} }, attack_debug: { files: [] } });

    renderPage();

    await waitFor(() => expect(mockListCaseIndex).toHaveBeenCalled());
    expect((await screen.findAllByText("vlr-clip_hf-t2i-38")).length).toBeGreaterThanOrEqual(2);
    expect((await screen.findAllByText(/COCO val2017 完整验证子集/)).length).toBeGreaterThanOrEqual(2);
  });

  it("does not reset to the first page while an uncached next page is loading", async () => {
    let resolvePage2: (value: { total: number; items: ReturnType<typeof pageItems> }) => void = () => {};
    mockListRuns.mockResolvedValue({ items: [] });
    mockGetCaseDetail.mockResolvedValue({ case_bundle: { task_kind: "vlr", artifact_refs: {}, inputs: {}, outputs: {}, metrics: {} }, attack_debug: { files: [] } });
    mockListCaseIndex.mockImplementation((params = {}) => {
      if (params.page === 2) {
        return new Promise((resolve) => {
          resolvePage2 = resolve;
        });
      }
      return Promise.resolve({ total: 1292, items: pageItems(1) });
    });

    renderPage();

    expect(await screen.findByText("1 - 50 / 1292")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "下一页" }));

    await waitFor(() => expect(mockListCaseIndex).toHaveBeenCalledWith(expect.objectContaining({ page: 2 })));
    expect(await screen.findByText("51 - 100 / 1292")).toBeInTheDocument();

    resolvePage2({ total: 1292, items: pageItems(51) });
    expect((await screen.findAllByText("case-51")).length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText("51 - 100 / 1292")).toBeInTheDocument();
    expect(mockListCaseIndex.mock.calls.filter(([params]) => params.page === 1)).toHaveLength(1);
  });
});
