// 文件说明：该文件属于前端页面，集中实现 JobCenterPage.failureDisplay.test 相关逻辑。
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, render, screen, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import JobCenterPage from "./JobCenterPage";

const failureMessage = "HTTPConnectionPool(host='127.0.0.1', port=8012): Max retries exceeded with url: /v1/chat/completions";
const staleProgressMessage = "正在评测正常输入在各受测模型上的表现：openai_qwen3_vl，已完成 2048/250000 对图文配对。";

const apiState = vi.hoisted(() => ({
  jobs: { total: 0, items: [] as any[] },
  progress: {} as any,
  logs: { total: 0, items: [] as any[] },
}));

vi.mock("../lib/api", () => ({
  cancelJob: vi.fn(),
  listJobs: vi.fn(async () => apiState.jobs),
  getJobProgress: vi.fn(async () => apiState.progress),
  listJobLogs: vi.fn(async () => apiState.logs),
}));

/** 整理 `set failure scenario` 前端辅助逻辑，保持数据转换和展示口径一致。 */
function setFailureScenario() {
  apiState.jobs = {
    total: 1,
    items: [
      {
        id: "job-failed",
        job_type: "run_vlr",
        status: "failed",
        created_at: "2026-05-02T19:55:34+00:00",
        started_at: "2026-05-02T19:55:34+00:00",
        finished_at: "2026-05-02T20:34:47+00:00",
        config_path: "configs/bench/bootstrap_full_vlr_advedm_plus_cuda.yaml",
        override_json: JSON.stringify({
          dataset: { max_items: 500 },
          plugins: { attack: "advedm_plus", model_adapter: "clip_hf" },
          runner: {
            max_samples: 500,
            max_pairs: 250000,
            victim_model_adapters: ["openai_qwen3_vl"],
          },
        }),
        run_id: "",
        error_code: "job_failed",
        error_message: failureMessage,
        benchmark_mode: false,
      },
    ],
  };
  apiState.progress = {
    job_id: "job-failed",
    job_type: "run_vlr",
    status: "failed",
    queue_position: 0,
    elapsed_seconds: 2353,
    eta_seconds: 0,
    estimated_ready_at: "",
    current_stage: "victim_evaluation",
    progress_percent: 100,
    current_stage_message: staleProgressMessage,
    current_stage_units_done: 2048,
    current_stage_units_total: 250000,
    current_stage_progress_percent: 0.82,
    current_stage_updated_at: "2026-05-02T20:34:35+00:00",
    last_log: `job failed: ${failureMessage}`,
    run_id: "",
    stages: [
      { stage_key: "queued", stage_label: "排队中", state: "success", progress_percent: 10, message: "等待结束，任务开始执行。", updated_at: "" },
      { stage_key: "model_preflight", stage_label: "模型预检查", state: "success", progress_percent: 16, message: "模型预检查完成。", updated_at: "" },
      { stage_key: "config_validation", stage_label: "配置校验", state: "success", progress_percent: 26, message: "配置校验完成。", updated_at: "" },
      { stage_key: "dataset_loading", stage_label: "数据集装载", state: "success", progress_percent: 38, message: "数据集装载完成，共纳入 500 条样本。", updated_at: "" },
      { stage_key: "attack_execution", stage_label: "执行攻击", state: "pending", progress_percent: 0, message: "", updated_at: "" },
      { stage_key: "victim_evaluation", stage_label: "受测模型评测", state: "running", progress_percent: 46, message: staleProgressMessage, updated_at: "" },
      { stage_key: "result_aggregation", stage_label: "结果汇总", state: "pending", progress_percent: 0, message: "", updated_at: "" },
      { stage_key: "report_writing", stage_label: "报告写入", state: "pending", progress_percent: 0, message: "", updated_at: "" },
      { stage_key: "completed", stage_label: "完成", state: "failed", progress_percent: 100, message: `任务失败：${failureMessage}`, updated_at: "" },
    ],
  };
  apiState.logs = {
    total: 1,
    items: [{ id: 1, job_id: "job-failed", ts: "2026-05-02T20:34:47+00:00", level: "error", message: `job failed: ${failureMessage}` }],
  };
}

/** 整理 `set 生成式评测 success scenario` 前端辅助逻辑，保持数据转换和展示口径一致。 */
function setGenerationSuccessScenario() {
  apiState.jobs = {
    total: 1,
    items: [
      {
        id: "job-caption",
        job_type: "run_caption",
        status: "success",
        created_at: "2026-05-04T00:26:33+08:00",
        started_at: "2026-05-04T00:26:33+08:00",
        finished_at: "2026-05-04T00:26:50+08:00",
        config_path: "configs/bench/bootstrap_quick_caption.yaml",
        override_json: JSON.stringify({
          dataset: { max_items: 1 },
          plugins: { attack: "advedm_plus" },
          runner: { victim_model_adapter: "openai_qwen35_9b" },
          extra: { ui_task_name: "真实图像描述测评" },
        }),
        run_id: "20260504_002634_657073",
        error_code: "",
        error_message: "",
        benchmark_mode: false,
      },
    ],
  };
  apiState.progress = {
    job_id: "job-caption",
    job_type: "run_caption",
    status: "success",
    queue_position: 0,
    elapsed_seconds: 17,
    eta_seconds: 0,
    estimated_ready_at: "",
    current_stage: "report_writing",
    progress_percent: 100,
    current_stage_message: "正在写入报告。",
    current_stage_units_done: 0,
    current_stage_units_total: 0,
    current_stage_progress_percent: 0,
    current_stage_updated_at: "2026-05-04T00:26:50+08:00",
    last_log: "job finished",
    run_id: "20260504_002634_657073",
    stages: [
      { stage_key: "queued", stage_label: "排队中", state: "success", progress_percent: 10, message: "等待结束，任务开始执行。", updated_at: "" },
      { stage_key: "model_preflight", stage_label: "模型预检查", state: "success", progress_percent: 16, message: "生成模型与代理模型预检查完成。", updated_at: "" },
      { stage_key: "config_validation", stage_label: "配置校验", state: "success", progress_percent: 26, message: "生成式评测配置校验完成。", updated_at: "" },
      { stage_key: "dataset_loading", stage_label: "图像描述样本装载", state: "success", progress_percent: 38, message: "正在读取生成式评测样本：data/coco2014/generation/coco_caption_object_val.jsonl", updated_at: "" },
      { stage_key: "attack_execution", stage_label: "执行攻击与描述生成", state: "success", progress_percent: 83, message: "已完成 1 / 1 条生成式样本", updated_at: "" },
      { stage_key: "result_aggregation", stage_label: "结果汇总", state: "success", progress_percent: 90, message: "正在汇总生成式评测结果。", updated_at: "" },
      { stage_key: "report_writing", stage_label: "报告写入", state: "running", progress_percent: 97, message: "正在写入报告。", updated_at: "" },
      { stage_key: "completed", stage_label: "完成", state: "success", progress_percent: 100, message: "任务执行完成，运行编号：20260504_002634_657073", updated_at: "" },
    ],
  };
  apiState.logs = {
    total: 1,
    items: [{ id: 1, job_id: "job-caption", ts: "2026-05-04T00:26:50+08:00", level: "info", message: "job finished" }],
  };
}

/** 整理 `set 样本 生成式评测 only scenario` 前端辅助逻辑，保持数据转换和展示口径一致。 */
function setSampleGenerationOnlyScenario() {
  apiState.jobs = {
    total: 1,
    items: [
      {
        id: "job-assets-only",
        job_type: "generate_sample_assets",
        status: "success",
        created_at: "2026-05-07T15:36:10+08:00",
        started_at: "2026-05-07T15:36:10+08:00",
        finished_at: "2026-05-07T15:36:16+08:00",
        config_path: "configs/bench/bootstrap_full_vlr_fgsm_cuda.yaml",
        override_json: JSON.stringify({
          dataset: { max_items: 1 },
          plugins: { attack: "fgsm", model_adapter: "clip_hf" },
          runner: {
            surrogate_model_adapter: "clip_hf",
            victim_model_adapters: [],
          },
          extra: {
            ui_task_name: "对抗样本集生成",
            workflow_type: "sample_generation_only",
          },
        }),
        run_id: "20260507_153610_133285",
        error_code: "",
        error_message: "",
        benchmark_mode: false,
      },
    ],
  };
  apiState.progress = {
    job_id: "job-assets-only",
    job_type: "generate_sample_assets",
    status: "success",
    queue_position: 0,
    elapsed_seconds: 6,
    eta_seconds: 0,
    estimated_ready_at: "",
    current_stage: "completed",
    progress_percent: 100,
    current_stage_message: "sample generation success: 运行编号=20260507_153610_133285 assets=1",
    current_stage_units_done: 1,
    current_stage_units_total: 1,
    current_stage_progress_percent: 100,
    current_stage_updated_at: "2026-05-07T15:36:16+08:00",
    last_log: "任务已完成。",
    run_id: "20260507_153610_133285",
    stages: [
      { stage_key: "queued", stage_label: "排队中", state: "success", progress_percent: 10, message: "等待结束，任务开始执行。", updated_at: "" },
      { stage_key: "model_preflight", stage_label: "模型预检查", state: "success", progress_percent: 18, message: "攻击生成依赖预检查完成。", updated_at: "" },
      { stage_key: "config_validation", stage_label: "配置校验", state: "success", progress_percent: 30, message: "配置校验完成。", updated_at: "" },
      { stage_key: "dataset_loading", stage_label: "数据集装载", state: "success", progress_percent: 45, message: "数据集装载完成，共纳入 1 条样本。", updated_at: "" },
      { stage_key: "attack_execution", stage_label: "执行攻击", state: "success", progress_percent: 80, message: "已生成 1 / 1 条待测评样本。", updated_at: "" },
      { stage_key: "result_aggregation", stage_label: "样本入库", state: "success", progress_percent: 95, message: "已写入 1 条待测评样本资产。", updated_at: "" },
      { stage_key: "completed", stage_label: "完成", state: "success", progress_percent: 100, message: "任务执行完成，运行编号：20260507_153610_133285", updated_at: "" },
    ],
  };
  apiState.logs = {
    total: 1,
    items: [{ id: 1, job_id: "job-assets-only", ts: "2026-05-07T15:36:16+08:00", level: "info", message: "sample generation success: 运行编号=20260507_153610_133285 assets=1" }],
  };
}

/** 构建 `create client` 结构，供页面渲染或测试断言复用。 */
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

/** 整理 `行记录 所属` 前端辅助逻辑，保持数据转换和展示口径一致。 */
function rowFor(label: string): HTMLElement {
  const rows = screen.getAllByRole("row");
  const row = rows.find((item) => within(item).queryByText(label));
  if (!row) throw new Error(`Row not found: ${label}`);
  return row;
}

describe("JobCenterPage display", () => {
  beforeEach(() => {
    cleanup();
    setFailureScenario();
  });

  it("shows the real failure reason and only marks the failed phase red", async () => {
    render(
      <QueryClientProvider client={createClient()}>
        <JobCenterPage />
      </QueryClientProvider>,
    );

    expect((await screen.findAllByText(failureMessage)).length).toBeGreaterThan(0);
    expect(within(rowFor("自动测评")).getByText("失败")).toBeInTheDocument();
    expect(within(rowFor("报告生成")).getByText("等待中")).toBeInTheDocument();
  });

  it("maps generation jobs to the real generation stage instead of showing auto evaluation as 0 percent", async () => {
    setGenerationSuccessScenario();

    render(
      <QueryClientProvider client={createClient()}>
        <JobCenterPage />
      </QueryClientProvider>,
    );

    await screen.findByText("图像描述测评");
    expect(screen.getByText("已完成")).toBeInTheDocument();
    expect(within(rowFor("数据准备")).getByText("成功")).toBeInTheDocument();
    expect(within(rowFor("数据准备")).getByText("100%")).toBeInTheDocument();
    expect(within(rowFor("样本生成")).getByText("成功")).toBeInTheDocument();
    expect(within(rowFor("样本生成")).getByText("100%")).toBeInTheDocument();
    expect(within(rowFor("自动测评")).getByText("成功")).toBeInTheDocument();
    expect(within(rowFor("自动测评")).getByText("100%")).toBeInTheDocument();
    expect(screen.getByText("分析输出差异")).toBeInTheDocument();
    expect(within(rowFor("自动测评")).getByText("分析模型在正常输入与对抗输入下的输出差异。")).toBeInTheDocument();
    expect(within(rowFor("指标统计")).getByText("成功")).toBeInTheDocument();
    expect(within(rowFor("指标统计")).getByText("100%")).toBeInTheDocument();
    expect(within(rowFor("报告生成")).getByText("成功")).toBeInTheDocument();
    expect(within(rowFor("报告生成")).getByText("100%")).toBeInTheDocument();
  });

  it("shows no-check sample generation as pending assets instead of completed evaluation metrics", async () => {
    setSampleGenerationOnlyScenario();

    render(
      <QueryClientProvider client={createClient()}>
        <JobCenterPage />
      </QueryClientProvider>,
    );

    await screen.findByText("待测评样本生成任务");
    expect(screen.getByText("样本生成流程进度")).toBeInTheDocument();
    expect(screen.getByText("不校验")).toBeInTheDocument();
    expect(within(rowFor("数据准备")).getByText("成功")).toBeInTheDocument();
    expect(within(rowFor("样本生成")).getByText("成功")).toBeInTheDocument();
    expect(within(rowFor("样本入库")).getByText("成功")).toBeInTheDocument();
    expect(within(rowFor("样本入库")).getByText("写入待测评样本资产，不计算模型风险。")).toBeInTheDocument();
    expect(within(rowFor("完成")).getByText("生成批次已进入待测评状态，选择受测模型后再生成风险、报告和案例判断。")).toBeInTheDocument();
    expect(screen.queryByText("自动测评")).not.toBeInTheDocument();
    expect(screen.queryByText("指标统计")).not.toBeInTheDocument();
    expect(screen.queryByText("报告生成")).not.toBeInTheDocument();
  });
});
