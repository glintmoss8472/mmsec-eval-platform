import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";

import { createJob } from "../lib/api";
import ExperimentStudioPage from "./ExperimentStudioPage";

afterEach(() => {
  cleanup();
  window.localStorage.clear();
  vi.mocked(createJob).mockReset();
});

const modelAdapters = [
  "fixture_vlm",
  "clip_hf",
  "blip_itm",
  "vilt_itm",
  "openai_qwen35_9b",
  "openai_qwen3_vl",
  "openai_qwen25_vl",
  "openai_internvl35",
  "openai_minicpm_v",
  "openai_ovis25",
  "openai_gemma3_12b",
];

const readyRequirement = (label: string, required = true) => ({
  label,
  required,
  configured: true,
  exists: true,
  status: "ready",
  path: `/mock/${label}`,
  note: "已配置并存在",
});

const notRequiredRequirement = (label: string) => ({
  label,
  required: false,
  configured: false,
  exists: false,
  status: "not_required",
  path: "",
  note: "该方法不需要此项",
});

const externalStatus = Object.fromEntries(
  ["vqa_visual_corruption", "xtransfer_uap", "foa_attack", "anyattack", "mpc_attack", "m_attack"].map((attack) => [
    attack,
    {
      attack_id: attack,
      display_name: attack,
      config_path: `configs/bench/bootstrap_standard_${attack}_cuda.yaml`,
      config_exists: true,
      command_template_configured: true,
      runnable: true,
      repo: attack === "xtransfer_uap" ? readyRequirement("官方仓库", false) : readyRequirement("官方仓库"),
      checkpoint: attack === "xtransfer_uap" || attack === "anyattack" ? readyRequirement("权重/UAP") : notRequiredRequirement("权重/UAP"),
      target: attack === "foa_attack" || attack === "anyattack" || attack === "mpc_attack" || attack === "m_attack" ? readyRequirement("目标图/目标文本") : notRequiredRequirement("目标图/目标文本"),
      messages: [],
    },
  ]),
);

vi.mock("../lib/api", () => ({
  getSystemOverview: vi.fn(async () => ({
    models: modelAdapters.map((adapter) => ({
      adapter,
      display_name: adapter,
      family: adapter === "clip_hf" || adapter === "blip_itm" || adapter === "vilt_itm" ? "经典检索模型" : "视觉语言模型",
      launch_mode: "本地或自托管",
      health_status: adapter.startsWith("openai_") && adapter !== "openai_gemma3_12b" ? "launchable" : "ready",
      endpoint_or_source: adapter,
      model_name: adapter,
      role: adapter === "clip_hf" ? "surrogate/local" : adapter === "fixture_vlm" ? "victim/demo" : "victim/local",
      formal_eval: adapter !== "fixture_vlm",
      task_capabilities:
        adapter === "fixture_vlm"
          ? []
          : adapter === "clip_hf" || adapter === "blip_itm" || adapter === "vilt_itm"
            ? ["vlr"]
            : ["vlr", "vqa", "caption"],
    })),
    datasets: [
      { key: "coco_subset", name: "COCO val2017 完整验证子集", ready: true, item_count: 5000, tier: "benchmark" },
      { key: "flickr1k", name: "Flickr1k", ready: true, item_count: 1000, tier: "benchmark" },
      { key: "mini_flickr", name: "Mini Flickr 演示集", ready: true, item_count: 4, tier: "demo" },
      { key: "vqa_v2_coco_val", name: "VQA v2 COCO 验证真实子集", ready: true, item_count: 300, tier: "generation" },
      { key: "coco_object_probe_val", name: "COCO 对象存在性探测真实子集", ready: true, item_count: 200, tier: "generation" },
      { key: "coco_caption_object_val", name: "COCO 图像描述对象级真实子集", ready: true, item_count: 100, tier: "generation" },
    ],
    attacks: [
      "advclip",
      "advedm",
      "advedm_plus",
      "vqa_visual_corruption",
      "xtransfer_uap",
      "foa_attack",
      "anyattack",
      "mpc_attack",
      "m_attack",
      "bim",
      "cw",
      "difgsm",
      "dtmifgsm",
      "fgsm",
      "mifgsm",
      "nifgsm",
      "pgd",
      "tifgsm",
      "tmm",
      "vmifgsm",
      "vnifgsm",
    ],
    external_attack_status: externalStatus,
  })),
  listJobs: vi.fn(async () => ({ total: 0, items: [] })),
  createJob: vi.fn(),
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

function renderPage() {
  return render(
    <MemoryRouter initialEntries={["/testing"]}>
      <QueryClientProvider client={createClient()}>
        <ExperimentStudioPage />
      </QueryClientProvider>
    </MemoryRouter>,
  );
}

async function findDatasetSelect() {
  await screen.findByRole("heading", { name: "选择数据集" });
  const datasetSelect = document.querySelector('select[name="datasetId"]') as HTMLSelectElement | null;
  expect(datasetSelect).not.toBeNull();
  return datasetSelect as HTMLSelectElement;
}

describe("ExperimentStudioPage model selection", () => {
  it("shows every backend model as a victim while keeping AdvEDM surrogate constrained", async () => {
    renderPage();

    const victimSelect = await screen.findByLabelText(/测评对象/);
    const victimOptions = within(victimSelect).getAllByRole("option");
    expect(victimOptions).toHaveLength(10);
    expect(victimOptions.map((item) => item.getAttribute("value"))).toEqual(modelAdapters.filter((adapter) => adapter !== "fixture_vlm"));

    expect(document.querySelector('select[name="surrogateModel"]')).toBeNull();
    fireEvent.click(await screen.findByRole("button", { name: /生成样本配置/ }));
    const surrogateSelect = document.querySelector('select[name="surrogateModel"]') as HTMLSelectElement | null;
    expect(surrogateSelect).not.toBeNull();
    if (!surrogateSelect) {
      throw new Error("代理模型选择框不存在");
    }
    const surrogateOptions = within(surrogateSelect).getAllByRole("option");
    expect(surrogateOptions.map((item) => item.getAttribute("value"))).toEqual(["clip_hf"]);

    fireEvent.click(await screen.findByRole("button", { name: /选择数据集/ }));
    const datasetSelect = await findDatasetSelect();
    const datasetValues = within(datasetSelect).getAllByRole("option").map((item) => item.getAttribute("value"));
    expect(datasetValues).toEqual(["coco_subset", "flickr1k"]);
    expect(datasetValues).not.toContain("mini_flickr");
    expect(datasetValues).not.toContain("vqa_v2_coco_val");
    expect(datasetValues).not.toContain("coco_object_probe_val");
    expect(datasetValues).not.toContain("coco_caption_object_val");
  });

  it("shows external paper attacks instead of truncating the attack selector", async () => {
    renderPage();

    fireEvent.click(await screen.findByRole("button", { name: /生成样本配置/ }));
    expect(await screen.findByRole("button", { name: /VQA Visual Robustness/ })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /X-Transfer UAP/ })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /FOA-Attack/ })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /AnyAttack/ })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /MPCAttack/ })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /M-Attack/ })).toBeInTheDocument();
  });

  it("hides retrieval-only and fixture models for VQA", async () => {
    renderPage();

    fireEvent.click(await screen.findByRole("button", { name: /视觉问答/ }));
    const victimSelect = await screen.findByLabelText(/测评对象/);
    const victimOptions = within(victimSelect).getAllByRole("option");
    const values = victimOptions.map((item) => item.getAttribute("value"));
    expect(values).not.toContain("fixture_vlm");
    expect(values).not.toContain("clip_hf");
    expect(values).not.toContain("blip_itm");
    expect(values).not.toContain("vilt_itm");
    expect(values).toContain("openai_qwen35_9b");

    fireEvent.click(await screen.findByRole("button", { name: /选择数据集/ }));
    const datasetSelect = await findDatasetSelect();
    expect(await screen.findByLabelText(/样本条数/)).not.toHaveAttribute("max");
    const datasetValues = within(datasetSelect).getAllByRole("option").map((item) => item.getAttribute("value"));
    expect(datasetValues).toEqual(["vqa_v2_coco_val", "coco_object_probe_val"]);
    expect(datasetValues).not.toContain("coco_caption_object_val");
    expect(screen.queryByRole("button", { name: /全部可提交模型/ })).toBeNull();
    expect(screen.queryByText(/全部可提交模型会覆盖/)).toBeNull();
    expect(screen.getByText(/生成式任务按真实样本逐条评测/)).toBeInTheDocument();
  });

  it("uses only the real caption dataset for image description", async () => {
    renderPage();

    fireEvent.click(await screen.findByRole("button", { name: /图像描述/ }));
    fireEvent.click(await screen.findByRole("button", { name: /选择数据集/ }));
    const datasetSelect = await findDatasetSelect();
    expect(datasetSelect).not.toBeDisabled();
    expect(await screen.findByLabelText(/样本条数/)).not.toHaveAttribute("max");
    const datasetValues = within(datasetSelect).getAllByRole("option").map((item) => item.getAttribute("value"));
    expect(datasetValues).toEqual(["coco_caption_object_val"]);
    expect(screen.queryByRole("button", { name: /全部可提交模型/ })).toBeNull();
    expect(screen.queryByText(/全部可提交模型会覆盖/)).toBeNull();
  });

  it("submits the selected object-probe VQA JSONL instead of the default VQA file", async () => {
    vi.mocked(createJob).mockResolvedValue({ id: "job-1", status: "queued" } as never);
    renderPage();

    fireEvent.click(await screen.findByRole("button", { name: /视觉问答/ }));
    fireEvent.click(await screen.findByRole("button", { name: /选择数据集/ }));
    const datasetSelect = await findDatasetSelect();
    fireEvent.change(datasetSelect, { target: { value: "coco_object_probe_val" } });
    fireEvent.click(await screen.findByRole("button", { name: /确认并提交/ }));
    fireEvent.click(await screen.findByRole("button", { name: /开始测评/ }));

    await waitFor(() => expect(createJob).toHaveBeenCalledTimes(1));
    const payload = vi.mocked(createJob).mock.calls[0][0];
    const override = payload.override as Record<string, unknown>;
    expect((override.task as Record<string, unknown>).cases_jsonl).toBe("data/coco2014/generation/coco_object_probe_val.jsonl");
    expect((override.dataset as Record<string, unknown>).benchmark_tag).toBe("coco_object_probe_val_real");
  });

  it("routes external paper attacks to standard configs without a frontend sample cap", async () => {
    vi.mocked(createJob).mockResolvedValue({ id: "job-external", status: "queued" } as never);
    renderPage();

    fireEvent.click(await screen.findByRole("button", { name: /生成样本配置/ }));
    fireEvent.click(await screen.findByRole("button", { name: /FOA-Attack/ }));
    fireEvent.click(await screen.findByRole("button", { name: /选择数据集/ }));
    const sampleInput = await screen.findByLabelText(/样本条数/);
    expect(sampleInput).not.toHaveAttribute("max");
    fireEvent.change(sampleInput, { target: { value: "75" } });
    fireEvent.click(await screen.findByRole("button", { name: /确认并提交/ }));
    fireEvent.click(await screen.findByRole("button", { name: /开始测评/ }));

    await waitFor(() => expect(createJob).toHaveBeenCalledTimes(1));
    const payload = vi.mocked(createJob).mock.calls[0][0];
    expect(payload.config_path).toBe("configs/bench/bootstrap_standard_vlr_foa_attack_cuda.yaml");
    const override = payload.override as Record<string, unknown>;
    expect(((override.runner as Record<string, unknown>).max_samples)).toBe(75);
    expect(((override.report as Record<string, unknown>).top_k_cases)).toBe(75);
  });


  it("submits FGSM with epsilon only, not unused step parameters", async () => {
    vi.mocked(createJob).mockResolvedValue({ id: "job-fgsm", status: "queued" } as never);
    renderPage();

    fireEvent.click(await screen.findByRole("button", { name: /生成样本配置/ }));
    fireEvent.click(await screen.findByRole("button", { name: /^快速梯度符号法（FGSM）/ }));
    fireEvent.click(await screen.findByRole("button", { name: "高级参数" }));
    expect(screen.getAllByLabelText(/图像扰动预算/).length).toBeGreaterThan(0);
    expect(screen.queryByLabelText(/优化步数/)).toBeNull();
    expect(screen.queryByLabelText(/单步步长/)).toBeNull();

    fireEvent.click(await screen.findByRole("button", { name: /确认并提交/ }));
    fireEvent.click(await screen.findByRole("button", { name: /开始测评/ }));

    await waitFor(() => expect(createJob).toHaveBeenCalledTimes(1));
    const payload = vi.mocked(createJob).mock.calls[0][0];
    const override = payload.override as Record<string, unknown>;
    const attackOverride = override.attack as Record<string, unknown>;
    expect(payload.config_path).toBe("configs/bench/bootstrap_full_vlr_cuda.yaml");
    expect(attackOverride.epsilon).toBeGreaterThan(0);
    expect(attackOverride).not.toHaveProperty("step_size");
    expect(attackOverride).not.toHaveProperty("steps");
  });

  it("submits VQA Visual Robustness advanced corruption parameters", async () => {
    vi.mocked(createJob).mockResolvedValue({ id: "job-advanced", status: "queued" } as never);
    renderPage();

    fireEvent.click(await screen.findByRole("button", { name: /视觉问答/ }));
    fireEvent.click(await screen.findByRole("button", { name: /生成样本配置/ }));
    fireEvent.click(await screen.findByRole("button", { name: /VQA Visual Robustness/ }));
    fireEvent.click(await screen.findByRole("button", { name: "高级参数" }));
    fireEvent.change(await screen.findByLabelText(/退化类型/), { target: { value: "jpeg_compression" } });
    fireEvent.change(await screen.findByLabelText(/严重度/), { target: { value: "4" } });
    fireEvent.change(await screen.findByLabelText(/随机种子/), { target: { value: "123" } });
    fireEvent.click(await screen.findByRole("button", { name: /确认并提交/ }));
    fireEvent.click(await screen.findByRole("button", { name: /开始测评/ }));

    await waitFor(() => expect(createJob).toHaveBeenCalledTimes(1));
    const payload = vi.mocked(createJob).mock.calls[0][0];
    const override = payload.override as Record<string, unknown>;
    const attackOverride = override.attack as Record<string, unknown>;
    const pluginsOverride = override.plugins as Record<string, unknown>;
    const runnerOverride = override.runner as Record<string, unknown>;
    expect(payload.job_type).toBe("run_vqa");
    expect(payload.config_path).toBe("configs/bench/bootstrap_standard_vqa_visual_corruption_cuda.yaml");
    expect(pluginsOverride.model_adapter).toBe("openai_qwen35_9b");
    expect(runnerOverride.surrogate_model_adapter).toBe("clip_hf");
    expect(runnerOverride.victim_model_adapters).toEqual(["openai_qwen35_9b"]);
    expect(attackOverride.corruption_type).toBe("jpeg_compression");
    expect(attackOverride.severity).toBe(4);
    expect(attackOverride.corruption_seed).toBe(123);
    expect(attackOverride).not.toHaveProperty("epsilon");
    expect(attackOverride).not.toHaveProperty("step_size");
    expect(attackOverride).not.toHaveProperty("steps");
  });

  it("submits external transfer attack advanced target and coordination parameters", async () => {
    vi.mocked(createJob).mockResolvedValue({ id: "job-mpc-advanced", status: "queued" } as never);
    renderPage();

    fireEvent.click(await screen.findByRole("button", { name: /生成样本配置/ }));
    fireEvent.click(await screen.findByRole("button", { name: /MPCAttack/ }));
    fireEvent.click(await screen.findByRole("button", { name: "高级参数" }));
    fireEvent.change(await screen.findByLabelText(/目标文本/), { target: { value: "a clean target caption" } });
    fireEvent.change(await screen.findByLabelText(/协同权重.*lam/), { target: { value: "0.7" } });
    fireEvent.change(await screen.findByLabelText(/温度系数.*tau/), { target: { value: "0.25" } });
    fireEvent.change(await screen.findByLabelText(/平衡项.*omega/), { target: { value: "2.5" } });
    fireEvent.click(await screen.findByRole("button", { name: /确认并提交/ }));
    fireEvent.click(await screen.findByRole("button", { name: /开始测评/ }));

    await waitFor(() => expect(createJob).toHaveBeenCalledTimes(1));
    const payload = vi.mocked(createJob).mock.calls[0][0];
    const override = payload.override as Record<string, unknown>;
    const attackOverride = override.attack as Record<string, unknown>;
    expect(payload.config_path).toBe("configs/bench/bootstrap_standard_vlr_mpc_attack_cuda.yaml");
    expect(attackOverride.target_text).toBe("a clean target caption");
    expect(attackOverride.clip_backbones).toEqual(["B16", "B32", "Laion"]);
    expect(attackOverride.lam).toBe(0.7);
    expect(attackOverride.tau).toBe(0.25);
    expect(attackOverride.omega).toBe(2.5);
  });

  it("submits built-in paper attack advanced multimodal parameters", async () => {
    vi.mocked(createJob).mockResolvedValue({ id: "job-tmm-advanced", status: "queued" } as never);
    renderPage();

    fireEvent.click(await screen.findByRole("button", { name: /生成样本配置/ }));
    fireEvent.click(await screen.findByRole("button", { name: /TMM/ }));
    fireEvent.click(await screen.findByRole("button", { name: "高级参数" }));
    fireEvent.change(await screen.findByLabelText(/攻击模式/), { target: { value: "B" } });
    fireEvent.change(await screen.findByLabelText(/注意阈值/), { target: { value: "0.65" } });
    fireEvent.change(await screen.findByLabelText(/非关键预算比例/), { target: { value: "0.3" } });
    fireEvent.change(await screen.findByLabelText(/文本替换预算/), { target: { value: "2" } });
    fireEvent.change(await screen.findByLabelText(/候选词数/), { target: { value: "18" } });
    fireEvent.click(await screen.findByRole("button", { name: /确认并提交/ }));
    fireEvent.click(await screen.findByRole("button", { name: /开始测评/ }));

    await waitFor(() => expect(createJob).toHaveBeenCalledTimes(1));
    const payload = vi.mocked(createJob).mock.calls[0][0];
    const override = payload.override as Record<string, unknown>;
    const attackOverride = override.attack as Record<string, unknown>;
    expect(payload.config_path).toBe("configs/bench/bootstrap_full_vlr_tmm_cuda.yaml");
    expect(attackOverride.mode).toBe("B");
    expect(attackOverride.lambda_att).toBe(0.65);
    expect(attackOverride.ratio_r).toBe(0.3);
    expect(attackOverride.eps_t).toBe(2);
    expect(attackOverride.text_candidates_k).toBe(18);
  });

  it("restores saved advanced attack fields without resetting them to defaults", async () => {
    renderPage();

    fireEvent.click(await screen.findByRole("button", { name: /生成样本配置/ }));
    fireEvent.click(await screen.findByRole("button", { name: /TMM/ }));
    fireEvent.click(await screen.findByRole("button", { name: "高级参数" }));
    fireEvent.change(await screen.findByLabelText(/注意阈值/), { target: { value: "0.65" } });
    fireEvent.change(await screen.findByLabelText(/文本替换预算/), { target: { value: "2" } });
    fireEvent.click(await screen.findByRole("button", { name: "保存草稿" }));
    expect(await screen.findByText(/草稿已保存/)).toBeInTheDocument();

    cleanup();
    renderPage();

    fireEvent.click(await screen.findByRole("button", { name: /生成样本配置/ }));
    expect(await screen.findByLabelText(/注意阈值/)).toHaveValue(0.65);
    expect(await screen.findByLabelText(/文本替换预算/)).toHaveValue(2);
  });

  it("uses one contextual side panel instead of repeating preview and recommendation panels", async () => {
    renderPage();

    expect(await screen.findByText("对象状态")).toBeInTheDocument();
    expect(screen.queryByText("本次任务预览")).toBeNull();
    expect(screen.queryByText("推荐说明")).toBeNull();

    fireEvent.click(await screen.findByRole("button", { name: /生成样本配置/ }));
    expect(await screen.findByText("样本生成参数")).toBeInTheDocument();
    expect(screen.queryByText("推荐说明")).toBeNull();
  });

});
