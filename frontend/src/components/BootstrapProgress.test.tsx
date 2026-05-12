import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { BootstrapProgress } from "./BootstrapProgress";

vi.mock("../lib/api", () => {
  const status = {
    state: "warming",
    started_at: "2026-02-13T00:00:00Z",
    updated_at: "2026-02-13T00:00:10Z",
    degraded_reason: "",
    steps: [
      { name: "seed_sync", state: "success", message: "", updated_at: "2026-02-13T00:00:01Z" },
      { name: "queue_benchmark_demo", state: "running", message: "", updated_at: "2026-02-13T00:00:08Z" },
    ],
    artifacts: {
      docs_index: "artifacts/docs_index.json",
      docs_snippets: "artifacts/docs_snippets.jsonl",
      seeded_runs: [],
      seeded_data: ["mini_flickr"],
    },
  };
  return {
    getBootstrapStatus: vi.fn(async () => status),
    getBootstrapLogs: vi.fn(async () => ({ items: [{ ts: "2026-02-13T00:00:09Z", level: "info", message: "bootstrap started" }] })),
    retryBootstrap: vi.fn(async () => status),
  };
});

describe("BootstrapProgress", () => {
  it("renders progress and logs", async () => {
    const qc = new QueryClient();
    render(
      <QueryClientProvider client={qc}>
        <BootstrapProgress />
      </QueryClientProvider>
    );

    expect(await screen.findByText("开箱预热进度")).toBeTruthy();
    expect(await screen.findByText("种子同步")).toBeTruthy();
    expect(await screen.findByText("预热日志")).toBeTruthy();
    expect(await screen.findByText(/bootstrap started/)).toBeTruthy();
  });
});
