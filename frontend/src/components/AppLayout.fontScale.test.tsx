import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { AppLayout } from "./AppLayout";

vi.mock("../lib/api", () => ({
  health: vi.fn(async () => ({
    status: "ok",
    version: "0.1.0",
    bootstrap_state: "ready",
    degraded_reason: "",
  })),
  getSystemOverview: vi.fn(async () => ({
    generated_at: "2026-05-03T08:00:00+00:00",
  })),
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

function renderLayout() {
  return render(
    <QueryClientProvider client={createClient()}>
      <MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
        <Routes>
          <Route element={<AppLayout />}>
            <Route path="/" element={<div>内容区域</div>} />
          </Route>
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe("AppLayout font scale", () => {
  beforeEach(() => {
    window.localStorage.clear();
    document.documentElement.style.removeProperty("--app-font-scale");
  });

  it("lets the user adjust and persist the global font scale", async () => {
    renderLayout();

    fireEvent.click(await screen.findByRole("button", { name: "用户菜单" }));
    expect(await screen.findByText("显示字号")).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("字号大小"), { target: { value: "120" } });

    await waitFor(() => {
      expect(document.documentElement.style.getPropertyValue("--app-font-scale")).toBe("1.2");
    });
    expect(window.localStorage.getItem("att-ui-font-scale")).toBe("1.2");
    expect(screen.getByText("120%")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "恢复标准字号" }));

    await waitFor(() => {
      expect(document.documentElement.style.getPropertyValue("--app-font-scale")).toBe("1");
    });
    expect(window.localStorage.getItem("att-ui-font-scale")).toBe("1");
  });
});
