import "@testing-library/jest-dom/vitest";
import { vi } from "vitest";

vi.mock("echarts-for-react", () => ({
  default: () => null,
}));
