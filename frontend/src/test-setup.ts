// 文件说明：该文件属于前端工程配置，集中实现 test setup 相关逻辑。
import "@testing-library/jest-dom/vitest";
import { vi } from "vitest";

vi.mock("echarts-for-react", () => ({
  default: () => null,
}));
