// 文件说明：该文件属于前端工程配置，集中实现 vitest.config 相关逻辑。
import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  test: {
    environment: "jsdom",
    setupFiles: "./src/test-setup.ts",
  },
});
