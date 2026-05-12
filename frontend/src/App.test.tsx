// 文件说明：该文件属于前端工程配置，集中实现 App.test 相关逻辑。
import { describe, expect, it } from "vitest";

import { LEGACY_TESTING_REDIRECTS, PRIMARY_NAV_ITEMS } from "./appRoutes";

describe("app route contract", () => {
  it("keeps the primary navigation entries", () => {
    expect(PRIMARY_NAV_ITEMS).toHaveLength(7);
    expect(PRIMARY_NAV_ITEMS.map((item) => item.to)).toEqual(["/", "/testing", "/samples", "/jobs", "/analysis", "/cases", "/reports"]);
  });

  it("redirects retired routes back to the testing page", () => {
    expect(LEGACY_TESTING_REDIRECTS).toEqual([
      "/experiments",
      "/experiments/compare",
    ]);
  });
});
