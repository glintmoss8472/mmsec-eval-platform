export const PRIMARY_NAV_ITEMS = [
  { to: "/", label: "首页总览", code: "home", entryId: "page-dashboard" },
  { to: "/testing", label: "新建测评", code: "plus", entryId: "page-testing" },
  { to: "/samples", label: "对抗样本库", code: "database", entryId: "page-samples" },
  { to: "/jobs", label: "任务监控", code: "check", entryId: "page-jobs" },
  { to: "/analysis", label: "结果分析", code: "chart", entryId: "page-analysis" },
  { to: "/cases", label: "案例库", code: "book", entryId: "page-cases" },
  { to: "/reports", label: "报告中心", code: "report", entryId: "page-reports" },
] as const;

export const LEGACY_TESTING_REDIRECTS = [
  "/experiments",
  "/experiments/compare",
] as const;
