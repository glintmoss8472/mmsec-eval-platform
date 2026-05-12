// 文件说明：该文件属于前端组件，集中实现 AppLayout 相关逻辑。
import { useQuery } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { NavLink, Outlet, useLocation } from "react-router-dom";

import { getSystemOverview, health } from "../lib/api";
import { PRIMARY_NAV_ITEMS } from "../appRoutes";
import { clearDismissedPanels } from "../hooks/useDismissible";

const PAGE_SUBTITLES: Record<string, string> = {
  "/": "用于展示系统运行情况、测评任务概况与整体风险态势",
  "/testing": "选择测评对象并调用样本集完成自动化测评",
  "/samples": "生成、筛选、复用和追踪多模态对抗样本集",
  "/jobs": "实时查看测评流程进度、阶段状态与当前运行情况",
  "/analysis": "跨运行统计攻击效果、风险分布、任务覆盖和证据置信度",
  "/cases": "按样本管理和复盘多模态对抗案例证据",
  "/reports": "管理单次运行报告、样本规模、风险结论和复现证据",
  "/glossary": "集中解释页面术语、指标口径和答辩时容易被追问的概念",
};
const FONT_SCALE_STORAGE_KEY = "att-ui-font-scale";
const FONT_SCALE_MIN = 0.9;
const FONT_SCALE_MAX = 1.3;
const FONT_SCALE_STEP = 0.05;

/** 整理 `clamp font 缩放` 前端辅助逻辑，保持数据转换和展示口径一致。 */
function clampFontScale(value: number) {
  if (!Number.isFinite(value)) return 1;
  return Math.min(FONT_SCALE_MAX, Math.max(FONT_SCALE_MIN, Math.round(value * 100) / 100));
}

/** 整理 `read stored font 缩放` 前端辅助逻辑，保持数据转换和展示口径一致。 */
function readStoredFontScale() {
  if (typeof window === "undefined") return 1;
  const stored = window.localStorage.getItem(FONT_SCALE_STORAGE_KEY);
  return clampFontScale(stored ? Number.parseFloat(stored) : 1);
}

/** 整理 `font 缩放 percent` 前端辅助逻辑，保持数据转换和展示口径一致。 */
function fontScalePercent(value: number) {
  return `${Math.round(value * 100)}%`;
}

/** 整理 `nav icon` 前端辅助逻辑，保持数据转换和展示口径一致。 */
function navIcon(code: string) {
  if (code === "home") {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M3 10.5 12 3l9 7.5" />
        <path d="M5 9.5V21h5v-6h4v6h5V9.5" />
      </svg>
    );
  }
  if (code === "plus") {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <circle cx="12" cy="12" r="9" />
        <path d="M12 8v8M8 12h8" />
      </svg>
    );
  }
  if (code === "check") {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <rect x="4" y="5" width="16" height="14" rx="2" />
        <path d="m8 13 2.5 2.5L16 10" />
      </svg>
    );
  }
  if (code === "database") {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <ellipse cx="12" cy="5" rx="7" ry="3" />
        <path d="M5 5v6c0 1.7 3.1 3 7 3s7-1.3 7-3V5" />
        <path d="M5 11v6c0 1.7 3.1 3 7 3s7-1.3 7-3v-6" />
      </svg>
    );
  }
  if (code === "chart") {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M4 19V5" />
        <path d="M4 19h16" />
        <path d="M8 16V9M12 16v-5M16 16V7" />
      </svg>
    );
  }
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M6 4h9l3 3v13H6z" />
      <path d="M9 9h6M9 13h6M9 17h4" />
    </svg>
  );
}

/** 整理 `page meta` 前端辅助逻辑，保持数据转换和展示口径一致。 */
function pageMeta(pathname: string) {
  if (pathname.startsWith("/reports/") && pathname.includes("/cases/")) {
    const cases = PRIMARY_NAV_ITEMS.find((item) => item.to === "/cases") ?? PRIMARY_NAV_ITEMS[0];
    return { ...cases, label: "案例复盘" };
  }
  if (pathname === "/reports") {
    return PRIMARY_NAV_ITEMS.find((item) => item.to === "/reports") ?? PRIMARY_NAV_ITEMS[0];
  }
  if (pathname.startsWith("/reports/")) {
    const reports = PRIMARY_NAV_ITEMS.find((item) => item.to === "/reports") ?? PRIMARY_NAV_ITEMS[0];
    return { ...reports, to: "/reports", label: "报告详情" };
  }
  if (pathname.startsWith("/glossary")) {
    return { to: "/glossary", label: "术语与指标详解", code: "book", entryId: "page-glossary" };
  }
  return PRIMARY_NAV_ITEMS.find((item) => (item.to === "/" ? pathname === "/" : pathname.startsWith(item.to))) ?? PRIMARY_NAV_ITEMS[0];
}

/** 渲染 `AppLayout` 组件，组织该区域的数据读取、交互状态和可访问性标记。 */
export function AppLayout() {
  const location = useLocation();
  const [noticeOpen, setNoticeOpen] = useState(false);
  const [accountOpen, setAccountOpen] = useState(false);
  const [fontScale, setFontScale] = useState(readStoredFontScale);
  const current = pageMeta(location.pathname);
  const healthQ = useQuery({
    queryKey: ["layout-health"],
    queryFn: health,
    retry: false,
    refetchInterval: 10000,
  });
  const overviewQ = useQuery({
    queryKey: ["layout-overview"],
    queryFn: getSystemOverview,
    retry: false,
    staleTime: 60000,
    refetchInterval: 60000,
  });

  const serviceReady = healthQ.data?.status === "ok" || healthQ.data?.bootstrap_state === "ready";
  const serviceText = healthQ.isError ? "系统连接异常" : serviceReady ? "系统运行正常" : "系统正在预热";
  const serviceDot = healthQ.isError ? "danger" : serviceReady ? "success" : "warn";
  const subtitle = PAGE_SUBTITLES[current.to] ?? PAGE_SUBTITLES["/"];
  const generatedAt = overviewQ.data?.generated_at ? new Date(overviewQ.data.generated_at).toLocaleString("zh-CN") : serviceReady ? "后端已连接" : "等待后端同步";
  /** 整理 `set and store font 缩放` 前端辅助逻辑，保持数据转换和展示口径一致。 */
  const setAndStoreFontScale = (value: number) => {
    setFontScale(clampFontScale(value));
  };

  useEffect(() => {
    const normalized = clampFontScale(fontScale);
    document.documentElement.style.setProperty("--app-font-scale", String(normalized));
    window.localStorage.setItem(FONT_SCALE_STORAGE_KEY, String(normalized));
  }, [fontScale]);

  return (
    <div className="gov-shell">
      <aside className="gov-sidebar">
        <div className="gov-brand">
          <div className="gov-brand-icon">
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <path d="M12 3 4 6.7v5.5c0 4.8 3.2 7.6 8 8.8 4.8-1.2 8-4 8-8.8V6.7z" />
              <path d="m8.5 12 2.2 2.2L15.8 9" />
            </svg>
          </div>
          <div>
            <div className="gov-brand-title">多模态大模型</div>
            <div className="gov-brand-title">对抗样本安全测评工具</div>
          </div>
        </div>

        <nav className="gov-nav" aria-label="主导航">
          {PRIMARY_NAV_ITEMS.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === "/"}
              className={({ isActive }) => {
                const reportActive = item.to === "/reports" && location.pathname.startsWith("/reports") && !location.pathname.includes("/cases/");
                const caseActive = item.to === "/cases" && location.pathname.startsWith("/reports/") && location.pathname.includes("/cases/");
                return `gov-nav-item ${isActive || reportActive || caseActive ? "active" : ""}`;
              }}
            >
              <span className="gov-nav-icon">{navIcon(item.code)}</span>
              <span>{item.label}</span>
            </NavLink>
          ))}
        </nav>

        <div className="gov-system-card">
          <div className="gov-system-icon">
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <path d="M12 3 5 6v5c0 4.4 2.8 7.1 7 8 4.2-.9 7-3.6 7-8V6z" />
              <path d="m8.8 12 2.1 2.1 4.5-5" />
            </svg>
          </div>
          <div className="gov-system-title">{serviceText}</div>
          <div className="gov-system-sub">
            <span className={`gov-dot ${serviceDot}`} />
            <span>本地服务运行中</span>
          </div>
          <div className="gov-system-time">{generatedAt}</div>
        </div>
      </aside>

      <div className="gov-workspace">
        <header className="gov-topbar">
          <div>
            <h1>{current.label}</h1>
            <p>{subtitle}</p>
          </div>
          <div className="gov-top-actions">
            <div className="gov-notice-wrap">
              <button
                className="gov-bell"
                type="button"
                aria-label="通知"
                aria-expanded={noticeOpen}
                onClick={() => {
                  setNoticeOpen((open) => !open);
                  setAccountOpen(false);
                }}
              >
                <svg viewBox="0 0 24 24" aria-hidden="true">
                  <path d="M18 8a6 6 0 0 0-12 0c0 7-3 7-3 9h18c0-2-3-2-3-9" />
                  <path d="M10 21h4" />
                </svg>
                <span />
              </button>
              {noticeOpen ? (
                <div className="gov-notice-popover" role="status">
                  <strong>系统通知</strong>
                  <p>{serviceText}，后端状态：{healthQ.data?.status || (healthQ.isError ? "异常" : "同步中")}。</p>
                  <p>最近同步时间：{generatedAt}</p>
                </div>
              ) : null}
            </div>
            <div className="gov-user-wrap">
              <button
                className="gov-user"
                type="button"
                aria-label="用户菜单"
                aria-expanded={accountOpen}
                onClick={() => {
                  setAccountOpen((open) => !open);
                  setNoticeOpen(false);
                }}
              >
                <span className="gov-avatar">
                  <svg viewBox="0 0 24 24" aria-hidden="true">
                    <circle cx="12" cy="8" r="4" />
                    <path d="M4.5 21a7.5 7.5 0 0 1 15 0" />
                  </svg>
                </span>
                <span className="gov-user-text">
                  <strong>答辩用户</strong>
                  <span>管理员</span>
                </span>
                <svg className="gov-chevron" viewBox="0 0 24 24" aria-hidden="true">
                  <path d="m7 10 5 5 5-5" />
                </svg>
              </button>
              {accountOpen ? (
                <div className="gov-user-popover" role="dialog" aria-label="用户设置">
                  <strong>账户信息</strong>
                  <p>当前身份：答辩用户 / 管理员</p>
                  <p>页面权限：本地测评与报告查看</p>
                  <div className="gov-font-size-panel">
                    <div className="gov-font-size-head">
                      <span>显示字号</span>
                      <strong>{fontScalePercent(fontScale)}</strong>
                    </div>
                    <div className="gov-font-size-control">
                      <button
                        type="button"
                        aria-label="减小字号"
                        onClick={() => setAndStoreFontScale(fontScale - FONT_SCALE_STEP)}
                      >
                        A-
                      </button>
                      <input
                        aria-label="字号大小"
                        type="range"
                        min={Math.round(FONT_SCALE_MIN * 100)}
                        max={Math.round(FONT_SCALE_MAX * 100)}
                        step={5}
                        value={Math.round(fontScale * 100)}
                        onChange={(event) => setAndStoreFontScale(Number(event.currentTarget.value) / 100)}
                      />
                      <button
                        type="button"
                        aria-label="增大字号"
                        onClick={() => setAndStoreFontScale(fontScale + FONT_SCALE_STEP)}
                      >
                        A+
                      </button>
                    </div>
                    <button type="button" className="gov-popover-action secondary" onClick={() => setAndStoreFontScale(1)}>
                      恢复标准字号
                    </button>
                  </div>
                  <button type="button" className="gov-popover-action" onClick={clearDismissedPanels}>
                    恢复隐藏面板
                  </button>
                </div>
              ) : null}
            </div>
          </div>
        </header>

        <main className="gov-page">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
