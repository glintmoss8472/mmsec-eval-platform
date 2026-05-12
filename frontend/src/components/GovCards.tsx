// 文件说明：该文件属于前端组件，集中实现 GovCards 相关逻辑。
import type { ReactNode } from "react";

type Tone = "blue" | "green" | "red" | "purple" | "orange";

/** 渲染 `GovIcon` 组件，组织该区域的数据读取、交互状态和可访问性标记。 */
export function GovIcon({ tone = "blue", children }: { tone?: Tone; children: ReactNode }) {
  return <span className={`gov-icon gov-icon-${tone}`}>{children}</span>;
}

/** 渲染 `GovPanel` 组件，组织该区域的数据读取、交互状态和可访问性标记。 */
export function GovPanel({
  title,
  children,
  className = "",
}: {
  title?: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`gov-panel ${className}`}>
      {title ? <h2 className="gov-panel-title">{title}</h2> : null}
      {children}
    </section>
  );
}

/** 渲染 `GovMetric` 组件，组织该区域的数据读取、交互状态和可访问性标记。 */
export function GovMetric({
  title,
  value,
  unit,
  tone = "blue",
  icon,
}: {
  title: string;
  value: string | number;
  unit?: string;
  tone?: Tone;
  icon: ReactNode;
}) {
  const valueText = String(value ?? "");
  const isTextValue = /[A-Za-z\u4e00-\u9fff]/.test(valueText);
  const cjkLength = valueText.match(/[\u4e00-\u9fff]/g)?.length ?? 0;
  const compactValue = valueText.includes("%") || valueText.includes("暂无数据") || valueText.includes("等待") || (isTextValue && (valueText.length >= 8 || cjkLength >= 3));
  const longValue = isTextValue && (valueText.length >= 14 || cjkLength >= 7);
  const veryLongValue = isTextValue && (valueText.length >= 22 || cjkLength >= 11);

  return (
    <GovPanel className="gov-metric">
      <GovIcon tone={tone}>{icon}</GovIcon>
      <div>
        <div className="gov-metric-title">{title}</div>
        <div className={`gov-metric-value text-${tone} ${compactValue ? "compact" : ""} ${longValue ? "long" : ""} ${veryLongValue ? "very-long" : ""}`}>
          {value}
          {unit ? <span>{unit}</span> : null}
        </div>
      </div>
    </GovPanel>
  );
}

/** 渲染 `MiniStatus` 组件，组织该区域的数据读取、交互状态和可访问性标记。 */
export function MiniStatus({ tone = "blue", children }: { tone?: Tone; children: ReactNode }) {
  return <span className={`gov-mini-status gov-mini-${tone}`}>{children}</span>;
}

/** 渲染 `ShieldIcon` 组件，组织该区域的数据读取、交互状态和可访问性标记。 */
export function ShieldIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 3 5 6v5c0 4.4 2.8 7.1 7 8 4.2-.9 7-3.6 7-8V6z" />
      <path d="m8.8 12 2.1 2.1 4.5-5" />
    </svg>
  );
}

/** 渲染 `CubeIcon` 组件，组织该区域的数据读取、交互状态和可访问性标记。 */
export function CubeIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 3 4 7.5v9L12 21l8-4.5v-9z" />
      <path d="M4 7.5 12 12l8-4.5M12 12v9" />
    </svg>
  );
}

/** 渲染 `DatabaseIcon` 组件，组织该区域的数据读取、交互状态和可访问性标记。 */
export function DatabaseIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <ellipse cx="12" cy="5" rx="7" ry="3" />
      <path d="M5 5v6c0 1.7 3.1 3 7 3s7-1.3 7-3V5" />
      <path d="M5 11v6c0 1.7 3.1 3 7 3s7-1.3 7-3v-6" />
    </svg>
  );
}

/** 渲染 `ClipboardIcon` 组件，组织该区域的数据读取、交互状态和可访问性标记。 */
export function ClipboardIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M8 5h8l1 3H7z" />
      <path d="M6 7H5v14h14V7h-1" />
      <path d="M9 13h6M9 17h5" />
    </svg>
  );
}

/** 渲染 `AlertIcon` 组件，组织该区域的数据读取、交互状态和可访问性标记。 */
export function AlertIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 3 2.8 20h18.4z" />
      <path d="M12 9v5M12 17.5v.1" />
    </svg>
  );
}

/** 渲染 `ChartIcon` 组件，组织该区域的数据读取、交互状态和可访问性标记。 */
export function ChartIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M4 19h16M6 16l4-4 3 3 5-7" />
      <path d="M18 8v5h-5" />
    </svg>
  );
}

/** 渲染 `ClockIcon` 组件，组织该区域的数据读取、交互状态和可访问性标记。 */
export function ClockIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <circle cx="12" cy="12" r="8" />
      <path d="M12 7v6l4 2" />
    </svg>
  );
}

/** 渲染 `CheckIcon` 组件，组织该区域的数据读取、交互状态和可访问性标记。 */
export function CheckIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="m5 12 4 4 10-10" />
    </svg>
  );
}
