// 文件说明：该文件属于前端组件，集中实现 StatCard 相关逻辑。
import { useDismissible } from "../hooks/useDismissible";

interface StatCardProps {
  title: React.ReactNode;
  value: React.ReactNode;
  hint?: React.ReactNode;
  dismissKey?: string;
}

/** 渲染 `StatCard` 组件，组织该区域的数据读取、交互状态和可访问性标记。 */
export function StatCard({ title, value, hint, dismissKey }: StatCardProps) {
  const { visible, dismiss, restore } = useDismissible(dismissKey);

  if (dismissKey && !visible) {
    return (
      <section className="metric-card dismissible-panel dismissible-panel-restore">
        <span>统计卡片已隐藏</span>
        <button type="button" className="panel-restore-button" onClick={restore}>
          显示统计卡片
        </button>
      </section>
    );
  }

  return (
    <section className={`metric-card ${dismissKey ? "dismissible-panel" : ""}`}>
      {dismissKey ? (
        <button type="button" className="panel-close-button" aria-label="关闭统计卡片" title="关闭统计卡片" onClick={dismiss}>
          ×
        </button>
      ) : null}
      <div className="metric-title">{title}</div>
      <div className="metric-value">{value}</div>
      {hint ? <div className="metric-hint">{hint}</div> : null}
    </section>
  );
}
