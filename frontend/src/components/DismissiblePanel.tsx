// 文件说明：该文件属于前端组件，集中实现 DismissiblePanel 相关逻辑。
import type { ReactNode } from "react";

import { useDismissible } from "../hooks/useDismissible";

type DismissiblePanelProps = {
  id: string;
  label: string;
  className?: string;
  children: ReactNode;
  as?: "section" | "div" | "article";
};

/** 渲染 `DismissiblePanel` 组件，组织该区域的数据读取、交互状态和可访问性标记。 */
export function DismissiblePanel({
  id,
  label,
  className = "section-card",
  children,
  as = "section",
}: DismissiblePanelProps) {
  const { visible, dismiss, restore } = useDismissible(id);
  const Component = as;

  if (!visible) {
    return (
      <Component className={`${className} dismissible-panel dismissible-panel-restore`} data-panel-id={id}>
        <span>{label}已隐藏</span>
        <button type="button" className="panel-restore-button" onClick={restore}>
          显示{label}
        </button>
      </Component>
    );
  }

  return (
    <Component className={`${className} dismissible-panel`} data-panel-id={id}>
      <button
        type="button"
        className="panel-close-button"
        aria-label={`关闭${label}`}
        title={`关闭${label}`}
        onClick={dismiss}
      >
        ×
      </button>
      {children}
    </Component>
  );
}
