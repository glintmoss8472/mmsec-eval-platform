import type { ReactNode } from "react";

import { useDismissible } from "../hooks/useDismissible";

type DismissiblePanelProps = {
  id: string;
  label: string;
  className?: string;
  children: ReactNode;
  as?: "section" | "div" | "article";
};

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
