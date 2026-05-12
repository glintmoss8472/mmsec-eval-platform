// 文件说明：该文件属于前端组件，集中实现 GlossaryLink 相关逻辑。
import clsx from "clsx";
import { Link } from "react-router-dom";

import { getGlossaryEntry, glossaryHref } from "../lib/glossaryRegistry";

type GlossaryLinkProps = {
  entryId: string;
  children?: React.ReactNode;
  className?: string;
  subtle?: boolean;
};

/** 渲染 `GlossaryLink` 组件，组织该区域的数据读取、交互状态和可访问性标记。 */
export function GlossaryLink({ entryId, children, className, subtle = false }: GlossaryLinkProps) {
  const entry = getGlossaryEntry(entryId);
  if (!entry) {
    return <span className={className}>{children ?? entryId}</span>;
  }

  return (
    <Link to={glossaryHref(entryId)} className={clsx("glossary-link", subtle && "glossary-link-subtle", className)} title={`查看术语详解：${entry.label}`}>
      {children ?? entry.label}
    </Link>
  );
}
