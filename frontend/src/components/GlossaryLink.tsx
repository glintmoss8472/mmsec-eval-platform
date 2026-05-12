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

/** 中文注释：实现 GlossaryLink 的核心流程，支撑前端组件中的业务语义和异常边界。 */
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
