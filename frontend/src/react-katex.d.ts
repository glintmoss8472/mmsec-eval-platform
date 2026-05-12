// 文件说明：该文件属于前端工程配置，集中实现 react katex.d 相关逻辑。
declare module "react-katex" {
  import type { ComponentType, ReactNode } from "react";

  export const BlockMath: ComponentType<{ math: string; errorColor?: string; renderError?: (error: Error) => ReactNode }>;
  export const InlineMath: ComponentType<{ math: string; errorColor?: string; renderError?: (error: Error) => ReactNode }>;
}
