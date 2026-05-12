// 文件说明：该文件属于前端页面，集中实现 AnalysisPage 相关逻辑。
import { RunRecordsView } from "./ReportCenterPage";

/** 渲染 `AnalysisPage` 组件，组织该区域的数据读取、交互状态和可访问性标记。 */
export default function AnalysisPage() {
  return <RunRecordsView mode="analysis" />;
}
