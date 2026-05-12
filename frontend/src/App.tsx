// 文件说明：该文件属于前端工程配置，集中实现 App 相关逻辑。
import { lazy, Suspense } from "react";
import { Navigate, Route, Routes } from "react-router-dom";

import { AppLayout } from "./components/AppLayout";
import { LEGACY_TESTING_REDIRECTS } from "./appRoutes";

const DashboardPage = lazy(() => import("./pages/DashboardPage"));
const ExperimentStudioPage = lazy(() => import("./pages/ExperimentStudioPage"));
const GlossaryPage = lazy(() => import("./pages/GlossaryPage"));
const JobCenterPage = lazy(() => import("./pages/JobCenterPage"));
const SampleLibraryPage = lazy(() => import("./pages/SampleLibraryPage"));
const AnalysisPage = lazy(() => import("./pages/AnalysisPage"));
const ReportsPage = lazy(() => import("./pages/ReportsPage"));
const CaseReviewPage = lazy(() => import("./pages/CaseReviewPage"));
const ReportDetailPage = lazy(() => import("./pages/ReportDetailPage"));
const CaseReplayPage = lazy(() => import("./pages/CaseReplayPage"));

/** 渲染 `PageFallback` 组件，组织该区域的数据读取、交互状态和可访问性标记。 */
function PageFallback() {
  return <div className="gov-empty-state">正在加载页面。</div>;
}

/** 渲染 `App` 组件，组织该区域的数据读取、交互状态和可访问性标记。 */
export default function App() {
  return (
    <Suspense fallback={<PageFallback />}>
      <Routes>
        <Route element={<AppLayout />}>
          <Route path="/" element={<DashboardPage />} />
          <Route path="/testing" element={<ExperimentStudioPage />} />
          <Route path="/samples" element={<SampleLibraryPage />} />
          <Route path="/jobs" element={<JobCenterPage />} />
          <Route path="/analysis" element={<AnalysisPage />} />
          <Route path="/results" element={<Navigate to="/analysis" replace />} />
          <Route path="/cases" element={<CaseReviewPage />} />
          <Route path="/reports" element={<ReportsPage />} />
          <Route path="/reports/:runId" element={<ReportDetailPage />} />
          <Route path="/reports/:runId/cases/:sampleId" element={<CaseReplayPage />} />
          <Route path="/glossary" element={<GlossaryPage />} />
          {LEGACY_TESTING_REDIRECTS.map((path) => (
            <Route key={path} path={path} element={<Navigate to="/testing" replace />} />
          ))}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </Suspense>
  );
}
