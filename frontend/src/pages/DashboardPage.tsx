import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";

import { AlertIcon, ChartIcon, ClipboardIcon, CubeIcon, DatabaseIcon, GovMetric, GovPanel } from "../components/GovCards";
import { getRunAnalytics, getSystemOverview, listRuns } from "../lib/api";
import { isDemoRun, isHighRisk, riskBucket, riskText, riskTone } from "../lib/runPresentation";
import { formatAdapterName, formatAttackName, formatRunDatasetName } from "../lib/uiLabels";

function percent(value: number | undefined, fallback = 0) {
  return Math.round(Number.isFinite(Number(value)) ? Number(value) * 100 : fallback);
}

export default function DashboardPage() {
  const overviewQ = useQuery({
    queryKey: ["system-overview"],
    queryFn: getSystemOverview,
    refetchInterval: 8000,
  });
  const analyticsQ = useQuery({
    queryKey: ["dashboard-run-analytics"],
    queryFn: () => getRunAnalytics({ exclude_demo: true }),
    refetchInterval: 8000,
  });
  const runsQ = useQuery({
    queryKey: ["dashboard-runs"],
    queryFn: () => listRuns({ page: 1, page_size: 100, sort_by: "created", sort_dir: "desc", exclude_demo: true }),
    refetchInterval: 8000,
  });
  const overview = overviewQ.data;
  const isOverviewLoading = overviewQ.isLoading && !overview;
  const formalModels = useMemo(() => (overview?.models ?? []).filter((model) => model.formal_eval !== false), [overview?.models]);
  const analytics = analyticsQ.data;
  const allRuns = runsQ.data?.items ?? analytics?.latest_runs ?? overview?.latest_runs ?? [];
  const runs = allRuns.filter((run) => !isDemoRun(run));
  const completedJobs = analytics?.total_runs ?? runsQ.data?.total ?? runs.length;
  const highRiskRuns = analytics?.high_risk_runs ?? runs.filter((item) => isHighRisk(item.risk_level)).length;
  const latest = runs[0];
  const latestRisk = riskText(latest?.risk_level);

  const trend = useMemo(() => {
    if (!runs.length) return [];
    const today = new Date();
    const days = Array.from({ length: 7 }, (_, idx) => {
      const date = new Date(today);
      date.setDate(today.getDate() - (6 - idx));
      return date.toISOString().slice(0, 10);
    });
    const counts = new Map(days.map((day) => [day, 0]));
    for (const run of runs) {
      const raw = run.created_at ? new Date(run.created_at) : undefined;
      if (!raw || Number.isNaN(raw.getTime())) continue;
      const key = raw.toISOString().slice(0, 10);
      if (counts.has(key)) counts.set(key, (counts.get(key) || 0) + 1);
    }
    return days.map((day) => counts.get(day) || 0);
  }, [runs]);

  const maxTrend = Math.max(1, ...trend);
  const riskBuckets = useMemo(() => {
    const source = analytics?.risk_distribution?.length
      ? analytics.risk_distribution.flatMap((item) => Array.from({ length: Number(item.count) || 0 }, () => ({ risk_level: item.key })))
      : runs;
    const low = source.filter((item) => riskBucket(item.risk_level) === "low").length;
    const mid = source.filter((item) => riskBucket(item.risk_level) === "medium").length;
    const high = source.filter((item) => riskBucket(item.risk_level) === "high").length;
    return [
      { label: "低风险", value: low, color: "#2cc58f" },
      { label: "中风险", value: mid, color: "#f6bd35" },
      { label: "高风险", value: high, color: "#ff4b55" },
    ];
  }, [analytics?.risk_distribution, runs]);
  const bucketTotal = riskBuckets.reduce((sum, item) => sum + item.value, 0);
  const lowEnd = bucketTotal ? (riskBuckets[0].value / bucketTotal) * 100 : 100;
  const midEnd = bucketTotal ? lowEnd + (riskBuckets[1].value / bucketTotal) * 100 : 100;
  const donutBackground = bucketTotal
    ? `conic-gradient(#2cc58f 0 ${lowEnd}%, #f6bd35 ${lowEnd}% ${midEnd}%, #ff4b55 ${midEnd}% 100%)`
    : "conic-gradient(#d8e3f5 0 100%)";
  const modelRiskRows = useMemo(() => {
    const analyticsGroups = analytics?.model_risk_groups;
    if (analyticsGroups?.length) {
      const grouped = new Map(analyticsGroups.map((item) => [item.model_adapter, item]));
      return formalModels.map((model) => {
        const item = grouped.get(model.adapter);
        const value = item ? Math.max(0, Math.min(100, Math.round(Number(item.avg_risk_score || 0) * 100))) : 0;
        const label = formatAdapterName(model.adapter).replace(/（.*?）/g, "").replace(/模型$/, "");
        return { label, value, measured: Number(item?.count || 0) > 0 };
      }).sort((a, b) => Number(b.measured) - Number(a.measured) || b.value - a.value || a.label.localeCompare(b.label, "zh-CN"));
    }
    const grouped = new Map<string, number[]>();
    for (const run of runs) {
      const key = run.victim_model_adapters?.[0] || run.model_adapter || "";
      if (!key) continue;
      const score = Number(run.risk_score || run.asr_attack || run.asr || 0);
      if (!Number.isFinite(score)) continue;
      grouped.set(key, [...(grouped.get(key) || []), Math.max(0, Math.min(100, Math.round(score * 100)))]);
    }
    return formalModels.map((model) => {
      const values = grouped.get(model.adapter) || [];
      const value = values.length ? Math.round(values.reduce((sum, item) => sum + item, 0) / values.length) : 0;
      const label = formatAdapterName(model.adapter).replace(/（.*?）/g, "").replace(/模型$/, "");
      return { label, value, measured: values.length > 0 };
    }).sort((a, b) => Number(b.measured) - Number(a.measured) || b.value - a.value || a.label.localeCompare(b.label, "zh-CN"));
  }, [analytics?.model_risk_groups, formalModels, runs]);

  return (
    <div className="gov-stack">
      <div className="gov-metric-grid four">
        <GovMetric title="已接入模型数量" value={isOverviewLoading ? "加载中..." : formalModels.length || overview?.supported_model_count || 0} unit={isOverviewLoading ? "" : "个"} tone="blue" icon={<CubeIcon />} />
        <GovMetric title="已接入正式数据集" value={isOverviewLoading ? "加载中..." : overview?.formal_dataset_count || overview?.dataset_total_count || overview?.datasets?.filter((item) => String(item.tier || "") !== "demo").length || 0} unit={isOverviewLoading ? "" : "个"} tone="green" icon={<DatabaseIcon />} />
        <GovMetric title="已完成测评任务" value={completedJobs || 0} unit="个" tone="blue" icon={<ClipboardIcon />} />
        <GovMetric title="高风险任务数量" value={highRiskRuns || 0} unit="个" tone="red" icon={<AlertIcon />} />
      </div>

      <div className="gov-grid-main">
        <GovPanel title="最近七天测评任务趋势" className="wide">
          <div className="gov-line-chart" aria-label="最近任务趋势">
            <div className="gov-chart-y">
              <span>50</span>
              <span>40</span>
              <span>30</span>
              <span>20</span>
              <span>10</span>
              <span>0</span>
            </div>
            <div className="gov-line-area">
              {trend.length ? (
                <>
                  {trend.map((value, idx) => {
                    const position = trend.length > 1 ? idx / (trend.length - 1) : 0.5;
                    return (
                      <div className="gov-line-point" key={`${value}-${idx}`} style={{ left: `${2 + position * 96}%`, bottom: `${(value / maxTrend) * 78 + 8}%` }}>
                        <span>{value}</span>
                      </div>
                    );
                  })}
                  <svg viewBox="0 0 600 220" preserveAspectRatio="none">
                    <polyline points={trend.map((value, idx) => {
                      const position = trend.length > 1 ? idx / (trend.length - 1) : 0.5;
                      return `${12 + position * 576},${210 - (value / maxTrend) * 170}`;
                    }).join(" ")} />
                  </svg>
                </>
              ) : (
                <div className="gov-empty-state">
                  {runs.length ? "当前只有一次真实运行，等待更多结果形成趋势。" : "暂无真实运行记录，等待后端生成测评结果。"}
                </div>
              )}
            </div>
          </div>
        </GovPanel>

        <GovPanel title="风险等级分布">
          <div className="gov-risk-donut">
            <div className="gov-donut" style={{ background: donutBackground }} />
            <div className="gov-risk-legend">
              {riskBuckets.map((item) => (
                <div key={item.label} className="gov-legend-row">
                  <i style={{ background: item.color }} />
                  <span>{item.label}</span>
                  <strong>
                    {item.value} 个（{bucketTotal ? Math.round((item.value / bucketTotal) * 100) : 0}%）
                  </strong>
                </div>
              ))}
              <div className="gov-high-chip">高风险任务占比 {bucketTotal ? Math.round(((riskBuckets[2]?.value || 0) / bucketTotal) * 100) : 0}%</div>
            </div>
          </div>
        </GovPanel>
      </div>

      <div className="gov-bottom-grid">
        <GovPanel title="各模型风险对比">
          <div className="gov-bar-list">
            {modelRiskRows.map((model) => {
              const value = model.value;
              return (
                <div className="gov-bar-row" key={model.label}>
                  <span>{model.label}</span>
                  <div>
                    <i style={{ width: `${value}%` }} />
                  </div>
                  <strong>{model.measured ? value : "未测"}</strong>
                </div>
              );
            })}
            {!modelRiskRows.length ? <div className="gov-empty-state">{isOverviewLoading ? "正在加载模型清单。" : "暂无模型清单。"}</div> : null}
          </div>
        </GovPanel>

        <GovPanel title="最近完成的测评任务" className="wide">
          <div className="gov-table-wrap">
            <table className="gov-table gov-table-roomy att-dashboard-task-table">
              <thead>
                <tr>
                  <th>任务名称</th>
                  <th>攻击方式</th>
                  <th>当前状态</th>
                  <th>总体结论</th>
                  <th>完成时间</th>
                </tr>
              </thead>
              <tbody>
                {(runs.length ? runs.slice(0, 5) : []).map((run) => (
                  <tr key={run.run_id}>
                    <td>{formatRunDatasetName(run.dataset_name || "", run.benchmark_tag || "", run.task_kind || "")}</td>
                    <td>{formatAttackName(run.attack || "-")}</td>
                    <td><span className="gov-dot-inline success" />已完成</td>
                    <td className={`text-${riskTone(run.risk_level)}`}>
                      {riskText(run.risk_level)}
                    </td>
                    <td>{run.created_at ? new Date(run.created_at).toLocaleString("zh-CN") : "未记录完成时间"}</td>
                  </tr>
                ))}
                {!runs.length ? (
                  <tr>
                    <td>暂无真实测评任务</td>
                    <td>未记录</td>
                    <td><span className="gov-dot-inline" />等待中</td>
                    <td>未记录</td>
                    <td>等待后端运行记录</td>
                  </tr>
                ) : null}
              </tbody>
            </table>
          </div>
          <Link to="/analysis" className="gov-link-more">查看更多任务</Link>
        </GovPanel>

        <GovPanel title="系统提示">
          <div className="gov-note-list">
            <div><span className="ok"><ChartIcon /></span>系统累计已有 {completedJobs || 0} 个任务完成报告生成。</div>
            <div><span className="warn"><AlertIcon /></span>{highRiskRuns > 0 ? `当前有 ${highRiskRuns} 个高风险任务，建议优先复核。` : latest ? `最近任务风险等级为${latestRisk}，建议按报告详情复核关键样本。` : "暂无可复核风险任务。"}</div>
            <div><span className="info"><ClipboardIcon /></span>案例复盘页可查看攻击前后对比。</div>
          </div>
        </GovPanel>
      </div>
    </div>
  );
}
