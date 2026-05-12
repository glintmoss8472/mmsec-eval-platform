// 文件说明：该文件属于前端页面，集中实现 ReportCenterPage 相关逻辑。
import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link, useLocation } from "react-router-dom";

import { AlertIcon, ChartIcon, GovMetric, GovPanel, ShieldIcon } from "../components/GovCards";
import { getRunAnalytics, getRunOptions, getRunSummary, listRuns, type RunItem, type RunOptionsResponse, type RunQueryParams } from "../lib/api";
import { isDemoRun, riskText, riskTone } from "../lib/runPresentation";
import { formatAdapterName, formatAttackName, formatEvalScope, formatRunDatasetName } from "../lib/uiLabels";

type FilterState = {
  task: string;
  model: string;
  dataset: string;
  attack: string;
  resultType: string;
  confidence: string;
  search: string;
};

type SortDirection = "asc" | "desc";
type RunSortKey = "created" | "run_id" | "result_type" | "task_dataset" | "model" | "attack" | "sample_confidence" | "metric" | "asr_risk" | "detail";
type RunSortState = { key: RunSortKey; direction: SortDirection };
type RunRecordsMode = "analysis" | "reports";

const DEFAULT_FILTERS: FilterState = {
  task: "",
  model: "",
  dataset: "",
  attack: "",
  resultType: "",
  confidence: "",
  search: "",
};
const PAGE_SIZES = [10, 20, 50];
const REPORT_REFRESH_MS = 30000;
const DEFAULT_RUN_SORT: RunSortState = { key: "created", direction: "desc" };
const SERVER_TIME_ZONE = "Asia/Shanghai";

/** 中文注释：实现 asNumber 的核心流程，支撑前端页面中的业务语义和异常边界。 */
function asNumber(value: unknown, fallback = 0): number {
  const num = Number(value);
  return Number.isFinite(num) ? num : fallback;
}

/** 中文注释：实现 percent 的核心流程，支撑前端页面中的业务语义和异常边界。 */
function percent(value: unknown): string {
  const num = asNumber(value, 0);
  return `${Math.round(Math.max(0, Math.min(1, num)) * 100)}%`;
}

/** 中文注释：实现 createdAtText 的核心流程，支撑前端页面中的业务语义和异常边界。 */
function createdAtText(value: string | undefined) {
  if (!value) return "未记录时间";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString("zh-CN", { hour12: false, timeZone: SERVER_TIME_ZONE });
}

/** 中文注释：实现 taskKindLabel 的核心流程，支撑前端页面中的业务语义和异常边界。 */
function taskKindLabel(kind: string | undefined) {
  if (kind === "vlr") return "图文检索";
  if (kind === "vqa") return "视觉问答";
  if (kind === "caption") return "图像描述";
  if (kind === "pairwise") return "图文配对";
  return "通用测评";
}

/** 中文注释：实现 confidenceLabel 的核心流程，支撑前端页面中的业务语义和异常边界。 */
function confidenceLabel(value: string | undefined) {
  if (value === "high") return "高置信";
  if (value === "medium") return "中置信";
  if (value === "low") return "低置信";
  return "未标注";
}

/** 中文注释：实现 confidenceTone 的核心流程，支撑前端页面中的业务语义和异常边界。 */
function confidenceTone(value: string | undefined): "green" | "orange" | "red" {
  if (value === "high") return "green";
  if (value === "medium") return "orange";
  return "red";
}

/** 中文注释：实现 resultTypeLabel 的核心流程，支撑前端页面中的业务语义和异常边界。 */
function resultTypeLabel(value: string | undefined) {
  if (value === "debug") return "调试结果";
  return "正式结果";
}

/** 中文注释：实现 resultTypeClass 的核心流程，支撑前端页面中的业务语义和异常边界。 */
function resultTypeClass(value: string | undefined) {
  return value === "debug" ? "att-chip att-chip-warn" : "att-chip att-chip-ok";
}

/** 中文注释：实现 datasetLabel 的核心流程，支撑前端页面中的业务语义和异常边界。 */
function datasetLabel(run: RunItem) {
  return formatRunDatasetName(run.dataset_name || "", run.benchmark_tag || "", run.task_kind || "");
}

/** 中文注释：实现 modelLabel 的核心流程，支撑前端页面中的业务语义和异常边界。 */
function modelLabel(run: RunItem) {
  return formatAdapterName(run.victim_model_adapters?.[0] || run.model_adapter || run.surrogate_model_adapter || "-");
}

/** 中文注释：实现 sampleCount 的核心流程，支撑前端页面中的业务语义和异常边界。 */
function sampleCount(run: RunItem) {
  const count = asNumber(run.evidence_sample_count || run.case_count || run.sample_pair_count, 0);
  return count > 0 ? count : 0;
}

/** 中文注释：实现 sampleScaleText 的核心流程，支撑前端页面中的业务语义和异常边界。 */
function sampleScaleText(run: RunItem | undefined) {
  if (!run) return "未记录";
  const count = sampleCount(run);
  if (count <= 0) return "未记录";
  if (run.task_kind === "vlr") return `${count} 对/例`;
  return `${count} 条`;
}

/** 中文注释：实现 baselineText 的核心流程，支撑前端页面中的业务语义和异常边界。 */
function baselineText(run: RunItem) {
  if (run.task_kind === "vqa") return `原始输入准确率 ${percent(run.clean_accuracy)}`;
  if (run.task_kind === "caption") return `描述文本相似度 ${percent(run.caption_text_similarity ?? run.semantic_preservation_rate ?? run.object_jaccard)}`;
  return `前一位召回率 ${percent(run.clean_r1_mean)}`;
}

/** 中文注释：实现 taskMetricText 的核心流程，支撑前端页面中的业务语义和异常边界。 */
function taskMetricText(run: RunItem) {
  if (run.task_kind === "vqa") {
    return `原始准确率 ${percent(run.clean_accuracy)} / 攻击后准确率 ${percent(run.attacked_accuracy)} / 答案变化率 ${percent(run.answer_change_rate)}`;
  }
  if (run.task_kind === "caption") {
    return `对象集合重合度 ${percent(run.object_jaccard)} / 文本相似度 ${percent(run.caption_text_similarity)} / 目标翻转率 ${percent(run.target_flip_rate)}`;
  }
  return `前一位召回率 ${percent(run.clean_r1_mean)} 到 ${percent(run.attacked_r1_mean)} / 平均排名变化 ${Number.isFinite(Number(run.rank_delta_mean)) ? Math.round(Number(run.rank_delta_mean)) : "未记录"}`;
}

/** 中文注释：实现 pageRangeText 的核心流程，支撑前端页面中的业务语义和异常边界。 */
function pageRangeText(page: number, pageSize: number, total: number) {
  if (total <= 0) return "0 / 0";
  const start = (page - 1) * pageSize + 1;
  const end = Math.min(total, page * pageSize);
  return `${start} - ${end} / ${total}`;
}

/** 中文注释：实现 nextRunSort 的核心流程，支撑前端页面中的业务语义和异常边界。 */
function nextRunSort(current: RunSortState, key: RunSortKey): RunSortState {
  return { key, direction: current.key === key && current.direction === "asc" ? "desc" : "asc" };
}

/** 中文注释：实现 ariaSort 的核心流程，支撑前端页面中的业务语义和异常边界。 */
function ariaSort(sort: RunSortState, key: RunSortKey): "none" | "ascending" | "descending" {
  if (sort.key !== key) return "none";
  return sort.direction === "asc" ? "ascending" : "descending";
}

/** 中文注释：实现 SortHeader 的核心流程，支撑前端页面中的业务语义和异常边界。 */
function SortHeader({ sort, sortKey, label, onSort }: { sort: RunSortState; sortKey: RunSortKey; label: string; onSort: (key: RunSortKey) => void }) {
  const active = sort.key === sortKey;
  const icon = active ? (sort.direction === "asc" ? "↑" : "↓") : "↕";
  return (
    <button type="button" className={`gov-sort-button ${active ? "active" : ""}`} onClick={() => onSort(sortKey)} aria-label={`${label}排序`}>
      <span>{label}</span>
      <i aria-hidden="true">{icon}</i>
    </button>
  );
}

/** 中文注释：实现 queryParams 的核心流程，支撑前端页面中的业务语义和异常边界。 */
function queryParams(filters: FilterState): Omit<RunQueryParams, "page" | "page_size" | "sort_by" | "sort_dir"> {
  return {
    task_kind: filters.task,
    model: filters.model,
    dataset: filters.dataset,
    attack: filters.attack,
    result_type: filters.resultType,
    confidence: filters.confidence,
    search: filters.search,
    exclude_demo: true,
  };
}

/** 中文注释：实现 optionValues 的核心流程，支撑前端页面中的业务语义和异常边界。 */
function optionValues(options: RunOptionsResponse | undefined, key: keyof RunOptionsResponse) {
  return (options?.[key] ?? []).map((item) => item.value).filter(Boolean);
}

/** 中文注释：实现 FilterBar 的核心流程，支撑前端页面中的业务语义和异常边界。 */
function FilterBar({ options, filters, setFilters }: { options?: RunOptionsResponse; filters: FilterState; setFilters: (next: FilterState) => void }) {
  const taskOptions = optionValues(options, "task_kinds");
  const attackOptions = optionValues(options, "attacks");
  /** 中文注释：实现 update 的核心流程，支撑前端页面中的业务语义和异常边界。 */
  const update = (patch: Partial<FilterState>) => setFilters({ ...filters, ...patch });
  return (
    <GovPanel className="gov-filter-panel att-filter-panel">
      <div className="gov-filter-field">
        <label htmlFor="filter-task">任务类型</label>
        <select id="filter-task" value={filters.task} onChange={(event) => update({ task: event.target.value })}>
          <option value="">全部任务</option>
          {taskOptions.map((item) => <option key={item} value={item}>{taskKindLabel(item)}</option>)}
        </select>
      </div>
      <div className="gov-filter-field">
        <label htmlFor="filter-attack">攻击方法</label>
        <select id="filter-attack" value={filters.attack} onChange={(event) => update({ attack: event.target.value })}>
          <option value="">全部攻击</option>
          {attackOptions.map((item) => <option key={item} value={item}>{formatAttackName(item)}</option>)}
        </select>
      </div>
      <div className="gov-filter-field">
        <label htmlFor="filter-result-type">结果类型</label>
        <select id="filter-result-type" value={filters.resultType} onChange={(event) => update({ resultType: event.target.value })}>
          <option value="">全部：正式 + 调试</option>
          <option value="formal">正式结果</option>
          <option value="debug">调试结果</option>
        </select>
      </div>
      <div className="gov-filter-field">
        <label htmlFor="filter-confidence">证据置信度</label>
        <select id="filter-confidence" value={filters.confidence} onChange={(event) => update({ confidence: event.target.value })}>
          <option value="">全部置信度</option>
          <option value="high">高置信</option>
          <option value="medium">中置信</option>
          <option value="low">低置信</option>
        </select>
      </div>
      <div className="gov-filter-field">
        <label htmlFor="filter-model">模型</label>
        <input id="filter-model" value={filters.model} onChange={(event) => update({ model: event.target.value })} placeholder="输入模型或适配器" />
      </div>
      <div className="gov-filter-field">
        <label htmlFor="filter-dataset">数据集</label>
        <input id="filter-dataset" value={filters.dataset} onChange={(event) => update({ dataset: event.target.value })} placeholder="输入数据集关键词" />
      </div>
      <div className="gov-filter-field att-filter-search">
        <label htmlFor="filter-search">全文搜索</label>
        <input id="filter-search" value={filters.search} onChange={(event) => update({ search: event.target.value })} placeholder="运行编号 / 模型 / 攻击" />
      </div>
      <div className="gov-filter-field att-filter-actions">
        <label>操作</label>
        <button type="button" onClick={() => setFilters(DEFAULT_FILTERS)}>重置筛选</button>
      </div>
      <small className="gov-filter-help-row">默认纳入全部真实运行；样本数为 1 或 2 的记录不会被剔除，但会显示低置信证据提示。</small>
    </GovPanel>
  );
}

/** 中文注释：实现 DistributionBars 的核心流程，支撑前端页面中的业务语义和异常边界。 */
function DistributionBars({ rows, labelFor }: { rows: Array<{ key: string; count: number }>; labelFor: (key: string) => string }) {
  const max = Math.max(1, ...rows.map((row) => row.count));
  return (
    <div className="att-bars">
      {rows.map((row) => (
        <div className="att-bar-row" key={row.key}>
          <span>{labelFor(row.key)}</span>
          <div><i style={{ width: `${Math.max(4, (row.count / max) * 100)}%` }} /></div>
          <strong>{row.count}</strong>
        </div>
      ))}
      {!rows.length ? <div className="gov-empty-state">暂无统计数据。</div> : null}
    </div>
  );
}

/** 中文注释：实现 RunTable 的核心流程，支撑前端页面中的业务语义和异常边界。 */
function RunTable({ rows, selectedRunId, onSelect, sort, onSort, loading }: { rows: RunItem[]; selectedRunId?: string; onSelect?: (run: RunItem) => void; sort: RunSortState; onSort: (key: RunSortKey) => void; loading?: boolean }) {
  const visibleRows = rows.filter((run) => !isDemoRun(run));
  return (
    <div className="gov-table-wrap">
      <table className="gov-table gov-table-roomy att-run-table">
        <thead>
          <tr>
            <th aria-sort={ariaSort(sort, "created")}><SortHeader sort={sort} sortKey="created" label="测评时间" onSort={onSort} /></th>
            <th aria-sort={ariaSort(sort, "run_id")}><SortHeader sort={sort} sortKey="run_id" label="运行编号" onSort={onSort} /></th>
            <th aria-sort={ariaSort(sort, "result_type")}><SortHeader sort={sort} sortKey="result_type" label="类型" onSort={onSort} /></th>
            <th aria-sort={ariaSort(sort, "task_dataset")}><SortHeader sort={sort} sortKey="task_dataset" label="任务 / 数据集" onSort={onSort} /></th>
            <th aria-sort={ariaSort(sort, "model")}><SortHeader sort={sort} sortKey="model" label="模型" onSort={onSort} /></th>
            <th aria-sort={ariaSort(sort, "attack")}><SortHeader sort={sort} sortKey="attack" label="攻击方式" onSort={onSort} /></th>
            <th aria-sort={ariaSort(sort, "sample_confidence")}><SortHeader sort={sort} sortKey="sample_confidence" label="样本规模 / 置信度" onSort={onSort} /></th>
            <th aria-sort={ariaSort(sort, "metric")}><SortHeader sort={sort} sortKey="metric" label="分任务指标解释" onSort={onSort} /></th>
            <th aria-sort={ariaSort(sort, "asr_risk")}><SortHeader sort={sort} sortKey="asr_risk" label="ASR / 风险" onSort={onSort} /></th>
            <th aria-sort={ariaSort(sort, "detail")}><SortHeader sort={sort} sortKey="detail" label="详情" onSort={onSort} /></th>
          </tr>
        </thead>
        <tbody>
          {loading ? <tr><td colSpan={10}>正在加载测评记录。</td></tr> : null}
          {!loading && visibleRows.map((run) => (
            <tr
              key={run.run_id}
              className={run.run_id === selectedRunId ? "is-selected" : ""}
              data-testid={`detail-row-${run.run_id}`}
              tabIndex={onSelect ? 0 : undefined}
              onClick={() => onSelect?.(run)}
              onKeyDown={(event) => {
                if (!onSelect) return;
                if (event.key === "Enter" || event.key === " ") {
                  event.preventDefault();
                  onSelect(run);
                }
              }}
            >
              <td>{createdAtText(run.created_at)}</td>
              <td className="font-mono text-xs">{run.run_id}</td>
              <td><span className={resultTypeClass(run.result_type)}>{resultTypeLabel(run.result_type)}</span></td>
              <td><strong>{taskKindLabel(run.task_kind)}</strong><br />{datasetLabel(run)}</td>
              <td>{modelLabel(run)}</td>
              <td>{formatAttackName(run.attack || "-")}</td>
              <td>
                <strong>{sampleScaleText(run)}</strong><br />
                <span className={`att-chip att-chip-${confidenceTone(run.evidence_confidence)}`}>{confidenceLabel(run.evidence_confidence)}</span>
              </td>
              <td><span>攻击前基线：{baselineText(run)}</span><br /><em>{taskMetricText(run)}</em></td>
              <td>
                <strong>{percent(run.asr_attack ?? run.asr)}</strong><br />
                <span className={`text-${riskTone(run.risk_level)}`}>{riskText(run.risk_level || "-")}</span>
              </td>
              <td><Link className="gov-inline-link" to={`/reports/${run.run_id}`} onClick={(event) => event.stopPropagation()}>打开</Link></td>
            </tr>
          ))}
          {!loading && !visibleRows.length ? <tr><td colSpan={10}>暂无符合筛选条件的测评记录。</td></tr> : null}
        </tbody>
      </table>
    </div>
  );
}

/** 中文注释：实现 getAnalyticsData 的核心流程，支撑前端页面中的业务语义和异常边界。 */
function getAnalyticsData(data: unknown) {
  return (data ?? {}) as {
    total_runs?: number;
    total_cases?: number;
    avg_asr_attack?: number;
    formal_runs?: number;
    debug_runs?: number;
    high_risk_runs?: number;
    runs_with_case_evidence?: number;
    low_confidence_runs?: number;
    task_groups?: Array<Record<string, unknown>>;
    risk_distribution?: Array<{ key: string; count: number }>;
  };
}

/** 中文注释：实现 taskRowsFromAnalytics 的核心流程，支撑前端页面中的业务语义和异常边界。 */
function taskRowsFromAnalytics(data: ReturnType<typeof getAnalyticsData> | undefined) {
  return (data?.task_groups ?? []).map((row) => ({ key: String(row.task_kind || "unknown"), count: asNumber(row.count, 0) }));
}

/** 中文注释：实现 useRunPageData 的核心流程，支撑前端页面中的业务语义和异常边界。 */
function useRunPageData(filters: FilterState, page: number, pageSize: number, sort: RunSortState) {
  const baseParams = useMemo(() => queryParams(filters), [filters]);
  const runsQ = useQuery({
    queryKey: ["runs-page", baseParams, page, pageSize, sort.key, sort.direction],
    queryFn: () => listRuns({ ...baseParams, page, page_size: pageSize, sort_by: sort.key, sort_dir: sort.direction }),
    staleTime: REPORT_REFRESH_MS,
    refetchInterval: REPORT_REFRESH_MS,
  });
  const analyticsQ = useQuery({
    queryKey: ["run-analytics", baseParams],
    queryFn: () => getRunAnalytics(baseParams),
    staleTime: REPORT_REFRESH_MS,
    refetchInterval: REPORT_REFRESH_MS,
  });
  const optionsQ = useQuery({
    queryKey: ["run-options"],
    queryFn: () => getRunOptions({ exclude_demo: true }),
    staleTime: REPORT_REFRESH_MS,
    refetchInterval: REPORT_REFRESH_MS,
  });
  return { runsQ, analyticsQ, optionsQ };
}

/** 中文注释：实现 AnalysisView 的核心流程，支撑前端页面中的业务语义和异常边界。 */
function AnalysisView() {
  const [filters, setFilters] = useState(DEFAULT_FILTERS);
  const [selectedRunId, setSelectedRunId] = useState<string>("");
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);
  const [sort, setSort] = useState<RunSortState>(DEFAULT_RUN_SORT);
  const [lastKnownTotalRows, setLastKnownTotalRows] = useState(0);
  const { runsQ, analyticsQ, optionsQ } = useRunPageData(filters, page, pageSize, sort);
  const runs = (runsQ.data?.items ?? []).filter((run) => !isDemoRun(run));
  const analytics = getAnalyticsData(analyticsQ.data);
  const totalRows = runsQ.data?.total ?? analytics.total_runs ?? lastKnownTotalRows;
  const totalPages = Math.max(1, Math.ceil(totalRows / pageSize));
  const safePage = totalRows ? Math.min(page, totalPages) : page;
  const selectedRun = runs.find((run) => run.run_id === selectedRunId) ?? runs[0];
  /** 中文注释：实现 handleSort 的核心流程，支撑前端页面中的业务语义和异常边界。 */
  const handleSort = (key: RunSortKey) => { setSort((current) => nextRunSort(current, key)); setPage(1); };
  const summaryQ = useQuery({ queryKey: ["analysis-summary", selectedRun?.run_id], queryFn: () => getRunSummary(String(selectedRun?.run_id || "")), enabled: Boolean(selectedRun?.run_id), staleTime: REPORT_REFRESH_MS, refetchInterval: REPORT_REFRESH_MS });
  const summary = summaryQ.data as Record<string, unknown> | undefined;
  const riskAudit = Array.isArray(summary?.risk_component_audit) ? summary.risk_component_audit as Array<Record<string, unknown>> : [];
  const riskRows = analytics.risk_distribution ?? [];
  const taskRows = taskRowsFromAnalytics(analytics);

  useEffect(() => {
    const total = runsQ.data?.total ?? analytics.total_runs;
    if (typeof total === "number") setLastKnownTotalRows(total);
  }, [runsQ.data?.total, analytics.total_runs]);

  useEffect(() => {
    if (!runsQ.data && !analyticsQ.data) return;
    if (page !== safePage) setPage(safePage);
  }, [runsQ.data, analyticsQ.data, page, safePage]);

  return (
    <div className="gov-stack">
      <FilterBar options={optionsQ.data} filters={filters} setFilters={(next) => { setFilters(next); setPage(1); }} />

      <div className="gov-metric-grid five">
        <GovMetric title="全量实验" value={analytics.total_runs ?? totalRows} tone="blue" icon={<ChartIcon />} />
        <GovMetric title="平均攻击成功率" value={analytics.avg_asr_attack === undefined ? "暂无数据" : percent(analytics.avg_asr_attack)} tone="red" icon={<AlertIcon />} />
        <GovMetric title="高风险记录" value={analytics.high_risk_runs ?? 0} tone="orange" icon={<AlertIcon />} />
        <GovMetric title="样本案例证据" value={analytics.total_cases ?? 0} tone="green" icon={<ShieldIcon />} />
        <GovMetric title="低置信提示" value={analytics.low_confidence_runs ?? 0} tone={analytics.low_confidence_runs ? "orange" : "green"} icon={<ChartIcon />} />
      </div>

      <div className="gov-analysis-context" data-testid="analysis-context">
        {selectedRun ? (
          <>
            <div>
              <span>当前核心记录</span>
              <strong>{modelLabel(selectedRun)} · {datasetLabel(selectedRun)}</strong>
              <p>{formatAttackName(selectedRun.attack || "-")} · {taskKindLabel(selectedRun.task_kind)} · {createdAtText(selectedRun.created_at)}</p>
            </div>
            <div className="gov-context-badges">
              <span>运行 {selectedRun.run_id}</span>
              <span>{resultTypeLabel(selectedRun.result_type)}</span>
              <span>样本规模 {sampleScaleText(selectedRun)}</span>
              <span>{confidenceLabel(selectedRun.evidence_confidence)}</span>
              <span>{formatEvalScope(selectedRun.risk_scenario || "general")}</span>
            </div>
          </>
        ) : (
          <div><span>当前核心记录</span><strong>暂无真实测评记录</strong><p>请先完成测评任务。</p></div>
        )}
      </div>

      {selectedRun?.evidence_note ? <div className="att-evidence-warning"><AlertIcon />{selectedRun.evidence_note}</div> : null}
      {runsQ.isError || analyticsQ.isError ? <div className="gov-empty-state">后端运行记录接口不可访问：{String(((runsQ.error || analyticsQ.error) as Error)?.message ?? runsQ.error ?? analyticsQ.error)}</div> : null}

      <div className="gov-analysis-grid">
        <GovPanel title="跨任务运行分布" className="wide">
          <DistributionBars rows={taskRows} labelFor={taskKindLabel} />
          <div className="gov-table-note">后端聚合接口当前返回 {analytics.total_runs ?? totalRows} 条运行、{analytics.total_cases ?? 0} 条案例证据。</div>
        </GovPanel>
        <GovPanel title="风险与证据解释">
          <DistributionBars rows={riskRows} labelFor={(key) => riskText(key, "未标注")} />
          <div className="att-risk-audit">
            {riskAudit.slice(0, 5).map((item) => (
              <span key={String(item.key || item.label_zh)}>{String(item.label_zh || item.key)} {percent(item.value)}</span>
            ))}
            {!riskAudit.length ? <span>当前运行未记录风险分解明细。</span> : null}
          </div>
        </GovPanel>
      </div>

      <GovPanel title="分任务指标明细" className="wide">
        <div className="gov-table-note">图文检索、视觉问答、图像描述使用各自指标解释；统一风险分只作为跨任务比较入口，不替代样本级证据。</div>
        <div className="gov-table-toolbar">
          <span>共 {totalRows} 条记录。样本数 1 或 2 的记录仍在默认视图中，只标注证据置信度。</span>
          <div className="gov-table-pagination">
            <label>每页行数<select value={pageSize} onChange={(event) => { setPageSize(Number(event.target.value)); setPage(1); }}>{PAGE_SIZES.map((size) => <option key={size} value={size}>{size}</option>)}</select></label>
            <button type="button" disabled={safePage <= 1} onClick={() => setPage(Math.max(1, safePage - 1))}>上一页</button>
            <strong>{pageRangeText(safePage, pageSize, totalRows)}</strong>
            <button type="button" disabled={safePage >= totalPages} onClick={() => setPage(Math.min(totalPages, safePage + 1))}>下一页</button>
          </div>
        </div>
        <RunTable rows={runs} selectedRunId={selectedRun?.run_id} onSelect={(run) => setSelectedRunId(run.run_id)} sort={sort} onSort={handleSort} loading={runsQ.isLoading} />
      </GovPanel>
    </div>
  );
}

/** 中文注释：实现 ReportsView 的核心流程，支撑前端页面中的业务语义和异常边界。 */
function ReportsView() {
  const [filters, setFilters] = useState(DEFAULT_FILTERS);
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);
  const [sort, setSort] = useState<RunSortState>(DEFAULT_RUN_SORT);
  const [lastKnownTotalRows, setLastKnownTotalRows] = useState(0);
  const { runsQ, analyticsQ, optionsQ } = useRunPageData(filters, page, pageSize, sort);
  const runs = (runsQ.data?.items ?? []).filter((run) => !isDemoRun(run));
  const analytics = getAnalyticsData(analyticsQ.data);
  const totalRows = runsQ.data?.total ?? analytics.total_runs ?? lastKnownTotalRows;
  const totalPages = Math.max(1, Math.ceil(totalRows / pageSize));
  const safePage = totalRows ? Math.min(page, totalPages) : page;
  /** 中文注释：实现 handleSort 的核心流程，支撑前端页面中的业务语义和异常边界。 */
  const handleSort = (key: RunSortKey) => { setSort((current) => nextRunSort(current, key)); setPage(1); };

  useEffect(() => {
    const total = runsQ.data?.total ?? analytics.total_runs;
    if (typeof total === "number") setLastKnownTotalRows(total);
  }, [runsQ.data?.total, analytics.total_runs]);

  useEffect(() => {
    if (!runsQ.data && !analyticsQ.data) return;
    if (page !== safePage) setPage(safePage);
  }, [runsQ.data, analyticsQ.data, page, safePage]);

  return (
    <div className="gov-stack">
      <FilterBar options={optionsQ.data} filters={filters} setFilters={(next) => { setFilters(next); setPage(1); }} />
      <div className="gov-metric-grid five">
        <GovMetric title="报告总数" value={analytics.total_runs ?? totalRows} tone="blue" icon={<ChartIcon />} />
        <GovMetric title="正式报告" value={analytics.formal_runs ?? 0} tone="green" icon={<ShieldIcon />} />
        <GovMetric title="调试报告" value={analytics.debug_runs ?? 0} tone="orange" icon={<AlertIcon />} />
        <GovMetric title="高风险报告" value={analytics.high_risk_runs ?? 0} tone="red" icon={<AlertIcon />} />
        <GovMetric title="有案例证据" value={analytics.runs_with_case_evidence ?? 0} tone="green" icon={<ShieldIcon />} />
      </div>
      {runsQ.isError || analyticsQ.isError ? <div className="gov-empty-state">后端运行记录接口不可访问：{String(((runsQ.error || analyticsQ.error) as Error)?.message ?? runsQ.error ?? analyticsQ.error)}</div> : null}
      <GovPanel title="报告中心" className="wide">
        <div className="gov-table-note">这里管理单次运行报告；点击“打开”进入报告详情，继续查看实验结论、分任务指标解释和样本入口。</div>
        <div className="gov-table-toolbar">
          <span>共 {totalRows} 份报告</span>
          <div className="gov-table-pagination">
            <label>每页行数<select value={pageSize} onChange={(event) => { setPageSize(Number(event.target.value)); setPage(1); }}>{PAGE_SIZES.map((size) => <option key={size} value={size}>{size}</option>)}</select></label>
            <button type="button" disabled={safePage <= 1} onClick={() => setPage(Math.max(1, safePage - 1))}>上一页</button>
            <strong>{pageRangeText(safePage, pageSize, totalRows)}</strong>
            <button type="button" disabled={safePage >= totalPages} onClick={() => setPage(Math.min(totalPages, safePage + 1))}>下一页</button>
          </div>
        </div>
        <RunTable rows={runs} sort={sort} onSort={handleSort} loading={runsQ.isLoading} />
      </GovPanel>
    </div>
  );
}

/** 中文注释：实现 RunRecordsView 的核心流程，支撑前端页面中的业务语义和异常边界。 */
export function RunRecordsView({ mode }: { mode: RunRecordsMode }) {
  return mode === "reports" ? <ReportsView /> : <AnalysisView />;
}

export default function ReportCenterPage() {
  const location = useLocation();
  return <RunRecordsView mode={location.pathname === "/reports" ? "reports" : "analysis"} />;
}
