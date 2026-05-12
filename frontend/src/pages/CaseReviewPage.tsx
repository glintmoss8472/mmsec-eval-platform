import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";

import { AlertIcon, CheckIcon, GovMetric, GovPanel, ShieldIcon } from "../components/GovCards";
import { getCaseDetail, getRunOptions, listCaseIndex, runAssetUrl, type CaseIndexItem } from "../lib/api";
import { caseInputLabel, caseInputText, caseOutputLabel, caseOutputText } from "../lib/caseBundleText";
import { riskText, riskTone } from "../lib/runPresentation";
import { formatAdapterName, formatAttackName, formatRunDatasetName } from "../lib/uiLabels";

type SortDirection = "asc" | "desc";
type CaseSortKey = "created" | "sample_id" | "task_dataset" | "model_attack" | "status" | "artifact" | "report";
type CaseSortState = { key: CaseSortKey; direction: SortDirection };

const DEFAULT_CASE_SORT: CaseSortState = { key: "created", direction: "desc" };
const SERVER_TIME_ZONE = "Asia/Shanghai";
const CASE_PAGE_SIZES = [20, 50, 100];
const CASE_REFRESH_MS = 30000;

function asRecord(value: unknown): Record<string, unknown> {
  return typeof value === "object" && value !== null ? value as Record<string, unknown> : {};
}

function relativeRunPath(path: string) {
  return path.replace(/^.*runs[\\/][^\\/]+[\\/]/, "");
}

function caseAsset(runId: string, path: unknown) {
  const raw = String(path || "");
  return raw ? runAssetUrl(runId, relativeRunPath(raw)) : "";
}

function taskKindLabel(kind: string | undefined) {
  if (kind === "vlr") return "图文检索";
  if (kind === "vqa") return "视觉问答";
  if (kind === "caption") return "图像描述";
  return "通用测评";
}

function confidenceLabel(value: string | undefined) {
  if (value === "high") return "高置信";
  if (value === "medium") return "中置信";
  if (value === "low") return "低置信";
  return "未标注";
}

function artifactLabel(value: string | undefined) {
  if (value === "complete") return "证据完整";
  if (value === "partial") return "证据部分";
  if (value === "summary_only") return "仅摘要";
  return "未标注";
}

function successLabel(value: unknown, taskKind = "") {
  const ok = value === true || String(value).toLowerCase() === "true";
  if (taskKind === "vlr") return ok ? "攻击成功" : "未形成检索失败";
  return ok ? "攻击成功" : "攻击未成功";
}

function statusClass(ok: unknown) {
  const success = ok === true || String(ok).toLowerCase() === "true";
  return success ? "att-chip att-chip-red" : "att-chip att-chip-ok";
}

function datasetLabel(item: CaseIndexItem) {
  return formatRunDatasetName(item.dataset_name || "", item.benchmark_tag || "", item.task_kind || "");
}

function modelLabel(item: CaseIndexItem) {
  return formatAdapterName(item.model_adapter || "-");
}

function fixed(value: unknown, digits = 4) {
  const num = Number(value);
  return Number.isFinite(num) ? num.toFixed(digits) : "未记录";
}

function unique(values: Array<string | undefined>) {
  return [...new Set(values.filter((value): value is string => Boolean(value)))].sort();
}

function createdAtText(value: string | undefined) {
  if (!value) return "未记录时间";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString("zh-CN", { hour12: false, timeZone: SERVER_TIME_ZONE });
}

function nextCaseSort(current: CaseSortState, key: CaseSortKey): CaseSortState {
  return { key, direction: current.key === key && current.direction === "asc" ? "desc" : "asc" };
}

function caseAriaSort(sort: CaseSortState, key: CaseSortKey): "none" | "ascending" | "descending" {
  if (sort.key !== key) return "none";
  return sort.direction === "asc" ? "ascending" : "descending";
}

function CaseSortHeader({ sort, sortKey, label, onSort }: { sort: CaseSortState; sortKey: CaseSortKey; label: string; onSort: (key: CaseSortKey) => void }) {
  const active = sort.key === sortKey;
  const icon = active ? (sort.direction === "asc" ? "↑" : "↓") : "↕";
  return (
    <button type="button" className={`gov-sort-button ${active ? "active" : ""}`} onClick={() => onSort(sortKey)} aria-label={`${label}排序`}>
      <span>{label}</span>
      <i aria-hidden="true">{icon}</i>
    </button>
  );
}


export default function CaseReviewPage() {
  const [taskKind, setTaskKind] = useState("");
  const [attack, setAttack] = useState("");
  const [success, setSuccess] = useState("");
  const [riskLevel, setRiskLevel] = useState("");
  const [artifactStatus, setArtifactStatus] = useState("");
  const [search, setSearch] = useState("");
  const [selectedKey, setSelectedKey] = useState("");
  const [sort, setSort] = useState<CaseSortState>(DEFAULT_CASE_SORT);
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(50);
  const [lastKnownTotalCases, setLastKnownTotalCases] = useState(0);

  const optionsQ = useQuery({ queryKey: ["case-library-options"], queryFn: () => getRunOptions({ exclude_demo: true }), staleTime: CASE_REFRESH_MS, refetchInterval: CASE_REFRESH_MS });
  const casesQ = useQuery({
    queryKey: ["case-library", taskKind, attack, success, riskLevel, artifactStatus, search, page, pageSize, sort.key, sort.direction],
    queryFn: () => listCaseIndex({ page, page_size: pageSize, sort_by: sort.key, sort_dir: sort.direction, task_kind: taskKind, attack, success, risk_level: riskLevel, artifact_status: artifactStatus, search, exclude_demo: true }),
    staleTime: CASE_REFRESH_MS,
    refetchInterval: CASE_REFRESH_MS,
  });
  const cases = casesQ.data?.items ?? [];
  const totalCases = casesQ.data?.total ?? lastKnownTotalCases;
  const totalPages = Math.max(1, Math.ceil(totalCases / pageSize));
  const safePage = totalCases ? Math.min(page, totalPages) : page;
  const selectedCase = cases.find((item) => `${item.run_id}::${item.sample_id}` === selectedKey) ?? cases[0];
  const handleSort = (key: CaseSortKey) => { setSort((current) => nextCaseSort(current, key)); setPage(1); };

  useEffect(() => {
    if (typeof casesQ.data?.total === "number") setLastKnownTotalCases(casesQ.data.total);
  }, [casesQ.data?.total]);

  useEffect(() => {
    if (!casesQ.data) return;
    if (page !== safePage) setPage(safePage);
  }, [casesQ.data, page, safePage]);

  useEffect(() => {
    if (!selectedCase) return;
    const key = `${selectedCase.run_id}::${selectedCase.sample_id}`;
    if (!selectedKey || !cases.some((item) => `${item.run_id}::${item.sample_id}` === selectedKey)) {
      setSelectedKey(key);
    }
  }, [cases, selectedCase, selectedKey]);

  const detailQ = useQuery({
    queryKey: ["case-library-detail", selectedCase?.run_id, selectedCase?.sample_id],
    queryFn: () => getCaseDetail(String(selectedCase?.run_id || ""), String(selectedCase?.sample_id || "")),
    enabled: Boolean(selectedCase?.run_id && selectedCase?.sample_id),
  });

  const bundle = asRecord(detailQ.data?.case_bundle);
  const refs = asRecord(bundle.artifact_refs);
  const cleanImage = caseAsset(String(selectedCase?.run_id || ""), refs.clean_image);
  const advImage = caseAsset(String(selectedCase?.run_id || ""), refs.adv_image || refs.attack_visualization);
  const metrics = asRecord(bundle.metrics);
  const capability = Array.isArray(bundle.artifact_capability) ? bundle.artifact_capability.map(asRecord) : [];
  const visibleCapability = capability.filter((item) => String(item.status || "").toLowerCase() === "available");
  const cleanOutput = caseOutputText(bundle, "clean");
  const advOutput = caseOutputText(bundle, "adv");
  const cleanInput = caseInputText(bundle, "clean");
  const advInput = caseInputText(bundle, "adv");
  const taskOptions = unique((optionsQ.data?.task_kinds ?? []).map((item) => item.value));
  const attackOptions = unique((optionsQ.data?.attacks ?? []).map((item) => item.value));
  const riskOptions = unique((optionsQ.data?.risk_levels ?? []).map((item) => item.value));
  const successCount = cases.filter((item) => item.judge_success).length;
  const partialCount = cases.filter((item) => item.artifact_status !== "complete").length;

  return (
    <div className="gov-stack">
      <GovPanel className="gov-filter-panel att-filter-panel">
        <div className="gov-filter-field">
          <label htmlFor="case-task-kind">任务类型</label>
          <select id="case-task-kind" value={taskKind} onChange={(event) => { setTaskKind(event.target.value); setPage(1); }}>
            <option value="">全部任务</option>
            {taskOptions.map((item) => <option key={item} value={item}>{taskKindLabel(item)}</option>)}
          </select>
        </div>
        <div className="gov-filter-field">
          <label htmlFor="case-attack">攻击方法</label>
          <select id="case-attack" value={attack} onChange={(event) => { setAttack(event.target.value); setPage(1); }}>
            <option value="">全部攻击</option>
            {attackOptions.map((item) => <option key={item} value={item}>{formatAttackName(item)}</option>)}
          </select>
        </div>
        <div className="gov-filter-field">
          <label htmlFor="case-success">成功状态</label>
          <select id="case-success" value={success} onChange={(event) => { setSuccess(event.target.value); setPage(1); }}>
            <option value="">全部案例</option>
            <option value="success">攻击成功</option>
            <option value="failed">攻击未成功</option>
          </select>
        </div>
        <div className="gov-filter-field">
          <label htmlFor="case-risk-level">风险等级</label>
          <select id="case-risk-level" value={riskLevel} onChange={(event) => { setRiskLevel(event.target.value); setPage(1); }}>
            <option value="">全部风险</option>
            {riskOptions.map((item) => <option key={item} value={item}>{riskText(item)}</option>)}
          </select>
        </div>
        <div className="gov-filter-field">
          <label htmlFor="case-artifact">证据状态</label>
          <select id="case-artifact" value={artifactStatus} onChange={(event) => { setArtifactStatus(event.target.value); setPage(1); }}>
            <option value="">全部证据</option>
            <option value="complete">证据完整</option>
            <option value="partial">证据部分</option>
          </select>
        </div>
        <div className="gov-filter-field att-filter-search">
          <label htmlFor="case-search">搜索</label>
          <input id="case-search" value={search} onChange={(event) => { setSearch(event.target.value); setPage(1); }} placeholder="样本 / 运行编号 / 文本 / 模型" />
        </div>
        <div className="gov-filter-field att-filter-actions">
          <label>操作</label>
          <button type="button" onClick={() => { setTaskKind(""); setAttack(""); setSuccess(""); setRiskLevel(""); setArtifactStatus(""); setSearch(""); setPage(1); }}>重置筛选</button>
        </div>
        <small className="gov-filter-help-row">案例库默认展示真实测评记录中的有证据样本；图文检索历史运行没有落盘案例索引时，会从报告和攻击调试产物派生代表性复盘案例。</small>
      </GovPanel>

      <div className="gov-metric-grid five">
        <GovMetric title="案例总数" value={casesQ.data?.total ?? cases.length} tone="blue" icon={<ShieldIcon />} />
        <GovMetric title="当前页成功" value={successCount} tone="red" icon={<AlertIcon />} />
        <GovMetric title="当前页证据部分" value={partialCount} tone={partialCount ? "orange" : "green"} icon={<AlertIcon />} />
        <GovMetric title="当前页任务覆盖" value={unique(cases.map((item) => item.task_kind)).length} tone="green" icon={<CheckIcon />} />
        <GovMetric title="已选样本" value={selectedCase ? taskKindLabel(selectedCase.task_kind) : "暂无"} tone="blue" icon={<ShieldIcon />} />
      </div>

      <div className="att-case-library-grid">
        <GovPanel title="案例列表" className="wide">
          <div className="gov-table-note">点击任意案例会在右侧展示证据预览；“打开复盘”进入完整样本页面。排序由后端按全量案例执行，前端只渲染当前页。</div>
          <div className="gov-table-toolbar">
            <span>共 {totalCases} 条案例，当前页 {cases.length} 条</span>
            <div className="gov-table-pagination">
              <label>每页行数<select value={pageSize} onChange={(event) => { setPageSize(Number(event.target.value)); setPage(1); }}>{CASE_PAGE_SIZES.map((size) => <option key={size} value={size}>{size}</option>)}</select></label>
              <button type="button" disabled={safePage <= 1} onClick={() => setPage(Math.max(1, safePage - 1))}>上一页</button>
              <strong>{totalCases ? `${(safePage - 1) * pageSize + 1} - ${Math.min(safePage * pageSize, totalCases)} / ${totalCases}` : "0 / 0"}</strong>
              <button type="button" disabled={safePage >= totalPages} onClick={() => setPage(Math.min(totalPages, safePage + 1))}>下一页</button>
            </div>
          </div>
          <div className="gov-table-wrap">
            <table className="gov-table gov-table-roomy att-case-table">
              <thead>
                <tr>
                  <th aria-sort={caseAriaSort(sort, "created")}><CaseSortHeader sort={sort} sortKey="created" label="测评时间" onSort={handleSort} /></th>
                  <th aria-sort={caseAriaSort(sort, "sample_id")}><CaseSortHeader sort={sort} sortKey="sample_id" label="样本编号" onSort={handleSort} /></th>
                  <th aria-sort={caseAriaSort(sort, "task_dataset")}><CaseSortHeader sort={sort} sortKey="task_dataset" label="任务 / 数据集" onSort={handleSort} /></th>
                  <th aria-sort={caseAriaSort(sort, "model_attack")}><CaseSortHeader sort={sort} sortKey="model_attack" label="模型 / 攻击" onSort={handleSort} /></th>
                  <th aria-sort={caseAriaSort(sort, "status")}><CaseSortHeader sort={sort} sortKey="status" label="状态" onSort={handleSort} /></th>
                  <th aria-sort={caseAriaSort(sort, "artifact")}><CaseSortHeader sort={sort} sortKey="artifact" label="证据" onSort={handleSort} /></th>
                  <th aria-sort={caseAriaSort(sort, "report")}><CaseSortHeader sort={sort} sortKey="report" label="报告" onSort={handleSort} /></th>
                </tr>
              </thead>
              <tbody>
                {casesQ.isLoading ? <tr><td colSpan={7}>正在加载案例列表。</td></tr> : null}
                {!casesQ.isLoading && cases.map((item) => {
                  const key = `${item.run_id}::${item.sample_id}`;
                  return (
                    <tr key={key} className={key === `${selectedCase?.run_id}::${selectedCase?.sample_id}` ? "is-selected" : ""} onClick={() => setSelectedKey(key)} tabIndex={0}>
                      <td>{createdAtText(item.created_at)}</td>
                      <td><strong className="font-mono text-xs">{item.sample_id}</strong><br /><span>{item.run_id}</span></td>
                      <td><strong>{taskKindLabel(item.task_kind)}</strong><br />{datasetLabel(item)}</td>
                      <td>{modelLabel(item)}<br />{formatAttackName(item.attack || "-")}</td>
                      <td>
                        <span className={statusClass(item.judge_success)}>{successLabel(item.judge_success, item.task_kind)}</span><br />
                        <span className={`text-${riskTone(item.risk_level)}`}>{riskText(item.risk_level)}</span>
                      </td>
                      <td><span className="att-chip">{artifactLabel(item.artifact_status)}</span><br /><span>{confidenceLabel(item.evidence_confidence)}</span></td>
                      <td><Link className="gov-inline-link" to={`/reports/${item.run_id}`} onClick={(event) => event.stopPropagation()}>报告</Link></td>
                    </tr>
                  );
                })}
                {!casesQ.isLoading && !cases.length ? <tr><td colSpan={7}>暂无符合筛选条件的案例。</td></tr> : null}
              </tbody>
            </table>
          </div>
        </GovPanel>

        <GovPanel title="案例证据预览">
          {!selectedCase ? <div className="gov-empty-state">请选择一个案例。</div> : null}
          {selectedCase ? (
            <div className="att-case-preview">
              <div className="att-case-head">
                <span className={statusClass(selectedCase.judge_success)}>{successLabel(selectedCase.judge_success, selectedCase.task_kind)}</span>
                <span className="att-chip">{artifactLabel(selectedCase.artifact_status)}</span>
                <span className="att-chip">{confidenceLabel(selectedCase.evidence_confidence)}</span>
              </div>
              <h3>{selectedCase.sample_id}</h3>
              <p>{taskKindLabel(selectedCase.task_kind)} · {datasetLabel(selectedCase)} · {formatAttackName(selectedCase.attack || "-")}</p>
              <div className="att-thumb-pair">
                <figure>{cleanImage ? <img src={cleanImage} alt="原始样本" /> : <div className="gov-empty-state gov-empty-state-compact">无原始图</div>}<figcaption>原始样本</figcaption></figure>
                <figure>{advImage ? <img src={advImage} alt="对抗样本" /> : <div className="gov-empty-state gov-empty-state-compact">无对抗图</div>}<figcaption>对抗样本</figcaption></figure>
              </div>
              <div className="att-output-diff">
                <div><strong>{caseInputLabel("clean", String(selectedCase.task_kind || ""), cleanInput || selectedCase.text)}</strong><p>{cleanInput || selectedCase.text || "未记录"}</p><strong>{caseOutputLabel("clean", String(selectedCase.task_kind || ""), cleanOutput)}</strong><p>{cleanOutput || "未记录"}</p></div>
                <div><strong>{caseInputLabel("adv", String(selectedCase.task_kind || ""), advInput || selectedCase.text)}</strong><p>{advInput || selectedCase.text || "未记录"}</p><strong>{caseOutputLabel("adv", String(selectedCase.task_kind || ""), advOutput)}</strong><p>{advOutput || "未记录"}</p></div>
              </div>
              <dl className="att-metric-list">
                <div><dt>扰动 L2</dt><dd>{fixed(metrics.perturbation_l2 ?? selectedCase.perturbation_l2)}</dd></div>
                <div><dt>扰动 Linf</dt><dd>{fixed(metrics.perturbation_linf ?? selectedCase.perturbation_linf)}</dd></div>
                <div><dt>风险分数</dt><dd>{fixed(selectedCase.risk_score)}</dd></div>
              </dl>
              {visibleCapability.length ? <div className="att-artifact-list">{visibleCapability.map((item) => <span key={String(item.key)} className={`att-chip att-artifact-${String(item.status)}`}>{String(item.label)}：{String(item.reason)}</span>)}</div> : null}
              <div className="att-action-row">
                <Link className="gov-inline-link" to={`/reports/${selectedCase.run_id}/cases/${selectedCase.sample_id}`}>打开复盘</Link>
                <Link className="gov-inline-link" to={`/reports/${selectedCase.run_id}`}>查看报告</Link>
              </div>
            </div>
          ) : null}
        </GovPanel>
      </div>
    </div>
  );
}
