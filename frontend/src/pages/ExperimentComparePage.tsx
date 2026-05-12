// 文件说明：该文件属于前端页面，集中实现 ExperimentComparePage 相关逻辑。
import ReactECharts from "echarts-for-react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";

import { compareRuns, listRuns, RunItem } from "../lib/api";
import { formatAdapterName, formatAttackName, formatDatasetName, formatJobType, formatRiskLevel, formatWrapped } from "../lib/uiLabels";

type MetricKey = "attack_drop" | "attacked_recall" | "clean_recall";

type MetricPoint = {
  attack_drop: number;
  attacked_recall: number;
  clean_recall: number;
};

const METRIC_OPTIONS: Array<{ value: MetricKey; label: string }> = [
  { value: "attack_drop", label: "攻击降幅" },
  { value: "attacked_recall", label: "攻击后召回率" },
  { value: "clean_recall", label: "原始召回率" },
];

/** 整理 `as record` 前端辅助逻辑，保持数据转换和展示口径一致。 */
function asRecord(value: unknown): Record<string, unknown> {
  return typeof value === "object" && value !== null ? (value as Record<string, unknown>) : {};
}

/** 整理 `as 数值` 前端辅助逻辑，保持数据转换和展示口径一致。 */
function asNum(value: unknown): number {
  const n = Number(value);
  return Number.isFinite(n) ? n : 0;
}

/** 转换 `parse 指标 point` 输入，保证接口数据能被页面安全使用。 */
function parseMetricPoint(value: unknown): MetricPoint {
  const row = asRecord(value);
  const cleanRecall = asNum(row.clean_recall);
  const attackedRecall = asNum(row.attacked_recall);
  return {
    attack_drop: asNum(row.attack_drop),
    attacked_recall: attackedRecall,
    clean_recall: cleanRecall,
  };
}

/** 整理 `short 运行记录 id` 前端辅助逻辑，保持数据转换和展示口径一致。 */
function shortRunId(runId: string): string {
  return runId.length > 16 ? `${runId.slice(0, 8)}...${runId.slice(-6)}` : runId;
}

/** 生成 `指标 label` 展示值，统一页面标签、颜色和缺省文案。 */
function metricLabel(metric: MetricKey): string {
  return METRIC_OPTIONS.find((item) => item.value === metric)?.label ?? metric;
}

/** 渲染 `ExperimentComparePage` 组件，组织该区域的数据读取、交互状态和可访问性标记。 */
export default function ExperimentComparePage() {
  const runsQ = useQuery({ queryKey: ["runs-compare"], queryFn: () => listRuns({ page: 1, page_size: 200 }) });
  const [selected, setSelected] = useState<string[]>([]);
  const [manualIds, setManualIds] = useState("");
  const [heatMetric, setHeatMetric] = useState<MetricKey>("attack_drop");
  const [focusVictim, setFocusVictim] = useState<string>("");

  const compare = useMutation({
    mutationFn: async () => {
      const manual = manualIds
        .split(",")
        .map((x) => x.trim())
        .filter(Boolean);
      const ids = manual.length >= 2 ? manual : selected;
      return compareRuns(ids);
    },
  });

  const rows = runsQ.data?.items ?? [];
  const runMap = useMemo(() => {
    const out = new Map<string, RunItem>();
    for (const row of rows) out.set(row.run_id, row);
    return out;
  }, [rows]);
  const selectable = useMemo(() => rows.map((x) => x.run_id), [rows]);

  const runIds = compare.data?.run_ids ?? [];
  const compareRoot = useMemo(() => asRecord(compare.data?.compare), [compare.data?.compare]);
  const victimsRoot = useMemo(() => asRecord(compareRoot.victims), [compareRoot.victims]);
  const victimNames = useMemo(() => Object.keys(victimsRoot), [victimsRoot]);

  useEffect(() => {
    if (focusVictim && victimNames.includes(focusVictim)) return;
    setFocusVictim(victimNames[0] ?? "");
  }, [focusVictim, victimNames]);

  const perVictimPerRun = useMemo(() => {
    const out: Record<string, Record<string, MetricPoint>> = {};
    for (const victim of victimNames) {
      const node = asRecord(victimsRoot[victim]);
      const runsNode = asRecord(node.runs);
      const row: Record<string, MetricPoint> = {};
      for (const runId of runIds) {
        row[runId] = parseMetricPoint(runsNode[runId]);
      }
      out[victim] = row;
    }
    return out;
  }, [runIds, victimNames, victimsRoot]);

  const runAggregate = useMemo(() => {
    return runIds.map((runId) => {
      const vals = victimNames.map((v) => perVictimPerRun[v]?.[runId]).filter((x) => !!x) as MetricPoint[];
      const n = vals.length || 1;
      const sum = vals.reduce(
        (acc, x) => {
          acc.attack_drop += x.attack_drop;
          acc.attacked_recall += x.attacked_recall;
          acc.clean_recall += x.clean_recall;
          return acc;
        },
        { attack_drop: 0, attacked_recall: 0, clean_recall: 0 },
      );
      return {
        run_id: runId,
        attack_drop: sum.attack_drop / n,
        attacked_recall: sum.attacked_recall / n,
        clean_recall: sum.clean_recall / n,
      };
    });
  }, [perVictimPerRun, runIds, victimNames]);

  const victimAggregate = useMemo(() => {
    return victimNames.map((victim) => {
      const vals = runIds.map((r) => perVictimPerRun[victim]?.[r]).filter((x) => !!x) as MetricPoint[];
      const n = vals.length || 1;
      const sum = vals.reduce(
        (acc, x) => {
          acc.attack_drop += x.attack_drop;
          acc.attacked_recall += x.attacked_recall;
          acc.clean_recall += x.clean_recall;
          return acc;
        },
        { attack_drop: 0, attacked_recall: 0, clean_recall: 0 },
      );
      return {
        victim,
        attack_drop: sum.attack_drop / n,
        attacked_recall: sum.attacked_recall / n,
        clean_recall: sum.clean_recall / n,
      };
    });
  }, [perVictimPerRun, runIds, victimNames]);

  const runBarOpt = useMemo(() => {
    return {
      tooltip: { trigger: "axis" },
      legend: { data: METRIC_OPTIONS.map((item) => item.label) },
      xAxis: { type: "category", data: runAggregate.map((x) => shortRunId(x.run_id)) },
      yAxis: { type: "value" },
      series: [
        { name: metricLabel("attack_drop"), type: "bar", data: runAggregate.map((x) => x.attack_drop) },
        { name: metricLabel("attacked_recall"), type: "bar", data: runAggregate.map((x) => x.attacked_recall) },
        { name: metricLabel("clean_recall"), type: "bar", data: runAggregate.map((x) => x.clean_recall) },
      ],
    };
  }, [runAggregate]);

  const heatData = useMemo(() => {
    const rowsData: Array<[number, number, number]> = [];
    for (let yi = 0; yi < victimNames.length; yi += 1) {
      const victim = victimNames[yi];
      for (let xi = 0; xi < runIds.length; xi += 1) {
        const runId = runIds[xi];
        const val = perVictimPerRun[victim]?.[runId]?.[heatMetric] ?? 0;
        rowsData.push([xi, yi, val]);
      }
    }
    return rowsData;
  }, [heatMetric, perVictimPerRun, runIds, victimNames]);

  const heatMax = useMemo(() => {
    const maxAbs = heatData.reduce((m, x) => Math.max(m, Math.abs(x[2])), 0);
    return maxAbs > 0 ? maxAbs : 1;
  }, [heatData]);

  const heatOpt = useMemo(() => {
    return {
      tooltip: {
        formatter: (params: { value: [number, number, number] }) => {
          const [xi, yi, v] = params.value;
          const runId = runIds[xi] ?? "-";
          const victim = victimNames[yi] ?? "-";
          return `受测模型=${formatAdapterName(victim)}<br/>运行编号=${runId}<br/>${metricLabel(heatMetric)}=${Number(v).toFixed(4)}`;
        },
      },
      xAxis: { type: "category", data: runIds.map((x) => shortRunId(x)) },
      yAxis: { type: "category", data: victimNames.map((item) => formatAdapterName(item)) },
      visualMap: {
        min: -heatMax,
        max: heatMax,
        orient: "horizontal",
        left: "center",
        bottom: 0,
        calculable: true,
        inRange: { color: ["#c0262d", "#f8fafc", "#0f766e"] },
      },
      series: [
        {
          type: "heatmap",
          data: heatData,
          label: { show: true, formatter: (p: { value: [number, number, number] }) => Number(p.value[2]).toFixed(2) },
        },
      ],
    };
  }, [heatData, heatMax, heatMetric, runIds, victimNames]);

  const victimSeries = useMemo(() => {
    if (!focusVictim) return [];
    return runIds.map((r) => perVictimPerRun[focusVictim]?.[r] ?? { attack_drop: 0, attacked_recall: 0, clean_recall: 0 });
  }, [focusVictim, perVictimPerRun, runIds]);

  const victimLineOpt = useMemo(() => {
    return {
      tooltip: { trigger: "axis" },
      legend: { data: METRIC_OPTIONS.map((item) => item.label) },
      xAxis: { type: "category", data: runIds.map((x) => shortRunId(x)) },
      yAxis: { type: "value" },
      series: [
        { name: metricLabel("attack_drop"), type: "line", smooth: true, data: victimSeries.map((x) => x.attack_drop) },
        { name: metricLabel("attacked_recall"), type: "line", smooth: true, data: victimSeries.map((x) => x.attacked_recall) },
        { name: metricLabel("clean_recall"), type: "line", smooth: true, data: victimSeries.map((x) => x.clean_recall) },
      ],
    };
  }, [runIds, victimSeries]);

  const avg = useMemo(() => {
    const n = victimAggregate.length || 1;
    const sum = victimAggregate.reduce(
      (acc, x) => {
        acc.attack_drop += x.attack_drop;
        acc.attacked_recall += x.attacked_recall;
        acc.clean_recall += x.clean_recall;
        return acc;
      },
      { attack_drop: 0, attacked_recall: 0, clean_recall: 0 },
    );
    return {
      attack_drop: sum.attack_drop / n,
      attacked_recall: sum.attacked_recall / n,
      clean_recall: sum.clean_recall / n,
    };
  }, [victimAggregate]);

  const runRiskRows = useMemo(() => {
    return runIds.map((id) => {
      const meta = runMap.get(id);
      return {
        run_id: id,
        risk_score: asNum(meta?.risk_score),
        asr_attack: asNum(meta?.asr_attack ?? meta?.asr),
        perturbation_l2: asNum(meta?.avg_l2),
      };
    });
  }, [runIds, runMap]);

  const riskAvg = useMemo(() => {
    if (runRiskRows.length === 0) return 0;
    return runRiskRows.reduce((s, x) => s + x.risk_score, 0) / runRiskRows.length;
  }, [runRiskRows]);

  const runRiskOpt = useMemo(() => {
    return {
      tooltip: { trigger: "axis" },
      legend: { data: ["风险分数", "攻击成功率", "扰动指标"] },
      xAxis: { type: "category", data: runRiskRows.map((x) => shortRunId(x.run_id)) },
      yAxis: { type: "value", min: 0, max: 1 },
      series: [
        { name: "风险分数", type: "bar", data: runRiskRows.map((x) => x.risk_score) },
        { name: "攻击成功率", type: "line", smooth: true, data: runRiskRows.map((x) => x.asr_attack) },
        { name: "扰动指标", type: "line", smooth: true, data: runRiskRows.map((x) => x.perturbation_l2) },
      ],
    };
  }, [runRiskRows]);

  return (
    <div className="space-y-6">
      <section className="section-card p-6">
        <h2 className="section-title">实验对比</h2>
        <p className="section-subtitle">不再只看 JSON，直接查看运行级和受测模型级的攻击对照图。</p>

        <div className="mt-3 grid gap-3 md:grid-cols-2">
          <div className="space-y-2">
            <div className="text-sm text-[var(--ink-soft)]">快速选择（至少 2 个）</div>
            <div className="max-h-56 overflow-auto rounded-xl border border-line p-2">
              {selectable.map((id) => (
                <label key={id} className="flex items-center gap-2 py-1 text-sm">
                  <input
                    type="checkbox"
                    checked={selected.includes(id)}
                    onChange={(e) => {
                      setSelected((prev) => (e.target.checked ? [...prev, id] : prev.filter((x) => x !== id)));
                    }}
                  />
                  <span className="font-mono">{id}</span>
                </label>
              ))}
            </div>
          </div>

          <div className="space-y-2">
            <div className="text-sm text-[var(--ink-soft)]">或手动输入运行编号（逗号分隔）</div>
            <textarea
              rows={8}
              className="w-full rounded-xl border border-line bg-[var(--panel-strong)] px-3 py-2 font-mono text-sm"
              value={manualIds}
              onChange={(e) => setManualIds(e.target.value)}
              placeholder="20260216_...,20260216_..."
            />
          </div>
        </div>

        <div className="mt-4">
          <button
            className="rounded-xl bg-accent px-5 py-2 text-white disabled:opacity-50"
            disabled={compare.isPending || (manualIds.trim().length === 0 && selected.length < 2)}
            onClick={() => compare.mutate()}
          >
            生成对比
          </button>
          {compare.error ? <div className="mt-2 text-sm text-rose-700">对比失败：{String((compare.error as Error).message ?? compare.error)}</div> : null}
        </div>
      </section>

      {compare.data ? (
        <>
          <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
            <div className="section-card p-4">
              <div className="text-sm text-[var(--ink-soft)]">运行数量</div>
              <div className="mt-1 text-2xl font-semibold">{runIds.length}</div>
            </div>
            <div className="section-card p-4">
              <div className="text-sm text-[var(--ink-soft)]">受测模型数量</div>
              <div className="mt-1 text-2xl font-semibold">{victimNames.length}</div>
            </div>
            <div className="section-card p-4">
              <div className="text-sm text-[var(--ink-soft)]">平均攻击降幅</div>
              <div className="mt-1 text-2xl font-semibold">{avg.attack_drop.toFixed(4)}</div>
            </div>
            <div className="section-card p-4">
              <div className="text-sm text-[var(--ink-soft)]">平均攻击后召回率</div>
              <div className="mt-1 text-2xl font-semibold">{avg.attacked_recall.toFixed(4)}</div>
            </div>
            <div className="section-card p-4">
              <div className="text-sm text-[var(--ink-soft)]">平均风险分数</div>
              <div className="mt-1 text-2xl font-semibold">{riskAvg.toFixed(4)}</div>
            </div>
          </section>

          <section className="section-card p-5">
            <h3 className="text-xl font-semibold">运行级攻击对照（受测模型均值）</h3>
            <div className="mt-2 text-sm text-[var(--ink-soft)]">每个运行的攻击降幅、攻击后召回率和原始召回率平均值。</div>
            <ReactECharts option={runBarOpt} style={{ height: 340 }} />
          </section>

          <section className="section-card p-5">
            <h3 className="text-xl font-semibold">运行综合风险评分</h3>
            <div className="mt-2 text-sm text-[var(--ink-soft)]">将风险总分与攻击成功率、扰动指标放在同一图里快速识别高风险运行。</div>
            <ReactECharts option={runRiskOpt} style={{ height: 320 }} />
          </section>

          <section className="grid gap-4 xl:grid-cols-2">
            <div className="section-card p-5">
              <div className="flex items-center justify-between gap-3">
                <h3 className="text-xl font-semibold">受测模型 × 运行热力图</h3>
                <select
                  className="rounded-xl border border-line bg-[var(--panel-strong)] px-3 py-2 text-sm"
                  value={heatMetric}
                  onChange={(e) => setHeatMetric(e.target.value as MetricKey)}
                >
                  {METRIC_OPTIONS.map((item) => (
                    <option key={item.value} value={item.value}>
                      {item.label}
                    </option>
                  ))}
                </select>
              </div>
              <div className="mt-2 text-sm text-[var(--ink-soft)]">红色偏高，绿色偏低，用于看不同受测模型在不同运行里的差异。</div>
              <ReactECharts option={heatOpt} style={{ height: 420 }} />
            </div>

            <div className="section-card p-5">
              <div className="flex items-center justify-between gap-3">
                <h3 className="text-xl font-semibold">单模型趋势</h3>
                <select
                  className="rounded-xl border border-line bg-[var(--panel-strong)] px-3 py-2 text-sm"
                  value={focusVictim}
                  onChange={(e) => setFocusVictim(e.target.value)}
                >
                  {victimNames.map((v) => (
                    <option key={v} value={v}>
                      {formatAdapterName(v)}
                    </option>
                  ))}
                </select>
              </div>
              <div className="mt-2 text-sm text-[var(--ink-soft)]">观察同一受测模型在不同运行下的攻击指标变化。</div>
              <ReactECharts option={victimLineOpt} style={{ height: 420 }} />
            </div>
          </section>

          <section className="section-card p-5">
            <h3 className="text-xl font-semibold">受测模型指标表</h3>
            <div className="mt-3 overflow-x-auto">
              <table className="w-full att-detail-table">
                <thead>
                  <tr className="text-left text-[var(--ink-soft)]">
                    <th>受测模型</th>
                    <th>平均攻击降幅</th>
                    <th>平均攻击后召回率</th>
                    <th>平均原始召回率</th>
                  </tr>
                </thead>
                <tbody>
                  {victimAggregate.map((row) => (
                    <tr key={row.victim} className="border-t border-line">
                      <td>{formatAdapterName(row.victim)}</td>
                      <td>{row.attack_drop.toFixed(4)}</td>
                      <td>{row.attacked_recall.toFixed(4)}</td>
                      <td>{row.clean_recall.toFixed(4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          <section className="section-card p-5">
            <h3 className="text-xl font-semibold">运行元信息</h3>
            <div className="mt-3 overflow-x-auto">
              <table className="w-full att-detail-table">
                <thead>
                  <tr className="text-left text-[var(--ink-soft)]">
                    <th>运行编号</th>
                    <th>任务</th>
                    <th>攻击</th>
                    <th>风险分数</th>
                    <th>风险等级</th>
                    <th>实验编号</th>
                    <th>数据集</th>
                  </tr>
                </thead>
                <tbody>
                  {runIds.map((id) => {
                    const meta = runMap.get(id);
                    return (
                      <tr key={id} className="border-t border-line">
                        <td className="font-mono text-sm">{id}</td>
                        <td>{formatJobType(String(meta?.task_kind ?? "-"))}</td>
                        <td>{formatAttackName(String(meta?.attack ?? "-"))}</td>
                        <td>{asNum(meta?.risk_score).toFixed(4)}</td>
                        <td>{formatRiskLevel(String(meta?.risk_level ?? "-"))}</td>
                        <td>{formatWrapped("实验编号", String(meta?.experiment_id ?? "-"))}</td>
                        <td>{formatDatasetName(String(meta?.dataset_name ?? "-"))}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </section>

        </>
      ) : null}
    </div>
  );
}
