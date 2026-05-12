import ReactECharts from "echarts-for-react";

import { GlossaryLink } from "./GlossaryLink";
import { useDismissible } from "../hooks/useDismissible";

function asRecord(value: unknown): Record<string, unknown> {
  return typeof value === "object" && value !== null ? (value as Record<string, unknown>) : {};
}

function asRows(value: unknown): Array<Record<string, unknown>> {
  return Array.isArray(value) ? value.map((item) => asRecord(item)) : [];
}

function asNum(value: unknown) {
  const n = Number(value);
  return Number.isFinite(n) ? n : 0;
}

type RunInsightChartsProps = {
  runId?: string;
  summary?: Record<string, unknown>;
  reportData?: Record<string, unknown>;
  cases?: Record<string, unknown>[];
  panelId?: string;
};

export function RunInsightCharts({ runId, summary = {}, reportData = {}, cases = [], panelId }: RunInsightChartsProps) {
  const { visible, dismiss, restore } = useDismissible(panelId);

  if (panelId && !visible) {
    return (
      <section className="section-card p-5 dismissible-panel dismissible-panel-restore">
        <span>结果图表已隐藏</span>
        <button type="button" className="panel-restore-button" onClick={restore}>
          显示结果图表
        </button>
      </section>
    );
  }

  if (!runId) {
    return (
      <section className={`section-card p-5 ${panelId ? "dismissible-panel" : ""}`}>
        {panelId ? (
          <button type="button" className="panel-close-button" aria-label="关闭结果图表" title="关闭结果图表" onClick={dismiss}>
            ×
          </button>
        ) : null}
        <div className="panel-label">
          <GlossaryLink entryId="section-result-insights">结果解读</GlossaryLink>
        </div>
        <div className="mt-3 text-sm text-[var(--ink-soft)]">任务完成后，这里会自动切换到最新运行并显示结果图表。</div>
      </section>
    );
  }

  const asrAttack = asNum(summary.asr_attack ?? summary.asr);
  const riskScore = asNum(summary.risk_score);
  const avgL2 = asNum(summary.avg_l2);

  const victimsRoot = asRecord(summary.victims);
  const modelRows = Object.entries(victimsRoot).map(([name, raw]) => {
    const clean = asRecord(asRecord(raw).clean);
    const attacked = asRecord(asRecord(raw).attacked);
    return {
      name,
      attackAsr: asNum(attacked["ir_asr@1"]),
      rankDrop: asNum(attacked.mean_rank_ir) - asNum(clean.mean_rank_ir),
    };
  });

  const failureCases = asRows(asRecord(reportData.vlr).failure_cases).slice(0, 20);
  const failureCount = failureCases.filter((item) => !Boolean(item.judge_success)).length;
  const successCount = Math.max(0, failureCases.length - failureCount);

  const stageMetrics = asRecord(reportData.stage_metrics);
  const attackedStage = asRecord(stageMetrics.attacked);
  const cleanStage = asRecord(stageMetrics.clean);

  const averageStageValue = (node: Record<string, unknown>, key: string) => {
    const values = Object.values(node)
      .map((value) => asNum(asRecord(value)[key]))
      .filter((value) => Number.isFinite(value));
    if (!values.length) return 0;
    return values.reduce((sum, value) => sum + value, 0) / values.length;
  };

  const metricOverviewOpt = {
    tooltip: { trigger: "axis" },
    xAxis: { type: "category", data: ["攻击成功率", "风险分数", "平均二范数"] },
    yAxis: { type: "value", min: 0 },
    series: [{ type: "bar", data: [asrAttack, riskScore, avgL2], itemStyle: { color: "#1768ff" } }],
  };

  const stageCompareOpt = {
    tooltip: { trigger: "axis" },
    xAxis: { type: "category", data: ["正常输入", "攻击后"] },
    yAxis: { type: "value", min: 0, max: 1 },
    series: [
      {
        type: "bar",
        data: [averageStageValue(cleanStage, "ir_r@1"), averageStageValue(attackedStage, "ir_asr@1")],
        itemStyle: { color: "#1bbcff" },
      },
    ],
  };

  const modelDiffOpt = {
    tooltip: { trigger: "axis" },
    legend: { data: ["攻击后攻击成功率（ASR）", "平均排名下降"] },
    xAxis: { type: "category", data: modelRows.map((item) => item.name) },
    yAxis: { type: "value" },
    series: [
      { name: "攻击后攻击成功率（ASR）", type: "bar", data: modelRows.map((item) => item.attackAsr) },
      { name: "平均排名下降", type: "line", smooth: true, data: modelRows.map((item) => item.rankDrop) },
    ],
  };

  const sampleDistributionOpt = {
    tooltip: { trigger: "item" },
    series: [
      {
        type: "pie",
        radius: ["45%", "72%"],
        data: [
          { value: successCount, name: "成功样本" },
          { value: failureCount, name: "失败样本" },
        ],
      },
    ],
  };

  return (
    <section className={`section-card p-5 ${panelId ? "dismissible-panel" : ""}`}>
      {panelId ? (
        <button type="button" className="panel-close-button" aria-label="关闭结果图表" title="关闭结果图表" onClick={dismiss}>
          ×
        </button>
      ) : null}
      <div className="workspace-header">
        <div>
          <div className="panel-label">
            <GlossaryLink entryId="section-result-insights">结果解读</GlossaryLink>
          </div>
          <h3 className="section-title mt-2">运行结果图表 | {runId}</h3>
          <div className="section-subtitle">这里只保留答辩最常用的四类图：整体结论、阶段对比、模型差异和样本分布。</div>
        </div>
      </div>

      <div className="mt-5 grid gap-4 xl:grid-cols-2">
        <div className="surface-soft">
          <div className="font-semibold text-[var(--ink)]">
            <GlossaryLink entryId="chart-metric-overview">指标总览图</GlossaryLink>
          </div>
          <div className="mt-2 text-sm text-[var(--ink-soft)]">先看这次运行的整体风险结论。</div>
          <ReactECharts option={metricOverviewOpt} style={{ height: 300 }} />
        </div>

        <div className="surface-soft">
          <div className="font-semibold text-[var(--ink)]">
            <GlossaryLink entryId="chart-stage-compare">阶段对比图</GlossaryLink>
          </div>
          <div className="mt-2 text-sm text-[var(--ink-soft)]">对比正常输入和攻击后输入的表现变化。</div>
          <ReactECharts option={stageCompareOpt} style={{ height: 300 }} />
        </div>

        <div className="surface-soft">
          <div className="font-semibold text-[var(--ink)]">
            <GlossaryLink entryId="chart-model-difference">模型差异图</GlossaryLink>
          </div>
          <div className="mt-2 text-sm text-[var(--ink-soft)]">回答“这次攻击主要对哪些受测模型更有效”。</div>
          <ReactECharts option={modelDiffOpt} style={{ height: 320 }} />
        </div>

        <div className="surface-soft">
          <div className="font-semibold text-[var(--ink)]">
            <GlossaryLink entryId="chart-sample-distribution">样本分布图</GlossaryLink>
          </div>
          <div className="mt-2 text-sm text-[var(--ink-soft)]">把整体指标落回到成功样本和失败样本数量。</div>
          <ReactECharts option={sampleDistributionOpt} style={{ height: 320 }} />
        </div>
      </div>

      {cases.length ? (
        <div className="mt-5 rounded-2xl border border-line bg-[var(--panel-strong)] p-4 text-sm text-[var(--ink-soft)]">
          当前结果加载了 {cases.length} 条样本索引记录。页面只展示图表摘要，详细样本放在独立结果接口里。
        </div>
      ) : null}
    </section>
  );
}
