// 文件说明：该文件属于前端页面，集中实现 GlossaryPage 相关逻辑。
import { useEffect, useMemo, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";

import { DismissiblePanel } from "../components/DismissiblePanel";
import { GlossaryFormula } from "../components/GlossaryFormula";
import { GlossaryLink } from "../components/GlossaryLink";
import { getGlossaryEntry, glossaryCategoryLabel, glossaryEntries, glossaryGroupOrder, glossarySearchText, type GlossaryEntry } from "../lib/glossaryRegistry";

const CURATED_ENTRY_IDS = new Set([
  "platform-main",
  "page-dashboard",
  "page-testing",
  "page-glossary",
  "section-attack-matrix",
  "section-model-matrix",
  "section-dataset-matrix",
  "section-latest-results",
  "section-parameter-settings",
  "section-result-insights",
  "workflow-job-progress",
  "workflow-task-launch",
  "workflow-surrogate-model",
  "workflow-current-stage",
  "workflow-queue-position",
  "workflow-run-id",
  "workflow-victim-model",
  "mode-standard-eval",
  "parameter-epsilon",
  "parameter-step-size",
  "parameter-steps",
  "parameter-patch-size",
  "parameter-text-budget",
  "parameter-max-pairs",
  "metric-validated-model-count",
  "metric-elapsed-time",
  "metric-eta-time",
  "metric-estimated-ready-at",
  "metric-asr",
  "metric-risk-score",
  "chart-metric-overview",
  "chart-stage-compare",
  "chart-model-difference",
  "chart-sample-distribution",
  "status-online",
  "status-ready",
  "attack-advclip",
  "attack-tmm",
  "attack-advedm",
  "attack-advedm-plus",
  "attack-fgsm",
  "attack-bim",
  "attack-pgd",
  "attack-mifgsm",
  "attack-nifgsm",
  "attack-difgsm",
  "attack-tifgsm",
  "attack-dtmifgsm",
  "attack-vmifgsm",
  "attack-vnifgsm",
  "attack-cw",
  "model-clip",
  "model-blip",
  "model-vilt",
  "model-qwen35-9b",
  "model-qwen3-vl",
  "model-qwen25-vl",
  "model-internvl35",
  "model-minicpm-v",
  "model-ovis25",
  "model-gemma3-12b",
  "dataset-coco-subset",
  "dataset-flickr30k",
  "dataset-flickr1k",
  "dataset-vqa-v2-coco-val",
  "dataset-coco-object-probe-val",
  "dataset-coco-caption-object-val",
  "dataset-mini-flickr",
]);

/** 封装 `useActiveAnchor` Hook，把页面状态、副作用和持久化逻辑集中管理。 */
function useActiveAnchor() {
  const location = useLocation();
  const [activeAnchor, setActiveAnchor] = useState("");

  useEffect(() => {
    const hash = location.hash.replace(/^#/, "");
    if (!hash) return;
    const timer = window.setTimeout(() => {
      const node = document.getElementById(hash);
      if (node) {
        node.scrollIntoView({ behavior: "smooth", block: "start" });
        setActiveAnchor(hash);
        window.setTimeout(() => setActiveAnchor((current) => (current === hash ? "" : current)), 2800);
      }
    }, 80);
    return () => window.clearTimeout(timer);
  }, [location.hash]);

  return activeAnchor;
}

/** 整理 `short paragraphs` 前端辅助逻辑，保持数据转换和展示口径一致。 */
function shortParagraphs(entry: GlossaryEntry) {
  return entry.detailSections
    .slice(0, 1)
    .flatMap((section) => section.paragraphs.slice(0, 2));
}

/** 判断 `是否需要 render formula` 状态，支撑页面分支渲染或按钮可用性。 */
function shouldRenderFormula(entry: GlossaryEntry) {
  return entry.category === "metric" || entry.category === "parameter";
}

/** 渲染 `GlossaryEntryCard` 组件，组织该区域的数据读取、交互状态和可访问性标记。 */
function GlossaryEntryCard({ entry, active }: { entry: GlossaryEntry; active: boolean }) {
  const shortCopy = shortParagraphs(entry);
  const formulaBlocks = shouldRenderFormula(entry) ? entry.formulaBlocks ?? [] : [];

  return (
    <article id={entry.anchor} className={`glossary-entry ${active ? "glossary-entry-active" : ""}`}>
      <div className="workspace-header">
        <div>
          <div className="panel-label">{glossaryCategoryLabel[entry.category]}</div>
          <h3 className="section-title mt-2">{entry.label}</h3>
        </div>
        <div className="quick-link-row">
          <span className="tag-chip">{entry.shortLabel}</span>
          {entry.aliases.slice(0, 2).map((alias) => (
            <span key={`${entry.id}-${alias}`} className="tag-chip">
              {alias}
            </span>
          ))}
        </div>
      </div>

      <div className="mt-4 grid gap-3 xl:grid-cols-2">
        <div className="surface-soft">
          <div className="panel-label">出现页面</div>
          <div className="mt-2 flex flex-wrap gap-2">
            {entry.sourcePages.map((page) => (
              <span key={`${entry.id}-${page}`} className="tag-chip">
                {page === "dashboard" ? "00 总览" : page === "testing" ? "01 对抗测试" : page === "layout" ? "公共布局" : "02 术语页"}
              </span>
            ))}
          </div>
        </div>
        {entry.relatedIds?.length ? (
          <div className="surface-soft">
            <div className="panel-label">相关术语</div>
            <div className="mt-2 flex flex-wrap gap-2">
              {entry.relatedIds.map((id) => (
                <GlossaryLink key={`${entry.id}-${id}`} entryId={id} className="tag-chip">
                  {getGlossaryEntry(id)?.label ?? id}
                </GlossaryLink>
              ))}
            </div>
          </div>
        ) : null}
      </div>

      <div className="mt-5 space-y-4">
        <section className="surface-soft">
          <div className="glossary-subtitle">答辩页保留原因</div>
          <div className="glossary-copy">
            {shortCopy.map((line) => (
              <p key={`${entry.id}-${line}`}>{line}</p>
            ))}
            {!shortCopy.length ? <p>这个词条用于解释主页面里出现的专有名词，避免现场口头补定义。</p> : null}
          </div>
        </section>

        {formulaBlocks.map((block) => (
          <GlossaryFormula key={`${entry.id}-${block.title}`} block={block} />
        ))}
      </div>
    </article>
  );
}

/** 渲染 `GlossaryPage` 组件，组织该区域的数据读取、交互状态和可访问性标记。 */
export default function GlossaryPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const activeAnchor = useActiveAnchor();
  const [keyword, setKeyword] = useState("");

  const filteredEntries = useMemo(() => {
    const query = keyword.trim().toLowerCase();
    const visibleEntries = glossaryEntries.filter((entry) => CURATED_ENTRY_IDS.has(entry.id));
    if (!query) return visibleEntries;
    return visibleEntries.filter((entry) => glossarySearchText(entry).includes(query));
  }, [keyword]);

  const groups = useMemo(
    () =>
      glossaryGroupOrder
        .map((group) => ({
          group,
          label: glossaryCategoryLabel[group],
          items: filteredEntries.filter((entry) => entry.category === group),
        }))
        .filter((group) => group.items.length > 0),
    [filteredEntries],
  );

  return (
    <div className="space-y-6">
      <DismissiblePanel id="glossary-hero" label="术语页导语" className="section-card">
        <div className="workspace-header">
          <div>
            <div className="eyebrow">术语说明</div>
            <h2 className="section-title mt-3">术语与指标详解</h2>
            <div className="section-subtitle">这里只保留总览页和测试页里真正出现的术语说明，作为答辩时的备用页面。</div>
          </div>
          <button className="action-button action-button-secondary" onClick={() => navigate(-1)}>
            返回上一页
          </button>
        </div>
      </DismissiblePanel>

      <DismissiblePanel id="glossary-index" label="术语目录" className="section-card p-5">
        <div className="workspace-header">
          <div>
            <div className="panel-label">术语目录</div>
            <h3 className="section-title mt-2">搜索与快速定位</h3>
          </div>
          <span className="tag-chip">共 {filteredEntries.length} 项</span>
        </div>
        <div className="mt-4">
          <label className="panel-label" htmlFor="glossary-search">搜索术语</label>
          <input id="glossary-search" name="glossarySearch" className="mt-2" value={keyword} onChange={(event) => setKeyword(event.target.value)} placeholder="输入攻击方法、模型、数据集、指标或参数名" />
        </div>
        <div className="mt-5 space-y-4">
          {groups.map((group) => (
            <section key={group.group} className="surface-soft">
              <div className="glossary-subtitle">{group.label}</div>
              <div className="mt-3 flex flex-wrap gap-2">
                {group.items.map((entry) => (
                  <GlossaryLink key={`${group.group}-${entry.id}`} entryId={entry.id} className={location.hash === `#${entry.anchor}` ? "glossary-link glossary-link-active" : undefined}>
                    {entry.label}
                  </GlossaryLink>
                ))}
              </div>
            </section>
          ))}
        </div>
      </DismissiblePanel>

      <div className="space-y-5">
        {groups.map((group) => (
          <DismissiblePanel key={`detail-${group.group}`} id={`glossary-group-${group.group}`} label={group.label} className="section-card p-5">
            <div className="panel-label">{group.label}</div>
            <div className="mt-4 grid gap-5 xl:grid-cols-2">
              {group.items.map((entry) => (
                <GlossaryEntryCard key={entry.id} entry={entry} active={activeAnchor === entry.anchor} />
              ))}
            </div>
          </DismissiblePanel>
        ))}
      </div>
    </div>
  );
}
