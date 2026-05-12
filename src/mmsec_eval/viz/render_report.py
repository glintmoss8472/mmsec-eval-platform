# 文件说明：该文件属于报告可视化层，集中实现 render report 相关逻辑。
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


# 汇总 `摘要 block`，从运行记录和指标中提炼页面展示所需的分析结果。
def _summary_block(summary: dict[str, Any]) -> str:
    items = "".join(
        f"<div class='kpi'><b>{html.escape(str(k))}</b>: {html.escape(str(v))}</div>"
        for k, v in summary.items()
    )
    pretty = html.escape(json.dumps(summary, ensure_ascii=False, indent=2))
    return f"<h2>Summary</h2><div>{items}</div><pre>{pretty}</pre>"


# 整理 `rows table` 行记录，把原始结果转换成列表接口和报告可消费的结构。
def _rows_table(rows: list[dict[str, Any]], n: int = 20) -> str:
    head = rows[:n]
    if not head:
        return "<p>No results.</p>"
    keys = sorted({k for row in head for k in row.keys() if k != "artifact_refs"})
    th = "".join(f"<th>{html.escape(k)}</th>" for k in keys)
    tr = []
    for row in head:
        tr.append("<tr>" + "".join(f"<td>{html.escape(str(row.get(k, '')))}</td>" for k in keys) + "</tr>")
    return "<table><thead><tr>" + th + "</tr></thead><tbody>" + "".join(tr) + "</tbody></table>"


# 执行 `检索 k values` 辅助逻辑，保持报告可视化层中的输入处理和结果输出一致。
def _retrieval_k_values(summary: dict[str, Any]) -> list[int]:
    ks = summary.get("retrieval_k", [1, 5, 10])
    ks2 = [int(x) for x in ks if int(x) > 0]
    if not ks2:
        ks2 = [1, 5, 10]
    return ks2


# 标记 `均值` 阶段，区分 clean、attacked 和 defended 样本。
def _vlr_victim_table(victims: dict[str, Any], ks2: list[int]) -> str:
    # 封装 _stage_mean 的内部步骤，让报告可视化层主流程保持清晰并隔离边界细节。
    def _stage_mean(stage: dict[str, Any], k: int) -> float:
        ir = float(stage.get(f"ir_r@{k}", 0.0))
        trr = float(stage.get(f"tr_r@{k}", 0.0))
        return 0.5 * (ir + trr)

    head = ["victim", "status"]
    for k in ks2:
        head.append(f"Recall at {k} clean")
    for k in ks2:
        head.append(f"Recall at {k} attacked")
    head += ["conditional ASR at 1", "mean_rank clean", "mean_rank attacked"]

    th = "".join(f"<th>{html.escape(x)}</th>" for x in head)
    tr: list[str] = []
    for victim, payload in sorted(victims.items()):
        node = payload if isinstance(payload, dict) else {}
        clean = node.get("clean", {}) if isinstance(node.get("clean"), dict) else {}
        attacked = node.get("attacked", {}) if isinstance(node.get("attacked"), dict) else {}
        conditional = node.get("conditional", {}) if isinstance(node.get("conditional"), dict) else {}
        status = node.get("status", {}) if isinstance(node.get("status"), dict) else {}
        status_txt = f"clean={status.get('clean', '-')}, attacked={status.get('attacked', '-')}"

        cells = [html.escape(str(victim)), html.escape(status_txt)]
        for k in ks2:
            cells.append(f"{_stage_mean(clean, k):.4f}")
        for k in ks2:
            cells.append(f"{_stage_mean(attacked, k):.4f}")
        cond_asr1 = 0.5 * (
            float(conditional.get("ir_cond_asr@1", 0.0) or 0.0)
            + float(conditional.get("tr_cond_asr@1", 0.0) or 0.0)
        )
        cells.append(f"{cond_asr1:.4f}")
        mr_c = 0.5 * (float(clean.get("mean_rank_ir", 0.0)) + float(clean.get("mean_rank_tr", 0.0)))
        mr_a = 0.5 * (float(attacked.get("mean_rank_ir", 0.0)) + float(attacked.get("mean_rank_tr", 0.0)))
        cells.append(f"{mr_c:.3f}")
        cells.append(f"{mr_a:.3f}")
        tr.append("<tr>" + "".join(f"<td>{x}</td>" for x in cells) + "</tr>")

    return "<table><thead><tr>" + th + "</tr></thead><tbody>" + "".join(tr) + "</tbody></table>"


# 执行 `图文检索 failure 案例` 辅助逻辑，保持报告可视化层中的输入处理和结果输出一致。
def _vlr_failure_cases(rows: list[dict[str, Any]]) -> str:
    fail_rows = [r for r in rows if str(r.get("query_type", "")) == "t2i" and not bool(r.get("judge_success", True))]
    fail_rows = fail_rows[:10]
    if not fail_rows:
        return ""
    fail_head = (
        "<tr><th>受测模型（victim model）</th><th>文本编号（text id）</th>"
        "<th>真实图像编号（ground-truth image id）</th><th>前五图像编号（top five image ids）</th></tr>"
    )
    fail_body = ""
    for r in fail_rows:
        top5 = r.get("top5_image_ids", [])
        top5_txt = ", ".join(str(x) for x in top5[:5]) if isinstance(top5, list) else str(top5)
        fail_body += (
            "<tr>"
            f"<td>{html.escape(str(r.get('victim', '')))}</td>"
            f"<td>{html.escape(str(r.get('text_id', '')))}</td>"
            f"<td>{html.escape(str(r.get('gt_image_id', '')))}</td>"
            f"<td>{html.escape(top5_txt)}</td>"
            "</tr>"
        )
    return "<h3>典型失败案例（text-to-image top five failures）</h3><table><thead>" + fail_head + "</thead><tbody>" + fail_body + "</tbody></table>"


# 执行 `semantic preservation table` 辅助逻辑，保持报告可视化层中的输入处理和结果输出一致。
def _semantic_preservation_table(summary: dict[str, Any]) -> str:
    semantic = summary.get("semantic_preservation", {})
    if not isinstance(semantic, dict) or not semantic:
        return ""
    return (
        "<h3>语义保持程度（semantic preservation）</h3>"
        "<table><thead><tr><th>综合分数（combined）</th><th>图像-图像相似度（image-image similarity）</th><th>文本相似度（text similarity）</th><th>无穷范数代理值（L-infinity proxy）</th></tr></thead>"
        "<tbody><tr>"
        f"<td>{float(semantic.get('combined_semantic_preservation', 0.0) or 0.0):.4f}</td>"
        f"<td>{html.escape(str(semantic.get('clip_image_image_similarity', '')))}</td>"
        f"<td>{html.escape(str(semantic.get('text_similarity', '')))}</td>"
        f"<td>{float(semantic.get('pixel_linf_preservation_proxy', 0.0) or 0.0):.4f}</td>"
        "</tr></tbody></table>"
    )


# 执行 `object decision proxy table` 辅助逻辑，保持报告可视化层中的输入处理和结果输出一致。
def _object_decision_proxy_table(summary: dict[str, Any]) -> str:
    object_proxy = summary.get("object_decision_proxy", {})
    if not isinstance(object_proxy, dict) or not object_proxy.get("available"):
        return ""
    return (
        "<h3>Object-Level Decision Proxy</h3>"
        "<table><thead><tr><th>cases</th><th>clean present</th><th>attacked present</th>"
        "<th>flip rate</th><th>valid-wrong rate</th></tr></thead><tbody><tr>"
        f"<td>{int(object_proxy.get('num_cases', 0) or 0)}</td>"
        f"<td>{float(object_proxy.get('clean_present_rate', 0.0) or 0.0):.4f}</td>"
        f"<td>{float(object_proxy.get('attacked_present_rate', 0.0) or 0.0):.4f}</td>"
        f"<td>{float(object_proxy.get('decision_flip_rate', 0.0) or 0.0):.4f}</td>"
        f"<td>{float(object_proxy.get('valid_wrong_rate', 0.0) or 0.0):.4f}</td>"
        "</tr></tbody></table>"
    )


# 执行 `图文检索 block` 辅助逻辑，保持报告可视化层中的输入处理和结果输出一致。
def _vlr_block(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    if str(summary.get("task_kind", "")).lower() != "vlr":
        return ""
    victims = summary.get("victims", {})
    if not isinstance(victims, dict) or not victims:
        return ""

    return (
        "<h2>VLR Metrics</h2>"
        + _vlr_victim_table(victims, _retrieval_k_values(summary))
        + _semantic_preservation_table(summary)
        + _object_decision_proxy_table(summary)
        + _vlr_failure_cases(rows)
    )


# 执行 `plots block` 辅助逻辑，保持报告可视化层中的输入处理和结果输出一致。
def _plots_block(run_dir: str) -> str:
    imgs = []
    names = [
        "metric_curve_l2.png",
        "metric_curve_linf.png",
        "asr_bar.png",
        "attack_compare_bar.png",
        "stage_compare_asr.png",
        "vlr_stage_compare_asr.png",
        "vlr_defense_recovery_curve.png",
    ]
    base = Path(run_dir)
    names += [p.relative_to(base).as_posix() for p in sorted(base.glob("vlr_*.png"))]
    # AdvCLIP patch training preview (if present).
    names += [p.relative_to(base).as_posix() for p in sorted(base.glob("attack_debug/advclip_patch_preview.png"))]

    seen = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        p = Path(run_dir) / name
        if p.exists():
            imgs.append(f"<div><img src='{name}' style='max-width: 48%; margin-right: 8px;' /></div>")
    if not imgs:
        return ""
    return "<h2>Plots</h2>" + "".join(imgs)


# 执行 `mode table` 辅助逻辑，保持报告可视化层中的输入处理和结果输出一致。
def _mode_table(rows: list[dict[str, Any]]) -> str:
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        attack = str(row.get("attack_name") or row.get("attack") or "unknown")
        mode = str(row.get("attack_mode") or "A")
        grouped.setdefault((attack, mode), []).append(1.0 if bool(row.get("judge_success", False)) else 0.0)

    if not grouped:
        return ""

    lines = ["<h2>A/B Mode Comparison</h2>", "<table><thead><tr><th>attack</th><th>mode</th><th>count</th><th>asr</th></tr></thead><tbody>"]
    for (attack, mode), vals in sorted(grouped.items()):
        asr = sum(vals) / max(1, len(vals))
        lines.append(
            f"<tr><td>{html.escape(attack)}</td><td>{html.escape(mode)}</td><td>{len(vals)}</td><td>{asr:.4f}</td></tr>"
        )
    lines.append("</tbody></table>")
    return "".join(lines)


# 定位 `安全 rel 路径`，把配置值或请求上下文转换成实际文件系统路径。
def _safe_rel_path(path: str, run_dir: str) -> str:
    try:
        p = Path(path)
        return str(p.relative_to(Path(run_dir))).replace("\\", "/")
    except (OSError, ValueError):
        return path.replace("\\", "/")


# 执行 `样本 panel` 辅助逻辑，保持报告可视化层中的输入处理和结果输出一致。
def _sample_panel(rows: list[dict[str, Any]], run_dir: str, n: int = 6) -> str:
    head = rows[:n]
    if not head:
        return ""

    cards: list[str] = []
    for row in head:
        refs = row.get("artifact_refs") or {}
        clean_img = refs.get("clean_image", "")
        adv_img = refs.get("adv_image", "")
        clean_src = _safe_rel_path(clean_img, run_dir) if clean_img else ""
        adv_src = _safe_rel_path(adv_img, run_dir) if adv_img else ""
        diagnostics = row.get("diagnostics") or {}
        cards.append(
            "".join(
                [
                    "<div style='border:1px solid #ddd;padding:10px;margin:8px 0;'>",
                    f"<div><b>{html.escape(str(row.get('sample_id', '')))}</b></div>",
                    f"<div>clean: {html.escape(str(row.get('clean_text', '')))}</div>",
                    f"<div>adv: {html.escape(str(row.get('adv_text', '')))}</div>",
                    f"<div>text_diff_score: {html.escape(str(diagnostics.get('text_diff_score', 0.0)))}</div>",
                    f"<div>embedding_shift: {html.escape(str(diagnostics.get('embedding_shift', 0.0)))}</div>",
                    (
                        f"<div><img src='{clean_src}' style='max-width: 180px; margin-right:8px;' />"
                        f"<img src='{adv_src}' style='max-width: 180px;' /></div>"
                        if clean_src and adv_src
                        else ""
                    ),
                    "</div>",
                ]
            )
        )
    return "<h2>Clean/Adv Sample Panel</h2>" + "".join(cards)


# 执行 `防御 block` 辅助逻辑，保持报告可视化层中的输入处理和结果输出一致。
def _defense_block(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    if not bool(summary.get("defense_enabled", False)):
        return ""
    asr_attack = float(summary.get("asr_attack", summary.get("asr", 0.0)) or 0.0)
    asr_defended = float(summary.get("asr_defended", summary.get("asr", 0.0)) or 0.0)
    defense_gain = float(summary.get("defense_gain", 0.0) or 0.0)
    utility_text = float(summary.get("clean_utility_text_diff", 0.0) or 0.0)
    utility_emb = float(summary.get("clean_utility_embedding_shift", 0.0) or 0.0)

    head = (
        "<h2>Defense Compare</h2>"
        "<table><thead><tr><th>asr_attack</th><th>asr_defended</th><th>defense_gain</th>"
        "<th>clean_utility_text_diff</th><th>clean_utility_embedding_shift</th></tr></thead>"
        "<tbody>"
        f"<tr><td>{asr_attack:.4f}</td><td>{asr_defended:.4f}</td><td>{defense_gain:.4f}</td>"
        f"<td>{utility_text:.4f}</td><td>{utility_emb:.4f}</td></tr>"
        "</tbody></table>"
    )

    recovered = [r for r in rows if bool(r.get("recovery_success", False))]
    failed = [r for r in rows if bool(r.get("judge_success_attack", r.get("judge_success", False))) and not bool(r.get("judge_success_defended", False))]
    recovered = recovered[:5]
    failed = failed[:5]

    # 执行 `mk 案例 table` 辅助逻辑，保持报告可视化层中的输入处理和结果输出一致。
    def _mk_case_table(title: str, case_rows: list[dict[str, Any]]) -> str:
        if not case_rows:
            return f"<h3>{html.escape(title)}</h3><p>暂无样本（none）。</p>"
        tr = "".join(
            "<tr>"
            f"<td>{html.escape(str(x.get('sample_id', '')))}</td>"
            f"<td>{html.escape(str(x.get('judge_success_attack', x.get('judge_success', ''))))}</td>"
            f"<td>{html.escape(str(x.get('judge_success_defended', '')))}</td>"
            f"<td>{html.escape(str(x.get('defense_gain_sample', '')))}</td>"
            "</tr>"
            for x in case_rows
        )
        return (
            f"<h3>{html.escape(title)}</h3>"
            "<table><thead><tr><th>样本编号（sample id）</th><th>攻击是否成功（attack success）</th><th>防御是否成功（defended success）</th><th>防御收益（defense gain）</th></tr></thead>"
            f"<tbody>{tr}</tbody></table>"
        )

    return head + _mk_case_table("典型恢复案例（top recovered cases）", recovered) + _mk_case_table("典型失败案例（top failed cases）", failed)


# 执行 `reproduction card` 辅助逻辑，保持报告可视化层中的输入处理和结果输出一致。
def _reproduction_card(summary: dict[str, Any]) -> str:
    fidelity = summary.get("reproduction_fidelity", {})
    if not isinstance(fidelity, dict) or not fidelity:
        return ""
    rows = "".join(
        f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>"
        for k, v in fidelity.items()
    )
    return "<h2>Reproduction Fidelity</h2><table><thead><tr><th>module</th><th>status</th></tr></thead><tbody>" + rows + "</tbody></table>"


# 执行 `风险 block` 辅助逻辑，保持报告可视化层中的输入处理和结果输出一致。
def _risk_block(summary: dict[str, Any]) -> str:
    score = float(summary.get("risk_score", 0.0) or 0.0)
    level = str(summary.get("risk_level", ""))
    scenario = str(summary.get("risk_scenario", ""))
    breakdown = summary.get("risk_breakdown", {})
    weights = summary.get("risk_weights", {})
    if not isinstance(breakdown, dict):
        breakdown = {}
    if not isinstance(weights, dict):
        weights = {}
    recs = summary.get("risk_recommendations", [])
    if not isinstance(recs, list):
        recs = []
    if not breakdown and not level:
        return ""

    rows = []
    for k in sorted(set(list(breakdown.keys()) + list(weights.keys()))):
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(k))}</td>"
            f"<td>{float(breakdown.get(k, 0.0) or 0.0):.4f}</td>"
            f"<td>{float(weights.get(k, 0.0) or 0.0):.4f}</td>"
            "</tr>"
        )
    table = (
        "<table><thead><tr><th>component</th><th>value</th><th>weight</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )
    rec_html = ""
    if recs:
        li = "".join(f"<li>{html.escape(str(x))}</li>" for x in recs)
        rec_html = f"<h3>Recommendations</h3><ul>{li}</ul>"

    return (
        "<h2>风险分数（工程辅助指标）</h2>"
        f"<div><b>risk_score</b>: {score:.4f} / <b>risk_level</b>: {html.escape(level)} / "
        f"<b>risk_scenario</b>: {html.escape(scenario)}</div>"
        "<p>该分数是用于报告分诊的内部工程指标，应结合 ASR 和受测模型级结果一起阅读，"
        "不应把它当成可以单独成立的科学指标。</p>"
        + table
        + rec_html
    )


# 渲染 `报告 HTML`，把结构化结果转换成页面或报告片段。
def render_report_html(summary: dict[str, Any], rows: list[dict[str, Any]], run_dir: str) -> str:
    tpl = Path("assets/templates/report_template.html")
    if tpl.exists():
        template = tpl.read_text(encoding="utf-8")
    else:
        template = "<html><body>{{SUMMARY_BLOCK}}{{MODE_TABLE}}{{VLR_BLOCK}}{{PLOTS_BLOCK}}{{SAMPLE_PANEL}}{{REPRODUCTION_CARD}}{{RISK_BLOCK}}{{RESULTS_TABLE}}</body></html>"

    html_text = template.replace("{{SUMMARY_BLOCK}}", _summary_block(summary))
    html_text = html_text.replace("{{MODE_TABLE}}", _mode_table(rows))
    html_text = html_text.replace("{{VLR_BLOCK}}", _vlr_block(summary, rows))
    html_text = html_text.replace("{{PLOTS_BLOCK}}", _plots_block(run_dir))
    html_text = html_text.replace("{{SAMPLE_PANEL}}", _sample_panel(rows, run_dir=run_dir, n=6))
    html_text = html_text.replace("{{REPRODUCTION_CARD}}", _reproduction_card(summary))
    html_text = html_text.replace("{{RISK_BLOCK}}", _risk_block(summary))
    defense_block = _defense_block(summary, rows)
    html_text = html_text.replace("{{DEFENSE_BLOCK}}", defense_block)
    html_text = html_text.replace("{{RESULTS_TABLE}}", defense_block + _rows_table(rows, n=summary.get("num_samples", 20)))
    return html_text
