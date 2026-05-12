# 文件说明：该文件属于报告可视化层，集中实现 plots 相关逻辑。
from __future__ import annotations

from pathlib import Path


_CJK_FONT_READY: bool | None = None


# 中文注释：封装 _configure_cjk_font 的内部步骤，让报告可视化层主流程保持清晰并隔离边界细节。
def _configure_cjk_font(matplotlib) -> bool:
    """Use a CJK-capable font when one is available on the server."""
    global _CJK_FONT_READY
    if _CJK_FONT_READY is not None:
        return _CJK_FONT_READY

    from matplotlib import font_manager

    preferred_names = [
        "Noto Sans CJK SC",
        "Noto Sans CJK",
        "Source Han Sans SC",
        "Microsoft YaHei",
        "SimHei",
        "WenQuanYi Zen Hei",
        "AR PL UMing CN",
    ]
    for font in font_manager.fontManager.ttflist:
        if font.name in preferred_names:
            matplotlib.rcParams["font.family"] = [font.name]
            matplotlib.rcParams["axes.unicode_minus"] = False
            _CJK_FONT_READY = True
            return True

    candidate_paths = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
    ]
    for raw_path in candidate_paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        font_manager.fontManager.addfont(str(path))
        prop = font_manager.FontProperties(fname=str(path))
        matplotlib.rcParams["font.family"] = [prop.get_name()]
        matplotlib.rcParams["axes.unicode_minus"] = False
        _CJK_FONT_READY = True
        return True

    matplotlib.rcParams["axes.unicode_minus"] = False
    _CJK_FONT_READY = False
    return False


# 中文注释：封装 _safe_text 的内部步骤，让报告可视化层主流程保持清晰并隔离边界细节。
def _safe_text(plt, value: object, fallback: str) -> str:
    """Avoid unreadable square glyphs when the runtime has no CJK font."""
    text = str(value or "").strip()
    if getattr(plt, "_mmsec_has_cjk_font", False):
        return text or fallback
    ascii_text = "".join(ch if ord(ch) < 128 else " " for ch in text)
    ascii_text = " ".join(ascii_text.split())
    return ascii_text or fallback


# 中文注释：封装 _safe_labels 的内部步骤，让报告可视化层主流程保持清晰并隔离边界细节。
def _safe_labels(plt, labels: list[str]) -> list[str]:
    return [_safe_text(plt, label, f"item {idx + 1}") for idx, label in enumerate(labels)]


# 中文注释：封装 _import_plt 的内部步骤，让报告可视化层主流程保持清晰并隔离边界细节。
def _import_plt():
    try:
        import matplotlib
    except ModuleNotFoundError:
        return None

    matplotlib.use("Agg", force=True)
    has_cjk_font = _configure_cjk_font(matplotlib)
    import matplotlib.pyplot as plt

    plt._mmsec_has_cjk_font = has_cjk_font  # type: ignore[attr-defined]
    return plt

# 中文注释：实现 plot_metric_curve 的核心流程，支撑报告可视化层中的业务语义和异常边界。
def plot_metric_curve(values: list[float], title: str, out_path: str) -> str:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt = _import_plt()
    if plt is None:
        return ""
    plt.figure(figsize=(6, 3))
    plt.plot(list(range(len(values))), values)
    plt.title(_safe_text(plt, title, "metric curve"))
    plt.xlabel("sample")
    plt.ylabel("value")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path


# 中文注释：实现 plot_asr_bar 的核心流程，支撑报告可视化层中的业务语义和异常边界。
def plot_asr_bar(metrics: dict[str, float], out_path: str) -> str:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt = _import_plt()
    if plt is None:
        return ""
    raw_keys = list(metrics.keys())
    keys = _safe_labels(plt, raw_keys)
    vals = [float(metrics[k]) for k in raw_keys]
    plt.figure(figsize=(4, 3))
    plt.bar(keys, vals)
    plt.ylim(0, 1)
    plt.ylabel("ratio")
    plt.title("Attack success")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path


# 中文注释：实现 plot_attack_comparison 的核心流程，支撑报告可视化层中的业务语义和异常边界。
def plot_attack_comparison(mode_stats: dict[str, dict[str, float]], out_path: str) -> str:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt = _import_plt()
    if plt is None:
        return ""
    keys = list(mode_stats.keys())
    vals = [float(mode_stats[k].get("asr", 0.0)) for k in keys]
    if not keys:
        raise ValueError("mode_stats is empty")
    display_keys = _safe_labels(plt, keys)
    plt.figure(figsize=(max(5, len(keys) * 1.4), 3.2))
    plt.bar(display_keys, vals)
    plt.ylim(0, 1)
    plt.ylabel("attack success rate")
    plt.title("Attack method comparison")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path


# 中文注释：实现 plot_grouped_bar 的核心流程，支撑报告可视化层中的业务语义和异常边界。
def plot_grouped_bar(
    *,
    labels: list[str],
    series: dict[str, list[float]],
    out_path: str,
    title: str = "",
    ylim: tuple[float, float] | None = (0.0, 1.0),
) -> str:
    """Generic grouped bar plot (used by VLR reports)."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt = _import_plt()
    if plt is None:
        return ""
    if not labels or not series:
        raise ValueError("labels/series cannot be empty")

    keys = list(series.keys())
    display_keys = _safe_labels(plt, keys)
    display_labels = _safe_labels(plt, labels)
    n = len(labels)
    x = list(range(n))
    width = 0.8 / max(1, len(keys))

    plt.figure(figsize=(max(6, n * 0.8), 3.4))
    for i, k in enumerate(keys):
        vals = [float(v) for v in (series.get(k) or [])]
        if len(vals) != n:
            vals = (vals + [0.0] * n)[:n]
        xs = [xi + (i - (len(keys) - 1) / 2.0) * width for xi in x]
        plt.bar(xs, vals, width=width, label=display_keys[i])

    plt.xticks(x, display_labels, rotation=20, ha="right")
    if ylim is not None:
        plt.ylim(float(ylim[0]), float(ylim[1]))
    if title:
        plt.title(_safe_text(plt, title, "grouped bar"))
    plt.tight_layout()
    plt.legend()
    plt.savefig(out_path)
    plt.close()
    return out_path


# 中文注释：实现 plot_stage_compare_bar 的核心流程，支撑报告可视化层中的业务语义和异常边界。
def plot_stage_compare_bar(
    *,
    out_path: str,
    asr_attack: float,
    asr_defended: float,
    title: str = "Stage Compare",
) -> str:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt = _import_plt()
    if plt is None:
        return ""
    labels = ["clean", "attacked", "defended"]
    vals = [0.0, float(asr_attack), float(asr_defended)]
    plt.figure(figsize=(5.2, 3.2))
    plt.bar(labels, vals)
    plt.ylim(0, 1)
    plt.title(_safe_text(plt, title, "stage comparison"))
    plt.ylabel("attack success rate")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path


# 中文注释：实现 plot_defense_recovery_curve 的核心流程，支撑报告可视化层中的业务语义和异常边界。
def plot_defense_recovery_curve(
    *,
    labels: list[str],
    attacked_vals: list[float],
    defended_vals: list[float],
    out_path: str,
    title: str = "Defense Recovery",
) -> str:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt = _import_plt()
    if plt is None:
        return ""
    n = min(len(labels), len(attacked_vals), len(defended_vals))
    if n <= 0:
        raise ValueError("recovery curve requires non-empty inputs")
    xs = list(range(n))
    display_labels = _safe_labels(plt, labels[:n])
    plt.figure(figsize=(max(5.0, n * 1.2), 3.2))
    plt.plot(xs, [float(x) for x in attacked_vals[:n]], marker="o", label="attacked")
    plt.plot(xs, [float(x) for x in defended_vals[:n]], marker="o", label="defended")
    plt.xticks(xs, display_labels, rotation=20, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("attack success rate avg")
    plt.title(_safe_text(plt, title, "defense recovery"))
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path
