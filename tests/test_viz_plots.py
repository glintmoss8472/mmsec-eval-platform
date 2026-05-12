from __future__ import annotations

from pathlib import Path

from mmsec_eval.viz.plots import plot_attack_comparison, plot_metric_curve


def test_plot_metric_curve(tmp_path: Path):
    out = tmp_path / "curve.png"
    plot_metric_curve([1, 2, 3], "x", str(out))
    assert out.exists()


def test_plot_attack_comparison(tmp_path: Path):
    out = tmp_path / "attack.png"
    plot_attack_comparison({"advedm:A": {"asr": 0.5}, "advclip:B": {"asr": 0.7}}, str(out))
    assert out.exists()
