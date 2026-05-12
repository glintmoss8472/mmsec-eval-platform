# 文件说明：该文件属于自动化测试，集中实现 test viz plots 相关逻辑。
from __future__ import annotations

from pathlib import Path

from mmsec_eval.viz.plots import plot_attack_comparison, plot_metric_curve


# 中文注释：验证 test_plot_metric_curve 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_plot_metric_curve(tmp_path: Path):
    out = tmp_path / "curve.png"
    plot_metric_curve([1, 2, 3], "x", str(out))
    assert out.exists()


# 中文注释：验证 test_plot_attack_comparison 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_plot_attack_comparison(tmp_path: Path):
    out = tmp_path / "attack.png"
    plot_attack_comparison({"advedm:A": {"asr": 0.5}, "advclip:B": {"asr": 0.7}}, str(out))
    assert out.exists()
