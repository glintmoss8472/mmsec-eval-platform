# 文件说明：该文件属于自动化测试，集中实现 test eval runner defense metrics 相关逻辑。
from __future__ import annotations

import json
from pathlib import Path

from mmsec_eval.config.loader import load_config
from mmsec_eval.config.validate import validate_config
from mmsec_eval.plugins.builtin import register_builtin_plugins
from mmsec_eval.runner.eval_runner import run


# 验证 `评测 runner 防御 指标` 场景，防止相关行为在后续修改中退化。
def test_eval_runner_defense_metrics(tmp_path: Path):
    register_builtin_plugins()
    cfg = load_config("configs/mvp.yaml")
    cfg.artifacts_dir = str(tmp_path / "artifacts")
    cfg.dataset.kind = "toy_shapes"
    cfg.dataset.num_samples = 3
    cfg.dataset.image_size = 64
    cfg.runner.max_samples = 3
    cfg.attack.steps = 1
    cfg.attack.patch_size = 8
    cfg.defense.enabled = True
    cfg.defense.apply_on_clean = True
    cfg.defense.apply_on_attacked = True
    validate_config(cfg)

    out = run(cfg)
    summary = json.loads(Path(out.summary_path).read_text(encoding="utf-8"))
    report_data = json.loads((Path(out.run_dir) / "report_data.json").read_text(encoding="utf-8"))

    assert "asr_attack" in summary
    assert "asr_defended" in summary
    assert "defense_gain" in summary
    assert "risk_score" in summary
    assert "risk_breakdown" in summary
    assert float(summary["asr"]) == float(summary["asr_attack"])
    assert "stage_metrics" in report_data
    assert "defense_compare" in report_data
    assert "risk" in report_data
