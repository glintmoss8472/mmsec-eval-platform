from __future__ import annotations

import json
from pathlib import Path

from mmsec_eval.config.loader import load_config
from mmsec_eval.config.validate import validate_config
from mmsec_eval.plugins.builtin import register_builtin_plugins
from mmsec_eval.runner.eval_runner import run


def test_report_data_consistency(tmp_path: Path):
    register_builtin_plugins()
    cfg = load_config("configs/mvp.yaml")
    cfg.artifacts_dir = str(tmp_path / "artifacts")
    cfg.dataset.num_samples = 4
    cfg.runner.max_samples = 4
    cfg.runner.continue_on_error = False
    cfg.attack.steps = 1
    cfg.attack.patch_size = 8
    validate_config(cfg)

    out = run(cfg)
    report_data_path = Path(out.run_dir) / "report_data.json"
    assert report_data_path.exists()

    report_data = json.loads(report_data_path.read_text(encoding="utf-8"))
    assert "summary" in report_data
    assert "mode_stats" in report_data
    assert report_data["summary"]["run_id"] == out.run_id
