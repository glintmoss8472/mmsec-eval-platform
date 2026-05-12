from __future__ import annotations

import json
from pathlib import Path

from mmsec_eval.config.loader import load_config
from mmsec_eval.config.validate import validate_config
from mmsec_eval.plugins.builtin import register_builtin_plugins
from mmsec_eval.runner.retrieval_runner import run as run_vlr


def test_vlr_runner_defense_metrics(tmp_path: Path):
    register_builtin_plugins()
    cfg = load_config("configs/mvp.yaml")
    cfg.artifacts_dir = str(tmp_path / "artifacts")
    cfg.task.kind = "vlr"
    cfg.task.eval_scope = "image"
    cfg.plugins.attack = "advedm"
    cfg.dataset.kind = "toy_shapes"
    cfg.dataset.num_samples = 6
    cfg.dataset.image_size = 64
    cfg.runner.max_samples = 6
    cfg.runner.surrogate_model_adapter = "clip_hf"
    cfg.runner.victim_model_adapters = ["clip_hf"]
    cfg.attack.steps = 1
    cfg.defense.enabled = True
    cfg.defense.apply_on_attacked = True
    cfg.defense.apply_on_clean = True
    validate_config(cfg)

    out = run_vlr(cfg)
    summary = json.loads(Path(out.summary_path).read_text(encoding="utf-8"))
    report_data = json.loads((Path(out.run_dir) / "report_data.json").read_text(encoding="utf-8"))

    assert summary.get("task_kind") == "vlr"
    assert "asr_attack" in summary
    assert "asr_defended" in summary
    assert "defense_gain" in summary
    assert "risk_score" in summary
    assert "risk_breakdown" in summary
    assert "stage_metrics" in report_data
    assert "defense_compare" in report_data
    assert "risk" in report_data
    assert "feature_projection" in report_data
