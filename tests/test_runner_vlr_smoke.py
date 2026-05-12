# 文件说明：该文件属于自动化测试，集中实现 test runner vlr smoke 相关逻辑。
from __future__ import annotations

import json
from pathlib import Path

from mmsec_eval.config.loader import load_config
from mmsec_eval.config.validate import validate_config
from mmsec_eval.attacks.advclip.train import train_advclip_patch
from mmsec_eval.plugins.builtin import register_builtin_plugins
from mmsec_eval.runner.retrieval_runner import run as run_vlr


# 中文注释：验证 test_runner_vlr_smoke 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_runner_vlr_smoke(tmp_path: Path):
    register_builtin_plugins()
    cfg = load_config("configs/mvp.yaml")
    cfg.artifacts_dir = str(tmp_path / "artifacts")
    cfg.task.kind = "vlr"
    cfg.task.eval_scope = "image"
    cfg.task.retrieval_k = [1, 5, 10]
    cfg.runner.surrogate_model_adapter = "clip_hf"
    cfg.runner.victim_model_adapters = ["clip_hf"]
    cfg.runner.max_pairs = 0
    cfg.plugins.attack = "advclip"
    cfg.dataset.kind = "toy_shapes"
    cfg.dataset.num_samples = 6
    cfg.runner.max_samples = 6
    cfg.runner.continue_on_error = False
    cfg.attack.patch_train_steps = 5

    validate_config(cfg)
    train_advclip_patch(cfg)
    out = run_vlr(cfg)
    assert Path(out.results_path).exists()
    assert Path(out.summary_path).exists()
    assert Path(out.report_path).exists()
    assert Path(out.run_index_path).exists()
    assert (Path(out.run_dir) / "report_data.json").exists()
    assert list((Path(out.run_dir) / "cases").glob("*/case_bundle.json"))

    summary = json.loads(Path(out.summary_path).read_text(encoding="utf-8"))
    assert summary.get("task_kind") == "vlr"
