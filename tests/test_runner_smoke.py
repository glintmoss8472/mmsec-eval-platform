# 文件说明：该文件属于自动化测试，集中实现 test runner smoke 相关逻辑。
from __future__ import annotations

from pathlib import Path

from mmsec_eval.config.loader import load_config
from mmsec_eval.config.validate import validate_config
from mmsec_eval.plugins.builtin import register_builtin_plugins
from mmsec_eval.runner.eval_runner import run


# 验证 `runner smoke` 场景，防止相关行为在后续修改中退化。
def test_runner_smoke(tmp_path: Path):
    register_builtin_plugins()
    cfg = load_config("configs/mvp.yaml")
    cfg.artifacts_dir = str(tmp_path / "artifacts")
    cfg.dataset.kind = "toy_shapes"
    cfg.dataset.num_samples = 2
    cfg.dataset.image_size = 64
    cfg.runner.max_samples = 2
    cfg.runner.continue_on_error = False
    cfg.attack.steps = 1
    cfg.attack.patch_size = 8
    validate_config(cfg)
    out = run(cfg)
    assert Path(out.results_path).exists()
    assert Path(out.summary_path).exists()
    assert Path(out.report_path).exists()
    assert Path(out.run_index_path).exists()
