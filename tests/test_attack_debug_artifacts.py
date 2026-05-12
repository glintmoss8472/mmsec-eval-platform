# 文件说明：该文件属于自动化测试，集中实现 test attack debug artifacts 相关逻辑。
from __future__ import annotations

from pathlib import Path

from mmsec_eval.config.loader import load_config
from mmsec_eval.config.validate import validate_config
from mmsec_eval.plugins.builtin import register_builtin_plugins
from mmsec_eval.runner.eval_runner import run


# 中文注释：验证 test_attack_debug_artifacts_exist 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_attack_debug_artifacts_exist(tmp_path: Path):
    register_builtin_plugins()
    cfg = load_config("configs/mvp.yaml")
    cfg.artifacts_dir = str(tmp_path / "artifacts")
    cfg.dataset.num_samples = 3
    cfg.runner.max_samples = 3
    cfg.plugins.attack = "advedm"
    cfg.runner.continue_on_error = False
    cfg.attack.steps = 1
    cfg.attack.patch_size = 8
    validate_config(cfg)

    out = run(cfg)
    debug_root = Path(out.run_dir) / "attack_debug"
    assert debug_root.exists()
    sample_dirs = [p for p in debug_root.iterdir() if p.is_dir()]
    assert sample_dirs

    debug_json = sample_dirs[0] / "debug.json"
    mask_png = sample_dirs[0] / "advedm_mask.png"
    assert debug_json.exists()
    assert mask_png.exists()
