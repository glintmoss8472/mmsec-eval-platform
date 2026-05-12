# 文件说明：该文件属于自动化测试，集中实现 test train advclip use gan fallback 相关逻辑。
from __future__ import annotations

import json
from pathlib import Path

from mmsec_eval.attacks.advclip.registry import make_key, read_registry
from mmsec_eval.attacks.advclip.train import train_advclip_patch
from mmsec_eval.config.loader import load_config
from mmsec_eval.config.validate import validate_config
from mmsec_eval.io.jsonl_io import read_jsonl
from mmsec_eval.plugins.builtin import register_builtin_plugins


# 中文注释：验证 test_train_advclip_use_gan 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_train_advclip_use_gan(tmp_path: Path) -> None:
    register_builtin_plugins()

    cfg = load_config("configs/mvp.yaml")
    cfg.artifacts_dir = str(tmp_path / "artifacts")
    cfg.plugins.model_adapter = "clip_hf"
    cfg.attack.mode = "A"
    cfg.attack.patch_size = 32
    cfg.attack.use_gan = True
    cfg.attack.gan_steps = 1
    cfg.attack.patch_train_steps = 5
    cfg.dataset.kind = "toy_shapes"
    cfg.dataset.num_samples = 2
    cfg.runner.max_samples = 2
    cfg.runner.continue_on_error = False
    validate_config(cfg)

    out = train_advclip_patch(cfg)
    assert Path(out.results_path).exists()
    assert Path(out.summary_path).exists()
    assert Path(out.report_path).exists()

    summary = json.loads(Path(out.summary_path).read_text(encoding="utf-8"))
    assert summary["trained"] is True
    assert summary["use_gan"] is True
    assert Path(summary["patch_path"]).exists()
    assert Path(summary["gan_state_path"]).exists()

    rows = read_jsonl(out.results_path)
    assert rows
    assert "loss_gan_d" in rows[0]
    assert "loss_gan_g" in rows[0]

    reg_key = make_key(
        clip_model_name=str(cfg.model.clip_model_name),
        mode=str(cfg.attack.mode),
        patch_size=int(cfg.attack.patch_size),
    )
    reg = read_registry(str(cfg.artifacts_dir))
    assert reg_key in reg.get("entries", {})
    assert bool(reg["entries"][reg_key].get("use_gan")) is True
