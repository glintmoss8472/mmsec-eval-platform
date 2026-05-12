# 文件说明：该文件属于自动化测试，集中实现 test defense sanitize v1 相关逻辑。
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mmsec_eval.config.loader import load_config
from mmsec_eval.defenses.sanitize_v1 import SanitizeDefenseV1
from mmsec_eval.types import DefenseContext, Sample


# 中文注释：验证 test_sanitize_v1_deterministic_and_range 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_sanitize_v1_deterministic_and_range(tmp_path: Path):
    cfg = load_config("configs/mvp.yaml")
    cfg.defense.enabled = True
    cfg.defense.resize_ratio = 0.9
    cfg.defense.bit_depth = 5
    cfg.defense.jpeg_quality = 80
    cfg.defense.blur_sigma = 0.8
    cfg.defense.text_normalize = True

    rng = np.random.default_rng(123)
    img = rng.random((64, 64, 3), dtype=np.float32)
    sample = Sample(sample_id="s1", image=img, text="  HELLO   WORLD ", target_text="", metadata={})
    d = SanitizeDefenseV1()

    out1 = d.defend(
        sample,
        DefenseContext(config=cfg, model_adapter=None, stage="attacked", run_dir=str(tmp_path), sample_debug_dir=str(tmp_path / "dbg")),
    )
    out2 = d.defend(
        sample,
        DefenseContext(config=cfg, model_adapter=None, stage="attacked", run_dir=str(tmp_path), sample_debug_dir=str(tmp_path / "dbg2")),
    )

    assert np.allclose(out1.sample.image, out2.sample.image)
    assert out1.sample.text == "hello world"
    assert float(out1.sample.image.min()) >= 0.0
    assert float(out1.sample.image.max()) <= 1.0

    trace_path = Path(out1.artifact_refs.get("defense_trace", ""))
    assert trace_path.exists()
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    assert payload.get("defense") == "sanitize_v1"
    assert payload.get("image_candidates")
    assert payload.get("selection", {}).get("selected_recipe")
