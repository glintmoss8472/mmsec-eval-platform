# 文件说明：该文件属于自动化测试，集中实现 test jsonl sweep 相关逻辑。
from __future__ import annotations

from pathlib import Path

from mmsec_eval.cli import cmd_run_sweep
from mmsec_eval.io.jsonl_io import write_jsonl


# 中文注释：验证 test_run_sweep_smoke 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_run_sweep_smoke(tmp_path: Path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "\n".join(
            [
                "seed: 1",
                "artifacts_dir: '" + str(tmp_path / "artifacts").replace("\\", "/") + "'",
                "plugins:",
                "  model_adapter: clip_hf",
                "  attack: advedm",
                "  metric: basic",
                "  judge: rule",
                "dataset:",
                "  kind: toy_shapes",
                "  num_samples: 4",
                "  image_size: 64",
                "attack:",
                "  steps: 1",
                "  patch_size: 8",
                "  eps_t: 0",
                "runner:",
                "  continue_on_error: false",
            ]
        ),
        encoding="utf-8",
    )
    sweep = tmp_path / "sweep.jsonl"
    write_jsonl(
        str(sweep),
        [
            {"plugins": {"attack": "advedm"}, "attack": {"mode": "B", "steps": 1}},
            {"plugins": {"attack": "tmm"}, "attack": {"eps_t": 0, "steps": 1}},
        ],
    )
    rc = cmd_run_sweep(str(cfg), str(sweep))
    assert rc == 0
    assert (tmp_path / "artifacts" / "runs" / "run_index.jsonl").exists()
