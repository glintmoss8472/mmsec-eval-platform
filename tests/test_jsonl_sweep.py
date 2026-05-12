from __future__ import annotations

from pathlib import Path

from mmsec_eval.cli import cmd_run_sweep
from mmsec_eval.io.jsonl_io import write_jsonl


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
