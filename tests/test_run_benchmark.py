from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from mmsec_eval.cli import cmd_run_benchmark


def test_run_benchmark_smoke(tmp_path: Path):
    root = tmp_path / "bench_data"
    images = root / "images"
    images.mkdir(parents=True, exist_ok=True)

    for i in range(2):
        arr = np.zeros((32, 32, 3), dtype=np.uint8)
        arr[..., i % 3] = 180
        Image.fromarray(arr).save(images / f"s{i}.png")

    cap = root / "captions_index.jsonl"
    rows = [
        {"id": "s0", "image": "s0.png", "caption": "caption zero", "split": "test"},
        {"id": "s1", "image": "s1.png", "caption": "caption one", "split": "test"},
    ]
    cap.write_text("\n".join(json.dumps(x) for x in rows) + "\n", encoding="utf-8")

    cfg = tmp_path / "bench.yaml"
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
                "  kind: flickr30k",
                "  root: '" + str(root).replace("\\", "/") + "'",
                "  image_dir: images",
                "  captions_file: captions_index.jsonl",
                "  split: test",
                "  max_items: 2",
                "  benchmark_tag: bench_smoke",
                "attack:",
                "  steps: 1",
                "  patch_size: 8",
                "runner:",
                "  max_samples: 2",
                "  continue_on_error: false",
            ]
        ),
        encoding="utf-8",
    )

    rc = cmd_run_benchmark(str(cfg))
    assert rc == 0

    runs = sorted((tmp_path / "artifacts" / "runs").glob("*/benchmark_summary.json"))
    assert runs
