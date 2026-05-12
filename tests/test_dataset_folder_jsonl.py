from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from mmsec_eval.datasets.folder_jsonl import load_folder_jsonl


def test_folder_jsonl_loader(tmp_path: Path):
    image_path = tmp_path / "a.png"
    arr = (np.ones((32, 32, 3)) * 200).astype("uint8")
    Image.fromarray(arr).save(image_path)
    jsonl = tmp_path / "samples.jsonl"
    jsonl.write_text(json.dumps({"id": "1", "image_path": str(image_path), "text": "hello"}) + "\n", encoding="utf-8")

    rows = load_folder_jsonl(str(jsonl))
    assert len(rows) == 1
    assert rows[0].sample_id == "1"

