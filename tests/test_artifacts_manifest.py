from __future__ import annotations

import json
from pathlib import Path

from scripts.build_artifacts_manifest import build_manifest


def test_artifacts_manifest_links_run_files(tmp_path: Path):
    run_dir = tmp_path / "artifacts" / "runs" / "run_001"
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(
        json.dumps({"dataset_name": "mini_flickr", "attack_name": "advclip", "model_adapter": "clip_hf"}),
        encoding="utf-8",
    )
    (run_dir / "report.html").write_text("<html></html>", encoding="utf-8")
    aux_dir = tmp_path / "artifacts" / "embodied_decision_demo"
    aux_dir.mkdir()
    (aux_dir / "decision_summary.json").write_text(json.dumps({"case_count": 1}), encoding="utf-8")
    interaction_dir = tmp_path / "artifacts" / "vqa_dialogue_demo"
    interaction_dir.mkdir()
    (interaction_dir / "interaction_summary.json").write_text(json.dumps({"case_count": 1}), encoding="utf-8")
    payload = build_manifest(tmp_path, tmp_path / "artifacts")
    assert payload["run_count"] == 1
    row = payload["runs"][0]
    assert row["run_id"] == "run_001"
    assert row["dataset"] == "mini_flickr"
    assert "summary.json" in row["files"]
    assert row["files"]["summary.json"]["sha256"]
    groups = {item["group"] for item in payload["auxiliary_artifacts"]}
    assert {"embodied_decision", "vqa_dialogue"}.issubset(groups)
