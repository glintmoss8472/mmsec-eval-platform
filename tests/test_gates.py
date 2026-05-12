from __future__ import annotations

from pathlib import Path

from mmsec_eval.runner.gates import gate1_ok, gate2_ok, gate3_ok


def test_gate_helpers(tmp_path: Path):
    art = tmp_path / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    assert gate1_ok(str(art))

    run = art / "runs" / "r1"
    run.mkdir(parents=True, exist_ok=True)
    (run / "results.jsonl").write_text("", encoding="utf-8")
    (run / "summary.json").write_text("{}", encoding="utf-8")
    (run / "report.html").write_text("<html></html>", encoding="utf-8")
    assert gate2_ok(str(run))

    (art / "runs" / "run_index.jsonl").write_text("", encoding="utf-8")
    assert gate3_ok(str(art))
