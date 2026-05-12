from __future__ import annotations

from pathlib import Path

from mmsec_eval.logging import setup_logging


def test_setup_logging(tmp_path: Path):
    setup_logging(str(tmp_path))
    assert (tmp_path / "mmsec.log").exists()

