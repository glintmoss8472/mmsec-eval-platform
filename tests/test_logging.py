# 文件说明：该文件属于自动化测试，集中实现 test logging 相关逻辑。
from __future__ import annotations

from pathlib import Path

from mmsec_eval.logging import setup_logging


# 中文注释：验证 test_setup_logging 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_setup_logging(tmp_path: Path):
    setup_logging(str(tmp_path))
    assert (tmp_path / "mmsec.log").exists()

