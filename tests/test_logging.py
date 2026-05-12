# 文件说明：该文件属于自动化测试，集中实现 test logging 相关逻辑。
from __future__ import annotations

from pathlib import Path

from mmsec_eval.logging import setup_logging


# 验证 `setup logging` 场景，防止相关行为在后续修改中退化。
def test_setup_logging(tmp_path: Path):
    setup_logging(str(tmp_path))
    assert (tmp_path / "mmsec.log").exists()

