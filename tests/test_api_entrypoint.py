# 文件说明：该文件属于自动化测试，集中实现 test api entrypoint 相关逻辑。
from __future__ import annotations

import subprocess
import sys


# 验证 `mmsec API module help does not start 服务` 场景，防止相关行为在后续修改中退化。
def test_mmsec_api_module_help_does_not_start_server() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "mmsec_api", "--help"],
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert result.returncode == 0
    assert "Run the ATT-project FastAPI service" in result.stdout
    assert "--host" in result.stdout
    assert "--port" in result.stdout
