from __future__ import annotations

import subprocess
import sys


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
