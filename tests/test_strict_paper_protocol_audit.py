# 文件说明：该文件属于自动化测试，集中实现 test strict paper protocol audit 相关逻辑。
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


# 中文注释：验证 test_strict_paper_protocol_audit_blocks_when_assets_are_missing 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_strict_paper_protocol_audit_blocks_when_assets_are_missing(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    advclip_root = tmp_path / "AdvCLIP"
    tmm_root = tmp_path / "TMM"
    out_dir = tmp_path / "audit"
    project_root.mkdir()
    advclip_root.mkdir()
    tmm_root.mkdir()

    script = Path(__file__).resolve().parents[1] / "scripts" / "audit_strict_paper_protocol.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--project-root",
            str(project_root),
            "--advclip-root",
            str(advclip_root),
            "--tmm-root",
            str(tmm_root),
            "--out-dir",
            str(out_dir),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert proc.returncode == 2
    result = json.loads(proc.stdout)
    assert result["status"] == "blocked"
    audit = json.loads((out_dir / "strict_paper_reproduction_audit.json").read_text(encoding="utf-8"))
    assert audit["advclip"]["status"] == "blocked"
    assert audit["tmm"]["status"] == "blocked"
    assert (out_dir / "advclip_official_matrix.sh").exists()
    assert (out_dir / "tmm_official_matrix.sh").exists()
    assert (out_dir / "strict_paper_reproduction_audit.md").exists()
