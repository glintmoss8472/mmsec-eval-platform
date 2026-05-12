# 文件说明：该文件属于自动化测试，集中实现 test docs ingest 相关逻辑。
from __future__ import annotations

import json
from pathlib import Path

from mmsec_eval.cli import cmd_ingest_docs


# 验证 `文档 ingest smoke` 场景，防止相关行为在后续修改中退化。
def test_docs_ingest_smoke(tmp_path: Path):
    txt = tmp_path / "a.txt"
    txt.write_text("hello docs ingest", encoding="utf-8")
    cfg = tmp_path / "cfg.yaml"
    txt_posix = str(txt).replace("\\", "/")
    artifacts_posix = str(tmp_path / "artifacts").replace("\\", "/")
    cfg.write_text(
        "\n".join(
            [
                "seed: 1",
                "artifacts_dir: '" + artifacts_posix + "'",
                "docs:",
                "  paths:",
                "    - '" + txt_posix + "'",
            ]
        ),
        encoding="utf-8",
    )
    rc = cmd_ingest_docs(str(cfg))
    assert rc == 0
    assert (tmp_path / "artifacts" / "docs_index.json").exists()
    data = json.loads((tmp_path / "artifacts" / "docs_index.json").read_text(encoding="utf-8"))
    assert len(data) == 1
