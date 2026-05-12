# 文件说明：该文件属于运维与实验脚本，集中实现 cleanup failed jobs 相关逻辑。
Param(
  [string]$DbPath = "artifacts/app.db"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $DbPath)) {
  throw "DB not found: $DbPath"
}

$env:MMSEC_CLEANUP_DB_PATH = (Resolve-Path $DbPath).Path
$tmpPy = [System.IO.Path]::ChangeExtension([System.IO.Path]::GetTempFileName(), ".py")
$py = @'
import json
import os
import sqlite3
from pathlib import Path

p = Path(os.environ["MMSEC_CLEANUP_DB_PATH"])
conn = sqlite3.connect(str(p))
cur = conn.cursor()


def count_by_status():
    out = {}
    for st in ("queued", "running", "success", "failed", "cancelled"):
        out[st] = int(cur.execute("select count(1) from jobs where status=?", (st,)).fetchone()[0])
    return out


before = count_by_status()
bad_ids = [r[0] for r in cur.execute("select id from jobs where status in ('failed','cancelled')").fetchall()]
if bad_ids:
    q = ",".join(["?"] * len(bad_ids))
    cur.execute(f"delete from job_logs where job_id in ({q})", bad_ids)
    cur.execute(f"delete from jobs where id in ({q})", bad_ids)
    conn.commit()
after = count_by_status()
print(json.dumps({"before": before, "after": after, "removed": len(bad_ids)}, ensure_ascii=False))
'@

Set-Content -Path $tmpPy -Value $py -Encoding UTF8
# 处理 `try` 步骤，封装脚本中的可复用命令片段。
try {
  .\.venv313\Scripts\python.exe $tmpPy
} finally {
  Remove-Item -Path $tmpPy -Force -ErrorAction SilentlyContinue | Out-Null
  Remove-Item Env:MMSEC_CLEANUP_DB_PATH -ErrorAction SilentlyContinue | Out-Null
}
