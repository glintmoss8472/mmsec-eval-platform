Param(
  [string]$Config = "configs/mvp.yaml",
  [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "_python_env.ps1")

$projectRoot = Get-ProjectRoot -ScriptRoot $PSScriptRoot
Set-Location $projectRoot

if ([string]::IsNullOrWhiteSpace($PythonExe)) {
  $PythonExe = Get-ProjectPython -ProjectRoot $projectRoot -EnsureVenv
}

& $PythonExe -m mmsec_eval ingest-docs --config $Config
if ($LASTEXITCODE -ne 0) { exit 1 }

& $PythonExe -m mmsec_eval run-eval --config $Config
if ($LASTEXITCODE -ne 0) { exit 1 }

& $PythonExe -m mmsec_eval run-sweep --config $Config --sweep configs/sweep/examples.jsonl
if ($LASTEXITCODE -ne 0) { exit 1 }

# Gate-3+ benchmark smoke (public benchmark loader path via tiny local fixture).
$benchCfg = "artifacts/_gate_benchmark_smoke.yaml"
$py = @"
from pathlib import Path
import json
import numpy as np
from PIL import Image

root = Path("artifacts/_gate_bench_data")
img_dir = root / "images"
img_dir.mkdir(parents=True, exist_ok=True)
rows = []
for i in range(2):
    arr = np.zeros((48, 48, 3), dtype=np.uint8)
    arr[..., i % 3] = 190
    p = img_dir / f"gate_{i}.png"
    Image.fromarray(arr).save(p)
    rows.append({"id": f"gate-{i}", "image": p.name, "caption": f"sample caption {i}", "split": "test"})

captions = root / "captions_index.jsonl"
with captions.open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

cfg = Path("$benchCfg")
cfg.parent.mkdir(parents=True, exist_ok=True)
cfg.write_text("""
seed: 1
artifacts_dir: "artifacts"
plugins:
  model_adapter: "clip_hf"
  attack: "advedm"
  metric: "basic"
  judge: "rule"
dataset:
  kind: "flickr30k"
  root: "artifacts/_gate_bench_data"
  image_dir: "images"
  captions_file: "captions_index.jsonl"
  split: "test"
  max_items: 2
  benchmark_tag: "gate_smoke"
runner:
  max_samples: 2
  continue_on_error: false
""".strip() + "\n", encoding="utf-8")
print(cfg)
"@
$py | & $PythonExe -
if ($LASTEXITCODE -ne 0) { exit 1 }

& $PythonExe -m mmsec_eval run-benchmark --config $benchCfg
if ($LASTEXITCODE -ne 0) { exit 1 }

& $PythonExe -m pytest
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "ALL GATES PASS"
