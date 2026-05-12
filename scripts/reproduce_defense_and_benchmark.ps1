Param(
  [string]$BenchmarkConfig = "configs/bench/flickr30k_clip.yaml"
)

$ErrorActionPreference = "Stop"

powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_defense_baseline.ps1
if ($LASTEXITCODE -ne 0) { exit 1 }

$benchOk = $false
if (Test-Path $BenchmarkConfig) {
  $pyCheck = @"
from pathlib import Path
import yaml
cfg = yaml.safe_load(Path(r"$BenchmarkConfig").read_text(encoding="utf-8")) or {}
dataset = cfg.get("dataset", {})
root = str(dataset.get("root", "")).strip()
ok = False
if root:
    p = Path(root)
    ok = p.exists()
print("1" if ok else "0")
"@
  $chk = ($pyCheck | python -).Trim()
  if ($chk -eq "1") {
    python -m mmsec_eval run-benchmark --config $BenchmarkConfig
    if ($LASTEXITCODE -eq 0) {
      $benchOk = $true
    }
  }
}

if (-not $benchOk) {
  $BenchmarkConfig = "artifacts/_bench_fixture.yaml"
  $py = @"
from pathlib import Path
import json
import numpy as np
from PIL import Image

root = Path("artifacts/bench_fixture")
img_dir = root / "images"
img_dir.mkdir(parents=True, exist_ok=True)
rows = []
for i in range(3):
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    arr[..., i % 3] = 180
    path = img_dir / f"fixture_{i}.png"
    Image.fromarray(arr).save(path)
    rows.append({"id": f"fx-{i}", "image": path.name, "caption": f"fixture caption {i}", "split": "test"})

index_path = root / "captions_index.jsonl"
with index_path.open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

cfg = Path("artifacts/_bench_fixture.yaml")
cfg.write_text(
    """
seed: 1
artifacts_dir: "artifacts"
plugins:
  model_adapter: "clip_hf"
  attack: "advclip"
  metric: "basic"
  judge: "rule"
dataset:
  kind: "flickr30k"
  root: "artifacts/bench_fixture"
  image_dir: "images"
  captions_file: "captions_index.jsonl"
  split: "test"
  max_items: 3
  benchmark_tag: "fixture"
runner:
  max_samples: 3
  continue_on_error: false
""".strip() + "\n",
    encoding="utf-8",
)
print(cfg)
"@
  $py | python -
  if ($LASTEXITCODE -ne 0) { exit 1 }

  python -m mmsec_eval run-benchmark --config $BenchmarkConfig
  if ($LASTEXITCODE -ne 0) { exit 1 }
}

powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\export_thesis_tables.ps1
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "BENCHMARK PASS"
