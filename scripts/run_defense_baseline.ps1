Param(
  [string]$BaseConfig = "configs/profiles/cpu_smoke.yaml",
  [string]$ArtifactsDir = "artifacts/defense_baseline"
)

$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Path $ArtifactsDir -Force | Out-Null

$cfgA = Join-Path $ArtifactsDir "defense_mode_a.yaml"
$cfgB = Join-Path $ArtifactsDir "defense_mode_b.yaml"

$py = @"
from pathlib import Path
import yaml

base = yaml.safe_load(Path(r"$BaseConfig").read_text(encoding="utf-8")) or {}

def write_cfg(path, mode):
    cfg = dict(base)
    cfg["artifacts_dir"] = r"$ArtifactsDir"
    attack = dict(cfg.get("attack", {}))
    attack["mode"] = mode
    cfg["attack"] = attack
    Path(path).write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")

write_cfg(r"$cfgA", "A")
write_cfg(r"$cfgB", "B")
"@
$py | python -
if ($LASTEXITCODE -ne 0) { exit 1 }

python -m mmsec_eval run-eval --config $cfgA
if ($LASTEXITCODE -ne 0) { exit 1 }

python -m mmsec_eval run-eval --config $cfgB
if ($LASTEXITCODE -ne 0) { exit 1 }

$pySummary = @"
from pathlib import Path
import json

root = Path(r"$ArtifactsDir") / "runs"
runs = sorted(root.glob("*/summary.json"))
if len(runs) < 2:
    raise SystemExit(1)
a = json.loads(runs[-2].read_text(encoding="utf-8"))
b = json.loads(runs[-1].read_text(encoding="utf-8"))
out = {
    "mode_a_run": a.get("run_id", ""),
    "mode_b_run": b.get("run_id", ""),
    "mode_a_asr": a.get("asr", 0.0),
    "mode_b_asr": b.get("asr", 0.0),
    "delta_asr": float(b.get("asr", 0.0)) - float(a.get("asr", 0.0)),
}
out_path = Path(r"$ArtifactsDir") / "defense_baseline_summary.json"
out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(out_path)
"@
$pySummary | python -
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "[DEFENSE] baseline completed"
