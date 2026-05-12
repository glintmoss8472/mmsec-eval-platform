# 文件说明：该文件属于运维与实验脚本，集中实现 reproduce p3 full 相关逻辑。
Param(
  [string]$ExperimentId = "",
  [switch]$SkipBaselinePrefetch
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "_python_env.ps1")

$projectRoot = Get-ProjectRoot -ScriptRoot $PSScriptRoot
Set-Location $projectRoot
$pythonExe = Get-ProjectPython -ProjectRoot $projectRoot -EnsureVenv

if ([string]::IsNullOrWhiteSpace($ExperimentId)) {
  $ExperimentId = "p3_" + (Get-Date -Format "yyyyMMdd_HHmmss")
}

if (-not $SkipBaselinePrefetch) {
  Write-Host "[P3] Running baseline strong VLR workflow (includes HF prefetch + AdvCLIP patch training)..."
  & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "reproduce_vlr_strong.ps1") -Profile full
  if ($LASTEXITCODE -ne 0) { exit 1 }
}

$cfgDir = Join-Path $projectRoot "artifacts\\tmp_p3_configs"
New-Item -ItemType Directory -Force -Path $cfgDir | Out-Null

$pyGen = @"
from pathlib import Path
import yaml

root = Path(r"$projectRoot")
cfg_dir = Path(r"$cfgDir")
exp_id = r"$ExperimentId"

pairs = [
    ("advclip", root / "configs/bench/bootstrap_full_vlr_gan_cuda.yaml"),
    ("tmm", root / "configs/bench/bootstrap_full_vlr_tmm_cuda.yaml"),
    ("advedm", root / "configs/bench/bootstrap_full_vlr_cuda.yaml"),
]

out = []
for attack, src in pairs:
    cfg = yaml.safe_load(src.read_text(encoding="utf-8")) or {}
    cfg.setdefault("plugins", {})["attack"] = attack
    cfg.setdefault("task", {})["kind"] = "vlr"
    cfg["task"]["eval_scope"] = "image"
    cfg.setdefault("runner", {})["experiment_id"] = exp_id
    cfg.setdefault("defense", {})["enabled"] = False
    cfg["defense"]["apply_on_attacked"] = True
    cfg["defense"]["apply_on_clean"] = True
    cfg.setdefault("dataset", {})["benchmark_tag"] = f"{exp_id}_{attack}_attacked"

    p_att = cfg_dir / f"{attack}_attacked.yaml"
    p_att.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    out.append(str(p_att))

    cfg2 = yaml.safe_load(p_att.read_text(encoding="utf-8")) or {}
    cfg2.setdefault("defense", {})["enabled"] = True
    cfg2.setdefault("dataset", {})["benchmark_tag"] = f"{exp_id}_{attack}_defended"
    p_def = cfg_dir / f"{attack}_defended.yaml"
    p_def.write_text(yaml.safe_dump(cfg2, allow_unicode=True, sort_keys=False), encoding="utf-8")
    out.append(str(p_def))

print("\\n".join(out))
"@
$cfgList = (& $pythonExe -c $pyGen)
if ($LASTEXITCODE -ne 0) { exit 1 }

$cfgPaths = @($cfgList -split "`r?`n" | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
foreach ($cfg in $cfgPaths) {
  Write-Host "[P3] run-vlr --config $cfg"
  & $pythonExe -m mmsec_eval run-vlr --config $cfg
  if ($LASTEXITCODE -ne 0) { exit 1 }
}

$pyCompare = @"
from pathlib import Path
import json

project = Path(r"$projectRoot")
exp_id = r"$ExperimentId"
runs_root = project / "artifacts" / "runs"
exp_root = project / "artifacts" / "experiments" / exp_id
exp_root.mkdir(parents=True, exist_ok=True)

rows = []
for p in sorted(runs_root.glob("*/summary.json")):
    try:
        s = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        continue
    if str(s.get("experiment_id", "")) != exp_id:
        continue
    rows.append({
        "run_id": str(s.get("run_id", p.parent.name)),
        "attack": str(s.get("attack", "")),
        "defense_enabled": bool(s.get("defense_enabled", False)),
        "asr_attack": float(s.get("asr_attack", s.get("asr", 0.0)) or 0.0),
        "asr_defended": float(s.get("asr_defended", s.get("asr", 0.0)) or 0.0),
        "defense_gain": float(s.get("defense_gain", 0.0) or 0.0),
        "summary_path": str(p),
    })

compare = {"experiment_id": exp_id, "rows": rows}
(exp_root / "compare.json").write_text(json.dumps(compare, ensure_ascii=False, indent=2), encoding="utf-8")

html = "<html><body><h2>P3 Experiment Compare</h2><pre>" + json.dumps(compare, ensure_ascii=False, indent=2) + "</pre></body></html>"
(exp_root / "compare.html").write_text(html, encoding="utf-8")
(exp_root / "manifest.json").write_text(json.dumps({"experiment_id": exp_id, "num_runs": len(rows)}, ensure_ascii=False, indent=2), encoding="utf-8")
print(exp_root / "compare.html")
"@
$compareOut = (& $pythonExe -c $pyCompare)
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "[P3] done. experiment_id=$ExperimentId"
Write-Host "[P3] compare=$compareOut"
