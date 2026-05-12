# 文件说明：该文件属于运维与实验脚本，集中实现 reproduce mvp 相关逻辑。
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "_python_env.ps1")

$projectRoot = Get-ProjectRoot -ScriptRoot $PSScriptRoot
Set-Location $projectRoot

$pythonExe = Get-ProjectPython -ProjectRoot $projectRoot -EnsureVenv
$env:MMSEC_SEED = "123"

& $pythonExe -m mmsec_eval ingest-docs --config configs/mvp.yaml
if ($LASTEXITCODE -ne 0) { exit 1 }

& $pythonExe -m mmsec_eval run-eval --config configs/mvp.yaml
if ($LASTEXITCODE -ne 0) { exit 1 }
