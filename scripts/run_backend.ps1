# 文件说明：该文件属于运维与实验脚本，集中实现 run backend 相关逻辑。
Param(
  [string]$ListenHost = "127.0.0.1",
  [int]$Port = 8000,
  [string]$PythonExe = "",
  [string]$BootstrapEnabled = "0",
  [switch]$Reload
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "_python_env.ps1")

$projectRoot = Get-ProjectRoot -ScriptRoot $PSScriptRoot
Set-Location $projectRoot

if ([string]::IsNullOrWhiteSpace($PythonExe)) {
  $PythonExe = Get-ProjectPython -ProjectRoot $projectRoot -EnsureVenv
}

Write-Host "[INFO] Backend Python: $PythonExe"
$env:MMSEC_BOOTSTRAP_ENABLED = "$BootstrapEnabled"
Write-Host "[INFO] MMSEC_BOOTSTRAP_ENABLED=$($env:MMSEC_BOOTSTRAP_ENABLED)"
if ($Reload) {
  & $PythonExe -m uvicorn mmsec_api.main:app --host $ListenHost --port $Port --reload
} else {
  & $PythonExe -m uvicorn mmsec_api.main:app --host $ListenHost --port $Port
}
