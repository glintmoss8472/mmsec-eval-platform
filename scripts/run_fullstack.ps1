# 文件说明：该文件属于运维与实验脚本，集中实现 run fullstack 相关逻辑。
Param(
  [int]$ApiPort = 8000,
  [int]$WebPort = 5173,
  [switch]$SkipBootstrap
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "_python_env.ps1")

$projectRoot = Get-ProjectRoot -ScriptRoot $PSScriptRoot
Set-Location $projectRoot

$pythonExe = Get-ProjectPython -ProjectRoot $projectRoot -EnsureVenv

if (-not $SkipBootstrap) {
  Write-Host "[INFO] Installing backend dependencies for zero-manual startup ..."
  powershell -NoProfile -ExecutionPolicy Bypass -File .\bootstrap_all.ps1 -SkipTests
  if ($LASTEXITCODE -ne 0) { exit 1 }
}

Start-Process powershell -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File .\scripts\run_backend.ps1 -Port $ApiPort -PythonExe `"$pythonExe`""

$ready = $false
for ($i = 0; $i -lt 25; $i++) {
  Start-Sleep -Milliseconds 800
  # 中文注释：实现 try 的核心流程，支撑运维与实验脚本中的业务语义和异常边界。
  try {
    $resp = Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:$ApiPort/api/v1/health" -TimeoutSec 3
    if ($resp.StatusCode -eq 200) {
      $ready = $true
      break
    }
  } catch {
    # keep waiting
  }
}

if (-not $ready) {
  Write-Host "[ERR] Backend failed to become healthy at http://127.0.0.1:$ApiPort/api/v1/health"
  Write-Host "[ERR] Do NOT open frontend only; run .\\scripts\\run_backend.ps1 manually and inspect error output."
  exit 1
}

Start-Process powershell -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File .\scripts\run_frontend.ps1 -Port $WebPort"
Write-Host "Backend: http://127.0.0.1:$ApiPort"
Write-Host "Frontend: http://127.0.0.1:$WebPort"
