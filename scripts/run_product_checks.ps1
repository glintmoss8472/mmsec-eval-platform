Param(
  [string]$PythonExe = "",
  [switch]$UseSystemProxy
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "_python_env.ps1")

$projectRoot = Get-ProjectRoot -ScriptRoot $PSScriptRoot
Set-Location $projectRoot

if ([string]::IsNullOrWhiteSpace($PythonExe)) {
  $PythonExe = Get-ProjectPython -ProjectRoot $projectRoot -EnsureVenv
}

Write-Host "[INFO] Ensuring Python requirements ..."
if (-not $UseSystemProxy) { Use-DirectProxyForPip }
& $PythonExe -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { exit 1 }
if (Test-Path ".\requirements-dev.txt") {
  & $PythonExe -m pip install -r requirements-dev.txt
  if ($LASTEXITCODE -ne 0) { exit 1 }
}

& $PythonExe -m pytest -q
if ($LASTEXITCODE -ne 0) { exit 1 }

if (-not (Get-Command pnpm -ErrorAction SilentlyContinue)) {
  corepack enable | Out-Null
  corepack prepare pnpm@9.12.3 --activate | Out-Null
}

pnpm -C frontend install
if ($LASTEXITCODE -ne 0) { exit 1 }

pnpm -C frontend test
if ($LASTEXITCODE -ne 0) { exit 1 }

pnpm -C frontend build
if ($LASTEXITCODE -ne 0) { exit 1 }

$apiSmoke = @'
from fastapi.testclient import TestClient
from mmsec_api.main import app

client = TestClient(app)
r = client.get("/api/v1/health")
assert r.status_code == 200
r2 = client.get("/api/v1/bootstrap/status")
assert r2.status_code == 200
print("API SMOKE PASS")
'@
$apiSmoke | & $PythonExe -
if ($LASTEXITCODE -ne 0) { exit 1 }

powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_gate_checks.ps1 -PythonExe "$PythonExe"
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "PRODUCT CHECKS PASS"
