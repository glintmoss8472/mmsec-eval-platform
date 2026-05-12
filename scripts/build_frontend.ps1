$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $projectRoot

if (-not (Get-Command pnpm -ErrorAction SilentlyContinue)) {
  corepack enable | Out-Null
  corepack prepare pnpm@9.12.3 --activate | Out-Null
}

pnpm -C frontend install
if ($LASTEXITCODE -ne 0) { exit 1 }

pnpm -C frontend build
if ($LASTEXITCODE -ne 0) { exit 1 }
