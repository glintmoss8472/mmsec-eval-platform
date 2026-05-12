# 文件说明：该文件属于运维与实验脚本，集中实现 build frontend 相关逻辑。
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
