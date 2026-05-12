Param(
  [switch]$SkipTests,
  [switch]$SkipInstall,
  [switch]$GpuFull,
  [switch]$UseSystemProxy
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "scripts\_python_env.ps1")

function Assert-Command($name) {
  if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
    Write-Host "[ERR] Missing command: $name"
    exit 1
  }
}

try {
  Write-Host "[INFO] Bootstrapping mmsec-eval-platform ..."
  Assert-Command "git"

  $projectRoot = Get-ProjectRoot -ScriptRoot $PSScriptRoot
  Set-Location $projectRoot

  $pythonExe = Get-ProjectPython -ProjectRoot $projectRoot -EnsureVenv
  Write-Host "[INFO] Using Python: $pythonExe"
  $pyv = (& $pythonExe -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')").Trim()
  Write-Host "[INFO] Python version = $pyv (target: 3.11+)"

  if (-not $SkipInstall) {
    if (-not $UseSystemProxy) { Use-DirectProxyForPip }

    & $pythonExe -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "Failed to upgrade pip." }

    if (Test-Path ".\requirements.txt") {
      & $pythonExe -m pip install -r requirements.txt
      if ($LASTEXITCODE -ne 0) { throw "Failed to install requirements.txt" }
    }
    if ($GpuFull) {
      Write-Host "[INFO] Ensuring CUDA-enabled PyTorch (GPU-only requested)..."
      & (Join-Path $projectRoot "scripts\\install_torch_cuda.ps1")
      if ($LASTEXITCODE -ne 0) { throw "Failed to install CUDA-enabled PyTorch." }
    }
    if ($GpuFull -and (Test-Path ".\requirements-gpu.txt")) {
      & $pythonExe -m pip install -r requirements-gpu.txt
      if ($LASTEXITCODE -ne 0) { throw "Failed to install requirements-gpu.txt" }
    }
    if (Test-Path ".\requirements-dev.txt") {
      & $pythonExe -m pip install -r requirements-dev.txt
      if ($LASTEXITCODE -ne 0) { throw "Failed to install requirements-dev.txt" }
    }
  } else {
    Write-Host "[WARN] SkipInstall enabled. Dependency installation was skipped."
  }

  Write-Host "[INFO] Using in-repo package shim (no editable install required)."
  if (-not $SkipTests) {
    & $pythonExe -m pytest
    if ($LASTEXITCODE -ne 0) { throw "pytest failed." }
  } else {
    Write-Host "[INFO] SkipTests enabled"
  }

  Write-Host "BOOTSTRAP OK"
  exit 0
} catch {
  Write-Host "[BOOTSTRAP FAILED]"
  Write-Host $_.Exception.Message
  exit 1
}
