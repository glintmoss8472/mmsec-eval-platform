# 文件说明：该文件属于运维与实验脚本，集中实现 python env 相关逻辑。
Set-StrictMode -Version Latest

# 中文注释：实现 Get-ProjectRoot 的核心流程，支撑运维与实验脚本中的业务语义和异常边界。
function Get-ProjectRoot {
  param([string]$ScriptRoot)
  if ([string]::IsNullOrWhiteSpace($ScriptRoot)) {
    return (Resolve-Path ".").Path
  }
  $selfPath = (Resolve-Path $ScriptRoot).Path
  if (Test-Path (Join-Path $selfPath "pyproject.toml")) {
    return $selfPath
  }
  $parentPath = (Resolve-Path (Join-Path $selfPath "..")).Path
  if (Test-Path (Join-Path $parentPath "pyproject.toml")) {
    return $parentPath
  }
  return $selfPath
}

# 中文注释：实现 Get-VenvPythonCandidates 的核心流程，支撑运维与实验脚本中的业务语义和异常边界。
function Get-VenvPythonCandidates {
  param([string]$ProjectRoot)
  return @(
    (Join-Path $ProjectRoot ".venv313\Scripts\python.exe"),
    (Join-Path $ProjectRoot ".venv312\Scripts\python.exe"),
    (Join-Path $ProjectRoot ".venv311\Scripts\python.exe"),
    (Join-Path $ProjectRoot ".venv\Scripts\python.exe")
  )
}

# 中文注释：实现 Find-ProjectPython 的核心流程，支撑运维与实验脚本中的业务语义和异常边界。
function Find-ProjectPython {
  param([string]$ProjectRoot)
  foreach ($candidate in (Get-VenvPythonCandidates -ProjectRoot $ProjectRoot)) {
    if (Test-Path $candidate) { return $candidate }
  }
  return ""
}

# 中文注释：实现 Find-SystemPython 的核心流程，支撑运维与实验脚本中的业务语义和异常边界。
function Find-SystemPython {
  $versions = @("3.13", "3.12", "3.11")
  foreach ($ver in $versions) {
    # 中文注释：实现 try 的核心流程，支撑运维与实验脚本中的业务语义和异常边界。
    try {
      $exe = (& py "-$ver" -c "import sys; print(sys.executable)" 2>$null).Trim()
      if (-not [string]::IsNullOrWhiteSpace($exe)) {
        return @{ Launcher = "py"; Version = $ver; Executable = $exe }
      }
    } catch {
      continue
    }
  }

  $fallback = (Get-Command python -ErrorAction SilentlyContinue)
  if ($null -ne $fallback) {
    return @{ Launcher = "python"; Version = "default"; Executable = $fallback.Source }
  }
  throw "Python 3.11+ not found. Install Python 3.11/3.12/3.13 first."
}

# 中文注释：实现 Ensure-ProjectVenv 的核心流程，支撑运维与实验脚本中的业务语义和异常边界。
function Ensure-ProjectVenv {
  param([string]$ProjectRoot)
  $existing = Find-ProjectPython -ProjectRoot $ProjectRoot
  if (-not [string]::IsNullOrWhiteSpace($existing)) { return $existing }

  $sysPy = Find-SystemPython
  $targetDir = ".venv"
  if ($sysPy.Version -eq "3.13") { $targetDir = ".venv313" }
  elseif ($sysPy.Version -eq "3.12") { $targetDir = ".venv312" }
  elseif ($sysPy.Version -eq "3.11") { $targetDir = ".venv311" }

  $venvPath = Join-Path $ProjectRoot $targetDir
  if (-not (Test-Path $venvPath)) {
    Write-Host "[INFO] Creating virtual env: $venvPath"
    if ($sysPy.Launcher -eq "py") {
      & py "-$($sysPy.Version)" -m venv $venvPath
    } else {
      & python -m venv $venvPath
    }
    if ($LASTEXITCODE -ne 0) { throw "Failed to create virtual environment: $venvPath" }
  }

  $venvPython = Join-Path $venvPath "Scripts\python.exe"
  if (-not (Test-Path $venvPython)) {
    throw "Virtual environment python not found: $venvPython"
  }
  return $venvPython
}

# 中文注释：实现 Get-ProjectPython 的核心流程，支撑运维与实验脚本中的业务语义和异常边界。
function Get-ProjectPython {
  param(
    [string]$ProjectRoot,
    [switch]$EnsureVenv
  )

  $current = Find-ProjectPython -ProjectRoot $ProjectRoot
  if (-not [string]::IsNullOrWhiteSpace($current)) { return $current }
  if ($EnsureVenv) { return (Ensure-ProjectVenv -ProjectRoot $ProjectRoot) }
  return ""
}

# 中文注释：实现 Use-DirectProxyForPip 的核心流程，支撑运维与实验脚本中的业务语义和异常边界。
function Use-DirectProxyForPip {
  $env:HTTP_PROXY = ""
  $env:HTTPS_PROXY = ""
  $env:ALL_PROXY = ""
  $env:NO_PROXY = "*"
}
