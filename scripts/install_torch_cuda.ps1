# 文件说明：该文件属于运维与实验脚本，集中实现 install torch cuda 相关逻辑。
Param(
  [string[]]$CudaIndices = @("cu126", "cu124", "cu121"),
  [switch]$Nightly
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "_python_env.ps1")

$projectRoot = Get-ProjectRoot -ScriptRoot $PSScriptRoot
Set-Location $projectRoot

$pythonExe = Get-ProjectPython -ProjectRoot $projectRoot -EnsureVenv

# 处理 `get PyTorch info` 步骤，封装脚本中的可复用命令片段。
function Get-TorchInfo {
  param([string]$Py)
  # 处理 `try` 步骤，封装脚本中的可复用命令片段。
  try {
    $raw = & $Py -c "import json; import torch; print(json.dumps({'torch': torch.__version__, 'cuda': torch.version.cuda, 'cuda_available': bool(torch.cuda.is_available())}))"
    if ($LASTEXITCODE -ne 0) { return $null }
    return ($raw | ConvertFrom-Json)
  } catch {
    return $null
  }
}

# 处理 `assert PyTorch CUDA ready` 步骤，封装脚本中的可复用命令片段。
function Assert-TorchCudaReady {
  param([string]$Py)
  $ti = Get-TorchInfo -Py $Py
  if ($null -eq $ti) { return $false }
  if (($ti.cuda -eq $null) -or (-not $ti.cuda_available)) { return $false }
  return $true
}

Write-Host "[TORCH] Python: $pythonExe"
$before = Get-TorchInfo -Py $pythonExe
if ($null -ne $before) {
  Write-Host "[TORCH] Current: torch=$($before.torch) cuda=$($before.cuda) cuda_available=$($before.cuda_available)"
}

Write-Host "[TORCH] Uninstalling existing torch packages (if any)..."
& $pythonExe -m pip uninstall -y torch torchvision torchaudio | Out-Null

Write-Host "[TORCH] Upgrading pip..."
& $pythonExe -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Failed to upgrade pip." }

$pypi = "https://pypi.org/simple"

$variants = @()
if ($Nightly) {
  $variants = @(@{ name = "nightly"; base = "https://download.pytorch.org/whl/nightly" })
} else {
  $variants = @(
    @{ name = "stable"; base = "https://download.pytorch.org/whl" },
    @{ name = "nightly"; base = "https://download.pytorch.org/whl/nightly" }
  )
}

$lastErr = ""
foreach ($v in $variants) {
  $base = $v.base
  Write-Host "[TORCH] Channel: $($v.name)"
  foreach ($idx in $CudaIndices) {
    $url = "$base/$idx"
    Write-Host "[TORCH] Installing CUDA build from: $url"
    # 处理 `try` 步骤，封装脚本中的可复用命令片段。
    try {
      # Only install torch here (torchvision/torchaudio are optional and make downloads much heavier).
      & $pythonExe -m pip install --no-cache-dir --index-url $url --extra-index-url $pypi torch
      if ($LASTEXITCODE -ne 0) {
        $lastErr = "pip exit $LASTEXITCODE"
        continue
      }
    } catch {
      $lastErr = $_.Exception.Message
      continue
    }

    if (Assert-TorchCudaReady -Py $pythonExe) {
      $after = Get-TorchInfo -Py $pythonExe
      Write-Host "[TORCH] OK: torch=$($after.torch) cuda=$($after.cuda) cuda_available=$($after.cuda_available)"
      exit 0
    }

    $cur = Get-TorchInfo -Py $pythonExe
    if ($null -ne $cur) {
      Write-Host "[TORCH] Not CUDA-ready after install: torch=$($cur.torch) cuda=$($cur.cuda) cuda_available=$($cur.cuda_available)"
    } else {
      Write-Host "[TORCH] Not CUDA-ready after install."
    }
  }
}

Write-Host "[TORCH] Failed to install a CUDA-enabled PyTorch build."
if ($lastErr) { Write-Host "[TORCH] Last error: $lastErr" }
Write-Host "[TORCH] Tips:"
Write-Host "  - Ensure NVIDIA driver works (run nvidia-smi)."
Write-Host "  - Try again with a different CUDA index: -CudaIndices cu124,cu121"
Write-Host "  - If you use Python 3.13 and stable wheels are unavailable, nightly will be tried automatically."
exit 1
