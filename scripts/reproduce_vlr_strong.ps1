# 文件说明：该文件属于运维与实验脚本，集中实现 reproduce vlr strong 相关逻辑。
Param(
  [ValidateSet("full", "quick")]
  [string]$Profile = "full",
  [int]$MaxWorkers = 2,
  [int]$Retries = 6
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "_python_env.ps1")

$projectRoot = Get-ProjectRoot -ScriptRoot $PSScriptRoot
Set-Location $projectRoot

$pythonExe = Get-ProjectPython -ProjectRoot $projectRoot -EnsureVenv

# 中文注释：实现 Get-TorchInfo 的核心流程，支撑运维与实验脚本中的业务语义和异常边界。
function Get-TorchInfo {
  param([string]$Py)
  # 中文注释：实现 try 的核心流程，支撑运维与实验脚本中的业务语义和异常边界。
  try {
    $raw = & $Py -c "import json; import torch; print(json.dumps({'torch': torch.__version__, 'cuda': torch.version.cuda, 'cuda_available': bool(torch.cuda.is_available())}))"
    if ($LASTEXITCODE -ne 0) { return $null }
    return ($raw | ConvertFrom-Json)
  } catch {
    return $null
  }
}

$ti = Get-TorchInfo -Py $pythonExe
if ($null -eq $ti) {
  Write-Host "[CUDA] torch not installed or not usable in this venv."
  exit 1
}
if (($ti.cuda -eq $null) -or (-not $ti.cuda_available)) {
  Write-Host "[CUDA] CUDA torch not ready: torch=$($ti.torch) cuda=$($ti.cuda) cuda_available=$($ti.cuda_available)"
  Write-Host "[CUDA] Installing CUDA-enabled PyTorch (this may take a few minutes)..."
  & (Join-Path $PSScriptRoot "install_torch_cuda.ps1")
  if ($LASTEXITCODE -ne 0) { exit 1 }
  $ti = Get-TorchInfo -Py $pythonExe
  if (($ti.cuda -eq $null) -or (-not $ti.cuda_available)) {
    Write-Host "[CUDA] Still not ready after install: torch=$($ti.torch) cuda=$($ti.cuda) cuda_available=$($ti.cuda_available)"
    exit 1
  }
}

# Align adapter model names with configs unless explicitly overridden.
if ([string]::IsNullOrWhiteSpace($env:MMSEC_CLIP_MODEL_NAME)) { $env:MMSEC_CLIP_MODEL_NAME = "openai/clip-vit-base-patch32" }
if ([string]::IsNullOrWhiteSpace($env:MMSEC_BLIP_ITM_MODEL_NAME)) { $env:MMSEC_BLIP_ITM_MODEL_NAME = "Salesforce/blip-itm-base-coco" }
if ([string]::IsNullOrWhiteSpace($env:MMSEC_VILT_ITM_MODEL_NAME)) { $env:MMSEC_VILT_ITM_MODEL_NAME = "dandelin/vilt-b32-finetuned-coco" }
if ([string]::IsNullOrWhiteSpace($env:MMSEC_BERT_MLM_MODEL_NAME)) { $env:MMSEC_BERT_MLM_MODEL_NAME = "bert-base-uncased" }

# Keep HF caches inside repo artifacts/ for reproducible runs and easier cleanup.
if ([string]::IsNullOrWhiteSpace($env:HF_HOME)) { $env:HF_HOME = (Join-Path $projectRoot "artifacts\\hf") }
$env:HF_HUB_DISABLE_TELEMETRY = "1"
$env:HF_HUB_ENABLE_HF_TRANSFER = "1"
$env:TOKENIZERS_PARALLELISM = "false"


# Ensure prefetch runs online (offline flags will be enabled later after prefetch succeeds).
$env:HF_HUB_OFFLINE = ""
$env:TRANSFORMERS_OFFLINE = ""


# 中文注释：实现 Test-HFWeightsReady 的核心流程，支撑运维与实验脚本中的业务语义和异常边界。
function Test-HFWeightsReady {
  param([string]$LocalDir)
  if (Test-Path (Join-Path $LocalDir "pytorch_model.bin")) { return $true }
  $st = Get-ChildItem -Path $LocalDir -Filter *.safetensors -File -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($null -ne $st) { return $true }
  return $false
}

# 中文注释：实现 Prefetch-HFModel 的核心流程，支撑运维与实验脚本中的业务语义和异常边界。
function Prefetch-HFModel {
  param(
    [string]$RepoId,
    [string]$LocalDir,
    [string[]]$Exclude = @()
  )
  $hfExe = Join-Path (Split-Path $pythonExe) "hf.exe"
  if (-not (Test-Path $hfExe)) { return }
  if (Test-HFWeightsReady -LocalDir $LocalDir) { return }
  New-Item -ItemType Directory -Force -Path $LocalDir | Out-Null

  $savedEndpoint = $env:HF_ENDPOINT
  $endpoints = @()
  if ([string]::IsNullOrWhiteSpace($savedEndpoint)) {
    # Default endpoint first, then a mirror fallback for unstable/blocked networks.
    $endpoints = @("https://huggingface.co", "https://hf-mirror.com")
  } else {
    $endpoints = @($savedEndpoint)
  }

  foreach ($ep in $endpoints) {
    $env:HF_ENDPOINT = $ep
    for ($i = 1; $i -le $Retries; $i++) {
      $wait = [Math]::Min(30, 2 * $i)
      Write-Host "[HF] Prefetch: $RepoId -> $LocalDir (endpoint=$ep try=$i/$Retries workers=$MaxWorkers)"
      $rc = 1
      # 中文注释：实现 try 的核心流程，支撑运维与实验脚本中的业务语义和异常边界。
      try {
        $args = @("download", $RepoId, "--local-dir", $LocalDir, "--max-workers", "$MaxWorkers")
        foreach ($x in $Exclude) { $args += @("--exclude", "$x") }
        & $hfExe @args
        $rc = $LASTEXITCODE
      } finally {
        # Always clear locks so the next attempt doesn't appear "stuck".
        Clear-HFLocks -ProjectRootPath $projectRoot
      }

      if ($rc -eq 0 -and (Test-HFWeightsReady -LocalDir $LocalDir)) {
        if ([string]::IsNullOrWhiteSpace($savedEndpoint)) {
          # Keep the endpoint used for successful download inside this session (child processes only).
          $env:HF_ENDPOINT = $ep
        } else {
          $env:HF_ENDPOINT = $savedEndpoint
        }
        return
      }

      Write-Host "[HF] Download failed (exit=$rc). Retrying in ${wait}s ..."
      Start-Sleep -Seconds $wait
    }
  }

  if ([string]::IsNullOrWhiteSpace($savedEndpoint)) {
    Remove-Item Env:HF_ENDPOINT -ErrorAction SilentlyContinue | Out-Null
  } else {
    $env:HF_ENDPOINT = $savedEndpoint
  }
  throw "HF prefetch failed for $RepoId. If your network is unstable/blocked, try setting HF_ENDPOINT=https://hf-mirror.com and/or HF_TOKEN."
}

# 中文注释：实现 Clear-HFLocks 的核心流程，支撑运维与实验脚本中的业务语义和异常边界。
function Clear-HFLocks {
  param([string]$ProjectRootPath)
  $lockRoot = Join-Path $ProjectRootPath "artifacts\\hf\\hub\\.locks"
  if (-not (Test-Path $lockRoot)) { return }
  Get-ChildItem $lockRoot -Recurse -Filter *.lock -File -ErrorAction SilentlyContinue | ForEach-Object {
    cmd /c "del /f /q `"$($_.FullName)`"" | Out-Null
  }
}

# Prefetch models once to avoid "stuck" feeling during first from_pretrained download.
# Exclude large unused weights to reduce download size/time (TF/Flax checkpoints).
Prefetch-HFModel -RepoId $env:MMSEC_CLIP_MODEL_NAME -LocalDir (Join-Path $projectRoot "artifacts\\hf_models\\clip") -Exclude @("tf_model.h5", "flax_model.msgpack")
Prefetch-HFModel -RepoId $env:MMSEC_BLIP_ITM_MODEL_NAME -LocalDir (Join-Path $projectRoot "artifacts\\hf_models\\blip_itm") -Exclude @("tf_model.h5")
Prefetch-HFModel -RepoId $env:MMSEC_VILT_ITM_MODEL_NAME -LocalDir (Join-Path $projectRoot "artifacts\\hf_models\\vilt_itm")
Prefetch-HFModel -RepoId $env:MMSEC_BERT_MLM_MODEL_NAME -LocalDir (Join-Path $projectRoot "artifacts\\hf_models\\bert_mlm") -Exclude @("tf_model.h5", "flax_model.msgpack")
Clear-HFLocks -ProjectRootPath $projectRoot

# After prefetch, run fully offline to avoid lock contention/network jitter.
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

if ($Profile -eq "quick") {
  $cfgBase = "configs/bench/bootstrap_quick_vlr_cuda.yaml"
  $cfgGan = "configs/bench/bootstrap_quick_vlr_gan_cuda.yaml"
  $cfgTmm = "configs/bench/bootstrap_quick_vlr_tmm_cuda.yaml"
} else {
  $cfgBase = "configs/bench/bootstrap_full_vlr_cuda.yaml"
  $cfgGan = "configs/bench/bootstrap_full_vlr_gan_cuda.yaml"
  $cfgTmm = "configs/bench/bootstrap_full_vlr_tmm_cuda.yaml"
}

Write-Host "[VLR] Profile: $Profile"

Write-Host "[VLR] Training AdvCLIP patch..."
& $pythonExe -m mmsec_eval train-advclip --config $cfgBase
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "[VLR] Running VLR eval (AdvCLIP, multi-victim（多受测模型）)..."
& $pythonExe -m mmsec_eval run-vlr --config $cfgBase
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "[VLR] Training AdvCLIP patch (GAN branch)..."
& $pythonExe -m mmsec_eval train-advclip --config $cfgGan
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "[VLR] Running VLR eval (AdvCLIP GAN, multi-victim（多受测模型）)..."
& $pythonExe -m mmsec_eval run-vlr --config $cfgGan
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "[VLR] Running VLR eval (TMM, multi-victim（多受测模型）, image+text)..."
& $pythonExe -m mmsec_eval run-vlr --config $cfgTmm
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "[VLR] Done. See artifacts/runs/<run_id>/summary.json and report.html"
