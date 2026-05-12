# 文件说明：该文件属于运维与实验脚本，集中实现 run cli 相关逻辑。
Param(
  [Parameter(Mandatory = $true)][string]$Command,
  [string]$Config = "configs/mvp.yaml",
  [string]$Sweep = ""
)

$ErrorActionPreference = "Stop"

if ($Command -eq "ingest-docs") {
  python -m mmsec_eval ingest-docs --config $Config
} elseif ($Command -eq "run-eval") {
  python -m mmsec_eval run-eval --config $Config
} elseif ($Command -eq "run-sweep") {
  if ([string]::IsNullOrWhiteSpace($Sweep)) {
    python -m mmsec_eval run-sweep --config $Config
  } else {
    python -m mmsec_eval run-sweep --config $Config --sweep $Sweep
  }
} elseif ($Command -eq "run-benchmark") {
  python -m mmsec_eval run-benchmark --config $Config
} else {
  Write-Host "Unsupported command: $Command"
  exit 1
}
