Param(
  [string]$Config = "configs/mvp.yaml",
  [string]$Sweep = "configs/sweep/examples.jsonl"
)

$ErrorActionPreference = "Stop"
python -m mmsec_eval run-sweep --config $Config --sweep $Sweep

