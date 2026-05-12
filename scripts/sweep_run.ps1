# 文件说明：该文件属于运维与实验脚本，集中实现 sweep run 相关逻辑。
Param(
  [string]$Config = "configs/mvp.yaml",
  [string]$Sweep = "configs/sweep/examples.jsonl"
)

$ErrorActionPreference = "Stop"
python -m mmsec_eval run-sweep --config $Config --sweep $Sweep

