# 文件说明：该文件属于运维与实验脚本，集中实现 prepare coco subset 相关逻辑。
Param(
  [string]$Root = "data/coco",
  [string]$Split = "val2017",
  [int]$MaxItems = 500,
  [switch]$DownloadAnnotations,
  [switch]$DownloadImages,
  [string]$AutoDownload = "true",
  [string]$AllowSyntheticFallback = "false"
)

$ErrorActionPreference = "Stop"
$scriptPath = Join-Path $PSScriptRoot "prepare_coco_subset.py"
$args = @($scriptPath, "-Root", $Root, "-Split", $Split, "-MaxItems", $MaxItems.ToString(), "-AutoDownload", $AutoDownload, "-AllowSyntheticFallback", $AllowSyntheticFallback)
if ($DownloadAnnotations.IsPresent) { $args += "-DownloadAnnotations" }
if ($DownloadImages.IsPresent) { $args += "-DownloadImages" }
& python @args
exit $LASTEXITCODE
