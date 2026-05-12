# 文件说明：该文件属于运维与实验脚本，集中实现 prepare flickr1k 相关逻辑。
Param(
  [string]$Root = "data/flickr1k",
  [string]$SourceRoot = "data/flickr30k",
  [string]$SourceImageDir = "images",
  [string]$ImageDir = "images",
  [string]$OutputFile = "captions_index.jsonl",
  [string]$AutoDownload = "true",
  [int]$MaxItems = 1000,
  [switch]$SkipDownload,
  [string]$AllowSyntheticFallback = "false"
)

$ErrorActionPreference = "Stop"
$scriptPath = Join-Path $PSScriptRoot "prepare_flickr1k.py"
$args = @(
  $scriptPath,
  "-Root", $Root,
  "-SourceRoot", $SourceRoot,
  "-SourceImageDir", $SourceImageDir,
  "-ImageDir", $ImageDir,
  "-OutputFile", $OutputFile,
  "-AutoDownload", $AutoDownload,
  "-MaxItems", $MaxItems,
  "-AllowSyntheticFallback", $AllowSyntheticFallback
)
if ($SkipDownload.IsPresent) { $args += "-SkipDownload" }
& python @args
exit $LASTEXITCODE
