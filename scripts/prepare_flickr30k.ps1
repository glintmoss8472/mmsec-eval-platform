# 文件说明：该文件属于运维与实验脚本，集中实现 prepare flickr30k 相关逻辑。
Param(
  [string]$Root = "data/flickr30k",
  [string]$ImageDir = "images",
  [string]$CaptionsSource = "",
  [string]$OutputFile = "captions_index.jsonl",
  [string]$DownloadUrl = "",
  [string]$AutoDownload = "true",
  [switch]$SkipDownload,
  [string]$AllowSyntheticFallback = "false"
)

$ErrorActionPreference = "Stop"
$scriptPath = Join-Path $PSScriptRoot "prepare_flickr30k.py"
$args = @($scriptPath, "-Root", $Root, "-ImageDir", $ImageDir, "-OutputFile", $OutputFile, "-AutoDownload", $AutoDownload, "-AllowSyntheticFallback", $AllowSyntheticFallback)
if (-not [string]::IsNullOrWhiteSpace($CaptionsSource)) { $args += @("-CaptionsSource", $CaptionsSource) }
if (-not [string]::IsNullOrWhiteSpace($DownloadUrl)) { $args += @("-DownloadUrl", $DownloadUrl) }
if ($SkipDownload.IsPresent) { $args += "-SkipDownload" }
& python @args
exit $LASTEXITCODE
