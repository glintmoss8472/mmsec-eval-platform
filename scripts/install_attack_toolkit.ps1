Param()

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "_python_env.ps1")

$projectRoot = Get-ProjectRoot -ScriptRoot $PSScriptRoot
Set-Location $projectRoot
$pythonExe = Get-ProjectPython -ProjectRoot $projectRoot -EnsureVenv

Write-Host "[INFO] Installing optional attack toolkit with existing project dependencies."
& $pythonExe -m pip install --upgrade pip
& $pythonExe -m pip install --no-deps "torchattacks==3.5.1"
if ($LASTEXITCODE -ne 0) {
  throw "Failed to install torchattacks without dependencies."
}
Write-Host "[INFO] torchattacks installed without overriding project requests/torch pins."
