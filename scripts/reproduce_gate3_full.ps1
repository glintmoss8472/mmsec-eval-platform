$ErrorActionPreference = "Stop"

python -m mmsec_eval ingest-docs --config configs/mvp.yaml
if ($LASTEXITCODE -ne 0) { exit 1 }

python -m mmsec_eval run-eval --config configs/mvp.yaml
if ($LASTEXITCODE -ne 0) { exit 1 }

python -m mmsec_eval run-sweep --config configs/mvp.yaml --sweep configs/sweep/examples.jsonl
if ($LASTEXITCODE -ne 0) { exit 1 }

powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_gate_checks.ps1
if ($LASTEXITCODE -ne 0) { exit 1 }

pytest
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "REPRODUCE GATE3 FULL PASS"
