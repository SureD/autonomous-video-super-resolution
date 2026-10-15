# exp001: latency baselines, 960x540 -> 1920x1080 x2, batch 1.
#   powershell -ExecutionPolicy Bypass -File scripts\run_exp001.ps1 [-Trt]
param([switch]$Trt)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
$py = ".\.venv\Scripts\python.exe"
$out = "experiments\exp001-latency-baselines\results"
New-Item -ItemType Directory -Force $out | Out-Null
nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.mem,power.limit,temperature.gpu --format=csv | Out-File "$out\nvidia-smi.csv"
$models = @("espcn", "fsrcnn", "ecbsr", "ecbsr_plain", "plain")
& $py scripts\benchmark.py --models $models --dtype fp16 --e2e --onnx --out $out
& $py scripts\benchmark.py --models $models --dtype fp32 --out $out
if ($Trt) { & $py scripts\benchmark.py --models $models --dtype fp16 --trt --out $out }
Write-Host "Results JSON in $out"
