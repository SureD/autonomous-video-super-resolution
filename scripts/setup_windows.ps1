# Creates .venv and installs PyTorch CUDA wheels + deps. Run from repo root in PowerShell:
#   powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1 [-CudaTag cu128]
# Driver 610.88 / CUDA 13.3 UMD is backward compatible with cu12x/cu13x wheels. Check https://pytorch.org/get-started/locally/
param([string]$CudaTag = "cu128", [string]$Python = "python")
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
if (-not (Test-Path ".venv")) { & $Python -m venv .venv }
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install --upgrade pip
& $py -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/$CudaTag"
& $py -m pip install -r requirements.txt
& $py -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
& $py -m pytest tests -q
Write-Host "Optional: install TensorRT and put trtexec.exe on PATH for the TRT path (scripts\run_exp001.ps1 -Trt)."
