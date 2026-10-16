# exp001: preliminary latency screening (540p -> 1080p, x2)

- **Status:** queued (needs user at dscs2pc; no GPU on agent box)
- **Hardware:** dscs2pc, RTX 3070 8GB, driver 610.88, CUDA UMD 13.3
- **Purpose:** screen candidate network and selected-copy costs. This experiment cannot establish deployment compliance or select a deployment-qualified student.

## Models
`espcn`, `fsrcnn`, `ecbsr` (merged after reparam; m=4, ch=16), `ecbsr_plain` (same topology, plain convs — should match `ecbsr` latency), `plain` (32ch x 8 conv). Untrained weights: latency does not depend on weights.

## Commands (PowerShell, repo root)
```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1        # once
powershell -ExecutionPolicy Bypass -File scripts\run_exp001.ps1           # PyTorch FP16/FP32 + ONNX export
powershell -ExecutionPolicy Bypass -File scripts\run_exp001.ps1 -Trt      # + trtexec --fp16 if trtexec on PATH
```
Benchmark settings: input 1x3x540x960, warmup 50, 300 timed iters per repeat, synchronized wall time + CUDA events, `cudnn.benchmark=True`.

## Expected outputs (in `results/`)
- `bench_fp16_960x540_<time>.json`, `bench_fp32_960x540_<time>.json`: env (GPU, torch, CUDA, cuDNN), per-model params, `torch_net` list of repeat results {wall/CUDA mean/median/p95 ms, fps, per-frame samples}, `torch_e2e_copy` (fp16 run: uint8 H2D + normalize + net + uint8 D2H), ONNX path, exact `trtexec_cmd`.
- `*.onnx` (gitignored), `*_trtexec.log` + `trtexec_summary` lines when `-Trt`.
- `nvidia-smi.csv` (clocks/power at start).

## Interpretation

There is no deployment pass criterion in exp001. A separate TensorRT FP16
playback harness must measure decode through output buffer under ADR 0001 before
selecting the fixed deployment student. The current trtexec run excludes transfers.

Both network-only and `--e2e` copy-inclusive paths report synchronized
`perf_counter` wall mean/median/p95 and CUDA-event timing separately, with raw
per-frame samples in measurement order. Three independent warmup/measurement runs
are saved by default (`--repeats`). Input layouts are contiguous NCHW in both paths.
Copy-inclusive begins with a preallocated pinned RGB uint8 frame and includes
upload, normalization/layout conversion, network, clipping/rounding and uint8
output download. Decode, presentation and application output-buffer handling
are excluded. These measurements are preliminary screening only.

## Results
_None yet. Do not fill in without raw JSON committed alongside._
