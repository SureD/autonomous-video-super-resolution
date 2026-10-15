# exp001: latency baselines (540p -> 1080p, x2)

- **Status:** queued (needs user at dscs2pc; no GPU on agent box)
- **Hardware:** dscs2pc, RTX 3070 8GB, driver 610.88, CUDA UMD 13.3
- **Purpose:** measure which candidate architectures fit the 60 FPS budget before any training-side experiment (H1–H5 all need a fixed, budget-compliant student).

## Models
`espcn`, `fsrcnn`, `ecbsr` (merged after reparam; m=4, ch=16), `ecbsr_plain` (same topology, plain convs — should match `ecbsr` latency), `plain` (32ch x 8 conv). Untrained weights: latency does not depend on weights.

## Commands (PowerShell, repo root)
```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1        # once
powershell -ExecutionPolicy Bypass -File scripts\run_exp001.ps1           # PyTorch FP16/FP32 + ONNX export
powershell -ExecutionPolicy Bypass -File scripts\run_exp001.ps1 -Trt      # + trtexec --fp16 if trtexec on PATH
```
Benchmark settings: input 1x3x540x960, warmup 50, 300 timed iters, CUDA events, `cudnn.benchmark=True`.

## Expected outputs (in `results/`)
- `bench_fp16_960x540_<time>.json`, `bench_fp32_960x540_<time>.json`: env (GPU, torch, CUDA, cuDNN), per-model params, `torch_net` {mean/median/p95 ms, fps}, `torch_e2e_copy` (fp16 run: uint8 H2D + normalize + net + uint8 D2H), ONNX path, exact `trtexec_cmd`.
- `*.onnx` (gitignored), `*_trtexec.log` + `trtexec_summary` lines when `-Trt`.
- `nvidia-smi.csv` (clocks/power at start).

## Pass criteria
- A model **passes** if `torch_e2e_copy.fps_mean >= 60` (PyTorch FP16) or TRT FP16 throughput >= 60 qps with transfers added back; final decision uses the slower of the two available E2E numbers.
- Note: decode/display are **not** in this timing. Before any claim of "≥60 FPS end-to-end" per ADR 0001, a follow-up must add NVDEC/ffmpeg decode + present (exp001 is network + copies only).
- Experiment outcome: record which models pass; pick the largest passing student (with >=20% headroom, i.e. >=72 FPS) as the fixed student for H1–H5.

## Results
_None yet. Do not fill in without raw JSON committed alongside._
