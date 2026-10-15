# Autonomous Video Super-Resolution

Private research log for **real-time video super-resolution (VSR)** targeting **60 FPS on consumer NVIDIA GPUs (RTX 3070 class)**.

Owner: SureD (GitHub). Timezone: Asia/Shanghai.

## Goal

Build (or train) a VSR system that runs at **≥60 FPS end-to-end** on an RTX 3070 (8 GB), with quality competitive for consumer playback (streaming / local video), under a fixed real-time inference budget.

Scientific honesty rules:
- Never fabricate papers, numbers, or results.
- Every claim in literature notes must include a verifiable link.
- Experiments record raw outputs; STATUS.md is the single source of truth for hypotheses (verified / refuted / open).

## Current focus

**Training-side optimizations** that improve quality **without increasing** the real-time inference cost (same architecture / same fused ops / same TensorRT FP16 budget).

See:
- [STATUS.md](STATUS.md) — resumable project state
- [docs/literature/training-side-optimizations.md](docs/literature/training-side-optimizations.md) — survey
- [docs/decisions/](docs/decisions/) — Architecture Decision Records (ADRs)
- [docs/issues/](docs/issues/) — hypothesis drafts (to be opened as GitHub issues when `gh` is authenticated)
- [experiments/](experiments/) — one folder per experiment

## Quick start (Windows, dscs2pc)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
powershell -ExecutionPolicy Bypass -File scripts\run_exp001.ps1 [-Trt]
python scripts\train.py --config configs\train_ecbsr.yaml [--resume]
python scripts\eval.py --gt <hr_frames> --model ecbsr --ckpt experiments\<id>\last.pt --lpips
```
CPU smoke tests: `python -m pytest tests -q`.

## How to read this repo

1. Start at **STATUS.md** (current question, hypotheses, next steps, blockers).
2. Read the latest ADR under `docs/decisions/` for evaluation protocol.
3. Literature notes are under `docs/literature/` (citations with links only).
4. Each experiment folder is self-contained: config, env, raw results, notes.

## Hardware (as of 2026-10-14)

| Machine | Role | GPU |
|---------|------|-----|
| Agent box | Bootstrap / docs / git | **No GPU** |
| `dscs2pc` (Windows) | Training & TensorRT inference | **NVIDIA GeForce RTX 3070 8GB**, driver 610.88, CUDA UMD 13.3 |
| MacBook Air (M5) | Optional tooling (`gh`, git) — not used for this bootstrap | — |

## Repo status

Local git repo prepared at `/workspace/autonomous-video-super-resolution`.  
**Blocker:** no remote; `gh` on the agent box is **not authenticated**, so the private GitHub repo `SureD/autonomous-video-super-resolution` and hypothesis issues could not be created/pushed yet.
