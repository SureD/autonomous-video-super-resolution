# ADR 0001: Evaluation setup proposal

- **Status:** Proposed
- **Date:** 2026-10-14 (Asia/Shanghai)
- **Deciders:** SureD / autonomous research agent

## Context

We need a fixed evaluation protocol before training-side experiments, so quality and latency claims are comparable across runs. Target deployment: consumer PC GPU, specifically **RTX 3070 8GB**, **≥60 FPS** end-to-end.

## Decision (proposal)

### Upscale tasks (spatial)

| Task | Scale | Motivation |
|------|-------|------------|
| 540p → 1080p | ×2 | Common streaming upscale |
| 720p → 1440p | ×2 | High-refresh desktop / gaming capture |
| 1080p → 4K | ×2 | Stretch goal; may miss 60 FPS on 3070 |

Primary reporting task: **540p→1080p ×2**. Others are secondary.

### Quality metrics

- **PSNR / SSIM on Y** (YCbCr / YUV) — fidelity baseline; report means ± std over clips.
- **LPIPS** (Alex or VGG backbone; fix one and stick to it) — perceptual.
- **Temporal consistency:**
  - **tLPIPS** (LPIPS between consecutive SR frames vs consecutive GT), and/or
  - **Flow-warp error** (warp SR_{t+1}→t with GT or estimated flow; L1/L2 on valid pixels).
- Optional: VMAF when comparing against codec+upscale pipelines.

### Latency / FPS

- Measure **end-to-end** wall time: decode LR frame → preprocess → network → postprocess → write/display buffer.
- Primary inference path: **TensorRT FP16** (ONNX → TRT engine). Also log PyTorch FP16 for debugging.
- Report: mean ms/frame, p95, FPS, GPU mem peak, batch=1, no graph capture unless noted.
- Warmup: ≥50 frames discarded; measure ≥300 frames (or full clip).

### Datasets (proposed; to be confirmed when data is staged)

- Synthetic bicubic / BD: REDS / Vid4 / Vimeo-90K-T style splits (standard VSR).
- Real / compressed: VideoLQ-style or self-encoded AV1/H.264 at several QPs (see RealBasicVSR degradation).
- Hold out a small **consumer clip set** (anime + live-action + game capture) never used in training.

### Pass/fail for "real-time"

A configuration **passes** the budget if TensorRT FP16 end-to-end FPS ≥ 60 on RTX 3070 for 540p→1080p at batch 1.

## Consequences

- Training experiments that change **only** training (KD, losses, QAT, curriculum) must keep the **same** exportable inference graph.
- Architecture changes that alter MACs/latency need a new experiment ID and a fresh latency measurement.
- Metrics scripts and exact package versions must be pinned in each `experiments/*/env.md`.

## Open questions

- Exact consumer clip set licensing / sourcing.
- Whether ×3 (360p→1080p) is needed for streaming scenarios (AIM RTSR used ×3/×4).
- Whether to include INT8 TensorRT as a secondary budget track from day one.
