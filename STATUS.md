# STATUS

Last updated: 2026-10-20. Single source of truth for open hypotheses and next steps.

## Where things stand (2026-10-20)

The literature review and research notes 01–04 are on `main`: [independent review](docs/research/cursor-independent-review.md), [related work](docs/research/01-related-work-check.md), [roofline](docs/research/02-roofline-latency-model.md), [agenda](docs/research/03-research-agenda.md), [embedding-guided compression](docs/research/04-embedding-guided-compression.md). Metric contract for new runs is [ADR 0002](docs/decisions/0002-downsample-kernel.md) (shared Pillow bicubic) and [ADR 0003](docs/decisions/0003-metric-contract.md) (BT.709 full-range Y, peak 255, no border crop). ADR 0001 is still the proposed deployment protocol.

The EmbeddingGemma 2 broad CPU probe is **NO-GO for an EG2-specific training budget**. DINOv2 stays the embedding-guidance baseline. Report: [embeddinggemma2-probe-report.md](docs/research/embeddinggemma2-probe-report.md). The probe's PSNR-Y used the pre-ADR BT.601 limited-range formula; those scores were not recomputed. The conditional-go plan in PR #9 is superseded and was not merged.

First GPU results exist on a separate local RTX 4060 training machine. They are intentionally kept in a local-only repository on that machine and are **not in this GitHub repo**. Reported there, and not re-measured here:

- exp001 latency, PyTorch FP16: all 5 students under 16.7 ms. Examples: ECBSR merged 3.53 ms, ESPCN 4.66 ms, FSRCNN 9.70 ms, plain 32-channel ×8 10.67 ms.
- exp002 ECBSR ×2 baseline on a tiny 192-frame train set: val PSNR-Y 32.74 vs bicubic 30.90, SSIM-Y 0.926 vs 0.889.

Those 4060 latencies must be re-measured on the RTX 3070 before any deployment claim. exp001 in this repo remains preliminary screening and has no deployment pass. `docs/reports/gpu-budget-proposal.md` still names a ≥72 FPS headroom target; the exp001 README and the roofline note do not.

On 2026-10-20, `python3 -m pytest -q` with torch 2.14.1+cpu, onnx 1.23.2, and onnxruntime 1.31.0 reported 25 passed and 8 warnings (legacy TorchScript ONNX export). No GPU was used for that run.

## Goal

Real-time VSR at ≥60 FPS end-to-end on RTX 3070 8GB (TensorRT FP16), 540p→1080p ×2, quality competitive for consumer playback. Prefer training-side gains that do not raise inference cost.

## Current question

Which candidate students warrant deployment measurement after preliminary screening (exp001), and then which training-side changes (KD, reparam, degradation, export-aware training, temporal loss) improve quality at fixed inference cost?

## Hypotheses

| ID | Claim | State | Draft | Evidence |
|----|-------|-------|-------|----------|
| H0 | Fixed eval protocol (ADR 0001) is enough to compare training-only runs | open | docs/decisions/0001 | ADR proposed; not locked |
| H1 | Dual-teacher KD into fixed-budget ECBSR/SPAN-like student beats Charbonnier-only at equal TRT FP16 FPS | open | [H1](docs/issues/H1-dual-teacher-distillation.md) | none |
| H2 | Train-time multi-branch reparam beats plain training of merged arch | open | [H2](docs/issues/H2-train-time-reparam.md) | none (merge equivalence verified only, CPU) |
| H3 | H.264 degradation training improves LPIPS/tLPIPS on H.264 clips, ≤0.2 dB clean PSNR loss | open | [H3](docs/issues/H3-compression-degradation.md) | none |
| H4 | FP16-aware finetune / INT8 QAT recovers ≥70% of export drop | open | [H4](docs/issues/H4-fp16-int8-export-recovery.md) | none (drop not yet measured) |
| H5 | Temporal-consistency loss on 2–3 frame student lowers tLPIPS, ≤0.2 dB PSNR loss | open | [H5](docs/issues/H5-temporal-consistency-loss.md) | none |

## Experiments queued (ordered)

1. **exp001-latency-baselines** — [README](experiments/exp001-latency-baselines/README.md). Run `scripts\run_exp001.ps1` on dscs2pc. Gates everything else.
2. exp002 — Charbonnier baseline training of the chosen student (configs/train_ecbsr.yaml) + eval; needs training frames staged.
3. exp003 — H2 (ecbsr vs ecbsr_plain, 3 seeds).
4. exp004 — H4 step 1: measure TRT FP16/INT8 drop of exp002 model.
5. Then H3, H1, H5.

## Code state

- `src/vsr/`: models (ESPCN, FSRCNN, ECBSR reparam + merge, plain student), data (frame folder, bicubic, ffmpeg H.264), metrics (PSNR/SSIM-Y, Farneback warp error).
- `scripts/`: benchmark.py (FP16 CUDA-event timing, ONNX, trtexec), eval.py (PSNR/SSIM/LPIPS/tLPIPS/warp error), train.py (Charbonnier, YAML, resume, seed), setup_windows.ps1, run_exp001.ps1.
- `tests/`: contract, baseline, and smoke tests. The 2026-10-20 CPU run above is 25 passed when onnx and onnxruntime are installed. CUDA/TRT paths untested in that run.

## Decided / proposed

- Eval protocol: [ADR 0001](docs/decisions/0001-evaluation-setup.md) (Proposed). Downsample kernel and Y metrics for new comparisons: ADR 0002 and ADR 0003.
- GPU rental: [survey](docs/reports/gpu-rental-survey.md). Do not rent yet; use the local 3070 first. The 4060 numbers above are not a substitute.
- Literature: [training-side survey](docs/literature/training-side-optimizations.md), plus the research notes and independent review linked above.
- Embedding guidance: DINOv2 is the baseline to test later. EG2-specific training budget is no-go on the 2026-10-17 CPU probe.

## Hardware

| Machine | Role | GPU |
|---------|------|-----|
| Agent box | Docs / git / CPU smoke tests | none |
| dscs2pc (Windows) | Train + TRT inference | RTX 3070 8GB, driver 610.88, CUDA UMD 13.3 |

## Blockers

1. ~~GitHub repo missing~~ **Resolved (live 2026-10-15):** [SureD/autonomous-video-super-resolution](https://github.com/SureD/autonomous-video-super-resolution) exists; H1–H5 are issues #1–#5; exp001 is #6. Agent-box checkout now has `origin` (HTTPS). Local `main` and `origin/main` share the same tree; commit histories differ (local linear vs remote import). `gh` still not logged in on the agent box, so pushes need Cursor GitHub / cloud agent / user auth.
2. RTX 3070 measurement is still required. The 4060 PyTorch FP16 timings and the tiny-set exp002 scores are local-only and are not deployment evidence.
3. Training/eval datasets (REDS / Vimeo-90K / consumer H.264 clips) not staged on dscs2pc (path unknown). Sample clips (BBB + Xiph 1080p + Vid4) are prepared on the agent box under `/workspace/data/prepared/` for transfer later.
4. TensorRT/trtexec install status on dscs2pc unknown.

## Do not do yet

- Rent cloud GPUs.
- Deployment claims before a separate ADR 0001 pipeline measurement on the RTX 3070. exp001 only screens candidates. Do not treat the local 4060 timings as that measurement.
- An EG2-specific cache or student-training sweep. The 2026-10-17 probe is no-go; DINOv2 remains the baseline to test, and that test has not been run.

## Changes by Codex implementer

- Added ADR 0002 and shared Pillow antialiased bicubic for frame-folder training,
  preparation, and clean evaluation; eval now includes a bicubic PSNR-Y/SSIM-Y floor.
- Added configurable exp002 Charbonnier runner, provisional ECBSR config, artifact
  capture, and regression tests. No research results or latency eligibility claimed.
- This environment has Pillow/NumPy but lacks torch, cv2, yaml, and pytest.
  `python3 -m pytest -q` failed before collection: `No module named pytest`.
  Synthetic two-iteration exp002 attempt failed on import: `No module named cv2`.
  Dependency-light resize/preparation checks: 2 passed, 0 failed; compileall and
  diff whitespace checks passed. Full tests, training, real data and GPU checks pending.
