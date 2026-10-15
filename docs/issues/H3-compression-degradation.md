# H3: Video-compression degradation vs bicubic-only training

**Status:** open

## Hypothesis
Training the student on LR frames re-encoded with H.264 (random CRF) in addition to bicubic downsampling improves LPIPS and tLPIPS on H.264-compressed test clips, while losing <=0.2 dB PSNR-Y on clean (bicubic-only) test LR.

## Minimal experiment
- Arm A: bicubic LR. Arm B: bicubic then H.264 re-encode via ffmpeg (CRF uniform in [18,35]), p=0.5 (src/vsr/data.py --h264).
- Two test sets: clean bicubic LR; H.264 CRF 23 and 30 LR.

## Metrics
LPIPS, tLPIPS, flow-warp error, PSNR/SSIM-Y.

## Pass / fail
Pass: on H.264 sets B lowers LPIPS and tLPIPS (mean over 3 seeds), AND clean PSNR-Y drop <=0.2 dB. Fail otherwise.

## Novelty
Low. Real-ESRGAN https://arxiv.org/abs/2107.10833, RealBasicVSR https://arxiv.org/abs/2111.12704, AnimeSR https://arxiv.org/abs/2206.07038, RTSR https://arxiv.org/abs/2411.13362 (AV1).
