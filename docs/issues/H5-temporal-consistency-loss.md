# H5: Temporal-consistency loss on a 2–3 frame student

**Status:** open

## Hypothesis
Adding a flow-warped temporal-consistency loss (L1 between SR_t and warp(SR_{t-1}) on occlusion-masked pixels, with flow from GT) to a 2–3 frame input student lowers tLPIPS with <=0.2 dB PSNR-Y loss, at the same FPS.

## Minimal experiment
- Student takes concatenated frames t-1, t (and t+1) as channels; same architecture both arms.
- Arm A: Charbonnier. Arm B: Charbonnier + lambda*temporal loss (lambda in {0.05, 0.1}).
- Eval on REDS4-style clips at 540p->1080p.

## Metrics
tLPIPS, flow-warp error (scripts/eval.py), PSNR/SSIM-Y, LPIPS.

## Pass / fail
Pass: tLPIPS decreases (3-seed mean, non-overlapping ranges) and PSNR-Y drop <=0.2 dB. Fail otherwise.

## Novelty
Low–medium. TecoGAN temporal losses/tLPIPS https://arxiv.org/abs/1811.09393; LPIPS https://arxiv.org/abs/1801.03924; recurrent VSR BasicVSR++ https://arxiv.org/abs/2104.13371.
