# H1: Dual-teacher distillation into a fixed-budget student

**Status:** open

## Hypothesis
Distilling from two teachers (e.g., a fidelity-oriented large SR model and a perceptual/real-world model) into a fixed ECBSR/SPAN-like x2 student yields higher PSNR-Y and/or lower LPIPS than training the same student with Charbonnier only, at identical TRT FP16 FPS (same exported graph).

## Minimal experiment
- Student: fixed architecture from exp001 that reaches >=60 FPS E2E at 540p->1080p.
- Arm A: Charbonnier(SR, HR). Arm B: Charbonnier(SR,HR) + a*L1(SR,T1) + b*L1(SR,T2) (teacher outputs precomputed offline).
- Same data, iterations, seed (3 seeds each), same export path.

## Metrics
PSNR/SSIM-Y, LPIPS (alex), flow-warp error; TRT FP16 FPS (must be equal within noise since graph is identical).

## Pass / fail
Pass: B improves PSNR-Y by >=0.1 dB or LPIPS by >=0.005 mean over 3 seeds, with non-overlapping seed ranges. Fail otherwise.

## Novelty
Low–medium. Multi-teacher KD for SR: MTKD https://arxiv.org/abs/2404.09571; KD into real-time compressed-content SR: RTSR https://arxiv.org/abs/2411.13362; student design: SPAN https://arxiv.org/abs/2311.12770. Our angle is only the strict FPS-matched video setting.
