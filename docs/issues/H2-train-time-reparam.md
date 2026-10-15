# H2: Train-time multi-branch reparam vs plain training of the merged architecture

**Status:** open

## Hypothesis
Training with ECBSR-style multi-branch blocks and merging to plain 3x3 convs gives higher PSNR-Y than training the merged plain-conv architecture directly, with identical inference graph and FPS.

## Minimal experiment
- src/vsr/models.py: ECBSR(reparam=True) vs. the same net built with plain 3x3 (ECBSR merged topology trained from scratch).
- Same recipe, 3 seeds; merge, verify equivalence (max abs diff <1e-4, tests/), export ONNX/TRT.

## Metrics
PSNR/SSIM-Y on eval set; TRT FP16 FPS; also FP16 export drop (interacts with H4).

## Pass / fail
Pass: reparam arm >= +0.05 dB PSNR-Y mean over 3 seeds, FPS equal within 2%. Fail otherwise.

## Novelty
Low (replication). RepVGG https://arxiv.org/abs/2101.03697, ECBSR https://github.com/xindongzhang/ECBSR, RepSR https://arxiv.org/abs/2205.05671.
