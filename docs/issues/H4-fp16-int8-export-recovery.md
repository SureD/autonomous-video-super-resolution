# H4: FP16-aware finetune / INT8 QAT recovers export drop

**Status:** open

## Hypothesis
A short FP16-aware finetune (for FP16) or INT8 QAT (TensorRT Q/DQ) recovers >=70% of the PSNR-Y drop between the FP32 PyTorch model and the TensorRT FP16/INT8 engine.

## Minimal experiment
1. Measure drop D = PSNR(FP32 torch) - PSNR(TRT engine) for FP16 and INT8 (PTQ calibration).
2. If D_FP16 < 0.02 dB, FP16 arm is moot (record as such). Otherwise finetune with autocast FP16 forward.
3. INT8: QAT with NVIDIA pytorch-quantization, export Q/DQ ONNX, build TRT INT8 engine.
4. Recovery R = (PSNR_after - PSNR_before_engine) / D.

## Metrics
PSNR/SSIM-Y of engine outputs; engine FPS (INT8 must be >= FP16 FPS to matter).

## Pass / fail
Pass: R >= 0.7 for the precision under test with D >= 0.05 dB. Fail if R < 0.7. Inconclusive if D < 0.05 dB.

## Novelty
Low. QAT: https://arxiv.org/abs/1712.05877; TensorRT toolkit https://github.com/NVIDIA/TensorRT/tree/main/tools/pytorch-quantization. Reparam+quantization interaction is a known issue; measurement on our student is the contribution.
