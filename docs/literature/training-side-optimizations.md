# Training-side optimizations for real-time VSR (survey)

Last updated: 2026-10-15 (Asia/Shanghai). All links checked to resolve (arXiv abstract titles / HTTP 200) on 2026-10-15.
Scope: techniques that raise quality **without changing the deployed inference graph** (or that change only training-time structure that is merged away before export). No numbers below are our own; any quoted claim is the paper's own claim and must be re-measured under ADR 0001.

## 1. Knowledge distillation (KD)
- Hinton et al., *Distilling the Knowledge in a Neural Network* — https://arxiv.org/abs/1503.02531. Foundation of output-level KD.
- **RTSR**, *A Real-Time Super-Resolution Model for AV1 Compressed Content* — https://arxiv.org/abs/2411.13362. Real-time SR for compressed content; trained with distillation from larger teachers into a small deployable network. Closest prior work to our setting (compressed input, real-time budget).
- **MTKD**, *Multi-Teacher Knowledge Distillation for Image SR* — https://arxiv.org/abs/2404.09571. Combines several teacher outputs and uses a wavelet-domain distillation loss. Direct prior for H1 (dual-teacher).
- **SPAN**, *Swift Parameter-free Attention Network for Efficient SR* — https://arxiv.org/abs/2311.12770. Efficient SR network with parameter-free attention; NTIRE 2024 efficient-SR winner per the authors. Candidate student architecture.
- Takeaway: multi-teacher KD for SR is already published (novelty of H1 is low-medium: applying it under a strict TRT FP16 FPS budget for 540p→1080p video).

## 2. Structural re-parameterization
- **RepVGG** — https://arxiv.org/abs/2101.03697. Train multi-branch (3x3 + 1x1 + identity), merge to a single 3x3 at inference.
- **ECBSR**, *Edge-oriented Convolution Block for Real-time SR on Mobile Devices* (ACM MM 2021) — code: https://github.com/xindongzhang/ECBSR. Multi-branch block (3x3, 1x1-3x3, 1x1-Sobel/Laplacian) merged to one 3x3.
- **RepSR** — https://arxiv.org/abs/2205.05671. Reparam for SR, discusses BN in reparam blocks.
- Takeaway: "reparam beats plain training of merged arch" is the central claim of these papers; H2 is a replication on our budget (low novelty).

## 3. Quantization: FP16 / INT8, QAT
- Jacob et al., *Quantization and Training of NNs for Efficient Integer-Arithmetic-Only Inference* — https://arxiv.org/abs/1712.05877. QAT with fake quantization.
- NVIDIA TensorRT pytorch-quantization toolkit (QAT, Q/DQ export to ONNX) — https://github.com/NVIDIA/TensorRT/tree/main/tools/pytorch-quantization
- TensorRT docs, working with quantized types — https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-quantized-types.html
- **FP32 vs FP16/INT8 mismatch**: models trained in FP32 can lose PSNR after TRT FP16/INT8 export (overflow in large activations, reduced precision in pixel-shuffle residual paths, calibration error for INT8). Reparam-merged weights are known to be harder to quantize (RepVGG-style branches widen weight distributions). We have **no measured drop yet**; H4 must first measure the drop, then attempt recovery.

## 4. NAS / pruning / efficient design
- **FALSR**, NAS for lightweight SR — https://arxiv.org/abs/1901.07261.
- Li et al., *Pruning Filters for Efficient ConvNets* — https://arxiv.org/abs/1608.08710.
- **RLFN**, *Residual Local Feature Network for Efficient SR* — https://arxiv.org/abs/2205.07514 (NTIRE 2022 efficient-SR; also proposes improved feature-based contrastive loss and warm-start training — both training-side).
- Note: NAS/pruning change the inference graph; we treat them as producing a fixed student, then compare training recipes on it.

## 5. Losses
- **Charbonnier** (robust L1) as used in LapSRN — https://arxiv.org/abs/1704.03915.
- **Perceptual (VGG feature) loss** — https://arxiv.org/abs/1603.08155.
- **GAN loss** (SRGAN) — https://arxiv.org/abs/1609.04802; Real-ESRGAN (§6) for practical GAN training.
- **Frequency-domain / FFT loss** (Focal Frequency Loss) — https://arxiv.org/abs/2012.12821.
- **Temporal losses**: TecoGAN ping-pong loss and tLPIPS/tOF metrics — https://arxiv.org/abs/1811.09393. LPIPS metric — https://arxiv.org/abs/1801.03924.

## 6. Degradation modeling
- **Real-ESRGAN** — https://arxiv.org/abs/2107.10833. High-order synthetic degradation (blur, noise, resize, JPEG).
- **RealBasicVSR** — https://arxiv.org/abs/2111.12704. Real-world VSR; video compression degradations, cleaning module, training trade-offs (batch vs. sequence length).
- **AnimeSR** — https://arxiv.org/abs/2206.07038. Learned basic operators for animation degradations.
- Relevance to H3: training with H.264 re-encode degradation is established practice; our contribution is only measuring the clean-PSNR cost under a 60 FPS student.

## 7. Recurrent / multi-frame training
- **BasicVSR++** — https://arxiv.org/abs/2104.13371. Second-order grid propagation and flow-guided deformable alignment; far over our budget but a teacher candidate and reference for training schedules (long sequences, flow-network LR freezing).
- For a 60 FPS student we target 1–3 frame windows; temporal consistency is enforced via loss (H5) rather than heavy alignment.

## 8. Baselines (inference-cheap)
- **ESPCN** — https://arxiv.org/abs/1609.05158 (sub-pixel conv).
- **FSRCNN** — https://arxiv.org/abs/1608.00367.
- ECBSR (above), SPAN (above), RLFN (above).
- Bicubic upscale as the zero-cost floor.

## Open gaps (our hypotheses)
See docs/issues/H1..H5. All are applications/combinations of prior work; novelty assessed low–medium.
