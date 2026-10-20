# Embedding-guided compression of classical VSR

Research check: 2026-10-16. Target: **540p → 1080p ×2, ≥60 FPS end-to-end, RTX 3070 8 GB, TensorRT FP16**. This is a training-side research proposal, not an experiment report. No GPU training, memory profiling, TensorRT measurement, or quality result was produced for this document.

## Verdict

**Worth testing, with a narrower novelty claim than “foundation embeddings improve SR.”** Output KD, feature KD, latent alignment, VSR pruning, and training-only CLIP guidance for SR all have published precedents. The plausible contribution is a controlled demonstration that **frozen, spatial foundation representations recover quality lost when compressing a deterministic VSR teacher into a student selected by measured deployment latency**, beyond ordinary KD and matched regularization controls.

The owner's thesis has two separable parts. Classical VSR is compressible: the opened pruning and KD literature supports this. **Compression with quality matching the large model is an acceptance criterion, not an established consequence.** For example, the authors' Table 1 in [Structured Sparsity Learning, arXiv:2206.07687v2](https://arxiv.org/html/2206.07687v2) reports REDS4 bicubic PSNR of 31.42 dB for BasicVSR and 31.06 dB for SSL-bi. In image SR, Table 3 of [MTKD, arXiv:2404.09571v1](https://arxiv.org/html/2404.09571v1) reports ×2 Urban100 PSNR of 32.93 / 31.99 / 32.42 dB for EDSR Full / Compact / MTKD. These are **published results under their protocols**, not predictions for our ×2 video task. They demonstrate useful compression with residual gaps in those settings.

Use **DINOv2 dense HR patch targets** as the first foundation arm. Compare internal representation alignment against an output-image perceptual loss; these are different interventions with different training costs. Keep the HR reconstruction anchor, pixel KD, and VSR-teacher feature KD. Remove every auxiliary head and encoder before export. CLIP/SigLIP, SAM, and video encoders are secondary ablations. Text-only embeddings or an LLM are not the starting point: a caption does not specify the spatial detail that the fidelity target requires.

Diffusion-based VSR, diffusion teachers, and diffusion sampling are excluded. REPA is included only because its **alignment objective** motivates an experiment; its generative results do not validate our application.

## Repository constraints and missing evidence

This proposal follows [STATUS](../../STATUS.md), the [training-side survey](../literature/training-side-optimizations.md), [H1](../issues/H1-dual-teacher-distillation.md), [H2](../issues/H2-train-time-reparam.md), [H3](../issues/H3-compression-degradation.md), [H4](../issues/H4-fp16-int8-export-recovery.md), and [H5](../issues/H5-temporal-consistency-loss.md).

- **exp001 gates training.** No student is yet verified at 60 FPS on the RTX 3070. Its network/copy timings also omit decode/presentation; the full [ADR 0001](../decisions/0001-evaluation-setup.md) deployment measurement remains necessary. exp001 is preliminary screening and has no deployment pass and no ≥72 FPS gate. Require ≥60 FPS for the actual pipeline after that measurement.
- The current ECBSR/ESPCN/FSRCNN/plain implementations consume one frame. A causal three-frame variant is a **new student graph requiring its own latency check**, not an existing VSR implementation. SPAN is a literature candidate, not an implemented model here.
- The current trainer implements single-frame Charbonnier training. It lacks temporal sampling, teacher caches, KD hooks, auxiliary projectors, and the proposed AMP/accumulation path. The H.264 helper encodes individual frames; it does **not** simulate an inter-frame codec GOP. Sequence encoding must precede cropping for the proposed video degradation test.
- Current evaluation supplies PSNR/SSIM-Y, LPIPS-Alex, a consecutive-frame LPIPS discrepancy, and Farneback warp error. Warp validity currently checks image bounds, not occlusions. Scene-cut handling, sequence-aware pairing, and forward/backward flow masks are prerequisites for stronger temporal claims.
- Dataset staging, a validated ×2 large-teacher checkpoint, local TensorRT readiness, and actual 8 GB peaks are unverified. Do not rent hardware or start training ahead of the existing project gates.

## Literature actually opened

The following **20 papers from 2022–2025 fall within the requested 2022–2026 window**. Each linked primary source was opened during this research, not merely found in search results. “Full” means the method text was inspected in HTML/PDF; “abstract” limits verification to that abstract. Dates below are first arXiv years unless a venue date is distinguished. This is a focused literature check, not an exhaustive priority search; targeted 2026 searches did not establish an additional direct non-diffusion VSR embedding-compression precedent.

### SR distillation and VSR compression

| Paper / opened primary source | Read level and verified contribution | Consequence for this project |
|---|---|---|
| **Learning Knowledge Representation with Meta Knowledge Distillation for Single Image Super-Resolution** (2022), [2207.08356](https://arxiv.org/abs/2207.08356) | Abstract. Learnable knowledge representation networks use teacher/student correlations and texture-aware dynamic kernels; meta-learning adapts transferred knowledge to the student. | Learning how to represent SR teacher features is established. It is SISR; detailed reproduction settings remain unverified here. |
| **Data Upcycling Knowledge Distillation for Image Super-Resolution** (2023), [2309.14162](https://arxiv.org/abs/2309.14162) | Abstract. Upcycled in-domain data and paired invertible augmentations extend KD; explicitly recognizes that teacher outputs are imperfect GT approximations. | Pixel KD can be redundant or biased on paired training data. Retain GT and include a competent non-foundation KD baseline. |
| **MTKD: Multi-Teacher Knowledge Distillation for Image Super-Resolution** (2024), [2404.09571v1](https://arxiv.org/html/2404.09571v1) | Full. A training-time aggregation network combines teachers; wavelet-domain supervision transfers refined targets. | H1-style multi-teacher SR KD is prior art. Neither foundation semantics nor temporal VSR evidence. |
| **DIPNet: Efficiency Distillation and Iterative Pruning for Image Super-Resolution** (2023), [2304.07018v1](https://arxiv.org/html/2304.07018v1) | Full. Enhanced training targets, feature adapters for teacher supervision, reparameterization, and iterative pruning. | Combining feature KD with compression is established in SISR; it does not establish video or foundation-feature gains. |
| **RTSR: A Real-Time Super-Resolution Model for AV1 Compressed Content** (2024), [2411.13362v1](https://arxiv.org/html/2411.13362v1) | Full. Compact image/block CNN trained with output-level Laplacian KD from CVEGAN and EDSR_baseline after initial training. | Strong compressed-content precedent. Its ×3/×4 experiments and per-image architecture do not verify multi-frame ×2 performance on our GPU. |
| **Structured Sparsity Learning for Efficient Video Super-Resolution** (2022 preprint; CVPR 2023), [2206.07687v2](https://arxiv.org/html/2206.07687v2) | Full. Residual sparsity connections, grouped PixelShuffle pruning, and temporal finetuning aligning pruned/unpruned hidden states; studies bidirectional and unidirectional BasicVSR. | Direct VSR compression precedent and evidence that recurrent errors accumulate. Its parameter/FLOP table excludes SPyNet; do not treat those as complete deployment totals. |
| **RepSR: Training Efficient VGG-style Super-Resolution Networks with Structural Re-Parameterization and Batch Normalization** (2022), [2205.05671v1](https://arxiv.org/html/2205.05671v1) | Full. Richer training branches merge into ordinary convolutions; BN handling changes late in training. | Supports H2. Algebraic merging preserves the trained network's function; this is different from preserving an arbitrary large teacher's quality. |
| **Compiler-Aware Neural Architecture Search for On-Mobile Real-time Super-Resolution** (2022), [2207.12577v1](https://arxiv.org/html/2207.12577v1) | Full. Depth/width search uses compiler/device latency information. | Select a student using actual TensorRT timing. Its mobile device, 720p task, and 40 ms criterion do not transfer to our 16.67 ms/frame budget. |
| **Residual Local Feature Network for Efficient Super-Resolution** (2022), [2205.07514](https://arxiv.org/abs/2205.07514) | Abstract. Efficient aggregation plus contrastive-loss feature selection and multistage warm-start training. | Useful conventional training baseline. Architecture-internal “feature distillation” must not automatically be interpreted as teacher–student KD. |
| **Swift Parameter-free Attention Network for Efficient Super-Resolution** (2023), [2311.12770](https://arxiv.org/abs/2311.12770) | Abstract. Efficient image SR with parameter-free attention. | Possible student topology after benchmarking; changing topology is not a loss-only ablation. |
| **Recurrent Video Restoration Transformer with Guided Deformable Attention** (RVRT, 2022), [2206.02146](https://arxiv.org/abs/2206.02146) | Abstract. Combines within-clip parallel processing, recurrence across clips, and guided deformable attention. | Large non-diffusion video teacher alternative. Neither ×2 checkpoint availability nor 3070 feasibility is verified by this abstract. |

### Foundation guidance, perceptual losses, and alignment

| Paper / opened primary source | Read level and verified contribution | Consequence for this project |
|---|---|---|
| **Zooming Out on Zooming In: Advancing Super-Resolution for Remote Sensing** (2023), [2311.18082v1](https://arxiv.org/html/2311.18082v1) | Full, especially §6.1. Trains a small ESRGAN with an auxiliary L1 distance between CLIP RN50 features of SR and HR. | Direct **non-diffusion CLIP perceptual-loss precedent**. Remote-sensing findings do not establish natural-video PSNR parity or our convergence speed. |
| **SeD: Semantic-Aware Discriminator for Image Super-Resolution** (2024), [2402.19387v1](https://arxiv.org/html/2402.19387v1) | Full. Frozen CLIP RN50 third-stage HR spatial features condition a discriminator through spatial cross-attention. Discriminator/CLIP are absent from SR inference. | Training-only foundation semantics for SR already exists. This is GAN SISR, not compressed classical VSR; perceptual gains need not preserve PSNR. |
| **Exploring Semantic Feature Discrimination for Perceptual Image Super-Resolution and Opinion-Unaware No-Reference Image Quality Assessment** (SFD, 2025), [2503.19295v1](https://arxiv.org/html/2503.19295v1) | Full. Discriminates frozen CLIP intermediate SR/HR features; learned positive/negative text prompts also discriminate global embeddings. | Semantic feature-distribution and text-image guidance are prior art. It does not establish teacher-equivalent reconstruction or temporal consistency. |
| **DINOv2: Learning Robust Visual Features without Supervision** (2023), [2304.07193v2](https://arxiv.org/html/2304.07193v2) | Full. Self-supervised image/patch representations; dense features support spatial tasks, and smaller models are distilled from a larger model. ViT-S/14 uses width 384. | Best first target for dense alignment, as a hypothesis. The paper does not validate a DINO loss for VSR compression. |
| **Sigmoid Loss for Language Image Pre-Training** (SigLIP, 2023), [2303.15343](https://arxiv.org/abs/2303.15343) | Abstract and PDF. Pairwise sigmoid image–text pretraining instead of global softmax normalization. | Semantic comparator, not evidence that global embedding similarity preserves pixels. A proposed SigLIP SR loss remains unverified. |
| **Segment Anything** (2023), [2304.02643v1](https://arxiv.org/html/2304.02643v1) | Full. Promptable segmentation with spatial image embeddings; default 1024² image encoding produces a 64×64×256 embedding. | Candidate region/boundary supervision. Segmentation quality is not photometric fidelity; default resolution costs more than the small DINO target cache. |
| **VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training** (2022), [2203.12602v3](https://arxiv.org/html/2203.12602v3) | Full. Tube-masked video reconstruction pretraining; a ViT-B configuration uses 16 frames and spatiotemporal tokens. | Candidate clip representation loss. Recognition/reconstruction pretraining is not evidence of reduced VSR flicker. |
| **Revisiting Feature Prediction for Learning Visual Representations from Video** (V-JEPA, 2024), [2404.08471v1](https://arxiv.org/html/2404.08471v1) | Full. Masked feature prediction without pixel or text reconstruction; studies motion/appearance representations. | Temporal semantics may help, but invariance to unpredictable low-level detail can conflict with SR fidelity. Sampling stride and token timing must be aligned explicitly. |
| **Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think** (REPA, 2024; ICLR 2025), [2410.06940v4](https://arxiv.org/html/2410.06940v4) | Full. Projects intermediate generator tokens toward clean-image pretrained representations; examines patchwise cosine/contrastive alignment and alignment depth. | **Analogy only: diffusion/flow generation is outside our model scope.** Transfer the alignment idea, not the architecture, sampling, acceleration claims, or quality results. |

The requested BasicVSR++ teacher is described in the separately opened [BasicVSR++: Improving Video Super-Resolution with Enhanced Propagation and Alignment, arXiv:2104.13371](https://arxiv.org/abs/2104.13371), first posted in 2021 and published at CVPR 2022. Its abstract verifies second-order propagation and flow-guided deformable alignment. It is not counted among the 20 sources above, and does not establish our teacher's ×2 implementation or memory use.

## What is novel, and what embeddings can provide

| Claim | Assessment after this check |
|---|---|
| Distill large SR/VSR models; align their intermediate representations | Established; low novelty alone. |
| Use pretrained semantic features or CLIP losses only during SR training | Established by Zooming Out, SeD, and SFD. |
| Use a REPA-like projector to align a small network | An established general mechanism; applying its name is not a contribution. |
| Train with foundation features and achieve teacher-equivalent **classical VSR** at a fixed, measured 3070 deployment budget | Plausible project contribution, **unverified**. This search does not establish a “first” claim. |
| Prove gains originate in learned representations rather than extra regularization, training, or projector capacity | A valuable experimental contribution requiring the controls below. |

Foundation visual features can supply spatial grouping, edges/regions, and semantic invariances learned outside the small paired VSR training set. This is an inference motivating the experiment, not a demonstrated VSR result. They do not restore information uniquely determined by neither the LR observations nor the learned prior. If a teacher sees future frames or a longer history than the student, the gap includes missing observations as well as network capacity.

Keep these mechanisms separate:

1. **SR-teacher feature KD:** student intermediate features imitate task-specific VSR teacher features computed from LR video.
2. **Foundation latent alignment:** a disposable projector maps student intermediate features to frozen embeddings of the HR target. This avoids running the foundation encoder through the student's output at every training step.
3. **Foundation perceptual loss:** compare embeddings of the reconstructed image and HR. Gradients must pass through the frozen encoder on the reconstructed-image branch. Frozen parameters do not make that backward pass free.
4. **Embedding-conditioned teacher/discriminator:** foundation features condition an expensive training teacher or discriminator. SeD is a direct precedent. This requires a separately trained teacher/discriminator and a matched unconditioned control; it is a later experiment, not part of the initial loss comparison.
5. **Text/global embeddings:** CLIP/SigLIP similarities, prompts, or VLM captions supply semantic guidance but weak spatial supervision. Global agreement can coexist with incorrect text, repeated textures, or changed identities. Use image patch features first; do not add a deployed text encoder or foundation model.

## Concrete experiment: fixed student, factorial training ablations

All schedules, weights, thresholds, and memory policies below are **proposed experimental choices**, not literature results or measured optimal settings.

### 1. Freeze deployment and teacher definitions

**Student S:** select the largest passing ECBSR-style merged convolution topology from exp001, then benchmark a causal three-frame version taking RGB `(t−2,t−1,t)` as nine input channels and producing only frame `t`. The residual skip uses the latest RGB frame. Channel count/depth and reparameterization recipe are identical across all training arms. The nine-channel stem and latest-frame skip require implementation; the existing `c=3` API cannot simply be set to nine because output channels must remain RGB. If no three-frame candidate passes, run the same experiment with a one-frame student and explicitly label its information limitation. Do not change architecture between loss arms.

Use batch 1, 960×540 input frames, ×2 output, no lookahead, and fixed boundary handling (repeat the first frame at startup; reset on cuts). Freeze preprocessing, TensorRT versions/build flags, merge procedure, workspace, and timing protocol. Require the complete decode → preprocessing → network → output-buffer pipeline to sustain ≥60 FPS; report mean and p95 milliseconds, dropped frames, and peak GPU memory. A proposed stricter latency target is p95 ≤16.67 ms/frame. Measure frame latency as well as throughput; future-frame buffering cannot be hidden inside an FPS claim.

**Teacher T:** one frozen, fidelity-trained **BasicVSR++-class ×2 teacher**. RVRT is a replacement only if chosen before the ablation and given a new recorded teacher identity. Use seven LR frames `(t−3,…,t+3)` and supervise the center output. This intentionally tests compression from a stronger teacher; future context is available only in offline target generation. Record the complete checkpoint hash, code revision, training/degradation history, and feature hooks.

A verified ×2 checkpoint is a prerequisite. If only ×4 weights are available, adapt the upsampling head and train/finetune the ×2 teacher on training data, then evaluate and freeze it before student comparison. Do not relabel a resized ×4 output as ×2 teacher quality. Teacher preparation is a separate cost; no assumption that a published recipe can be reproduced cheaply on 8 GB is justified.

Also evaluate **T3**, the same teacher restricted to exactly the student's three available frames, outputting the latest frame. If the student falls back to one frame, use **T1** instead, with only the current observation and explicitly documented padding if the teacher implementation requires it. This matched-context reference diagnoses the information gap; T7 remains the primary large-model quality reference. If final gaps are substantial, repeat arms C/D with matched-context KD targets to distinguish capacity/optimization failure from unavailable temporal evidence. Reporting parity only against T3/T1 must not become a claim of matching T7. For cache generation and evaluation, T7 always means independently reset, seven-frame sliding windows; a longer recurrent history would be a different teacher.

### 2. Lock data and optimization

- Train on paired REDS training clips, with a separately fixed validation split excluding REDS4/test clips. Add Vimeo-90K only as a separately declared data expansion, not midway through the arm comparison. Split by source clip before cropping or cache construction.
- Phase I isolates losses with clean ×2 bicubic LR. Pin the exact resizing implementation: this repo's shared kernel is Pillow antialiased bicubic ([ADR 0002](../decisions/0002-downsample-kernel.md)), which is not automatically equivalent to a paper's MATLAB degradation. Do not compare PSNR across differing operators as if protocols matched.
- Predeclare **100,000 training samples**, each with seven synchronized LR crops of 112×112, HR target 224×224, clip/frame IDs, crop coordinates, augmentation, and degradation seed. Student loads only its three frames. This finite manifest supports exact cached targets and gives identical samples/order across paired arms. Cache multiple predetermined crop/flip variants inside that manifest; do not apply new augmentations to stale encoder features.
- Three paired seeds `{0,1,2}`; **200,000 optimizer updates**, effective batch eight target frames, Adam, learning rate `5e-4` with cosine decay to `1e-6`. Start with microbatch two and four accumulation steps. These replace the current default batch/patch for this proposal and require a new experiment config.
- Within each seed, share the first **20,000 Charbonnier-only updates**, then branch all arms, including A. Ramp auxiliary losses from zero to their fixed weights over the next 5,000 updates. All arms finish at the same 200,000 updates; sharing a warmup does not grant any arm additional data or training.
- Use RGB `[0,1]`, Charbonnier epsilon `1e-3`; average all losses over their elements/tokens. Compute loss reductions and normalizations in FP32 under AMP. Checkpoint optimizer, scheduler, scaler, RNG, sampler, and accumulation state. Use the final update for the primary result; validation selects loss settings, never test clips.

Phase II repeats the baseline C, foundation D, and strongest control with **50% clean / 50% H.264 CRF uniformly 18–35** LR, using identical sequence-level encodes, crop manifest, and teacher targets for each arm. Fix encoder version, preset, chroma format, and GOP. Evaluate clean and compressed strata separately. Staged E3 is H.264 CRF 18/23/28/33 and AV1 35/45. CRF 23 is in that set; CRF 30 is not. Generate and freeze any CRF that is not already staged before using it. Do not combine an H3 degradation improvement with an embedding effect.

### 3. Define losses and main arms

Let `y=S(x[t−2:t])`, `h=HR_t`, and `z=T7(x[t−3:t+3])`. All teacher and HR targets are stop-gradient.

- `Lrec = mean sqrt((y−h)^2 + 1e−6)`.
- `Lpix = mean abs(y−z)`. In a regression network this is **output/pixel KD**, the relevant analogue of “logit KD”; there are no classifier logits to soften.
- `Lfeat`: mean cosine distance between aligned student/teacher spatial tokens at **two fixed mid/late feature hooks**. Pool each LR feature map to 16×16; a training-only student 1×1 adapter matches the teacher width. Compare corresponding target-frame features; document hook names/shapes. This pooled design is a memory compromise, not full-resolution feature matching.
- `Lalign(E)`: pool one intermediate student feature map to 16×16 and apply a training-only pointwise MLP `Cs→128→384` with GELU. Match tokenwise L2-normalized frozen target embeddings with mean cosine distance. Primary target `E(h)` is the last patch-token layer of **DINOv2 ViT-S/14**, excluding CLS/register tokens: `16×16×384` at 224². Fix checkpoint, normalization, layer, and token-grid convention. The MLP sees student features only; target features cannot feed the SR reconstruction path.
- `Lout(E)`: mean cosine distance of corresponding normalized `E(y)` and `E(h)` patch tokens, using that same frozen DINO encoder. Target branch is cached/no-grad; the `E(y)` branch retains image gradients.

| Arm | Objective after warmup | Question |
|---|---|---|
| **A** | `Lrec` | Charbonnier baseline. |
| **B** | `Lrec + 0.1 Lpix` | Does output KD help? |
| **C** | `B + 0.01 Lfeat` | What does ordinary VSR feature KD recover? |
| **D** | `C + 0.01 Lalign(DINO-pretrained)` | Does foundation latent alignment add value? Primary comparison: D−C. |
| **E** | `C + 0.01 Lalign(DINO-untrained)` | Same architecture/layer, frozen random initialization: does learned representation matter? |
| **F** | `C + 0.01 Lalign(random-RGB-projection)` | Does a generic spatial auxiliary target explain the gain? |
| **G** | `C + 0.01 Lalign(shuffled-DINO)` | Does correct spatial correspondence matter? |
| **H** | `C + 0.01 Lalign(VGG)` | Does conventional pretrained perceptual supervision explain the gain? |
| **I** | `C + 0.01 Lout(DINO-pretrained)` | Is supervising output embeddings better than aligning internal states? |

For F, flatten nonoverlapping 14×14 RGB patches of the **same normalized HR crop** and apply one fixed Gaussian projection `588→384`, producing the same 16×16×384 target. For G, permute pretrained DINO patch positions with a fixed sample-keyed random permutation, destroying correspondence without altering the token distribution. For H, use fixed pretrained VGG-19 `relu4_4` features (512 channels), pool to 16×16, then apply a fixed seeded `512→384` projection whose matrix has orthonormal rows. E/F/G/H use exactly D's trainable MLP. Random encoder/projection seeds are recorded and paired; any promising result should survive a second random-target seed.

Normalize token vectors identically. To make raw target statistics less of a confound, calibrate a **fixed scalar** per added embedding loss on the same 128 training-only warmup samples so its median gradient norm at the tapped student feature is 10% of `Lrec`'s. Freeze that scalar, report it, and apply the same procedure to D–I, including I through the encoder. The displayed `0.01` is the initial multiplier; the calibrated effective multiplier must be recorded. Never tune E/F/G less carefully than D.

The core run is nine arms × three seeds = **27 student runs**, with reusable warmups. Before committing to that cost, a 30,000-update, one-seed screen checks numerical stability and memory for every arm. Screen results are provisional; confirmatory runs restart from the paired seed warmup and use the full schedule. If tuning is necessary, grant each embedding arm the same validation-only multiplier grid `{0.3,1,3}` around its calibration, then freeze settings before confirmation. Record total tuning/cache/teacher compute as well as final-run compute.

### 4. Temporal and encoder extensions

The first nine arms have no added temporal loss; measure temporal behavior in every arm. Then take C and D and run a paired **2×2 extension**: embedding off/on × temporal loss off/on, initialized from their corresponding completed checkpoints and finetuned for the same additional 20,000 updates. The off/off and on/off arms also receive those updates. Use three adjacent student outputs generated from overlapping causal windows, with identical extra samples for all four cells.

Construct a **new temporal manifest/cache**: outputs `(t−1,t,t+1)` need five student LR frames `(t−3,…,t+1)`; retaining independently reset T7 KD for all three targets needs nine LR frames `(t−4,…,t+4)`, three teacher outputs/feature records, and three HR embedding targets. The earlier seven-frame center-target cache is insufficient. Start with one three-target group per microbatch and accumulate four groups (12 target frames/update), averaging reconstruction/KD/alignment losses over outputs. Use this same batch and data budget in all four cells and account for the additional cache generation cost.

Use GT-derived, frozen flow with forward/backward consistency and out-of-frame masks. Define `r_t = y_t − warp(y_{t−1})` and `rGT_t = h_t − warp(h_{t−1})`; propose `Ltemp = mean_valid abs(r_t−rGT_t)` with weight `0.05`. Matching the GT temporal residual avoids insisting that illumination changes or disocclusions be constant. Exclude cuts. This is a proposed H5 variant, distinct from simply minimizing SR warp error; assess both reconstruction and flicker.

Only after D passes its controls, substitute dense CLIP features for DINO with matched adapters/gradient calibration; use SigLIP global embeddings as a semantic comparator. SAM region features are a boundary-oriented follow-up. VideoMAE/V-JEPA need a separate clip-target experiment: cache HR clip tokens, specify time stride and token alignment, and compare against an untrained video encoder and a time-shuffled control. If the pretrained encoder requires 16 frames, use an explicitly declared longer training clip and its matching controls; repeating three frames silently is not the same setting. All these encoders remain absent at deployment.

## RTX 3070 8 GB: feasible execution strategy and costs

**No fit or training-time number below is measured.** The plausible local route is cropped student training with **offline cached teacher and HR embeddings**, processing each large model in a separate GPU pass. Joint full-resolution large-teacher/student/foundation backpropagation is not the proposed workflow.

| Workload | Starting policy | Evidence boundary |
|---|---|---|
| Student A–H, cached supervision | AMP, LR 112², three inputs, microbatch 2, accumulation 4; teacher and encoders unloaded | Likely tractable for an exp001-sized CNN; actual peak, time, and reparameterized training branches must be profiled. |
| T7 cache generation | Frozen/eval/no-grad, seven LR crops, batch 1; save center output and two pooled features, release temporaries | Try one batch before generating caches. BasicVSR++/RVRT operators, context, and workspace can still cause OOM. |
| DINO HR-target generation | Frozen/eval/no-grad, HR 224², small batches in a separate pass | No target-encoder backward activations. Increase batches only after profiling. |
| Arm I output perceptual loss | Student plus frozen DINO; microbatch 1 with accumulation 8 if needed, encoder activation checkpointing | More expensive than D because gradients traverse `E(y)`; exact fit is unverified. Match effective batch/updates across arms. |
| Teacher ×2 adaptation/training | Separate job, AMP/checkpointing, LR 64² and seven-frame microbatch 1 as an initial probe | Potential major time cost. A complete published large-model training recipe may not fit 8 GB; its quality cannot be assumed. |
| Full-resolution teacher evaluation | Offline, batch 1, process independent seven-frame windows sequentially; CPU feature offload if supported | Preserve T7 resets/context. If spatial tiling is required, validate halos/seams and compare tiled versus untiled small clips. Do not silently redefine the teacher reference. |

Use a measured GPU peak budget of **≤7 GiB** as an initial policy to leave room for the Windows display/driver and allocator variation; this is headroom, not a hardware guarantee. Record both allocated/reserved PyTorch peaks and device-wide usage. OOM fallback is lower microbatch with accumulation, then checkpointing. A patch/context reduction changes the recipe and must be applied uniformly to affected comparisons, with a new manifest/cache identity. Do not report a run as equivalent after silently shortening the teacher context.

The following are **tensor-size calculations**, excluding activations saved for backward, workspaces, optimizer state, and allocator overhead:

| FP16 tensor/cache item | Calculation | Size |
|---|---|---|
| One full LR 64-channel feature | `540×960×64×2 bytes` | 63.3 MiB/frame; 443 MiB for seven frames at just one layer |
| Full HR RGB teacher output | `1080×1920×3×2` | 11.9 MiB/frame; about 1.13 TiB for 100,000 frames |
| One 224² RGB teacher crop | `224×224×3×2` | 294 KiB/sample; about 28.0 GiB for 100,000 |
| One DINO-S/14 token layer at 224² | `16×16×384×2` | 192 KiB/sample; about 18.3 GiB for 100,000 |
| Two pooled teacher maps, assuming width 64 | `2×16×16×64×2` | 64 KiB/sample; about 6.1 GiB for 100,000 |
| Default SAM image embedding | `64×64×256×2` | 2 MiB/sample; about 195 GiB for 100,000 |

Thus the primary T7-output + two teacher-map + one DINO-layer cache is approximately **52.4 GiB per 100,000 crop targets**, conditional on the stated teacher width. Teacher feature width must be read from actual hooks; more layers/contexts scale storage directly. HR/LR source data, controls, metadata, masks, and other encoder targets are additional. Store controls in separate sequential passes or regenerate them to limit disk use; stream bounded batches from sharded/memory-mapped disk caches rather than loading the full cache into host or GPU memory. VGG/DINO-untrained can also be target-only cached.

Parameters alone are misleading: 100 million frozen parameters occupy approximately 191 MiB in FP16, while 100 million trainable parameters with FP32 weights, gradients, and two Adam moments require roughly 1.49 GiB **before activations or extra precision copies**. The 63.3 MiB full-frame feature calculation explains why full-resolution temporal activations are the larger concern.

Cache identity must include dataset/clip split, frame order, crop/augmentation, exact decoded LR sequence and codec settings, teacher/checkpoint/context, feature hooks/pooling, encoder/checkpoint/layer/preprocessing, dtype, and code revision. Cache after the actual image augmentation: cropping or rotating a full-image ViT feature map is generally not equivalent to encoding the cropped/rotated image because attention and position context change. Audit a random sample against recomputation and compare FP16 caches to FP32 targets on a small subset before treating cache precision as harmless.

Measure wall-clock seconds/update and cache seconds/sample locally. For planning, `200,000 × measured seconds/update` gives one run's time; add the 27 runs, shared-warmup savings, tuning, teacher preparation, and cache passes. No defensible hours/days estimate exists yet. Report both training-step efficiency and total GPU-hours; a cached method can shift substantial cost into preprocessing.

## Evaluation, acceptance, and failure interpretation

Freeze source-clip holdouts, frame ordering, cut boundaries, exact degradation, output clamping/rounding, metric versions, and per-clip aggregation before running. The locked metric contract is [ADR 0003](../decisions/0003-metric-contract.md): BT.709 full-range Y (`0.2126 R + 0.7152 G + 0.0722 B`), PSNR peak 255, no border crop, SSIM 11×11 σ=1.5 on valid centers, rounded uint8 images when that is the display output under test. Do not mix those scores with papers that use another Y matrix, peak, or border, and do not put the earlier BT.601 limited-range formula or a 2-pixel border back into the evaluator.

Evaluate REDS4, Vid4, and a held-out 1080p consumer set covering live action, anime, game capture, text/HUDs, fine repeated textures, fast motion, and occlusion. Downsample consumer HR to 960×540 for the primary deployment-resolution quality track. For native-size REDS4/Vid4 quality, build separately recorded shape-specific TensorRT FP16 profiles/engines with the same graph and precision policy; the fixed 960×540 engine cannot accept those sizes directly. Validate numerical export parity for each shape and do not transfer native-size speed figures to the deployment track. Smaller standard datasets are useful diagnostics; they do not replace full-resolution evaluation. Keep clean and compressed strata separate. Staged E3 CRFs are 18/23/28/33, not an unstaged 23/30 pair. The proposed test set is not yet staged.

Report engine-output PSNR/SSIM-Y, LPIPS-Alex, and the repo's tLPIPS definition `abs(LPIPS(SR_t,SR_prev) − LPIPS(HR_t,HR_prev))`, averaged within clips. Add occlusion-masked warp/residual error using the same GT-derived flow for every model; keep the existing Farneback result as a separately labeled diagnostic. These metrics do not certify temporal fidelity individually. Inspect synchronized crops and difference videos for flicker, ghosting, texture changes, and text/identity errors. CLIP/DINO similarity may be diagnostic but cannot be the primary test when used as the training loss.

For three paired seeds, report clip-macro means, seed dispersion, per-clip deltas, and confidence intervals from a paired hierarchical bootstrap over seeds and clips. Frames from one clip are correlated; do not treat them as independent samples. Three seeds provide limited certainty. Predeclare D−C as the primary efficacy comparison, then D against E/F/G/H as the learned-representation controls; report all comparisons and adjust confirmatory multiple-comparison claims rather than selecting favorable datasets.

Proposed decision rules:

1. **Deployment:** every reported winner passes ≥60 FPS end-to-end on the 3070 at 540p→1080p, with peak VRAM <8 GB. Compare exported topology/op counts and benchmark each engine; identical graphs do not guarantee identical numerical behavior or perfectly identical measured times. Investigate FPS differences >2% under controlled repeated timing.
2. **Foundation-specific benefit:** D exceeds C and the strongest matched control by ≥0.05 dB PSNR-Y with a positive paired confidence interval, while LPIPS/tLPIPS do not materially worsen. Alternatively, a perceptual gain ≥0.005 LPIPS with ≤0.05 dB PSNR loss and no temporal regression is useful, but must be labeled a **perceptual tradeoff**, not improved fidelity. These effect-size thresholds are project choices.
3. **Large-teacher quality match:** preregister non-inferiority margins against T7 of **≤0.10 dB PSNR-Y deficit, ≤0.001 SSIM-Y deficit, ≤0.005 LPIPS deficit, and ≤0.005 tLPIPS deficit**, plus no >5% worsening of masked temporal residual error. Require the one-sided 95% upper bound on each deficit to stay within its margin in each primary clean/compressed stratum; if the teacher temporal error is near zero, use a predeclared absolute tolerance instead of an unstable ratio. This is operational equivalence within tolerances, not equality on every frame. Margins must be agreed before results, not relaxed afterward.
4. **Export fidelity:** compare PyTorch FP32, PyTorch AMP/FP16, and TensorRT FP16 quality. If the H4 export gap is material, apply the same recovery procedure to baseline and candidate, then repeat comparisons. Teacher parity must be assessed on the deployed student's outputs.

Track teacher-gap closure as `(Q_arm−Q_A)/(Q_T7−Q_A)` for higher-is-better metrics, only when the teacher exceeds A by a meaningful amount. Report the absolute gap too; do not aggregate incompatible metrics into one “quality matched” number. Close a small teacher gap and a large gap separately.

Risks and how to interpret outcomes:

| Risk / observation | Interpretation and response |
|---|---|
| D beats C but ties untrained/random/VGG controls | Evidence for auxiliary regularization or generic perceptual supervision; insufficient evidence that foundation semantics caused the gain. |
| Embedding similarity improves while PSNR falls or textures change | Representation invariance can reward semantically plausible but incorrect detail. Keep the pixel anchor, lower the auxiliary weight, and report a failed fidelity/parity criterion if it remains. |
| Frame metrics improve but flicker/ghosting worsens | Independent image embeddings do not enforce motion correspondence. Examine the temporal factorial extension; do not claim VSR success from still images. |
| Auxiliary loss falls but reconstruction does not improve | The disposable projector may absorb the alignment without useful change to the reconstruction path. Inspect gradients and compare I; low alignment loss is not the endpoint. |
| Matched-context T3/T1 gap closes but T7 gap persists | Missing temporal evidence is a plausible limit. Longer/recurrent student context is a separate architecture study requiring a fresh latency gate. |
| T7 itself fails on H.264, anime, or text | KD can transfer teacher errors and foundation domain bias. Preserve HR supervision; stratify errors and validate the teacher rather than assuming its targets are superior. |
| Offline caches or pretrained evaluation features leak test data | Construct split/caches from training IDs only; record encoder pretraining provenance where known. Unknown foundation pretraining overlap must be labeled unknown. |
| Pruning/NAS lowers MACs but misses latency | Export operators, memory traffic, flow modules, and TensorRT compilation determine deployment. Return to measured student selection; loss tuning cannot establish a speed claim. |

The next authorized research step is therefore to **finish exp001 and freeze a passing student**, validate a ×2 teacher, profile cache generation/training on the local 3070, and run the paired ablations. This document establishes the sources and experiment design; teacher-quality matching, VRAM fit, training time, and 60 FPS remain unverified until those measurements exist.
