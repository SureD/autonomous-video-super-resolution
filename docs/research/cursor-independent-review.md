# Independent review: training-side real-time VSR

- Date: 2026-10-16
- Author: an independent reviewer (not the agent family that wrote `STATUS.md`, `docs/literature/training-side-optimizations.md`, or `docs/issues/H1`–`H5`). Those notes were read as claims to audit, not as authority.
- Scope: fixed-latency real-time video super-resolution, 540p→1080p ×2, RTX 3070 8GB, TensorRT, training-side methods that do not change the exported graph. Diffusion VSR is out of scope for any method this review recommends. No GPU was available. Nothing below is a measured result of this repository.
- Citation rule: a paper is cited only if its full text or a full HTML/PDF rendering was opened in this session. Each citation has the arXiv id or the URL that was opened. Search snippets, abstracts seen only in a hit list, and the repository’s own bibliography are not evidence. Where a number is the paper’s, it is attributed. This review does not invent effect sizes for the 3070 setting.

## 0. How to read the verdicts

| Verdict | Meaning used here |
|---|---|
| Already done | A paper opened here states the same mechanism on SR or VSR, including the efficient / real-time setting where one exists. Re-running it on ECBSR/SPAN under a TensorRT FPS cap is a replication. |
| Incremental | The pieces exist. What remains is a specific interaction, a sign choice, or a measurement this literature did not isolate. Not a new principle. |
| Novel | Not used. Nothing audited here cleared that bar. |

Absence from the papers opened is not proof that no such paper exists. Those gaps are marked **not found in the papers opened**.

The owner’s thesis, as stated for this review, has three parts: (1) classical non-diffusion VSR models are large but compressible; (2) the compressed student must match the large model’s quality; (3) foundation-model embeddings might be the training signal that makes (2) true. Part (1) is supported as an engineering fact by the compression papers in §5. Part (2) is not supported by those papers, and two distillation papers opened here argue the opposite of naive matching (§5.3). Part (3) is tested in §5, not assumed.

## 1. H1 — Dual-teacher output distillation

**Claim.** Two teachers (fidelity and perceptual), L1 to each, into a fixed ECBSR/SPAN-like student, beat Charbonnier at the same TensorRT FP16 graph.

**Papers opened (2022–2026 unless noted).**

1. MTKD, multi-teacher KD for image SR, with a learned aggregator and a wavelet distillation loss. https://arxiv.org/abs/2404.09571
2. RTSR, dual-teacher KD (CVEGAN and EDSR baseline) for a small CNN on AV1 video, 360p→1080p and 540p→4K, AIM 2024 real-time VSR challenge. https://arxiv.org/abs/2411.13362
3. AIM 2024 efficient VSR for AV1 compressed content, the challenge report that contains that dual-teacher entry and other compact students. https://arxiv.org/abs/2409.17256
4. SPAN, NTIRE 2024 efficient-SR winner; the challenge model is trained with an NGswin teacher and L1. https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/papers/Wan_Swift_Parameter-free_Attention_Network_for_Efficient_Super-Resolution_CVPRW_2024_paper.pdf
5. NTIRE 2024 efficient SR challenge report. The report’s own conclusions name knowledge distillation and re-parameterization as central, and individual entries distill from HAT into re-parameterized students. https://arxiv.org/abs/2404.10343
6. NTIRE 2025 efficient SR challenge report. Distillation and re-parameterization are again listed as standard tools; SPAN variants remain a baseline. https://arxiv.org/abs/2504.10686
7. FACD, feature-domain adaptive contrastive distillation for lightweight EDSR/RCAN students. https://arxiv.org/abs/2211.15951
8. DUKD, a critique of output-level KD for SR: teacher images are noisy approximations of GT, so matching them is a weak channel for “dark knowledge.” https://arxiv.org/abs/2309.14162
9. Outside the window, but opened: space-time distillation for compact VSR (STD, CVPR 2021). https://openaccess.thecvf.com/content/CVPR2021/papers/Xiao_Space-Time_Distillation_for_Video_Super-Resolution_CVPR_2021_paper.pdf

**Closest prior.** RTSR. It is dual-teacher (one GAN, one fidelity CNN), real-time, compressed video, and one track is 540p. The differences from H1 are real and small: AV1 rather than a clean bicubic mix, ×3/×4 rather than ×2 540p→1080p, and a perceptual loss from CVEGAN rather than two L1 terms. MTKD is stronger than H1’s recipe on the image side: it does not stop at `a*L1(SR,T1)+b*L1(SR,T2)`. It trains an aggregation network and a wavelet loss, and it reports gains up to 0.46 dB PSNR over other KD methods, which is the paper’s number, not a prediction for this student.

**What remains.** The effect of this exact loss on this student at this FPS cap. That is a replication with a different codec and scale. SPAN already uses a single teacher inside the efficient-SR recipe H1 wants to adopt.

**Verdict: already done.** H1’s draft calls the novelty low–medium. The opened record supports “already done.” DUKD is a reason to expect a small or null PSNR gain from output L1, not a reason to expect a new result.

## 2. H2 — Train multi-branch, merge, beat plain 3×3

**Claim.** ECBSR-style multi-branch training beats training the merged plain network, same inference graph.

**Papers opened.**

1. RepSR. Train multi-branch with BN, merge to 3×3. The paper’s own reading of ECBSR: edge filters without BN “can only work well on tiny networks” in their experiments, and plain VGG-style stacks are the inference target. https://arxiv.org/abs/2205.05671
2. RepQ, QAT for re-parameterized nets, including ECBSR-m4c8, ×2, DIV2K. Defines plain, merged, and RepQ arms. https://arxiv.org/abs/2311.05317
3. QARepVGG. Re-parameterized nets quantize badly; the remedy is a quantization-friendly branch layout, not “train the merged net.” https://arxiv.org/abs/2212.01593
4. RT4KSR, structural re-parameterization inside a real-time 4K SR baseline (NTIRE 2023). https://openaccess.thecvf.com/content/CVPR2023W/NTIRE/papers/Zamfir_Towards_Real-Time_4K_Image_Super-Resolution_CVPRW_2023_paper.pdf
5. NTIRE 2023 real-time 4K SR challenge report. Runtime on commercial GPUs is the ranking axis. https://openaccess.thecvf.com/content/CVPR2023W/NTIRE/papers/Conde_Efficient_Deep_Models_for_Real-Time_4K_Image_Super-Resolution._NTIRE_2023_CVPRW_2023_paper.pdf
6. NTIRE 2024 efficient SR report (re-parameterization called out in the conclusions; ECB blocks merged to 3×3 in submitted students). https://arxiv.org/abs/2404.10343
7. NTIRE 2025 efficient SR report (same conclusion; EFDN/SPAN lineage uses re-parameterized branches). https://arxiv.org/abs/2504.10686
8. RepNet-VSR, re-parameterized real-time 4× VSR, CVPR 2025 MAI challenge winner on a mobile NPU. https://arxiv.org/abs/2504.15649
9. REPVSR, recurrent VSR whose SR and flow blocks are multi-branch in training and a single 3×3 at inference. https://www.scitepress.org/Papers/2025/131869/131869.pdf

**Closest prior.** RepSR for the image claim H2 actually states, and REPVSR / RepNet-VSR once the student is allowed to be video. RepSR’s warning matters: ECB without a BN story is exactly the block this repository implements, and RepSR reports that this family does not extend past tiny nets. H2 does not mention BN. A positive H2 result on a tiny ECBSR and a null result on a wider student would both be consistent with RepSR.

**What remains.** Whether, on the one architecture exp001 keeps, the multi-branch parameterization beats plain training under one optimizer. That comparison is the point of RepVGG-style papers. Doing it again is a replication. The pass bar of +0.05 dB is below the gap RepQ’s Table 3 shows between methods on ECBSR at 8-bit (hundredths of a dB; see §4), so a pass at +0.05 dB is a small effect even if real.

**Verdict: already done** as a claim. Useful only as a local replication, and only if BN handling is recorded. The repository’s own “low novelty / replication” label is right.

## 3. H3 — H.264 degradation versus bicubic

**Claim.** Random-CRF H.264 on the training LR improves LPIPS and tLPIPS on H.264 tests, with ≤0.2 dB clean PSNR loss.

**Implementation fact, from this repo, not from a paper.** `src/vsr/data.py` `h264_roundtrip` encodes **one** RGB frame as an mp4. That is an intra-coded still, not a GOP. `scripts/prepare_data.py` `codec_roundtrip` encodes the **frame sequence** with libx264 preset medium (default GOP) or SVT-AV1. H3 as drafted therefore trains on a different degradation from E3. This is a design bug. It is also why candidate (b) is the experiment H3 thought it was running.

**Papers opened.**

1. RealBasicVSR (CVPR 2022). Second-order image degradations plus video codecs (`libx264`, `h264`, `mpeg4`) at random bitrates. The paper states that video compression, unlike blur/noise/JPEG, depends on inter-frame prediction and is spatially and temporally varying. https://openaccess.thecvf.com/content/CVPR2022/papers/Chan_Investigating_Tradeoffs_in_Real-World_Video_Super-Resolution_CVPR_2022_paper.pdf and the supplement https://openaccess.thecvf.com/content/CVPR2022/supplemental/Chan_Investigating_Tradeoffs_in_CVPR_2022_supplemental.pdf
2. FTVSR, compressed VSR; training mixes uncompressed and H.264 CRF 15/25/35 sequences. https://arxiv.org/abs/2208.03012
3. Compression-aware VSR (CVPR 2023). H.264 on Vimeo sequences at CRF 0/15/25/35; the method consumes frame type, motion vectors, and residuals because I/P/B frames are not the same degradation. https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Compression-Aware_Video_Super-Resolution_CVPR_2023_paper.pdf
4. VCISR, synthetic **video-codec** artifacts as training degradation for blind SISR, motivated by the claim that JPEG-style image degradations miss inter-prediction artifacts. https://arxiv.org/abs/2311.00996
5. SRWD, blur/noise/binning plus video compression (`libx264`, `h264`, `mpeg4`, random bitrate) for blind VSR. https://arxiv.org/abs/2305.02660
6. AIM 2024 AV1 VSR challenge: full sequences encoded with SVT-AV1 at several CRF/QP, then cropped. https://arxiv.org/abs/2409.17256
7. RTSR, trained on those AV1 sequences. https://arxiv.org/abs/2411.13362
8. COMISR (ICCV 2021, opened because later papers define themselves against it). Training on compressed frames without an architecture change gave only modest gains in their experiments; they also discuss unknown intra-frame locations. https://arxiv.org/abs/2105.01237

**Closest prior.** RealBasicVSR for “add a video codec to the degradation soup,” and CAVSR / VCISR for the claim that a video codec is not JPEG. H3’s pass condition (better LPIPS and tLPIPS on compressed clips, bounded clean PSNR loss) is a standard robustness check, not a new degradation.

**What remains.** The clean-PSNR cost for **this** 60 FPS student. COMISR’s qualitative report (compressed-frame finetuning without architectural support can fail to help) is a reason to expect a null, not a reason to skip the measurement. The draft’s framewise encoder does not measure the degradation the hypothesis names.

**Verdict: already done.** The repo’s “low novelty” label is right. Fix the encoder before spending a training run (§6).

## 4. H4 — FP16 finetune / INT8 QAT recovers export drop

**Claim.** Short FP16 finetune or INT8 QAT recovers ≥70% of the PSNR drop from FP32 PyTorch to the TensorRT engine.

**Papers opened.**

1. RepQ, as above. On ECBSR-m4c8 ×2, their Table 3 (PSNR, DIV2K-trained) is:

   | bits | method | Set5 | Set14 | B100 | Urban100 |
   |---|---|---|---|---|---|
   | 32 | FP | 36.90 | 32.62 | 31.43 | 29.66 |
   | 8 | plain | 36.81 | 32.55 | 31.39 | 29.57 |
   | 8 | merged | 36.86 | 32.54 | 31.39 | 29.53 |
   | 8 | RepQ | 36.91 | 32.57 | 31.40 | 29.60 |

   These are the paper’s numbers. The 8-bit drop versus FP32 is a few hundredths of a dB. Their prose says the merged baseline is “inferior to plain” on ECBSR; the table is mixed (merged wins Set5, loses Urban100). https://arxiv.org/abs/2311.05317
2. QARepVGG. INT8 failure of RepVGG is attributed to the multi-branch weight range; their fix is structural. ImageNet, not SR. https://arxiv.org/abs/2212.01593
3. CADyQ, content-aware dynamic bit-width for SR. https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136670360.pdf
4. Tu et al., PTQ for image SR (CVPR 2023), with a comparison table against PAMS, FQSR, CADyQ, DAQ. https://openaccess.thecvf.com/content/CVPR2023/papers/Tu_Toward_Accurate_Post-Training_Quantization_for_Image_Super_Resolution_CVPR_2023_paper.pdf
5. CondiQuant, condition-number PTQ for low-bit SR. https://arxiv.org/abs/2502.15478
6. Outlier-aware PTQ for SR (ICCV 2025). Activation outliers in SR are tied, in their analysis, to color; they also compare with QAT methods (PAMS, DAQ, DDTB, ODM). https://openaccess.thecvf.com/content/ICCV2025/papers/Wang_Outlier-Aware_Post-Training_Quantization_for_Image_Super-Resolution_ICCV_2025_paper.pdf
7. NTIRE 2024 and 2025 efficient-SR reports list quantization among the compression tools, without making it the challenge metric. https://arxiv.org/abs/2404.10343 https://arxiv.org/abs/2504.10686
8. RT4KSR / NTIRE 2023 real-time 4K report: the deployment constraint that matches this project is FP16 runtime on a desktop GPU, not INT8. https://openaccess.thecvf.com/content/CVPR2023W/NTIRE/papers/Conde_Efficient_Deep_Models_for_Real-Time_4K_Image_Super-Resolution._NTIRE_2023_CVPRW_2023_paper.pdf

**Closest prior.** Generic QAT is old (the Jacob et al. 2018 paper is in the repo survey; it was **not** re-opened here). For this architecture, RepQ is the closest: ECBSR, fake-quant on the merged kernel, plain and merged controls. The SR quantization line (CADyQ through the 2025 PTQ paper) is about sub-8-bit image SR, where the drop is large enough to study. RepQ’s 8-bit ECBSR drop is not.

**What remains.** The unmeasured TensorRT FP16 and INT8 drop on the exp001 student, on this 3070. H4 is right that the drop must be measured first. The recovery clause is not a research question until that drop exceeds seed and clip noise (§6). A ≥70% recovery of a 0.02 dB drop is not a result.

**Verdict: already done** as a technique. The open work is a measurement. If `D < 0.05 dB`, the draft’s “inconclusive” branch should be the expected FP16 outcome for a plain conv student, not a surprise. INT8 on a re-parameterized student is the case RepQ and QARepVGG say can hurt; it is still a known case.

## 5. H5 — Occlusion-masked warp loss on a 2–3 frame student

**Claim.** L1 between `SR_t` and `warp(SR_{t-1})` on pixels the GT flow marks visible, plus Charbonnier, lowers tLPIPS with ≤0.2 dB PSNR loss.

**Papers opened.**

1. JFFRA (2025). An occlusion-aware temporal loss: warp the previous restoration with flow, mask by a GT-derived occlusion/stillness weight, L1 against the current restoration, added to a reconstruction loss. They set both weights to 0.2. The network is a multi-scale restorer, not a 60 FPS student. https://arxiv.org/abs/2505.16434
2. TMP, online (causal) VSR aimed at speed; temporal consistency is reported with tOF, TCC, and flow-warping error, not trained as an H5-style loss. https://arxiv.org/abs/2312.09909
3. RealBasicVSR. Under a fixed compute budget, longer training sequences beat larger batches for temporal stability. That is a data-schedule result, not a loss. https://openaccess.thecvf.com/content/CVPR2022/papers/Chan_Investigating_Tradeoffs_in_Real-World_Video_Super-Resolution_CVPR_2022_paper.pdf
4. SAM-guided VSR (SEEM). Semantic masks inside EDVR and BasicVSR. On REDS4 ×4 they report +0.0254 dB PSNR for EDVR and +0.0877 dB for BasicVSR; efficient tuning of SEEM alone is +0.0043 dB on EDVR. These are the paper’s averages. The module stays in the graph. https://arxiv.org/abs/2305.06524
5. MGLD-VSR. Occlusion-masked flow warping, but inside a diffusion sampler. Excluded as a method. https://arxiv.org/abs/2312.00853
6. StableVSR. Temporal consistency for diffusion VSR; excluded. tLPIPS is an evaluation metric in that line of work, which this repo already uses. https://arxiv.org/abs/2311.15908
7. D2-VR. Few-step diffusion video restoration; a temporal-LPIPS **loss** (match the frame-to-frame LPIPS change of the GT). Excluded as a method. Cited only to record that tLPIPS-as-loss exists in 2026 generative restoration. https://arxiv.org/abs/2602.08395
8. AIM 2024 efficient VSR report. Perceptual training losses (VIF/DLM, CVEGAN perceptual loss) on compact video students, sequence-level. https://arxiv.org/abs/2409.17256

TecoGAN (2018/2019), which the issue draft cites for ping-pong loss and tLPIPS, was **not** re-opened. It is not used as evidence here. The mechanism H5 writes down is the one in JFFRA, read in full.

**Closest prior.** JFFRA’s loss. H5 adds a small-student, GT-flow, 2–3 frame constraint. That constraint is not isolated in the papers opened.

**What remains.** Whether the loss still helps when the student sees only two or three frames and cannot represent the warp the loss demands. That is incremental, and it can come out negative: a warp loss is minimized by blurring. The draft does not compare against the GT warp error that `scripts/eval.py` already computes (`warp_err_gt`).

**Verdict: incremental.** The loss is published. The tiny-student test is not, in the papers opened. Do not describe it as a new temporal principle.

## 6. Candidate (a) — student-resolvability weights from teacher disagreement and occlusion

**Idea.** Training-only per-pixel (or per-patch) weights on a temporal distillation loss, down-weighting places where teachers disagree or where flow says occlusion, on the theory that those pixels are not representable by the student.

**Papers opened that bound it.**

1. FACD. Teachers are worse than the student versus GT on up to 5% of EDSR patches and 11% of RCAN patches in their count; those patches are **dropped** from distillation. The gate is distance-to-GT, not teacher–teacher disagreement. https://arxiv.org/abs/2211.15951
2. DUKD. Matching the teacher image is matching a noisy stand-in for GT. Their remedy is different data, not a spatial weight. https://arxiv.org/abs/2309.14162
3. MTKD. Multiple teachers are fused by a network, not by a disagreement map. https://arxiv.org/abs/2404.09571
4. RTSR / AIM 2024. Two teachers, scalar loss weights, no spatial resolvability mask. https://arxiv.org/abs/2411.13362 https://arxiv.org/abs/2409.17256
5. JFFRA. Occlusion mask on a photometric temporal loss, not on a distillation residual. https://arxiv.org/abs/2505.16434
6. FaceMat (face matting, not SR). Uncertainty **up**-weights distillation on ambiguous or occluded pixels. The opposite sign from “skip what the student cannot resolve.” https://arxiv.org/abs/2508.03055
7. STD (2021, outside the window). Spatial attention and ConvLSTM memory distilled uniformly. The student still has a temporal module. https://openaccess.thecvf.com/content/CVPR2021/papers/Xiao_Space-Time_Distillation_for_Video_Super-Resolution_CVPR_2021_paper.pdf
8. DreamSim. Reports that an **untrained** network can act as a perceptual metric (citing prior work). Relevant to the control in §8, and a warning that a structured weight map can help for reasons that are not “resolvability.” https://arxiv.org/abs/2306.09344
9. SAM-VSR. Occlusion/semantic structure from a foundation model is injected into the **inference** graph, and the PSNR movement on strong VSR is <0.1 dB in their tables. https://arxiv.org/abs/2305.06524

**Closest prior.** FACD’s “do not distill the teacher where the teacher is the worse reconstruction,” plus JFFRA’s occlusion mask. Neither uses teacher disagreement as a proxy for student capacity.

**What was not found** in the papers opened: a temporal KD weight defined as a decreasing function of teacher disagreement, justified as student resolvability, on a fixed real-time VSR graph.

**Why the sign is not free.** FACD drops the teacher when the teacher is farther from GT than the student. FaceMat puts more distillation where uncertainty is high. Disagreement is compatible with both stories: hard pixels (up-weight) or unresolvable pixels (down-weight). The candidate states one story. The other is the published alternative.

**Verdict: incremental.** A three-arm weight sign test is worth one training slot after H1, not a paper claim. Arms: uniform KD, disagreement down-weight, disagreement up-weight, plus FACD’s GT-distance gate as the published control. Falsify the resolvability story if the up-weight arm matches or beats the down-weight arm on the pre-registered metric. No expected dB is stated; FACD’s patch rates (5% and 11%) are the only scale this review can point at, and they are patch-discard rates, not quality deltas for a video student.

## 7. Candidate (b) — GOP-coherent codec degradation versus framewise

**Idea.** Train on multi-frame H.264/AV1 with a real GOP (I/P/B, motion-compensated residuals) and compare with independently coded frames.

**Papers opened.** The eight in §3. RealBasicVSR, FTVSR, CAVSR, VCISR, SRWD, AIM 2024, and RTSR all apply a **sequence** codec. CAVSR’s introduction is explicit that an I-frame resembles still-image coding and a P/B-frame does not. VCISR’s motivation is that frame-shuffled or image-only compression is the wrong model of streamed video. COMISR discusses intra-frame placement.

**What was not found** as a controlled ablation in those papers: the same CRF, the same preset, the same frames, sequence mux versus one-frame-per-file, on one frozen student. They compare “codec on” with “codec off,” or they change the network (CAVSR’s ranker, FTVSR’s frequency transformer).

**Verdict: incremental.** The degradation is published. The isolated GOP-versus-framewise contrast was not the experiment in the papers opened. It is still the right next measurement **for this repo**, because training and eval already implement the two arms by accident (`h264_roundtrip` versus `codec_roundtrip`). Run it as a correction of H3, not as a new degradation model.

**Minimal experiment.** One student from exp001. Three training degradations, same CRF marginal (for example the set {18, 23, 28, 33} already used in `prepare_data.py`), probability 1.0 of codec so the mix rate is not a second factor: (A) bicubic only; (B) bicubic then **single-frame** x264, today’s `h264_roundtrip`; (C) bicubic then **sequence** x264, today’s `codec_roundtrip`, GOP left at the encoder default and recorded (`ffprobe` keyframe interval in the run notes). Eval: E1 clean, and E3 sequence encodes. Do not use CRF 30 as a test point unless it is generated; the prepared set is 18/23/28/33.

**Falsify** “GOP structure matters for this student” if C and B are indistinguishable on E3 LPIPS and tLPIPS under the clip-level interval in §9. A win of C over A only reproduces H3 and does not support the GOP claim.

**Effect-size reasoning.** No dB number is predicted. CAVSR and FTVSR report gains for methods that also change the architecture; those gains are not transferable to a fixed ECBSR. The plausible outcomes are: null (the student is too small to use temporal codec structure), or a gain of C over B that is confined to P/B frames and disappears on the I-frame subset. Report I-frames and P-frames separately. If they are not separated, a positive average can be an I-frame effect that framewise training already covers.

## 8. Candidate (c) — train against the merged deployment kernel to beat ordinary QAT

**Idea.** Reduce INT8 damage by training on the re-parameterized, merged weights that TensorRT will run, beyond standard QAT.

**Papers opened.** RepQ and QARepVGG in full, plus the SR PTQ/QAT line in §4 (CADyQ, Tu et al., CondiQuant, outlier-aware PTQ), NTIRE 2024/2025 on re-parameterized students, RepSR on the merge itself.

**What RepQ already did.** During QAT they form the merged kernel as a differentiable function of the multi-branch parameters and fake-quantize that kernel. Their “merged” baseline is: train re-parameterized in FP, merge, then run QAT on the plain conv. Their “plain” baseline never re-parameterizes. On ECBSR, 8-bit RepQ is at or slightly above the FP32 Set5 number in Table 3 (36.91 vs 36.90); merged and plain sit a few hundredths lower. So:

- “QAT on the merged deployment weights” **is** the merged baseline.
- “Something beyond that QAT which respects the re-parameterization” **is** RepQ.
- On this architecture, at 8 bits, the published gap is hundredths of a dB, on still images, in their training recipe, not in TensorRT.

QARepVGG is a third option: change the training branches so the merged kernel has a friendlier range. That changes training structure and still merges to one conv. It was not demonstrated on ECBSR in the paper opened.

**Verdict: already done.** The candidate’s wording matches the baseline RepQ beat by a very small margin, not a gap beyond RepQ. A TensorRT-INT8 rerun on the exp001 net is a port, and only after H4 shows a drop larger than the clip interval. Do not expect RepQ’s “8-bit can exceed FP32” sentence (their classification tables, and their Set5 ECBSR cell) to appear on a 3070 engine; that sentence is a regularization anecdote in their discussion, not a deployment guarantee.

## 9. Three further training-side directions

Diffusion students, new inference modules, and NAS/pruning that change the graph are excluded. Each direction keeps the exp001 graph.

### 9.1 Same-degradation teacher versus clean-input teacher

**Hypothesis.** Under H.264 input, a teacher run on clean bicubic LR produces a target the compressed student cannot implement, so pixel KD increases hallucination. A teacher run on the same compressed clip is a consistent target.

**Why this is not H1.** H1 does not specify the teacher’s input degradation. RTSR trains teachers and student on the same AV1 material; it does not ablate clean-input teachers. DUKD’s claim that teacher pixels are a noisy proxy for GT is the closest argument, and it predicts that any pixel-KD gain is fragile, not that the clean teacher wins.

**Experiment.** One student, three arms, teachers precomputed offline (required on 8GB): (i) Charbonnier; (ii) plus L1 to a classical non-diffusion teacher evaluated on clean LR; (iii) plus L1 to the same teacher evaluated on the student’s sequence-encoded LR. No foundation encoder in this run.

**Falsify** if (iii) does not beat (ii) on E3 LPIPS with a clip-level interval that excludes zero, or if (iii) fails the clean-PSNR non-inferiority test against (i).

**Effect.** No dB figure. The result that would change the project is the sign of (iii)−(ii) on E3, not the magnitude.

### 9.2 Privileged flow versus flow the student can see

**Hypothesis.** H5’s GT-flow warp loss supervises motion that is not a function of the LR. The student spends capacity on a warp it cannot verify. A loss that uses Farneback on the LR (the estimator already in `src/vsr/metrics.py`) or a no-flow temporal L1 (identity warp, pure smoothness) may dominate.

**Experiment.** Four arms, same 2–3 frame student: Charbonnier; plus GT-flow occlusion-masked L1 (H5); plus LR-Farneback occlusion-masked L1; plus unwarped L1 between consecutive SR frames. One λ, chosen on the REDS validation split named in `docs/datasets.md`, not on E1.

**Falsify** the privileged-flow story if the GT-flow arm is best on both tLPIPS and PSNR. Falsify “temporal loss helps” if all three temporal arms tie with Charbonnier inside the clip interval. Report `warp_err_sr − warp_err_gt` so a blurrier video cannot win on warp error alone.

**Effect.** JFFRA uses weight 0.2 on a large restorer. That weight is not an effect size for ECBSR. No transfer.

### 9.3 Can a one-frame student match a multi-frame classical teacher?

**Hypothesis.** The thesis “the compressed student must match the large model” is false when the teacher’s advantage is temporal state the student does not have. Output KD from a multi-frame teacher then targets a conditional expectation over unseen frames and blurs. KD from a single-image teacher of similar spatial strength is the right ceiling.

**Experiment.** Student is the 1-frame exp001 net (lowest latency, no future frames). Two frozen classical teachers, outputs precomputed: T-img (single-image SR) and T-vid (a recurrent VSR such as the BasicVSR class; not a diffusion model). Arms: Charbonnier; L1 to T-img; L1 to T-vid. Report, per clip, the fraction of each teacher’s PSNR gain over bicubic that the student achieves. Pre-register failure of the thesis as: the upper end of the clip-bootstrap interval of that fraction is below 0.5. Also compare the two KD arms. STD (2021) distilled into students that still had temporal memory; it is not this experiment.

**Falsify** the blur prediction if T-vid KD beats T-img KD on both PSNR and tLPIPS. Falsify the pessimistic fraction if the student clears 0.5 of T-vid’s bicubic gain with the interval above 0.5.

**Effect.** Not a dB guess. The fraction is the estimand. SAM-VSR’s <0.1 dB from a semantic plug-in on EDVR/BasicVSR is the only VSR “extra signal” magnitude read in full here, and that plug-in changed the graph. A training-only loss on a smaller student should be treated as smaller than the teacher–bicubic gap until the pilot variance says otherwise.

## 10. Foundation embeddings and compression of classical VSR

This section answers the owner’s question: can CLIP, DINOv2, SigLIP, a VLM, SAM, VideoMAE, or V-JEPA guide training of a **fixed** classical student so that the student matches a large classical teacher? Diffusion VSR is not a candidate method. Papers that use embeddings inside a diffusion sampler are cited as boundary evidence and then set aside.

### 10.1 Papers opened for this section

Representation alignment and embedding losses:

1. REPA. Aligns a projection of diffusion-transformer hidden states to clean DINOv2 (also MAE, MoCov3) patch tokens with a training MLP head. The target is `f(x*)` of the clean image. Reported gains are generation FID and linear-probe accuracy, including a claimed >17.5× training speedup on SiT in the abstract. Not a restoration network, and not a CNN student. https://arxiv.org/abs/2410.06940
2. Align & Invert. Applies REPA **at diffusion inference** for inverse problems, including super-resolution, using a DINOv2 proxy because GT is unavailable at test time. They compare CLIP as an alternative encoder in an appendix and prefer DINOv2 in that setup. Excluded for deployment: it is a diffusion sampler. https://arxiv.org/abs/2511.16870
3. DreamSim. A perceptual metric: LoRA-tuned ensemble of CLIP, OpenCLIP, and DINO, trained on synthetic human judgments. Images are resized to 224. Before tuning, DINO/CLIP/OpenCLIP are less sensitive to low-level distortions than MAE in their BAPPS analysis; the metric attends to foreground and semantic layout. They also drive a diffusion reconstructor with DreamSim distance. They do not train a CNN SR student. https://arxiv.org/abs/2306.09344
4. DA-CLIP. A controller on frozen CLIP predicts a clean-content embedding and a degradation embedding from a **corrupted** input; those are injected by cross-attention (and a prompt) into a restoration net. The graph changes. They show this on IR-SDE (diffusion) and on NAFNet. On their ten-dataset unified-restoration average, NAFNet 26.34 dB PSNR versus NAFNet+DA-CLIP 27.22 dB. That number is not a real-time VSR result and the embedding is present at inference. Their motivation is important: stock CLIP embeddings of degraded inputs are poor (near-zero degradation classification in their table before the controller). https://arxiv.org/abs/2310.01018
5. CLIP-SR. Text and image both enter the SR network; CLIP-ViT and a CLIP discriminator stay in the method. Target regime is 8×–16×, where semantics invent detail. Not a training-only loss, and not ×2 video. https://arxiv.org/abs/2412.11609
6. “Can SAM Boost Video Super-Resolution?” SEEM reads SAM masks and is inserted into alignment, fusion, and reconstruction. Gains quoted in §5. Efficient tuning still leaves SEEM in the net. https://arxiv.org/abs/2305.06524
7. InstructIR. Human instructions, encoded and fed to a NAFNet-style restorer. SISR is an added task (bicubic, then the network enhances an already-upsampled image). Text is an inference input. https://arxiv.org/abs/2401.16468
8. HF-Diff. One-step diffusion SR with a high-frequency perceptual loss and a Jensen–Shannon match in DINOv2 embedding space. They claim, inside that diffusion setup, that the high-frequency loss beats LPIPS and VGG perceptual losses. Excluded as a method. Not evidence that DINOv2 matching helps ECBSR. https://arxiv.org/abs/2411.13548
9. V-JEPA. Latent feature prediction on video, no pixel reconstruction, compared by the authors with VideoMAE-style pixel prediction. No restoration loss, no SR experiment. The VideoMAE paper itself was **not** opened; only V-JEPA’s description of it was. https://arxiv.org/abs/2404.08471
10. Vision–language backbones for no-reference IQA, including CLIP, SigLIP2, DINOv2, DINOv3. This is quality **assessment**, not a training loss for SR. No SigLIP SR-training paper was opened. https://arxiv.org/abs/2509.17374

Compression of VSR and of efficient SR students (classical):

11. Structured sparsity learning for VSR. Prunes residual blocks, recurrent state, and pixel-shuffle of networks in the BasicVSR family, then temporal finetuning because pruning error accumulates in the hidden state. The pruned model is still a VSR model. https://arxiv.org/abs/2206.07687
12. RepNet-VSR. NAS plus re-parameterization for real-time 4× VSR. Their reported operating point is 27.79 dB on REDS validation, 180p→720p, 103 ms per 10 frames on a MediaTek Dimensity NPU. Not a 3070, not ×2 540p, not “matches the teacher.” https://arxiv.org/abs/2504.15649
13. REPVSR. Re-parameterized recurrent VSR (SR and flow), merged to 3×3, aimed at real-time. https://www.scitepress.org/Papers/2025/131869/131869.pdf
14. STD, CVPR 2021 (outside the window; opened). Feature distillation into compact VSR **without** changing inference. Students are VESPCN, VSR-Net, FastDVDnet, teacher EDVR. They report improvement over Charbonnier, not parity with EDVR. https://openaccess.thecvf.com/content/CVPR2021/papers/Xiao_Space-Time_Distillation_for_Video_Super-Resolution_CVPR_2021_paper.pdf
15. AIM 2024 efficient VSR report and RTSR. Compact students, dual-teacher KD, AV1. https://arxiv.org/abs/2409.17256 https://arxiv.org/abs/2411.13362
16. MTKD, FACD, DUKD. Image-SR KD. DUKD and FACD are the skeptical core: teacher outputs are noisy, and on a non-trivial fraction of patches the teacher is the worse reconstruction. https://arxiv.org/abs/2404.09571 https://arxiv.org/abs/2211.15951 https://arxiv.org/abs/2309.14162
17. SPAN and the NTIRE 2024/2025 reports. Single-teacher L1 (NGswin, HAT) into efficient image students is competition practice. https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/papers/Wan_Swift_Parameter-free_Attention_Network_for_Efficient_Super-Resolution_CVPRW_2024_paper.pdf https://arxiv.org/abs/2404.10343 https://arxiv.org/abs/2504.10686
18. RepSR and RepQ. Re-parameterization as the compression that does not change the merged graph; quantization of that graph. https://arxiv.org/abs/2205.05671 https://arxiv.org/abs/2311.05317

A J-STAGE record titled “Improving Perceptual Loss with CLIP for Super-Resolution” (2024) appeared in search. The full text was **not** opened. It is unverified and is not evidence.

SwinRep was named in a citation inside RepNet-VSR. The arXiv HTML fetched for that id was a different paper (solar MHD). SwinRep is **not** treated as read.

### 10.2 What “guide training” already means

Three different mechanisms are being conflated in the thesis. They should not be.

| Mechanism | Inference graph | Opened example | Matches a fixed 60 FPS student? |
|---|---|---|---|
| Embedding injected by cross-attention, prompt, or a plug-in | Changes | DA-CLIP, CLIP-SR, InstructIR, SAM-SEEM | No, unless the module is proven free under the FPS cap, which none of these papers do for a 3070 ×2 video path |
| Training loss: distance between a frozen encoder on the SR image and the same encoder on HR, head discarded | Unchanged | Not found for a classical VSR student. Closest discarded-head alignment is REPA, on a diffusion transformer | The experiment in §10.4 |
| Distillation of a large **SR/VSR** teacher’s pixels or features | Unchanged if the teacher is offline | MTKD, FACD, DUKD, SPAN, RTSR, STD | This is H1, and it does not use CLIP/DINO |

REPA’s head is an auxiliary projection of hidden states onto clean tokens. That pattern (train-time projector, frozen external encoder, clean target) is the one worth copying. Copying it onto a diffusion sampler (Align & Invert, HF-Diff) does not answer the question.

### 10.3 Novelty verdict

**Incremental, and possibly null. Not novel.**

- External pretrained features as a perceptual loss are older than this window (DreamSim treats LPIPS/VGG as the previous generation). Swapping the encoder for CLIP or DINOv2 is an encoder ablation, not a new training principle.
- Using those features **inside** the net is published (DA-CLIP, CLIP-SR, InstructIR, SAM-SEEM) and incompatible with a frozen TensorRT graph unless the module is removed. Removing it is the unpublished part, and it is the part most likely to drop the gain: DA-CLIP’s +0.88 dB average is with the cross-attention present, on a different task.
- REPA shows that alignment to DINOv2 changes diffusion-transformer training. It does not show that a 3×3 video student moves.
- No paper opened here trains a fixed-graph, non-diffusion, real-time VSR student with a frozen CLIP, DINOv2, SigLIP, SAM, VideoMAE, or V-JEPA loss and a random-encoder control.
- The matching thesis is already in trouble without embeddings. FACD discards teacher patches because the teacher is sometimes worse than the student. DUKD says output KD cannot carry much beyond GT. SSL’s pruned VSR is not claimed to equal the dense teacher. STD reports a gain over Charbonnier for compact VSR, not equality with EDVR. “Must match” should be retired as a requirement. The estimand is the fraction of the teacher–bicubic gap recovered (§9.3).

SAM is the wrong interface for a training-only loss: the opened VSR paper uses masks as features inside SEEM. A mask-agreement loss could be written, but that is a new proposal, and their PSNR deltas from actually inserting SAM are <0.1 dB.

V-JEPA and VideoMAE are not training signals for restoration in anything opened here. V-JEPA predicts latent video features; that could in principle be a temporal perceptual target. It has not been shown, and a ViT-L/H video encoder is a poor fit for an 8GB training step if it is run online (§10.5). SigLIP appears in the opened IQA paper as an assessor, not as an SR loss.

### 10.4 Controlled experiment

One student, the exp001 graph, identical ONNX after export (hash the graph). All auxiliary heads are training-only and are absent from the ONNX. Teachers and encoders are frozen. Diffusion teachers are not allowed. One pre-registered primary metric: LPIPS-Alex, because it is **not** any of the training encoders below. PSNR-Y is a non-inferiority constraint, not a second primary. tLPIPS is secondary. Do not select the winner by DINOv2 distance, CLIP cosine, or LPIPS if LPIPS was in the loss.

Arms, same iterations, paired crops, ≥5 seeds only after the variance pilot in §11.1. The issue drafts’ 3 seeds are not this experiment’s sample size.

| Arm | Loss | Role |
|---|---|---|
| A | Charbonnier(SR, HR) | Baseline |
| B | A + L1(SR, T) | Pixel KD. T is one classical teacher, precomputed on the **same** LR the student sees |
| C | B + feature L1 through a 1×1 projector on one teacher stage | Feature KD in the FACD/FAKD sense. If the teacher cannot be re-run, drop C and say so; do not substitute a foundation encoder and still call it feature KD |
| D | A + cosine distance between frozen E(SR) and E(HR) | Foundation alignment on the **image**, head is just the frozen encoder |
| E | B + the D term | Pixel KD plus foundation alignment |
| F | D with E replaced by the same architecture, randomly initialized, frozen | Untrained-encoder control |
| G | D with E replaced by a fixed random Gaussian projection of the pixels (same output dimension, no learned visual semantics) | Random-projection control |
| H | A + a VGG-19 feature L2 (ImageNet convolutional features), coefficient set by the gradient-norm match below | Classical perceptual control, so D must beat VGG and not merely beat pixels |

Encoder for D/E: DINOv2-B, or CLIP ViT-B/32 if DINOv2-B does not fit even as a no-grad pass. One encoder, chosen before looking at E1. A second encoder is a transfer **metric** only (train on DINOv2, report CLIP cosine), never a second primary.

Coefficient rule. At step 0, scale each auxiliary term so its gradient norm on the student is a fixed fraction of the Charbonnier gradient (suggest 0.1, frozen after the pilot). Otherwise the arms compare learning rates, not hypotheses. FACD’s finding that some teacher patches are harmful is handled by an optional pre-registered variant of B and C that zeros the KD term when `||T−HR|| > ||SR−HR||` (their α). That variant is not a new arm in the primary table; it is a sensitivity.

**Falsify** “foundation embeddings guide this student” if:

- the clip-bootstrap interval for D−A and for E−B on LPIPS-Alex contains 0, or
- F or G matches D within that interval (the gain is a regularizer), or
- D does not beat H on LPIPS-Alex.

**Falsify** “the student matches the teacher” if the fraction in §9.3 stays below 0.5 on the E arm. Matching is not the success criterion for the embedding. Beating A and B and H and the random controls is.

**Negative arm worth one extra run, not part of the primary eight:** replace E(HR) with E(LR). DA-CLIP’s result says the degraded embedding is the wrong target. If E(LR) ties E(HR), the encoder is not seeing the residual the student is supposed to learn.

### 10.5 8GB feasibility

The expensive objects are encoder weights and a live VSR teacher, not the loss value.

- Precompute T, the pixel teacher, offline. H1 already says this. Do not hold BasicVSR-class activations and the student optimizer on an 8GB card at once. SSL’s description of BasicVSR (bidirectional recurrent residual stacks, pixel-shuffle a large fraction of the compute) is enough to treat it as a cache, not a co-trained teacher.
- REPA’s target is `f(clean image)`, independent of the student. Cache it. Practical unit: the training crop, not a 1080p frame. A ViT is not translation-equivariant, so cropping a cached full-frame token map is **not** the same as encoding the crop. Either encode the crop under `torch.no_grad()` with a ViT-B or smaller, or freeze the crop coordinates and cache those encodings. Do not cache SAM-H, DINOv2-g, SigLIP-SO400M, or a V-JEPA ViT-L/H as an online forward on this card; those sizes were not benchmarked here, and they are the wrong risk to take before the ViT-B arm has a result. Exact VRAM was **not measured** (no GPU).
- Global CLIP embeddings are small but, as DreamSim’s pipeline makes obvious, they are computed at 224². A 224 resize of a 1080p frame (or of a 192² HR crop upscaled to 224) discards the high-frequency residual that ×2 SR exists to restore. That is a scientific limitation, not a memory win. Prefer patch tokens of the HR crop over a single global vector, and state the resize in the run notes.
- SAM masks, if ever used, are cheap **after** they are cached as low-resolution label maps. Running SAM during the training step is not the 8GB plan. The opened SAM-VSR paper keeps the mask pathway at inference, which this experiment must not do.
- Feature KD (arm C) stores one feature tensor per crop, not a video model. If that tensor does not fit the disk budget, drop C rather than silently switching to an embedding.

### 10.6 Risks

**PSNR drop.** DreamSim’s analysis says untuned DINO and CLIP are less sensitive to the low-level distortions in BAPPS than MAE is, and that tuning them toward mid-level human similarity does not make them high-level-category metrics either. A loss in that space can move layout and object appearance while the pixel residual fights Charbonnier. Non-inferiority on PSNR-Y is mandatory. A large LPIPS win with a PSNR loss outside the margin is a perceptual trade, not a distillation success, and it has to be reported as a trade.

**Hallucination.** CLIP-SR and the text-guided SR line exist because semantic guidance fills in detail that is not in the LR, at large scale factors. At ×2 the missing information is smaller, so the risk is lower, not zero. The failure mode to look for on E5 (the hold-out the datasets note already defines) is texture that matches the semantic class and contradicts the LR: text, gratings, faces. A random-projection control that produces the same artifacts means the loss weight is too high, not that semantics are the cause.

**Temporal flicker.** Arms D–H are per frame. Nothing in them penalizes a texture that flips between frames while both frames stay close in CLIP/DINO space. tLPIPS will catch some of this and must be reported even though it is secondary. V-JEPA was opened; it is a video feature model; it was not opened as a restoration loss, and it is not in the primary arms because a second temporal encoder doubles the comparison and the memory risk. If D improves LPIPS and worsens tLPIPS, the embedding is not a VSR training signal.

**Metric hacking.** Optimizing E and then ranking by E is circular. The primary metric is LPIPS-Alex only because none of D–G train it. If a future run puts LPIPS in the loss, the primary has to move.

**Wrong target.** Aligning to E(HR) uses privileged clean information. That is allowed at train time (REPA does it). Aligning to E(LR) reintroduces the failure DA-CLIP documents. Aligning the student to E(teacher SR) rather than E(HR) stacks two noisy proxies (DUKD’s objection, then an embedding of it). The primary arms use E(HR).

**Regularizer in disguise.** DreamSim records that untrained networks can score perceptual similarity. Arm F exists because of that. If F ties D, stop. Do not add a larger encoder.

## 11. Critique of the experimental design in `docs/issues`

`docs/datasets.md` is ahead of the issue drafts: it already says to report mean±std **over clips**, to tune on REDS validation clips 240–269 or a held-out Tears of Steel slice, and never on E1. The issue drafts do not follow that. The critique below is of the drafts as written (H1–H5), which are the documents that define pass/fail.

### 11.1 The unit of analysis is the clip, and there are few clips

E1 as specified in `docs/datasets.md` is five clips × 48 frames (four TUM plus a Sintel slice). Frames inside a clip are not independent. A mean over frames, or a standard deviation over frames, overstates precision. The drafts say “mean over 3 seeds” and “non-overlapping seed ranges.” With three numbers, range overlap has no calibrated false-positive rate. One bad seed fails a real effect; three lucky seeds pass a null.

**Replace the rule.** For each seed, reduce each clip to one number (mean over frames). The independent replicates are clips, nested in seeds. Analyze the paired difference (arm B − arm A) on the same clip. Interval: cluster bootstrap, resampling clips, keeping frames and pairing. Report the interval, not a range of three seed means. A mixed model `metric ~ arm + (1|clip) + (1|seed)` is the parametric version; with five clips it is still a small-sample interval, and it should be labelled as such.

Five clips cannot both tune λ and confirm a result. `docs/datasets.md` already forbids tuning on E1. Write that into every issue. λ ∈ {0.05, 0.1} (H5) chosen on E1 is a second look at the test set.

Three seeds cannot estimate a variance (two degrees of freedom). Practical plan on one 3070: a pilot of the Charbonnier student, **5 seeds**, used only to estimate clip-level SD. Then set thresholds from that SD. Then 5 seeds for the single pre-registered comparison. Do not set 0.05 dB or 0.10 dB before that pilot. Those numbers are aspirations. RepQ’s entire 8-bit ECBSR gap sits inside 0.1 dB, so a threshold below the pilot SD will “pass” noise.

### 11.2 One primary endpoint, then constraints

H1 passes if PSNR-Y improves by ≥0.1 dB **or** LPIPS by ≥0.005. That is two chances. H3 requires LPIPS and tLPIPS to move and PSNR to stay inside 0.2 dB, judged on point estimates. H5 requires tLPIPS down and PSNR inside 0.2 dB, again on means.

**Pre-register one primary per hypothesis**, matching the scientific claim:

| ID | Primary | Constraint (not a second win) |
|---|---|---|
| H1 | LPIPS-Alex, if the claim is perceptual; PSNR-Y if the claim is fidelity | The other metric is non-inferiority |
| H2 | PSNR-Y | FPS is not a quality metric; see below |
| H3 | LPIPS on the sequence-encoded E3 set | Clean PSNR-Y non-inferiority; tLPIPS secondary |
| H4 | Absolute PSNR of the engine | The ratio R only if D exceeds the pilot interval |
| H5 | tLPIPS | PSNR-Y non-inferiority; also report warp gap versus GT |

Non-inferiority is one-sided: conclude “drop ≤ 0.2 dB” only if the **upper** end of the interval of (baseline − treatment) on clean PSNR is ≤ 0.2 dB. A point estimate of 0.19 dB whose interval reaches 0.4 dB is a fail. The margin 0.2 dB stays arbitrary until the pilot; if the pilot SD of the clip mean is already ~0.2 dB, the margin does not demonstrate anything and must be widened or the clip set must grow. TUM has seven clips and up to 500 frames (`docs/datasets.md`). Use them before treating 0.2 dB as a decision.

H4’s `R = (PSNR_after − PSNR_engine) / D` has a denominator that the draft already suspects can be ~0. When D is smaller than the clip interval, R is noise. Keep the draft’s “inconclusive if D < 0.05 dB,” and also refuse R when D is smaller than the pilot’s clip interval, even if D ≥ 0.05. Report the absolute dB recovered.

### 11.3 Pair the arms

`scripts/train.py` seeds Python, NumPy, and Torch, and seeds the `DataLoader` generator. `FrameFolderSR.__getitem__` then calls the global `random` module for crops, flips, and CRF. With `num_workers > 0`, that stream is not paired across processes or across arms. SyntheticSR in the same file already uses a per-index generator. Training should do the same, so seed *k* of arm A and seed *k* of arm B see the same crops. Unpaired seeds turn a 0.05 dB gate into a comparison of data order.

H1’s teachers are deterministic if precomputed. The seed then only affects the student. That is the right design. Do not re-sample teacher inputs inside the training loop.

### 11.4 H3’s degradation, CRF list, and kernel

Three concrete confounders, each larger than a 0.05 dB gate:

1. Framewise training codec versus sequence eval codec (§3).
2. The draft tests CRF 23 and 30. Prepared E3 is CRF 18/23/28/33 for H.264 and 35/45 for AV1 (`docs/datasets.md`). CRF 30 is not in the set. Either generate it and freeze the command, or test the CRFs that exist.
3. `docs/datasets.md` already records that training LR is `cv2.INTER_CUBIC` and prepared eval LR defaults to PIL bicubic, and that these are not the same kernel. Until one function is used on both sides, no H1–H5 comparison at the draft’s thresholds is interpretable. This should be a blocker in STATUS, not an open caveat.

Also report bitrate, which the datasets note already measured for one clip. A CRF label without bitrate is not a degradation specification.

### 11.5 Temporal metrics can reward blur

`scripts/eval.py` implements tLPIPS as the absolute gap between consecutive-frame LPIPS of SR and of GT, which is a reasonable reading of a temporal LPIPS. `warp_error` runs Farneback on GT and scores the SR pair, and also scores the GT pair. H5’s pass line uses tLPIPS and does not require the SR warp error to approach the GT warp error. A smoother, worse video can reduce a raw warp L1. Primary temporal success should be tLPIPS (distance of temporal change from GT), with `warp_err_sr − warp_err_gt` reported beside it. Bicubic is a required arm on every temporal table: it is temporally stable and spatially poor. A method that does not beat bicubic on the temporal metric and the fidelity metric jointly has not won.

H5 trains with GT flow and evaluates with GT-referenced flow. That is privileged information at train time, which is allowed, but it is the same family of signal as the metric. The LR-flow arm in §9.2 is the control that breaks the circle.

### 11.6 FPS gates

H1: FPS “equal within noise.” H2: “within 2%.” For a merged graph that is identical by construction, a 2% FPS gap means the engine build changed (tactics, clocks, or a failed merge), so the run is invalid, not a quality failure. Lock GPU clocks, record the TensorRT tactic timing cache, and compare engine hashes. Do not spend the quality threshold on latency noise. The ADR’s warmup (≥50) and ≥300 timed frames are reasonable; the issue drafts do not point at them.

INT8 must be at least as fast as FP16 to matter (H4). Agree. Speed without a PSNR interval is not a result.

### 11.7 Multiplicity across H1–H5

Five hypotheses, several metrics, one five-clip test. Even in the planned order (exp001 → H2 → H4 → H3, H1, H5), later tests see the same clips. Pre-register the order and the primary, and do not reinterpret a failed primary with a secondary metric. A held-out confirmation slice does not exist at five clips. Growing E1 is the statistical fix; another seed is not.

### 11.8 What the drafts get right

Measuring the FP16 drop before any recovery claim (H4). Refusing to change the graph inside a training comparison. Requiring the clean-set cost of a degradation or a temporal loss, as a constraint. Precomputing teachers. Those should stay. The numerical gates around them should not be treated as analysis until the pilot SD exists.

## 12. Bottom line

| Item | Verdict | Closest work opened |
|---|---|---|
| H1 dual-teacher L1 | Already done | RTSR; MTKD is a stronger image version |
| H2 re-parameterization vs plain | Already done | RepSR; REPVSR / RepNet-VSR for video |
| H3 codec degradation | Already done, and the draft’s encoder is the wrong codec | RealBasicVSR, CAVSR, VCISR |
| H4 QAT / FP16 finetune | Already done; measure the drop before recovery | RepQ on ECBSR |
| H5 masked warp loss | Incremental (loss published, tiny student not isolated) | JFFRA |
| (a) resolvability weights | Incremental; the sign is not identified | FACD (opposite operationalization), FaceMat (opposite sign) |
| (b) GOP vs framewise | Incremental measurement; the right fix for H3 | CAVSR’s I/P/B distinction, not a frozen-student ablation |
| (c) merged-kernel QAT beyond ordinary QAT | Already done; the wording matches RepQ’s baseline | RepQ |
| Foundation-embedding alignment as a training-only loss | Incremental encoder swap; not shown for this student; matching the teacher is not supported | REPA (diffusion training), DA-CLIP (graph changes) |

Run order implied by this review, still training-only, still one graph: fix the bicubic kernel; pilot variance; H3 corrected as candidate (b); only then H1 with the same-degradation teacher control (§9.1); H5 only with the flow controls (§9.2); the embedding table (§10.4) after pixel KD has a measured gap to beat; H2 and H4 only as replications, H4 only if the engine drop clears the clip interval. Candidate (c) does not need a run distinct from RepQ unless that drop is real.

The compressed student does not have to match the large model for the project to succeed. It has to beat bicubic and beat Charbonnier on a pre-registered metric, inside a non-inferiority constraint, at the same TensorRT graph. The papers opened here do not show foundation embeddings doing that for classical real-time VSR. They show that injecting those embeddings into the network, or into a diffusion sampler, can move quality on other tasks, sometimes by a lot (DA-CLIP’s unified-restoration table) and sometimes by <0.1 dB (SAM-SEEM on VSR). Those numbers do not transfer. The random-encoder arm is what decides whether this project should keep the idea.
