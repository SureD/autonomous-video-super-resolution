# EmbeddingGemma 2: broader CPU go/no-go probe

Run on 2026-10-17. **NO-GO for spending EG2-specific training budget.** This broader result supersedes the provisional two-image screen retained at the end. This is an encoder sensitivity/association screen, not a trained SR or VSR result.

## What ran

- **40 distinct source frames from eight clips**, five uniformly spaced frames per clip; **both 192×192 center crops and full native-frame inputs for all 40 frames**. Clips: four Vid4 sequences, Xiph sunflower/rush_hour, BBB t=120 s, and Sintel trailer. No synthetic clip was included.
- Five fidelity degradations: Pillow bicubic down/up ×2/×4/×8 and **native-resolution H.264 CRF 23/33** (libx264 medium, YUV420p). All source HR frames in each contiguous clip were encoded before sampling/cropping; these are inter-frame GOP sequences, not five isolated I-frames. Codec compression is a separate family, not added on top of bicubic downsampling.
- Three sensitivity controls: one-pixel right translation (replicated left edge), mild HR unsharp masking, and the same unsharp mask on ×2. Sharpening is Pillow `UnsharpMask(radius=1, percent=50, threshold=0)`. Native images are degraded/shifted/sharpened before extracting their corresponding center crop.
- EG2 global vector, final 3×3-pooled vision tokens, final raw patch tokens; **DINOv2-base final patch tokens**; plus the same three EG2 streams after replacing **only its vision tower with a random initialization (seed 1729)**. The random global arm retains the pretrained projection/text stack, isolating the tower change; it is not an entirely random multimodal encoder.
- **CPU only, FP32; EG2/random four intra-op threads, DINO two**, evaluation/inference mode, audio disabled, no student training. Each encoder stage made **722 forwards** (80 HR references, 640 candidates, two unchanged repeats); 2,166 encoder forwards in total. All recorded features are finite. The metric stage measured 640 PSNR-Y/LPIPS pairs.
- **LPIPS-Alex v0.1 ran**, package 0.1.4. ImageNet AlexNet trunk and learned LPIPS calibration were both loaded from explicit local files; no random trunk metric. The `pnet_rand=True` constructor only suppresses its download path, after which all convolution weights/biases are replaced with the verified pretrained AlexNet tensors. LPIPS uses complete RGB views in [-1,1] with no resize.
- PSNR-Y in this probe uses the pre-ADR 0003 formula `Y=16+(65.481 R+128.553 G+24.966 B)/255` (BT.601 limited-range coefficients, peak 255, **border=0**, uint8 decoded RGB). ADR 0003 later locked BT.709 full-range Y with the same peak and border. These scores were not recomputed. Correlations use **negative PSNR-Y**, so positive correlation has the same fidelity direction as positive LPIPS correlation. HR identities and sensitivity controls are excluded from fidelity correlation tables.

Artifacts: [manifest and exact ffmpeg commands](embeddinggemma2-broad-manifest.json), [per-frame metrics](embeddinggemma2-broad-metrics.json), [EG2](embeddinggemma2-broad-eg2.json), [random tower](embeddinggemma2-broad-random.json), [DINO](embeddinggemma2-broad-dino.json), [joined raw measurements/statistics](embeddinggemma2-broad-summary.json), [decision](embeddinggemma2-broad-decision.json), [download hashes](embeddinggemma2-broad-downloads.json). [Probe script](../../scripts/probe_embeddinggemma2_broad.py), [report renderer](../../scripts/render_embeddinggemma2_broad_report.py).

Validation: saved LPIPS values matched the unmodified pretrained public constructor on crop/native blur/codec pairs with downloads disabled (absolute tolerance 1e−7). Rank calculations matched SciPy on tied/untied checks; an isolated identical-arm check returned exactly zero paired deltas/CIs. All 640 joined rows, 280 correlation summaries, 120 paired differences and three complete 80-case model stages passed finite-value/count checks. Image dimensions/pairing and script syntax checks passed.

## Geometry and statistical interpretation

“Native” denotes the complete native-resolution source view and native-resolution PSNR/LPIPS, **not an encoder operating on every native pixel without resizing**. EG2 uses the pinned official processor at B=280: the 192² crop is enlarged to 768² (48² raw patches, 16² pooled tokens); full native frames are resized to their aspect-preserving budget. DINO uses the complete same view, torchvision float bicubic/antialias resizing, ImageNet normalization, and **the same raw spatial token-grid dimensions** as EG2 (14-pixel rather than 16-pixel patches). Thus a square DINO raster is 672². This matches field of view and patch count, **not raster size, parameter count, or FLOPs**; it avoids DINO’s default central-224 crop discarding most of the native frame. Different encoder streams are not calibrated to a common distance scale.

EG2 global is the attention-masked mean of final 768-dimensional sequence states, including valid boundary tokens. Pooled tokens are the final vision-tower output after 3×3 pooling; raw tokens come from a hook on the final vision encoder layer before pooling. Padding is excluded. Dense distances average cosine distance at corresponding spatial positions; cosine normalization/accumulation uses float64 over FP32 features. DINO uses final post-layernorm patch tokens with CLS excluded. No intermediate layer was selected from the observed results.

| View | Source pixels | EG2 raster | DINO raster | Raw tokens (both) | EG2 pooled |
|---|---:|---:|---:|---:|---:|
| 192×192 crop | 192×192 | 768×768 | 672×672 | 2304 | 256 |
| Native input | 720×576 | 864×672 | 756×588 | 2268 | 252 |
| Native input | 704×576 | 864×720 | 756×630 | 2430 | 270 |
| Native input | 720×480 | 960×624 | 840×546 | 2340 | 260 |
| Native input | 1920×1080 | 1056×576 | 924×504 | 2376 | 264 |

Spearman ρ and Kendall τ-b below have **95% percentile CIs from 2,000 paired cluster-bootstrap replicates over eight clips**, seed 20261017. Each draw samples eight clips with replacement and retains every selected frame/level; duplicate clip draws receive duplicate weight. The same draws are used for model-minus-control deltas. Frames are not treated as independent bootstrap units. Only eight clips and one random initialization limit precision and generalization; CIs are descriptive and not adjusted for multiple comparisons.

- **Pooled across levels**: correlation over all 200 frame/degradation pairs per view. This mixes content and severity.
- **Across levels within frame**: mean of 40 per-frame correlations, each using the five fidelity levels; the clip bootstrap is applied to those per-frame statistics. This controls frame content but has only five points per correlation.
- **Within a fixed level**: correlation across 40 frames/content, reported separately for every level. **Within-level macro** is the arithmetic mean of those five correlations; each bootstrap replicate recomputes all five before averaging. This is the central check against a severity-only association.

## Quality levels and feature distances

Means across the 40 frames in each view (equal numbers per clip). All exact per-frame values are linked above.

| View | Degradation | PSNR-Y dB | LPIPS | EG2 global | EG2 pooled | EG2 raw | DINO patch | Random global | Random pooled | Random raw |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 192×192 crop | bicubic_x2 | 34.905 | 0.16163 | 0.03074 | 0.23256 | 0.41717 | 0.11763 | 0.00016 | 0.00008 | 0.00025 |
| 192×192 crop | bicubic_x4 | 29.805 | 0.37952 | 0.08073 | 0.40728 | 0.60777 | 0.25697 | 0.00051 | 0.00017 | 0.00068 |
| 192×192 crop | bicubic_x8 | 26.048 | 0.58488 | 0.11861 | 0.58349 | 0.74013 | 0.42647 | 0.00133 | 0.00036 | 0.00137 |
| 192×192 crop | h264_crf23 | 37.931 | 0.05189 | 0.02910 | 0.22055 | 0.40592 | 0.11063 | 0.00023 | 0.00003 | 0.00008 |
| 192×192 crop | h264_crf33 | 32.511 | 0.14941 | 0.05862 | 0.38663 | 0.59595 | 0.24976 | 0.00030 | 0.00007 | 0.00026 |
| 192×192 crop | shift_right1 | 28.944 | 0.03463 | 0.00871 | 0.12390 | 0.29161 | 0.04734 | 0.00023 | 0.00018 | 0.00075 |
| 192×192 crop | sharpen_hr | 41.242 | 0.02339 | 0.00457 | 0.03499 | 0.07457 | 0.00919 | 0.00008 | 0.00002 | 0.00006 |
| 192×192 crop | sharpen_x2 | 35.443 | 0.13347 | 0.02893 | 0.22699 | 0.41022 | 0.11671 | 0.00012 | 0.00006 | 0.00022 |
| Native input | bicubic_x2 | 34.953 | 0.16016 | 0.01014 | 0.09623 | 0.16781 | 0.02443 | 0.00009 | 0.00005 | 0.00023 |
| Native input | bicubic_x4 | 30.449 | 0.36871 | 0.04185 | 0.21721 | 0.34878 | 0.07712 | 0.00031 | 0.00016 | 0.00068 |
| Native input | bicubic_x8 | 27.168 | 0.55610 | 0.07574 | 0.33793 | 0.50901 | 0.18158 | 0.00079 | 0.00034 | 0.00127 |
| Native input | h264_crf23 | 38.831 | 0.06712 | 0.01129 | 0.08509 | 0.16703 | 0.03089 | 0.00022 | 0.00002 | 0.00006 |
| Native input | h264_crf33 | 33.611 | 0.16260 | 0.03413 | 0.19644 | 0.34389 | 0.09132 | 0.00022 | 0.00005 | 0.00020 |
| Native input | shift_right1 | 29.718 | 0.03259 | 0.00461 | 0.06669 | 0.13473 | 0.01465 | 0.00019 | 0.00014 | 0.00068 |
| Native input | sharpen_hr | 41.812 | 0.02076 | 0.00194 | 0.01808 | 0.03462 | 0.00219 | 0.00004 | 0.00001 | 0.00005 |
| Native input | sharpen_x2 | 35.424 | 0.13348 | 0.00912 | 0.09017 | 0.15771 | 0.02382 | 0.00007 | 0.00004 | 0.00019 |

## Pooled across all five fidelity levels

Each cell is estimate [95% CI].

| View | Encoder | ρ vs LPIPS | τ vs LPIPS | ρ vs −PSNR-Y | τ vs −PSNR-Y |
|---|---|---:|---:|---:|---:|
| 192×192 crop | EG2 global | 0.788 [0.697, 0.879] | 0.594 [0.495, 0.703] | 0.584 [0.378, 0.806] | 0.411 [0.258, 0.616] |
| 192×192 crop | EG2 pooled 3×3 | 0.843 [0.778, 0.912] | 0.651 [0.574, 0.743] | 0.625 [0.483, 0.825] | 0.448 [0.345, 0.649] |
| 192×192 crop | EG2 raw patches | 0.834 [0.769, 0.905] | 0.641 [0.564, 0.733] | 0.589 [0.443, 0.812] | 0.419 [0.322, 0.634] |
| 192×192 crop | DINOv2-B patches | 0.823 [0.726, 0.903] | 0.636 [0.541, 0.736] | 0.652 [0.554, 0.811] | 0.467 [0.396, 0.638] |
| 192×192 crop | Random tower → global | 0.469 [0.261, 0.676] | 0.326 [0.187, 0.482] | 0.658 [0.421, 0.816] | 0.487 [0.305, 0.630] |
| 192×192 crop | Random tower pooled | 0.619 [0.491, 0.829] | 0.436 [0.332, 0.644] | 0.936 [0.811, 0.988] | 0.805 [0.630, 0.916] |
| 192×192 crop | Random tower raw | 0.700 [0.563, 0.863] | 0.510 [0.405, 0.679] | 0.988 [0.967, 0.994] | 0.911 [0.855, 0.947] |
| Native input | EG2 global | 0.689 [0.583, 0.800] | 0.500 [0.410, 0.607] | 0.689 [0.562, 0.823] | 0.510 [0.399, 0.644] |
| Native input | EG2 pooled 3×3 | 0.749 [0.601, 0.888] | 0.554 [0.419, 0.708] | 0.777 [0.645, 0.878] | 0.595 [0.490, 0.707] |
| Native input | EG2 raw patches | 0.755 [0.611, 0.889] | 0.568 [0.427, 0.723] | 0.787 [0.666, 0.876] | 0.607 [0.513, 0.697] |
| Native input | DINOv2-B patches | 0.701 [0.576, 0.828] | 0.524 [0.401, 0.653] | 0.769 [0.635, 0.859] | 0.589 [0.470, 0.697] |
| Native input | Random tower → global | 0.491 [0.244, 0.723] | 0.349 [0.164, 0.543] | 0.573 [0.291, 0.763] | 0.420 [0.200, 0.594] |
| Native input | Random tower pooled | 0.553 [0.394, 0.841] | 0.391 [0.276, 0.649] | 0.927 [0.755, 0.988] | 0.798 [0.599, 0.917] |
| Native input | Random tower raw | 0.600 [0.467, 0.867] | 0.430 [0.341, 0.683] | 0.964 [0.885, 0.994] | 0.867 [0.752, 0.948] |

## Across fidelity levels within each frame

Each cell is estimate [95% CI].

| View | Encoder | ρ vs LPIPS | τ vs LPIPS | ρ vs −PSNR-Y | τ vs −PSNR-Y |
|---|---|---:|---:|---:|---:|
| 192×192 crop | EG2 global | 0.847 [0.782, 0.907] | 0.765 [0.695, 0.835] | 0.870 [0.805, 0.925] | 0.790 [0.725, 0.855] |
| 192×192 crop | EG2 pooled 3×3 | 0.907 [0.862, 0.952] | 0.830 [0.750, 0.910] | 0.952 [0.915, 0.980] | 0.905 [0.830, 0.960] |
| 192×192 crop | EG2 raw patches | 0.895 [0.855, 0.937] | 0.810 [0.740, 0.880] | 0.942 [0.910, 0.967] | 0.885 [0.820, 0.935] |
| 192×192 crop | DINOv2-B patches | 0.872 [0.807, 0.940] | 0.800 [0.715, 0.895] | 0.937 [0.900, 0.972] | 0.885 [0.825, 0.945] |
| 192×192 crop | Random tower → global | 0.527 [0.268, 0.747] | 0.485 [0.265, 0.685] | 0.607 [0.337, 0.835] | 0.570 [0.335, 0.780] |
| 192×192 crop | Random tower pooled | 0.802 [0.657, 0.922] | 0.740 [0.580, 0.885] | 0.855 [0.717, 0.970] | 0.805 [0.640, 0.940] |
| 192×192 crop | Random tower raw | 0.895 [0.860, 0.932] | 0.810 [0.760, 0.870] | 0.952 [0.930, 0.972] | 0.905 [0.860, 0.945] |
| Native input | EG2 global | 0.867 [0.822, 0.917] | 0.785 [0.720, 0.855] | 0.870 [0.840, 0.902] | 0.775 [0.735, 0.815] |
| Native input | EG2 pooled 3×3 | 0.862 [0.777, 0.935] | 0.765 [0.650, 0.870] | 0.870 [0.752, 0.962] | 0.795 [0.650, 0.925] |
| Native input | EG2 raw patches | 0.852 [0.772, 0.917] | 0.745 [0.640, 0.850] | 0.865 [0.750, 0.952] | 0.785 [0.650, 0.905] |
| Native input | DINOv2-B patches | 0.760 [0.652, 0.853] | 0.635 [0.520, 0.740] | 0.790 [0.655, 0.892] | 0.675 [0.545, 0.785] |
| Native input | Random tower → global | 0.487 [0.235, 0.717] | 0.425 [0.200, 0.655] | 0.522 [0.275, 0.750] | 0.465 [0.240, 0.690] |
| Native input | Random tower pooled | 0.815 [0.630, 0.940] | 0.750 [0.570, 0.895] | 0.837 [0.660, 0.968] | 0.790 [0.600, 0.940] |
| Native input | Random tower raw | 0.895 [0.835, 0.950] | 0.815 [0.720, 0.905] | 0.920 [0.862, 0.970] | 0.855 [0.755, 0.945] |

## Across frames/content within a fixed level: macro of five levels

Each cell is estimate [95% CI].

| View | Encoder | ρ vs LPIPS | τ vs LPIPS | ρ vs −PSNR-Y | τ vs −PSNR-Y |
|---|---|---:|---:|---:|---:|
| 192×192 crop | EG2 global | 0.501 [0.129, 0.702] | 0.354 [0.101, 0.521] | 0.271 [-0.200, 0.622] | 0.175 [-0.142, 0.445] |
| 192×192 crop | EG2 pooled 3×3 | 0.738 [0.400, 0.820] | 0.539 [0.274, 0.626] | 0.329 [-0.062, 0.645] | 0.221 [-0.033, 0.458] |
| 192×192 crop | EG2 raw patches | 0.749 [0.410, 0.833] | 0.557 [0.292, 0.650] | 0.277 [-0.164, 0.633] | 0.181 [-0.102, 0.451] |
| 192×192 crop | DINOv2-B patches | 0.749 [0.442, 0.818] | 0.555 [0.322, 0.640] | 0.433 [0.075, 0.688] | 0.308 [0.056, 0.518] |
| 192×192 crop | Random tower → global | 0.551 [0.145, 0.727] | 0.380 [0.074, 0.549] | 0.543 [0.321, 0.632] | 0.408 [0.257, 0.487] |
| 192×192 crop | Random tower pooled | 0.481 [0.131, 0.738] | 0.331 [0.083, 0.577] | 0.916 [0.764, 0.951] | 0.795 [0.650, 0.870] |
| 192×192 crop | Random tower raw | 0.537 [0.182, 0.779] | 0.391 [0.148, 0.628] | 0.974 [0.924, 0.983] | 0.894 [0.837, 0.936] |
| Native input | EG2 global | 0.357 [-0.033, 0.518] | 0.257 [-0.037, 0.393] | 0.541 [0.083, 0.771] | 0.391 [0.041, 0.584] |
| Native input | EG2 pooled 3×3 | 0.606 [0.098, 0.807] | 0.419 [0.052, 0.625] | 0.540 [0.274, 0.725] | 0.405 [0.197, 0.565] |
| Native input | EG2 raw patches | 0.639 [0.162, 0.850] | 0.464 [0.107, 0.668] | 0.572 [0.339, 0.751] | 0.450 [0.274, 0.593] |
| Native input | DINOv2-B patches | 0.652 [0.260, 0.828] | 0.486 [0.205, 0.655] | 0.657 [0.413, 0.834] | 0.535 [0.354, 0.695] |
| Native input | Random tower → global | 0.651 [0.317, 0.764] | 0.472 [0.205, 0.584] | 0.479 [0.105, 0.696] | 0.376 [0.096, 0.534] |
| Native input | Random tower pooled | 0.351 [0.134, 0.622] | 0.303 [0.108, 0.523] | 0.934 [0.713, 0.967] | 0.823 [0.594, 0.895] |
| Native input | Random tower raw | 0.384 [0.152, 0.667] | 0.334 [0.139, 0.558] | 0.963 [0.843, 0.985] | 0.888 [0.750, 0.945] |

## Does pretraining beat random initialization?

Paired **pretrained minus random-tower** correlation deltas. A positive lower CI bound supports superiority for that association; an interval spanning zero is inconclusive. This comparison isolates the EG2 tower, while global controls keep the trained text/projection stack.

| View | Association | Arm | Δρ LPIPS | Δτ LPIPS | Δρ −PSNR-Y | Δτ −PSNR-Y |
|---|---|---|---:|---:|---:|---:|
| 192×192 crop | all_levels | EG2 global | 0.319 [0.111, 0.551] | 0.268 [0.108, 0.431] | -0.074 [-0.277, 0.270] | -0.076 [-0.236, 0.196] |
| 192×192 crop | all_levels | EG2 pooled 3×3 | 0.224 [0.038, 0.342] | 0.214 [0.043, 0.313] | -0.311 [-0.464, -0.056] | -0.357 [-0.501, -0.078] |
| 192×192 crop | all_levels | EG2 raw patches | 0.134 [0.013, 0.243] | 0.132 [0.016, 0.205] | -0.399 [-0.545, -0.175] | -0.492 [-0.591, -0.283] |
| 192×192 crop | within_frame_across_levels | EG2 global | 0.320 [0.080, 0.607] | 0.280 [0.075, 0.515] | 0.262 [0.025, 0.540] | 0.220 [0.015, 0.455] |
| 192×192 crop | within_frame_across_levels | EG2 pooled 3×3 | 0.105 [-0.023, 0.255] | 0.090 [-0.065, 0.275] | 0.097 [-0.010, 0.235] | 0.100 [-0.030, 0.265] |
| 192×192 crop | within_frame_across_levels | EG2 raw patches | 0.000 [-0.042, 0.043] | 0.000 [-0.070, 0.075] | -0.010 [-0.040, 0.028] | -0.020 [-0.080, 0.055] |
| 192×192 crop | within_level_macro | EG2 global | -0.051 [-0.400, 0.270] | -0.026 [-0.291, 0.220] | -0.273 [-0.687, 0.111] | -0.233 [-0.528, 0.059] |
| 192×192 crop | within_level_macro | EG2 pooled 3×3 | 0.257 [-0.011, 0.474] | 0.208 [-0.052, 0.388] | -0.587 [-0.967, -0.236] | -0.574 [-0.833, -0.310] |
| 192×192 crop | within_level_macro | EG2 raw patches | 0.211 [-0.025, 0.439] | 0.166 [-0.071, 0.331] | -0.697 [-1.124, -0.324] | -0.713 [-0.986, -0.433] |
| Native input | all_levels | EG2 global | 0.198 [-0.013, 0.481] | 0.151 [-0.018, 0.369] | 0.116 [-0.114, 0.432] | 0.089 [-0.110, 0.363] |
| Native input | all_levels | EG2 pooled 3×3 | 0.195 [-0.023, 0.346] | 0.164 [-0.038, 0.283] | -0.150 [-0.194, -0.032] | -0.203 [-0.286, -0.055] |
| Native input | all_levels | EG2 raw patches | 0.155 [-0.023, 0.247] | 0.138 [-0.041, 0.215] | -0.177 [-0.232, -0.095] | -0.260 [-0.311, -0.169] |
| Native input | within_frame_across_levels | EG2 global | 0.380 [0.145, 0.617] | 0.360 [0.115, 0.585] | 0.347 [0.120, 0.587] | 0.310 [0.085, 0.530] |
| Native input | within_frame_across_levels | EG2 pooled 3×3 | 0.048 [-0.060, 0.200] | 0.015 [-0.090, 0.135] | 0.033 [-0.070, 0.180] | 0.005 [-0.095, 0.125] |
| Native input | within_frame_across_levels | EG2 raw patches | -0.042 [-0.115, 0.018] | -0.070 [-0.160, 0.020] | -0.055 [-0.138, 0.010] | -0.070 [-0.160, 0.020] |
| Native input | within_level_macro | EG2 global | -0.294 [-0.539, 0.044] | -0.215 [-0.427, 0.060] | 0.061 [-0.529, 0.516] | 0.014 [-0.452, 0.389] |
| Native input | within_level_macro | EG2 pooled 3×3 | 0.255 [-0.217, 0.487] | 0.116 [-0.251, 0.343] | -0.393 [-0.619, -0.172] | -0.418 [-0.623, -0.208] |
| Native input | within_level_macro | EG2 raw patches | 0.255 [-0.170, 0.480] | 0.130 [-0.220, 0.330] | -0.391 [-0.610, -0.204] | -0.438 [-0.622, -0.288] |

## Translation and sharpening sensitivity

Distance ratios are per-frame distance(control, HR) / distance(×2, HR), then median over frames; the ratio CI resamples clips. A median >1 means the perturbation produces a larger encoder distance than ×2 blur. A one-pixel translation itself can also reduce pixel fidelity: LPIPS/PSNR measurements are reported rather than assuming every shift is harmless. Sharpened ×2 additionally tests whether the distance direction agrees with the measured fidelity direction.

| View | Encoder | Shift/×2 [95% CI] | Shift > ×2 frames | HR sharpen/×2 [95% CI] | ×2 sharpen/×2 [95% CI] | ×2 sharpening direction agrees LPIPS / −PSNR |
|---|---|---:|---:|---:|---:|---:|
| 192×192 crop | EG2 global | 0.177 [0.132, 0.280] | 12.5% | 0.121 [0.085, 0.246] | 0.972 [0.909, 0.988] | 75.0% / 75.0% |
| 192×192 crop | EG2 pooled 3×3 | 0.510 [0.464, 0.603] | 0.0% | 0.146 [0.118, 0.207] | 0.976 [0.968, 0.987] | 90.0% / 90.0% |
| 192×192 crop | EG2 raw patches | 0.675 [0.650, 0.742] | 0.0% | 0.169 [0.141, 0.216] | 0.984 [0.977, 0.990] | 95.0% / 95.0% |
| 192×192 crop | DINOv2-B patches | 0.403 [0.328, 0.524] | 0.0% | 0.058 [0.042, 0.124] | 0.973 [0.957, 1.027] | 67.5% / 67.5% |
| 192×192 crop | Random tower → global | 2.435 [1.275, 5.683] | 90.0% | 0.627 [0.484, 1.215] | 0.927 [0.675, 1.319] | 57.5% / 57.5% |
| 192×192 crop | Random tower pooled | 2.806 [2.415, 4.799] | 100.0% | 0.312 [0.269, 0.342] | 0.913 [0.811, 0.967] | 82.5% / 82.5% |
| 192×192 crop | Random tower raw | 3.685 [3.004, 4.204] | 100.0% | 0.281 [0.238, 0.315] | 0.879 [0.817, 0.927] | 95.0% / 95.0% |
| Native input | EG2 global | 0.462 [0.365, 0.818] | 15.0% | 0.169 [0.129, 0.262] | 0.902 [0.888, 0.921] | 90.0% / 90.0% |
| Native input | EG2 pooled 3×3 | 0.807 [0.572, 1.483] | 37.5% | 0.196 [0.152, 0.408] | 0.927 [0.854, 0.956] | 95.0% / 95.0% |
| Native input | EG2 raw patches | 0.904 [0.650, 1.604] | 50.0% | 0.209 [0.156, 0.442] | 0.940 [0.877, 0.960] | 100.0% / 100.0% |
| Native input | DINOv2-B patches | 0.751 [0.534, 2.427] | 45.0% | 0.152 [0.048, 0.176] | 0.978 [0.967, 0.983] | 85.0% / 85.0% |
| Native input | Random tower → global | 3.192 [1.775, 10.401] | 95.0% | 0.761 [0.439, 1.582] | 0.811 [0.660, 1.009] | 70.0% / 70.0% |
| Native input | Random tower pooled | 3.584 [2.641, 14.869] | 100.0% | 0.388 [0.330, 0.762] | 0.792 [0.749, 0.849] | 100.0% / 100.0% |
| Native input | Random tower raw | 4.001 [2.876, 11.559] | 100.0% | 0.298 [0.274, 0.531] | 0.820 [0.792, 0.879] | 100.0% / 100.0% |

## Bicubic monotonicity

Within each frame: strictly increasing distance for ×2 → ×4 → ×8. Median per-frame ρ is against negative PSNR-Y. Codec levels are included in the five-level tables above, ranked by their actual fidelity rather than an assumed cross-family severity order.

| View | Encoder | Strictly ordered frames | Median per-frame ρ [95% CI] |
|---|---|---:|---:|
| 192×192 crop | EG2 global | 97.5% | 1.000 [1.000, 1.000] |
| 192×192 crop | EG2 pooled 3×3 | 100.0% | 1.000 [1.000, 1.000] |
| 192×192 crop | EG2 raw patches | 100.0% | 1.000 [1.000, 1.000] |
| 192×192 crop | DINOv2-B patches | 100.0% | 1.000 [1.000, 1.000] |
| 192×192 crop | Random tower → global | 100.0% | 1.000 [1.000, 1.000] |
| 192×192 crop | Random tower pooled | 100.0% | 1.000 [1.000, 1.000] |
| 192×192 crop | Random tower raw | 100.0% | 1.000 [1.000, 1.000] |
| Native input | EG2 global | 100.0% | 1.000 [1.000, 1.000] |
| Native input | EG2 pooled 3×3 | 100.0% | 1.000 [1.000, 1.000] |
| Native input | EG2 raw patches | 100.0% | 1.000 [1.000, 1.000] |
| Native input | DINOv2-B patches | 100.0% | 1.000 [1.000, 1.000] |
| Native input | Random tower → global | 100.0% | 1.000 [1.000, 1.000] |
| Native input | Random tower pooled | 100.0% | 1.000 [1.000, 1.000] |
| Native input | Random tower raw | 100.0% | 1.000 [1.000, 1.000] |

## Decision and limits

**NO-GO for spending EG2-specific training budget.**

The earlier blur-only pass is insufficient: every random-tower stream also strictly orders ×2/×4/×8 in 100% of frames in both views. EG2 dense and DINO do the same; EG2 global orders 97.5% of crops and 100% of native inputs.

Pretraining has a measurable benefit in some summaries: all three EG2 crop streams improve the pooled-across-level LPIPS Spearman correlation over their random-tower counterparts (positive paired CI lower bounds). Global EG2 also improves the mean within-frame five-level LPIPS ordering in both views. That does not establish robust fidelity discrimination across content at a fixed degradation.

**192×192 crop:** EG2 pooled 3×3 LPIPS Δρ 0.257 [-0.011, 0.474], −PSNR-Y Δρ -0.587 [-0.967, -0.236]; EG2 raw patches LPIPS Δρ 0.211 [-0.025, 0.439], −PSNR-Y Δρ -0.697 [-1.124, -0.324]. Thus the dense within-level LPIPS advantage is inconclusive, while its PSNR-ranking deficit is clear for these sampled clips. Global within-level LPIPS superiority is also unsupported.

**Native input:** EG2 pooled 3×3 LPIPS Δρ 0.255 [-0.217, 0.487], −PSNR-Y Δρ -0.393 [-0.619, -0.172]; EG2 raw patches LPIPS Δρ 0.255 [-0.170, 0.480], −PSNR-Y Δρ -0.391 [-0.610, -0.204]. Thus the dense within-level LPIPS advantage is inconclusive, while its PSNR-ranking deficit is clear for these sampled clips. Global within-level LPIPS superiority is also unsupported.

The learned encoders are less shift-sensitive than the random tower, but native-input translation remains material: one-pixel shift exceeds the ×2 distance in **37.5% of frames for EG2 pooled and 50% for EG2 raw** (DINO: 45%; EG2 global: 15%). Native median shift/×2 ratios are 0.807, 0.904, 0.751 and 0.462 respectively. In contrast, native LPIPS shift/×2 is 0.213; PSNR-Y penalizes that shift more than ×2 by a median 5.506 dB. This is a metric-dependent tradeoff, not proof that all translation sensitivity is wrong.

Mild ×2 sharpening improves measured fidelity: median PSNR-Y rises **0.531 dB on crops / 0.479 dB on native frames**, and median LPIPS ratios are **0.840 / 0.817**. EG2 pooled/raw distances decrease in 90%/95% of crops and 95%/100% of native frames, so that control is mostly sensible. Its median distance reductions are smaller: ratios 0.976/0.984 on crops and 0.927/0.940 on native inputs. This positive sensitivity result does not resolve the fixed-level control deficits.

**Budget recommendation:** do not allocate an EG2 cache/training sweep from these results. Retain DINO as the planned baseline, with its own training/gradient confirmation still required; these data do not statistically establish that DINO is superior to EG2. Revisiting EG2 would require fresh, training-only clip replication and a predeclared objective resolving pixel fidelity versus perceptual tolerance, rather than selecting a favorable summary from this probe.

At least one EG2 arm must pass both views: within-level macro LPIPS Spearman improvement over its random-tower counterpart has a positive paired 95% CI; no statistically clear PSNR correlation deficit; >=75% strict x2<x4<x8 ordering; median one-pixel-shift distance < x2 distance.

This is a conservative screening rule for whether to allocate EG2 training effort, not a validated law connecting rank correlation to SR gains. Neither a positive correlation nor a DINO point estimate proves that its loss has useful reconstruction gradients. Statistical significance of DINO-versus-EG2 differences was not tested. Same-level correlation across frames is still susceptible to clip/content structure; the bootstrap reflects the eight sampled clips and does not remove that confounding.

| Arm | View | Positive LPIPS Δρ CI | No clear PSNR Δρ deficit | ≥75% blur ordering | Median shift < ×2 |
|---|---|---|---|---|---|
| EG2 global | 192×192 crop | no | yes | yes | yes |
| EG2 global | Native input | no | yes | yes | yes |
| EG2 pooled 3×3 | 192×192 crop | no | no | yes | yes |
| EG2 pooled 3×3 | Native input | no | no | yes | yes |
| EG2 raw patches | 192×192 crop | no | no | yes | yes |
| EG2 raw patches | Native input | no | no | yes | yes |

**Did not run:** student/teacher SR training, learned SR outputs, GPU execution or VRAM profiling, encoder backward/gradient usefulness, temporal fidelity/flow metrics, full native-resolution transformer grids, EG2 B=1120 or intermediate-layer sweep, multiple random seeds, DINO random-init control, SigLIP/CLIP/VGG arms, and external held-out clip replication. The native-input encoder results cannot establish preservation of detail removed by their internal resizing. Existing exp001/teacher/deployment gates still apply before any training decision.

## Runtime, reproducibility and exact sample IDs

DINO and random-tower timings overlap; stage durations must not be summed as end-to-end wall time. Preparation: 126.76 s. PSNR/LPIPS pair evaluation: 70.64 s.

- eg2: load 6.34 s; inference/image-processing and record construction 2465.42 s; total stage 2471.94 s. Largest unchanged-repeat distance 4.32e-17.
- random: load 9.87 s; inference/image-processing and record construction 2187.17 s; total stage 2197.23 s. Largest unchanged-repeat distance 5.07e-17.
- dino: load 6.89 s; inference/image-processing and record construction 1644.53 s; total stage 1651.69 s. Largest unchanged-repeat distance 3.75e-17.

Environment: `{"Pillow": "12.3.0", "lpips": "0.1.4", "numpy": "2.5.2", "scipy": "1.18.1", "torch": "2.14.1+cpu", "torchvision": "0.29.1+cpu", "transformers": "5.21.0.dev0"}`. DINO revision `f9e44c814b77203eaa57a6bdbbd535f21ede1415` ([checkpoint](https://huggingface.co/facebook/dinov2-base)); LPIPS implementation [authors' repository](https://github.com/richzhang/PerceptualSimilarity). EG2 revision `914f7f89142e33e77833254d9c9b90c3cef7303b` is retained from the original probe; SHA256 `197a32965d4b1105faf060417baa899e193fb73cd401f42ec9295234d5553d79`. DINO and LPIPS weight hashes are in the linked manifests. Network acquisition occurred only before inference; model stages set `CUDA_VISIBLE_DEVICES=''`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` and use local-files-only model loads.

| Clip | Sampled frame names | Source dimensions |
|---|---|---|
| vid4_calendar | 00001, 00011, 00021, 00031, 00041 | 720×576 |
| vid4_city | 00001, 00009, 00017, 00026, 00034 | 704×576 |
| vid4_foliage | 00001, 00013, 00025, 00037, 00049 | 720×480 |
| vid4_walk | 00001, 00013, 00024, 00035, 00047 | 720×480 |
| sunflower | 00001, 00013, 00025, 00036, 00048 | 1920×1080 |
| rush_hour | 00001, 00013, 00025, 00036, 00048 | 1920×1080 |
| bbb_t120 | 00001, 00013, 00025, 00036, 00048 | 1920×1080 |
| sintel_trailer | 00001, 00013, 00025, 00036, 00048 | 1920×1080 |

With the local model/weight paths in the scripts available:

```bash
for stage in prepare metrics eg2 random; do
  /tmp/eg2-probe-venv/bin/python scripts/probe_embeddinggemma2_broad.py --stage "$stage" --threads 4 --bootstrap 2000
done
/tmp/eg2-probe-venv/bin/python scripts/probe_embeddinggemma2_broad.py --stage dino --threads 2
/tmp/eg2-probe-venv/bin/python scripts/probe_embeddinggemma2_broad.py --stage summarize --bootstrap 2000
/tmp/eg2-probe-venv/bin/python scripts/render_embeddinggemma2_broad_report.py
```

Model downloads were explicitly staged from the pinned public Hugging Face DINO revision and the torchvision AlexNet URL recorded in [downloads](embeddinggemma2-broad-downloads.json); EG2 reused the previously SHA256-checked local checkpoint. Installing `lpips==0.1.4 scipy==1.18.1` in the previous probe venv supplied the extra metric/statistics packages. Completed per-view records are checkpointed; to rerun a changed protocol, use a fresh output/cache directory instead of mixing resumed results.

<details>
<summary>Full per-level content correlations, including confidence intervals</summary>

### 192×192 crop / bicubic_x2

| Encoder | ρ LPIPS | τ LPIPS | ρ −PSNR-Y | τ −PSNR-Y |
|---|---:|---:|---:|---:|
| EG2 global | 0.499 [-0.215, 0.870] | 0.349 [-0.177, 0.676] | 0.213 [-0.544, 0.757] | 0.131 [-0.427, 0.570] |
| EG2 pooled 3×3 | 0.726 [0.187, 0.923] | 0.546 [0.107, 0.760] | 0.394 [-0.257, 0.820] | 0.256 [-0.200, 0.608] |
| EG2 raw patches | 0.685 [0.092, 0.943] | 0.533 [0.062, 0.810] | 0.288 [-0.468, 0.839] | 0.213 [-0.284, 0.642] |
| DINOv2-B patches | 0.867 [0.472, 0.955] | 0.692 [0.340, 0.839] | 0.752 [0.158, 0.910] | 0.510 [0.095, 0.750] |
| Random tower → global | 0.732 [0.171, 0.902] | 0.503 [0.069, 0.722] | 0.846 [0.462, 0.921] | 0.638 [0.309, 0.773] |
| Random tower pooled | 0.724 [0.220, 0.907] | 0.495 [0.111, 0.753] | 0.964 [0.833, 0.984] | 0.862 [0.702, 0.921] |
| Random tower raw | 0.733 [0.249, 0.910] | 0.503 [0.132, 0.767] | 0.978 [0.887, 0.992] | 0.900 [0.788, 0.956] |

### 192×192 crop / bicubic_x4

| Encoder | ρ LPIPS | τ LPIPS | ρ −PSNR-Y | τ −PSNR-Y |
|---|---:|---:|---:|---:|
| EG2 global | 0.398 [-0.323, 0.841] | 0.308 [-0.187, 0.658] | 0.308 [-0.381, 0.767] | 0.213 [-0.274, 0.582] |
| EG2 pooled 3×3 | 0.726 [0.115, 0.902] | 0.538 [0.040, 0.730] | 0.636 [0.092, 0.897] | 0.449 [0.074, 0.737] |
| EG2 raw patches | 0.744 [0.156, 0.915] | 0.551 [0.055, 0.749] | 0.577 [0.020, 0.872] | 0.374 [-0.008, 0.703] |
| DINOv2-B patches | 0.706 [0.176, 0.857] | 0.477 [0.063, 0.720] | 0.768 [0.270, 0.895] | 0.572 [0.176, 0.739] |
| Random tower → global | 0.783 [0.259, 0.920] | 0.579 [0.155, 0.765] | 0.839 [0.391, 0.933] | 0.649 [0.318, 0.801] |
| Random tower pooled | 0.821 [0.354, 0.931] | 0.600 [0.211, 0.796] | 0.983 [0.904, 0.998] | 0.921 [0.801, 0.987] |
| Random tower raw | 0.808 [0.371, 0.923] | 0.582 [0.227, 0.773] | 0.996 [0.959, 1.000] | 0.969 [0.883, 1.000] |

### 192×192 crop / bicubic_x8

| Encoder | ρ LPIPS | τ LPIPS | ρ −PSNR-Y | τ −PSNR-Y |
|---|---:|---:|---:|---:|
| EG2 global | 0.586 [0.015, 0.839] | 0.400 [-0.003, 0.632] | 0.624 [0.117, 0.846] | 0.418 [0.023, 0.656] |
| EG2 pooled 3×3 | 0.773 [0.269, 0.890] | 0.526 [0.124, 0.693] | 0.754 [0.309, 0.889] | 0.523 [0.179, 0.734] |
| EG2 raw patches | 0.737 [0.211, 0.864] | 0.474 [0.060, 0.660] | 0.700 [0.215, 0.878] | 0.467 [0.089, 0.713] |
| DINOv2-B patches | 0.682 [0.145, 0.877] | 0.513 [0.132, 0.697] | 0.703 [0.142, 0.880] | 0.510 [0.082, 0.715] |
| Random tower → global | 0.656 [0.148, 0.788] | 0.436 [0.078, 0.592] | 0.753 [0.260, 0.874] | 0.531 [0.184, 0.699] |
| Random tower pooled | 0.748 [0.328, 0.865] | 0.549 [0.232, 0.724] | 0.874 [0.560, 0.978] | 0.710 [0.443, 0.912] |
| Random tower raw | 0.843 [0.538, 0.941] | 0.672 [0.385, 0.829] | 0.964 [0.807, 0.996] | 0.869 [0.684, 0.976] |

### 192×192 crop / h264_crf23

| Encoder | ρ LPIPS | τ LPIPS | ρ −PSNR-Y | τ −PSNR-Y |
|---|---:|---:|---:|---:|
| EG2 global | 0.617 [0.283, 0.818] | 0.438 [0.195, 0.666] | 0.043 [-0.453, 0.561] | 0.003 [-0.332, 0.382] |
| EG2 pooled 3×3 | 0.846 [0.582, 0.940] | 0.669 [0.441, 0.815] | 0.020 [-0.589, 0.659] | 0.008 [-0.362, 0.476] |
| EG2 raw patches | 0.876 [0.630, 0.954] | 0.723 [0.519, 0.837] | -0.041 [-0.609, 0.634] | -0.036 [-0.379, 0.434] |
| DINOv2-B patches | 0.855 [0.601, 0.928] | 0.659 [0.434, 0.782] | -0.016 [-0.575, 0.663] | -0.013 [-0.340, 0.471] |
| Random tower → global | 0.269 [-0.249, 0.685] | 0.169 [-0.205, 0.532] | 0.067 [-0.299, 0.564] | 0.072 [-0.210, 0.480] |
| Random tower pooled | -0.030 [-0.604, 0.643] | -0.015 [-0.428, 0.537] | 0.798 [0.433, 0.954] | 0.631 [0.329, 0.855] |
| Random tower raw | 0.119 [-0.565, 0.801] | 0.097 [-0.357, 0.685] | 0.948 [0.857, 0.991] | 0.821 [0.708, 0.950] |

### 192×192 crop / h264_crf33

| Encoder | ρ LPIPS | τ LPIPS | ρ −PSNR-Y | τ −PSNR-Y |
|---|---:|---:|---:|---:|
| EG2 global | 0.403 [-0.059, 0.725] | 0.274 [-0.025, 0.519] | 0.165 [-0.398, 0.727] | 0.110 [-0.260, 0.545] |
| EG2 pooled 3×3 | 0.619 [0.255, 0.793] | 0.415 [0.155, 0.561] | -0.160 [-0.657, 0.560] | -0.133 [-0.462, 0.345] |
| EG2 raw patches | 0.701 [0.355, 0.838] | 0.503 [0.229, 0.634] | -0.138 [-0.688, 0.564] | -0.113 [-0.478, 0.385] |
| DINOv2-B patches | 0.636 [0.167, 0.870] | 0.433 [0.097, 0.691] | -0.044 [-0.639, 0.702] | -0.038 [-0.420, 0.505] |
| Random tower → global | 0.318 [-0.234, 0.696] | 0.213 [-0.179, 0.503] | 0.211 [-0.248, 0.619] | 0.151 [-0.147, 0.452] |
| Random tower pooled | 0.140 [-0.546, 0.644] | 0.026 [-0.397, 0.422] | 0.960 [0.810, 0.987] | 0.851 [0.658, 0.935] |
| Random tower raw | 0.183 [-0.484, 0.693] | 0.103 [-0.333, 0.500] | 0.986 [0.910, 0.993] | 0.913 [0.775, 0.960] |

### 192×192 crop / bicubic_only

| Encoder | ρ LPIPS | τ LPIPS | ρ −PSNR-Y | τ −PSNR-Y |
|---|---:|---:|---:|---:|
| EG2 global | 0.807 [0.659, 0.928] | 0.613 [0.448, 0.771] | 0.593 [0.381, 0.810] | 0.413 [0.261, 0.617] |
| EG2 pooled 3×3 | 0.884 [0.815, 0.935] | 0.687 [0.587, 0.767] | 0.664 [0.531, 0.837] | 0.478 [0.379, 0.650] |
| EG2 raw patches | 0.887 [0.816, 0.941] | 0.691 [0.579, 0.785] | 0.640 [0.488, 0.833] | 0.460 [0.349, 0.654] |
| DINOv2-B patches | 0.877 [0.790, 0.938] | 0.708 [0.620, 0.791] | 0.713 [0.628, 0.842] | 0.532 [0.448, 0.677] |
| Random tower → global | 0.812 [0.692, 0.895] | 0.614 [0.496, 0.718] | 0.906 [0.814, 0.940] | 0.730 [0.619, 0.794] |
| Random tower pooled | 0.803 [0.683, 0.910] | 0.602 [0.493, 0.745] | 0.979 [0.947, 0.991] | 0.889 [0.821, 0.931] |
| Random tower raw | 0.838 [0.732, 0.926] | 0.641 [0.535, 0.774] | 0.995 [0.986, 0.999] | 0.952 [0.924, 0.982] |

### 192×192 crop / codec_only

| Encoder | ρ LPIPS | τ LPIPS | ρ −PSNR-Y | τ −PSNR-Y |
|---|---:|---:|---:|---:|
| EG2 global | 0.733 [0.579, 0.862] | 0.533 [0.404, 0.674] | 0.428 [0.152, 0.736] | 0.278 [0.091, 0.547] |
| EG2 pooled 3×3 | 0.860 [0.771, 0.930] | 0.680 [0.583, 0.779] | 0.396 [0.161, 0.717] | 0.247 [0.112, 0.503] |
| EG2 raw patches | 0.886 [0.804, 0.942] | 0.716 [0.621, 0.809] | 0.339 [0.060, 0.703] | 0.211 [0.067, 0.491] |
| DINOv2-B patches | 0.876 [0.748, 0.949] | 0.683 [0.551, 0.808] | 0.400 [0.114, 0.751] | 0.259 [0.112, 0.546] |
| Random tower → global | 0.361 [0.064, 0.664] | 0.246 [0.043, 0.478] | 0.232 [-0.052, 0.577] | 0.169 [-0.017, 0.437] |
| Random tower pooled | 0.471 [0.139, 0.795] | 0.303 [0.077, 0.604] | 0.924 [0.797, 0.980] | 0.789 [0.624, 0.905] |
| Random tower raw | 0.448 [0.042, 0.801] | 0.313 [0.059, 0.634] | 0.969 [0.921, 0.992] | 0.866 [0.786, 0.942] |

### Native input / bicubic_x2

| Encoder | ρ LPIPS | τ LPIPS | ρ −PSNR-Y | τ −PSNR-Y |
|---|---:|---:|---:|---:|
| EG2 global | 0.573 [-0.096, 0.885] | 0.426 [-0.054, 0.712] | 0.759 [0.199, 0.880] | 0.541 [0.083, 0.705] |
| EG2 pooled 3×3 | 0.676 [0.020, 0.900] | 0.513 [0.040, 0.743] | 0.794 [0.304, 0.904] | 0.567 [0.213, 0.735] |
| EG2 raw patches | 0.754 [0.153, 0.931] | 0.597 [0.097, 0.812] | 0.864 [0.511, 0.935] | 0.677 [0.373, 0.807] |
| DINOv2-B patches | 0.768 [0.143, 0.945] | 0.577 [0.092, 0.801] | 0.921 [0.577, 0.974] | 0.744 [0.409, 0.877] |
| Random tower → global | 0.814 [0.255, 0.948] | 0.628 [0.153, 0.827] | 0.823 [0.234, 0.952] | 0.651 [0.154, 0.858] |
| Random tower pooled | 0.741 [0.083, 0.950] | 0.582 [0.074, 0.820] | 0.972 [0.779, 0.992] | 0.882 [0.624, 0.948] |
| Random tower raw | 0.766 [0.093, 0.977] | 0.628 [0.095, 0.896] | 0.991 [0.906, 1.000] | 0.954 [0.787, 0.997] |

### Native input / bicubic_x4

| Encoder | ρ LPIPS | τ LPIPS | ρ −PSNR-Y | τ −PSNR-Y |
|---|---:|---:|---:|---:|
| EG2 global | 0.788 [0.263, 0.887] | 0.549 [0.139, 0.705] | 0.855 [0.310, 0.927] | 0.641 [0.168, 0.787] |
| EG2 pooled 3×3 | 0.735 [0.069, 0.870] | 0.479 [0.014, 0.675] | 0.870 [0.474, 0.940] | 0.659 [0.287, 0.782] |
| EG2 raw patches | 0.762 [0.194, 0.880] | 0.518 [0.058, 0.697] | 0.895 [0.466, 0.950] | 0.708 [0.274, 0.817] |
| DINOv2-B patches | 0.802 [0.281, 0.933] | 0.592 [0.181, 0.787] | 0.852 [0.350, 0.957] | 0.674 [0.219, 0.841] |
| Random tower → global | 0.858 [0.282, 0.939] | 0.672 [0.132, 0.812] | 0.793 [0.151, 0.929] | 0.595 [0.131, 0.796] |
| Random tower pooled | 0.889 [0.434, 0.949] | 0.703 [0.246, 0.805] | 0.942 [0.638, 0.994] | 0.826 [0.487, 0.961] |
| Random tower raw | 0.888 [0.422, 0.957] | 0.731 [0.306, 0.822] | 0.980 [0.874, 0.997] | 0.910 [0.739, 0.976] |

### Native input / bicubic_x8

| Encoder | ρ LPIPS | τ LPIPS | ρ −PSNR-Y | τ −PSNR-Y |
|---|---:|---:|---:|---:|
| EG2 global | 0.513 [-0.352, 0.894] | 0.372 [-0.266, 0.743] | 0.538 [-0.183, 0.881] | 0.392 [-0.115, 0.691] |
| EG2 pooled 3×3 | 0.665 [-0.115, 0.845] | 0.377 [-0.124, 0.650] | 0.828 [0.347, 0.920] | 0.613 [0.235, 0.769] |
| EG2 raw patches | 0.701 [-0.032, 0.833] | 0.449 [-0.012, 0.624] | 0.808 [0.307, 0.927] | 0.592 [0.216, 0.780] |
| DINOv2-B patches | 0.806 [0.299, 0.919] | 0.590 [0.181, 0.777] | 0.950 [0.760, 0.977] | 0.821 [0.616, 0.901] |
| Random tower → global | 0.768 [0.249, 0.900] | 0.536 [0.162, 0.727] | 0.880 [0.498, 0.962] | 0.715 [0.352, 0.859] |
| Random tower pooled | 0.857 [0.359, 0.960] | 0.674 [0.240, 0.853] | 0.971 [0.797, 0.995] | 0.900 [0.677, 0.963] |
| Random tower raw | 0.854 [0.373, 0.960] | 0.677 [0.289, 0.850] | 0.977 [0.821, 1.000] | 0.933 [0.756, 0.997] |

### Native input / h264_crf23

| Encoder | ρ LPIPS | τ LPIPS | ρ −PSNR-Y | τ −PSNR-Y |
|---|---:|---:|---:|---:|
| EG2 global | -0.001 [-0.557, 0.500] | -0.018 [-0.407, 0.319] | 0.384 [-0.158, 0.743] | 0.272 [-0.118, 0.561] |
| EG2 pooled 3×3 | 0.551 [-0.125, 0.888] | 0.415 [-0.058, 0.739] | 0.325 [-0.281, 0.792] | 0.233 [-0.174, 0.643] |
| EG2 raw patches | 0.626 [-0.023, 0.929] | 0.497 [0.023, 0.791] | 0.301 [-0.236, 0.826] | 0.233 [-0.089, 0.682] |
| DINOv2-B patches | 0.444 [-0.207, 0.863] | 0.308 [-0.137, 0.678] | 0.423 [-0.055, 0.840] | 0.321 [0.009, 0.695] |
| Random tower → global | 0.440 [-0.121, 0.734] | 0.290 [-0.102, 0.532] | -0.083 [-0.696, 0.593] | -0.062 [-0.503, 0.419] |
| Random tower pooled | -0.256 [-0.685, 0.400] | -0.141 [-0.469, 0.321] | 0.869 [0.533, 0.940] | 0.708 [0.442, 0.825] |
| Random tower raw | -0.117 [-0.702, 0.554] | -0.067 [-0.453, 0.434] | 0.942 [0.820, 0.980] | 0.823 [0.697, 0.918] |

### Native input / h264_crf33

| Encoder | ρ LPIPS | τ LPIPS | ρ −PSNR-Y | τ −PSNR-Y |
|---|---:|---:|---:|---:|
| EG2 global | -0.088 [-0.596, 0.592] | -0.044 [-0.420, 0.421] | 0.168 [-0.563, 0.702] | 0.108 [-0.447, 0.545] |
| EG2 pooled 3×3 | 0.404 [-0.402, 0.852] | 0.313 [-0.226, 0.689] | -0.114 [-0.579, 0.616] | -0.049 [-0.385, 0.443] |
| EG2 raw patches | 0.352 [-0.464, 0.851] | 0.256 [-0.264, 0.689] | -0.006 [-0.521, 0.687] | 0.038 [-0.295, 0.518] |
| DINOv2-B patches | 0.441 [-0.305, 0.888] | 0.364 [-0.138, 0.732] | 0.140 [-0.373, 0.714] | 0.115 [-0.197, 0.561] |
| Random tower → global | 0.378 [-0.176, 0.719] | 0.236 [-0.140, 0.505] | -0.016 [-0.644, 0.531] | -0.018 [-0.426, 0.375] |
| Random tower pooled | -0.473 [-0.743, 0.275] | -0.303 [-0.533, 0.261] | 0.914 [0.593, 0.982] | 0.797 [0.514, 0.925] |
| Random tower raw | -0.473 [-0.757, 0.298] | -0.300 [-0.553, 0.261] | 0.926 [0.689, 0.994] | 0.821 [0.605, 0.961] |

### Native input / bicubic_only

| Encoder | ρ LPIPS | τ LPIPS | ρ −PSNR-Y | τ −PSNR-Y |
|---|---:|---:|---:|---:|
| EG2 global | 0.861 [0.765, 0.925] | 0.657 [0.538, 0.768] | 0.747 [0.603, 0.863] | 0.557 [0.438, 0.688] |
| EG2 pooled 3×3 | 0.877 [0.780, 0.940] | 0.677 [0.560, 0.780] | 0.872 [0.735, 0.937] | 0.685 [0.563, 0.787] |
| EG2 raw patches | 0.900 [0.804, 0.958] | 0.723 [0.587, 0.824] | 0.876 [0.776, 0.919] | 0.691 [0.605, 0.762] |
| DINOv2-B patches | 0.925 [0.852, 0.967] | 0.762 [0.648, 0.845] | 0.879 [0.807, 0.939] | 0.708 [0.630, 0.815] |
| Random tower → global | 0.868 [0.785, 0.925] | 0.679 [0.578, 0.767] | 0.881 [0.700, 0.964] | 0.707 [0.515, 0.844] |
| Random tower pooled | 0.833 [0.751, 0.909] | 0.637 [0.562, 0.742] | 0.976 [0.913, 0.991] | 0.879 [0.768, 0.940] |
| Random tower raw | 0.841 [0.726, 0.928] | 0.651 [0.551, 0.776] | 0.992 [0.959, 0.998] | 0.943 [0.862, 0.976] |

### Native input / codec_only

| Encoder | ρ LPIPS | τ LPIPS | ρ −PSNR-Y | τ −PSNR-Y |
|---|---:|---:|---:|---:|
| EG2 global | 0.405 [0.134, 0.701] | 0.271 [0.073, 0.526] | 0.504 [0.229, 0.785] | 0.354 [0.137, 0.612] |
| EG2 pooled 3×3 | 0.712 [0.419, 0.942] | 0.544 [0.279, 0.811] | 0.509 [0.264, 0.808] | 0.361 [0.185, 0.638] |
| EG2 raw patches | 0.725 [0.447, 0.952] | 0.558 [0.302, 0.831] | 0.532 [0.275, 0.840] | 0.387 [0.218, 0.669] |
| DINOv2-B patches | 0.708 [0.457, 0.933] | 0.532 [0.300, 0.794] | 0.589 [0.367, 0.845] | 0.431 [0.277, 0.682] |
| Random tower → global | 0.329 [-0.109, 0.651] | 0.222 [-0.078, 0.463] | -0.003 [-0.477, 0.423] | -0.001 [-0.311, 0.289] |
| Random tower pooled | -0.006 [-0.298, 0.618] | 0.017 [-0.161, 0.483] | 0.891 [0.707, 0.971] | 0.756 [0.567, 0.876] |
| Random tower raw | 0.070 [-0.288, 0.672] | 0.058 [-0.156, 0.513] | 0.951 [0.891, 0.984] | 0.839 [0.752, 0.921] |

</details>

<details>
<summary>Historical two-frame CPU screen (superseded)</summary>

# EmbeddingGemma 2: CPU SR sensitivity screen

Run on 2026-10-17. This is a **provisional pass for a larger controlled probe**, not evidence that the encoder improves trained SR. Two natural images, three degradation levels per image, and no learned SR outputs cannot select a teacher or establish statistical reliability.

## What actually ran

- Stdlib-only disk/dependency/HF metadata and weight-access preflight: [JSON](embeddinggemma2-preflight.json). HF was public/ungated; the pinned weight URL returned HTTP 200 and **1,488,915,288 bytes**. There was over 110 GB free before the model run. The file downloaded successfully; no GPU was used.
- Processor-only screen, without loading model weights: [JSON](embeddinggemma2-processor-only.json). Every bicubic degradation survived preprocessing with nonzero valid-pixel MSE. This checks that a feature-sensitivity probe is meaningful; it does not measure model sensitivity.
- Full float32 CPU feature probes: [224 crop JSON](embeddinggemma2-probe-crop224.json) and [native-frame JSON](embeddinggemma2-probe-native.json), including per-frame PSNR, all feature distances, timings, crop coordinates, masks/geometries, repeat, spatial permutation, wrong-image, and BF16-cache controls. [Script](../../scripts/probe_embeddinggemma2.py).

Pinned model: `google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b`; safetensors SHA256 `197a32965d4b1105faf060417baa899e193fb73cd401f42ec9295234d5553d79`.

Actual environment: Python 3.13.5, torch 2.14.1+cpu, torchvision 0.29.1+cpu, transformers 5.21.0.dev0 from source commit `8498ed425fa01bbee7d086efdf1a6634c2962aed`, Pillow 12.3.0, NumPy 2.5.2, four CPU threads. `config.audio_config=None` before model construction; actual loaded text/vision parameter count **438,760,448**, of which the vision tower has **167,364,608**. Model state was float32, evaluation mode, with gradients disabled and SDPA attention. LPIPS was **not computed**; the package/extra learned weights were not installed.

## Protocol and interpretation

Images were `/workspace/data/prepared/{sunflower,rush_hour}/HR/00001.png`, both 1920×1080. Center crop coordinates were `(848,428,1072,652)`. Each HR image/crop was bicubically downsampled by 2, 4, or 8 and bicubically restored to exactly the HR dimensions with Pillow. HR and degraded inputs then used the identical model processor and 280-soft-token budget.

Global distance is cosine distance between L2-normalized, attention-masked means of final 768-dimensional sequence states, including valid boundary tokens. Dense distances are mean cosine distances at corresponding positions: either final 768-dimensional vision states after 3×3 spatial pooling, or final raw 16×16-patch encoder-layer states captured before pooling by a forward hook. Padding patches are removed using `image_position_ids>=0`. Both dense streams were captured during the same complete global-embedding forward; no second teacher forward was required. Pixel MSE is calculated over valid processed pixels **before inference**. All returned features and processed pixels were finite.

The 224×224 crop is **upscaled to 768×768** by this processor: 2,304 valid raw patches plus 216 padded slots, then 256 valid pooled tokens and 260 total valid sequence tokens. Native 1920×1080 is **downscaled to 1056×576**: 2,376 valid raw patches plus 144 padded slots, then 264 pooled tokens and 268 valid sequence tokens. Thus these are different detail regimes; the stronger crop response must not be extrapolated to native full-frame detail.

Within each of the four image/view cases, all three feature distances strictly increased as RGB PSNR decreased: Spearman ρ=1.0 across the **three** degradation levels. This is descriptive ordering only. Global, pooled, and raw distances have different scales and should not be compared as if they were calibrated quality metrics.

## Measured values

Distances are to the matching HR target; values below are rounded. Full precision is in the JSON files.

| View | Image | Bicubic scale | RGB PSNR dB | Global cosine distance | Pooled dense distance | Raw patch distance |
|---|---|---:|---:|---:|---:|---:|
| 224 crop | sunflower | ×2 | 38.474 | 0.01831 | 0.17575 | 0.35708 |
| 224 crop | sunflower | ×4 | 33.396 | 0.08002 | 0.39400 | 0.59260 |
| 224 crop | sunflower | ×8 | 28.310 | 0.16313 | 0.52965 | 0.70557 |
| 224 crop | rush_hour | ×2 | 42.572 | 0.01061 | 0.18930 | 0.38673 |
| 224 crop | rush_hour | ×4 | 37.511 | 0.04633 | 0.27825 | 0.50356 |
| 224 crop | rush_hour | ×8 | 28.916 | 0.07189 | 0.40714 | 0.62609 |
| Native frame | sunflower | ×2 | 42.777 | 0.00197 | 0.02967 | 0.05359 |
| Native frame | sunflower | ×4 | 38.889 | 0.00832 | 0.10234 | 0.17799 |
| Native frame | sunflower | ×8 | 34.207 | 0.02700 | 0.21796 | 0.37296 |
| Native frame | rush_hour | ×2 | 42.544 | 0.00163 | 0.02850 | 0.05369 |
| Native frame | rush_hour | ×4 | 38.702 | 0.01229 | 0.06568 | 0.13867 |
| Native frame | rush_hour | ×8 | 33.048 | 0.05434 | 0.18840 | 0.35055 |

## Noise and correspondence controls

The largest unchanged-repeat cosine distances across both views and images were global **1.8e-08**, pooled **3.17e-08**, and raw **6.18e-08**. Fixed image selection/crops and seeds do not guarantee bitwise-identical multithreaded CPU arithmetic; these measured repeats establish the numerical floor. Repeated processed pixels had exactly zero MSE.

BF16 roundtrip cache noise was at most global **1.42e-06**, pooled **1.37e-06**, and raw **1.37e-06** cosine distance. Global vectors were normalized before caching/quantization; dense vectors were cached before the per-token normalization used by the distance. The smallest degraded-image signal was global **0.00163**, pooled **0.02850**, raw **0.05359**, well above both noise controls.

Valid processed-pixel MSE was positive throughout (**1.06e-05–0.00143**, RGB scaled to [0,1]), so the processor retained a measurable degradation signal. Randomly permuting HR dense feature positions raised correspondence distances to **0.669–0.888**. This is a feature-permutation control, not an additional shuffled-image model inference; the global vector is unchanged by construction. Different-image HR controls also had larger distances than the same-image mild degradation, with exact values in JSON.

The final crop and native runs took **28.53 s** and **36.78 s**, respectively (excluding package installation, weight download, and metadata preflight); each made ten complete forwards: two HR references, six degraded images, and two unchanged repeats.

## Decision and limitations

Both global and dense arms survive this initial sensitivity screen and can enter a larger, held-out probe. They still require DINOv2/SigLIP/CLIP controls at matched input geometry and teacher compute, translated/cropped/aligned spatial tests, codec/ringing/noise degradations, and PSNR plus LPIPS on more independent images before any VSR training allocation. This probe provides no evidence that greater teacher distance predicts improvements in reconstructed fine detail, that the global arm is better than the dense arm, or that either improves temporal stability. The samples are two first frames from two sequences, not a training/validation evaluation. No training was run.

## Reproduce

The exact commands used for the final model runs are below; model files were manually downloaded at the pinned revision to `/tmp/eg2-model` and SHA256-checked by the script. To let the script download instead, replace `--local-model /tmp/eg2-model` with `--download`; the default weight budget is 2 GB. Temporary files, environments, and weights are outside the repository.

```bash
UV_CACHE_DIR=/tmp/eg2-uv-cache uv venv /tmp/eg2-probe-venv --python /usr/bin/python3.13
UV_CACHE_DIR=/tmp/eg2-uv-cache uv pip install --python /tmp/eg2-probe-venv/bin/python torch==2.14.1+cpu torchvision==0.29.1+cpu --index-url https://download.pytorch.org/whl/cpu
UV_CACHE_DIR=/tmp/eg2-uv-cache uv pip install --python /tmp/eg2-probe-venv/bin/python numpy==2.5.2 pillow==12.3.0 https://github.com/huggingface/transformers/archive/8498ed425fa01bbee7d086efdf1a6634c2962aed.zip
/usr/bin/python3.13 scripts/probe_embeddinggemma2.py --preflight --output docs/research/embeddinggemma2-preflight.json
/tmp/eg2-probe-venv/bin/python scripts/probe_embeddinggemma2.py --processor-only --local-model /tmp/eg2-model --output docs/research/embeddinggemma2-processor-only.json
/tmp/eg2-probe-venv/bin/python scripts/probe_embeddinggemma2.py --local-model /tmp/eg2-model --output docs/research/embeddinggemma2-probe-crop224.json --threads 4
/tmp/eg2-probe-venv/bin/python scripts/probe_embeddinggemma2.py --local-model /tmp/eg2-model --crop 0 --output docs/research/embeddinggemma2-probe-native.json --threads 4
```

The first three lines reproduce the actual core versions/source, consolidated from the installation commands. The JSON also records the installed transformers archive URL. The script was syntax-checked with `py_compile`; normal model and processor-only paths ran successfully. Tied Spearman ranks return `None`, and the summary emits JSON `null` rather than NaN if a representation collapses.

</details>
