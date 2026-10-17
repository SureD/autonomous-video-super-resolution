# EmbeddingGemma 2 as training-only guidance for a 60 FPS VSR student

- Date: 2026-10-17
- Author: an independent pass, separate from the notes in `STATUS.md` and from `docs/research/cursor-independent-review.md`. Those documents were read. They are not the source of the interface facts below.
- Question: during training only, can the pretrained `google/embeddinggemma-2` (native 768-d multimodal embedding; vision tower stated at 170M) steer a compressed real-time student (540p→1080p ×2, ≥60 FPS on an RTX 3070 8GB, TensorRT) without remaining in the exported graph?
- Scope: training loss. The inference graph stays the student that exp001 keeps. Diffusion VSR is out of scope. No GPU was available. Nothing in this note is a measured PSNR, LPIPS, FPS, or VRAM result.
- Citation rule: a claim about the model is cited only if that page or file was opened on 2026-10-17. Repository code is cited as code. Arithmetic from opened fields is labeled arithmetic and was not executed as a forward. Unverified items are marked.

## 0. Decision

**Conditional go**, for one loss and one student.

The loss is an output perceptual loss: cosine distance between the official image-only 768-d L2-normalized mean-pooled embeddings of the SR crop and the HR crop. The encoder’s weights are frozen. The encoder is absent from ONNX. The student is whichever existing net in `src/vsr/models.py` clears the exp001 TensorRT FP16 gate, then frozen for this study.

**No-go**, before any training run, for:

- dense patch-feature alignment as the primary arm;
- the model’s video pathway (default 1 frame per second, 140 soft tokens per frame) as a temporal training loss;
- `float16` (the model card states that it returns NaNs or silently degraded embeddings);
- loading the audio tower on this card;
- a new student architecture;
- any claim that the student matches a large VSR teacher. EmbeddingGemma 2 is not a VSR model. The target is the embedding of the clean crop.

The conditional go becomes a **no-go of the whole scheme** if gate 0 in §7 fails. The expected scientific outcome, if the scheme is run, is a small or null LPIPS move and a PSNR risk. No decibel effect is predicted.

This note specializes the embedding experiment in §10 of `docs/research/cursor-independent-review.md` to one named encoder. It does not reopen H1–H5.

## 1. Sources opened, and what was not

Opened:

| Source | URL |
|---|---|
| Hugging Face model card (rendered) | https://huggingface.co/google/embeddinggemma-2 |
| Same card as raw README, including the front-matter license line | https://huggingface.co/google/embeddinggemma-2/raw/main/README.md |
| Google model card (same architecture and benchmark tables) | https://ai.google.dev/gemma/docs/embeddinggemma/model_card_2 |
| Checkpoint `config.json` | https://huggingface.co/google/embeddinggemma-2/raw/main/config.json |
| Transformers modeling source, `main`, fetched 2026-10-17, no commit SHA recorded | https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/embedding_gemma2/modeling_embedding_gemma2.py |
| Transformers model doc, raw `main` | https://raw.githubusercontent.com/huggingface/transformers/main/docs/source/en/model_doc/embedding_gemma2.md |
| Sentence Transformers tutorial (page states last updated 2026-10-06 UTC) | https://ai.google.dev/gemma/docs/embeddinggemma/multimodal-embeddinggemma-with-sentence-transformers |
| Gemma 4 vision section (generative family; used only for the shared vision geometry) | https://raw.githubusercontent.com/huggingface/transformers/main/docs/source/en/model_doc/gemma4.md |
| DINOv2-base model card | https://huggingface.co/facebook/dinov2-base |
| DINOv2-base `config.json` | https://huggingface.co/facebook/dinov2-base/raw/main/config.json |
| SigLIP transformers doc | https://huggingface.co/docs/transformers/model_doc/siglip |

Not opened, so not used as evidence:

- The raw `LICENSE` file. `https://huggingface.co/google/embeddinggemma-2/raw/main/LICENSE` returned 404. The README front matter says `license: apache-2.0`, and the card banner says “License: Apache 2.0”. The license text itself was not read. The card also names a Gemma Prohibited Use Policy; that policy page was not opened.
- Gemma 4 image-processor source. Resize behavior on a 192×192 crop and on a 1920×1080 frame was not executed.
- SigLIP modeling source, and `google/siglip-base-patch16-224`’s own `config.json`. The vision class defaults and the documented example were opened; a checkpoint parameter count was not.
- DINOv2 modeling source. The card and config were opened. A forward was not run.
- Any EmbeddingGemma 2 paper. The cards do not link one that was fetched here.

`transformers` `main` can move. The modeling description is the file fetched on this date.

Two documents disagree, and this note does not pick a winner:

| Field | Model card | Other opened file |
|---|---|---|
| Total parameters | 740M | Sentence Transformers tutorial prints `744371488` for the full model |
| Text + image, audio omitted | 440M | Transformers doc says 439M without the audio tower, 271M with neither tower |
| Sliding window | 1024 tokens | `config.json` `text_config.sliding_window` is `512` |

The tutorial also prints `576613664` parameters for a load with `vision_config=None` (audio kept, vision dropped). That is a third count, not a conflict with the 440M / 439M text+vision row.

## 2. What the vision tower exposes

### 2.1 The published embedding is one pooled vector

The model card’s output is a single 768-d vector. Pooling is mean pooling, then a projection 512→768. Supported Matryoshka truncations are 128, 256, 512, and 768, re-normalized after truncation.

The Sentence Transformers tutorial prints the stack that `encode` actually runs:

1. `Transformer`, every modality’s `method_output_name` set to `last_hidden_state`;
2. `Pooling`, `embedding_dimension` 768, `pooling_mode` mean, `include_prompt` True;
3. `Normalize`.

Printed shapes in that tutorial for image, interleaved, and media-only inputs are `(768,)`. The transformers doc’s text example prints `torch.Size([4, 768])` after mask-aware mean pooling and L2 norm in `float32`.

In the modeling source, `EmbeddingGemma2TextModel` applies `embedding_projection` (linear, `hidden_size` 512 → `embedding_dim` 768, no bias) to every token. The source comment says projecting per token is equivalent to projecting after mean pooling. `EmbeddingGemma2Model.forward` returns that projected `last_hidden_state`. The class docstring says the model is meant to be wrapped by Sentence Transformers’ mean pooling and normalization.

So the loss that matches the published model is cosine distance on that pooled, L2-normalized 768-d vector. Truncation is not part of the primary arm. The card’s MMEB v2 overall is 59.01 at 768-d, 56.24 at 256-d, and 45.65 at 128-d. 128-d is the dimension the card says degrades multimodal quality. If a later cache-size ablation truncates, it uses 256 and re-normalizes, and it is not the primary.

### 2.2 Soft tokens exist, and they are not a DINOv2 grid

`config.json` `vision_config`:

| Field | Value |
|---|---|
| `model_type` | `gemma4_vision` |
| `hidden_size` | 768 |
| `patch_size` | 16 |
| `pooling_kernel_size` | 3 |
| `default_output_length` | 280 |
| `num_hidden_layers` | 16 |
| `num_attention_heads` | 12 |
| `intermediate_size` | 3072 |
| `position_embedding_size` | 10240 |
| RoPE | `axial` |
| `dtype` | `bfloat16` |

Top-level `vision_soft_tokens_per_image` is 280. The card and the transformers doc both list soft-token budgets `{70, 140, 280, 560, 1120}`. Default image budget is 280 tokens (~29 images in the 8192 context at that budget; the card says a lower budget reaches ~114 images or frames). Default video budget is 140 tokens per frame. Video defaults in the transformers doc: `fps=1`, `max_frames=32`, `overflow_strategy="uniform"`, `add_timestamps=False`.

`get_image_features` does not return the pooled 768-d embedding. It runs the vision tower, then `embed_vision`. `EmbeddingGemma2MultimodalEmbedder` is an RMSNorm (no scale) plus a bias-free linear map from the vision hidden size (768, unless `output_proj_dims` is set; it is not set on the vision config) into the text hidden size (512). The method then splits that tensor on non-padding `image_position_ids`. Padding positions are `(-1, -1)`. The split tensors are stored on `pooler_output`. Those are variable-length soft tokens in text space, before the text encoder.

`forward` scatters those soft tokens into the text-token sequence at image placeholder positions, runs the bidirectional text encoder (sliding and full layers, per-layer embeddings), and returns:

- `last_hidden_state`: text-encoder outputs after the 512→768 projection, shape `(batch, sequence, 768)` before user-side mean pooling;
- `image_hidden_states`: the concatenated projected soft tokens from the scatter step, and only when `pixel_values` is not None.

The dataclass docstring says `image_hidden_states` has shape `(batch, num_images, sequence_length, hidden_size)`. The assignment in `forward` is the concatenated soft-token tensor produced for `masked_scatter`, not a stacked 4-D tensor. The docstring shape is inconsistent with that assignment. Treat the docstring shape as unverified. The implemented object is a variable-length 512-d sequence with padding removed.

The transformers doc prints one processor call: `processor(images=[IMAGE], max_soft_tokens=70)` → `input_ids` `[1, 67]`, `pixel_values` `[1, 630, 768]`. Patch pixels are `16×16×3 = 768`, so `pixel_values` are flattened patches before the 3×3 pool. `70×9 = 630`. The token sequence length is 67, not 70. A soft-token budget is not the sequence length of `last_hidden_state`.

The Gemma 4 doc (generative checkpoints, not this embedding checkpoint) states the geometry that matches this config: aspect ratio kept, height and width divisible by 48 (`16×3`), no ImageNet mean/std, 2-D position table of 10240 per axis. Its pixel-area table:

| Soft tokens | Patches before pooling | Approx. area |
|---:|---:|---:|
| 70 | 630 | ~161K pixels |
| 140 | 1,260 | ~323K pixels |
| 280 | 2,520 | ~645K pixels |
| 560 | 5,040 | ~1.3M pixels |
| 1,120 | 10,080 | ~2.6M pixels |

That table was not measured on `google/embeddinggemma-2`. The config matches (`patch_size` 16, `pooling_kernel_size` 3, default 280, position table 10240), and the printed `pixel_values` shape at budget 70 matches 630. The area column stays a Gemma 4 statement until gate 0 records the resized `H×W` for this processor.

### 2.3 Why dense alignment is a no-go as the primary arm

Four separate facts, all from the files above:

1. The trained objective that the cards publish is the mean-pooled 768-d vector. A dense loss would optimize a representation the checkpoint was not trained to expose as its output.
2. Soft tokens are 3×3-pooled 16-pixel patches, variable count, aspect-preserving. They are not a fixed grid lined up with a CNN feature map.
3. After they enter the text encoder, bidirectional attention mixes them with whatever else is in the sequence. Positions of `last_hidden_state` are not a spatial map. Mean pooling includes non-padding tokens (`include_prompt` is True). The printed budget-70 sequence is already a different length from the soft-token count.
4. `get_image_features().pooler_output` and `image_hidden_states` are 512-d, variable length, and pre-text-encoder. `image_position_ids` could support a hand-written matcher. That matcher is an unpublished loss. It is not this model’s embedding, and it is not the fixed patch grid DINOv2 returns.

An exploratory dense arm is allowed only after gate 0 prints a stable spatial index, and only as an appendix. It is not in the primary table. It does not get a second primary metric.

### 2.4 Prompts and video

The card says task prefixes apply to text. Images, video, and audio are passed with no prefix. `SentenceSimilarity` (`task: sentence similarity | query: {content}`) is a text prefix. The pooling module includes the prompt. On a short visual sequence a text prefix is a large share of the mean. The primary loss is image-only: the same `max_soft_tokens` on SR and HR, no text on either side. If a prefix is ever added, it is byte-identical on both sides and the run is not the primary arm.

Video mode is the wrong interface for a per-frame 60 FPS student. The default path samples 1 frame per second and spends 140 soft tokens per kept frame, then still returns one pooled vector for the clip. Temporal supervision in this study is the existing tLPIPS report, not a video embedding.

### 2.5 Compared with DINOv2-base and SigLIP

Opened this session, not taken from memory:

**DINOv2-base.** The card (written by Hugging Face, as the card itself says) describes a ViT: fixed-size patches, a `[CLS]` token prepended, absolute position embeddings. The usage example returns `last_hidden_state`. The card says the CLS token’s last hidden state is the usual image vector. `config.json`: `hidden_size` 768, `image_size` 518, `patch_size` 14, 12 layers, 12 heads. The card’s parameter count is 86.6M, dtype float32. Arithmetic from that config, not a forward: `518/14 = 37`, so 1369 patch tokens plus the CLS token. The input is a fixed square. Patch tokens are the rest of one sequence, on a regular grid.

**SigLIP.** `SiglipVisionConfig` defaults opened in the transformers doc: `image_size` 224, `patch_size` 16, `hidden_size` 768, 12 layers, 12 heads. `SiglipVisionModel.forward` documents both `last_hidden_state` `(batch, sequence, hidden)` and `pooler_output` `(batch, hidden)`. The vision example loads `google/siglip-base-patch16-224` and comments `pooler_output` as pooled features. The pooler docstring text about a BERT classification token is the generic autodoc paragraph; the text-model example separately comments its pooler as EOS. This note does not treat the SigLIP vision pooler as a CLS token. The classification-head sentence in that doc says the head sits on pooled final hidden states of the patch tokens. A SigLIP parameter count was not opened.

| | EmbeddingGemma 2, text+vision | DINOv2-base | SigLIP vision, as documented |
|---|---|---|---|
| What a normal call returns | one 768-d mean-pooled, L2-normalized vector | `last_hidden_state`; CLS is the card’s image vector | `last_hidden_state` and `pooler_output` |
| Spatial tokens available without a new head | soft tokens at 512-d, variable length, then mixed by a text encoder | regular patch grid, patch 14, canvas 518 | patch sequence on a square canvas, default patch 16, canvas 224 |
| Input geometry | aspect kept, sides multiple of 48, budget in `{70…1120}` | fixed square 518 | fixed square, default 224 |
| Weights to load for the image loss | card 440M / transformers doc 439M, audio off | 86.6M | not opened |
| Role in this study | the training encoder, pooled cosine only | the encoder to use if a dense arm is ever justified | same remark as DINOv2 for a dense arm; not the training encoder here |

DINOv2-base is the smaller and cleaner dense teacher. EmbeddingGemma 2’s relevant property is the single pooled multimodal vector and a documented switch to drop the audio tower (`audio_config=None`). This plan does not claim that EmbeddingGemma 2 is a better perceptual encoder than either model. The prior review already opened REPA and recorded that REPA aligns to DINOv2 patch tokens; REPA was not re-opened here, and no REPA number is repeated. SigLIP as an SR training loss was not opened.

The primary metric stays LPIPS-Alex, which none of these encoders is. Reporting DINOv2 cosine or SigLIP cosine as a transfer metric is allowed. Selecting the winner by EmbeddingGemma cosine is not.

## 3. Student

No new architecture. The candidates already constructed in `src/vsr/models.py`, with parameter counts taken from the layer definitions (torch is not installed here; the counts are integer arithmetic on those definitions):

| Name | As defined | Parameters |
|---|---|---:|
| `espcn` | RGB ESPCN, ×2 | 26,796 |
| `fsrcnn` | `d=56`, `s=12`, `m=4`, ×2 | 24,683 |
| `ecbsr` | `m=4`, `ch=16`, `reparam=True` (train-time branches) | 44,604 |
| `ecbsr` after `merge()`, and `ecbsr_plain` | plain 3×3 topology | 11,548 |
| `plain` | `ch=32`, `n=6` | 59,852 |

`configs/train_ecbsr.yaml` is the configured run: `ecbsr` with `m=4`, `ch=16`, LR patch 96 (HR crop 192 at ×2), batch 16, 200k iterations, Charbonnier is the training loss elsewhere in this repo. That net is the default candidate. It is not selected until exp001.

Selection rule, in order, before this study looks at E1:

1. exp001 must show a TensorRT FP16 engine, batch 1, 540p→1080p, at ≥60 FPS end to end. A 2% FPS gap on an identical merged graph is an invalid engine, as the prior review already argued, not a reason to switch nets.
2. Among nets that pass, keep the one with the better Charbonnier PSNR on the variance pilot (§7). Ties break toward the merged `ecbsr` graph, because that is the configured student and the merge is what export ships.
3. One student enters the embedding table. This study is not a model bake-off.
4. The loss does not change channel count, pixel-shuffle, the nearest residual, or `merge()`. Export hashes the ONNX graph against the Charbonnier export of the same student.

## 4. Teacher

There is no VSR teacher inside this checkpoint.

The training target is `E(HR crop)`: the frozen official embedding of the clean crop. That uses privileged clean information, which is allowed at train time and forbidden at inference. The SR argument of the loss is `E(student(LR crop))`, same crop, same budget.

A classical non-diffusion pixel teacher, precomputed on the same LR, is optional arm F. It is H1-style distillation. It is not an EmbeddingGemma condition. Diffusion teachers are excluded.

`E(LR)` is not a target. The prior review’s DA-CLIP reading is the reason: the degraded embedding is the wrong side of the residual. That negative is not a primary arm.

## 5. Loss, weights, arms, controls

### 5.1 Primary loss

For a paired crop, image-only, one frozen `max_soft_tokens` chosen in gate 0:

```
L = Charbonnier(SR, HR) + λ * (1 - cosine(E(SR), E(HR)))
```

`E` is the official pool: mask-aware mean of `last_hidden_state`, L2 norm in float32, full 768 dimensions. `λ` is not a tuned grid. At step 0, scale `λ` so the embedding term’s gradient norm on the student is 0.1 times the Charbonnier gradient norm, then freeze `λ`. The pilot may show that 0.1 saturates or vanishes; the replacement fraction is written down before E1 and is not chosen on E1.

Charbonnier stays in every arm so a semantic pull cannot ignore the pixels.

### 5.2 Arms

Same iterations, paired crops (the prior review’s pairing requirement: seed *k* of every arm sees the same crop stream), one student.

| Arm | Loss | What it answers |
|---|---|---|
| A | Charbonnier(SR, HR) | Pixel baseline |
| B | A + cosine(E(SR), E(HR)), pretrained E, image-only | The scheme under test |
| C | B, but E is this same architecture randomly initialized and frozen | Untrained-encoder control |
| D | B, but E is a fixed Gaussian map to 768-d | Random-projection control, no visual pretraining |
| E | A + VGG-19 feature L2, same 0.1 gradient-norm rule | Classical perceptual control. B has to beat this, not only pixels |
| F (optional, not required to call the scheme) | A + L1(SR, T), T a classical non-diffusion teacher, precomputed on the same LR | Separates “a teacher helps” from “this embedding helps” |

Arm D, specified so it can be implemented: bilinear-resize the crop to 16×16×3 (768 values), apply one frozen Gaussian matrix `768×768` drawn once and stored, L2-normalize. The matrix is not trained. The resize is fixed and identical for SR and HR.

No dense soft-token arm in this table.

### 5.3 Metrics

Inherited from the prior review, not weakened:

- Primary: LPIPS-Alex. It is not E, not VGG-19, and not in the loss.
- Constraint: PSNR-Y non-inferiority. The margin is a multiple of the clip-level SD from a 5-seed Charbonnier pilot. The issue-draft numbers 0.05 dB, 0.1 dB, and 0.2 dB are not reused as if they had been measured.
- Secondary: tLPIPS, plus `warp_err_sr − warp_err_gt`. Bicubic is a row on the temporal table.
- Unit: the clip. Paired difference, cluster bootstrap over clips. Five clips is a small-sample interval and is labeled as one.
- Sample size: 5 seeds after that variance pilot. Tuning stays off E1 (`docs/datasets.md`).
- Do not rank arms by EmbeddingGemma cosine.

## 6. 8GB arithmetic

Labeled arithmetic. Peak VRAM was not measured.

### 6.1 Weights

Audio off (`audio_config=None` / `config_kwargs={"audio_config": None}`).

| Load | Opened count | bf16 weights | fp32 weights |
|---|---:|---:|---:|
| Text + vision, card | 440M | 0.88 GB | 1.76 GB |
| Text + vision, transformers doc | 439M | 0.878 GB | 1.756 GB |
| Full model, printed parameter count | 744,371,488 | 1.489 GB | 2.977 GB |

The card forbids `float16`. The 3070 path is `bfloat16` when `torch.cuda.is_bf16_supported()` is true, otherwise `float32`. That check was not run. If it is false, the weight footprint is the fp32 column. Activations are extra and are not in the table.

Student weights are negligible next to this (the largest candidate above is 59,852 parameters). Optimizer state stays on the student only: EmbeddingGemma parameters are `requires_grad_(False)`.

### 6.2 What “frozen” does not save

The HR embedding is a constant. It can be computed under `torch.no_grad()`.

The SR embedding cannot. Cosine distance trains the student only if the graph from the SR image through the tower is retained for backward. Freezing the weights removes Adam moments for ~440M parameters. It does not remove the tower’s activations. That backward is the 8GB risk. Gradient checkpointing on the tower is the mitigation to try if a one-crop backward runs out of memory. It was not timed and it was not shown to fit.

The configured student batch is 16. The embedding term does not have to use all 16. Gate 0 picks the largest microbatch in `{1, 2, 4}` that completes one backward with the student optimizer resident. Charbonnier can keep batch 16. The microbatch size is frozen with `λ`.

### 6.3 Cache

The encoder is not translation-equivariant: axial positions and the aspect-preserving resize depend on the processed canvas. Cropping a cached full-frame token map is a different input from encoding the crop. Cache the crop.

Training crops in `FrameFolderSR` are random. The cache key is `(frame id, y, x, height, width, max_soft_tokens, dim, dtype)`, write-once on disk.

| Object | Arithmetic | Per 192-crop |
|---|---|---:|
| Pooled 768-d, bf16 | `768 × 2` | 1,536 B |
| Pooled 768-d, fp32 | `768 × 4` | 3,072 B |
| `last_hidden_state` upper bound | sequence length was **not** printed at budget 280. Budget-70 printed length 67. A bound of 320 tokens × 768 × 2 | 480 KiB |
| `pixel_values` at budget 280 | `2520 × 768 × 2`, using 630 patches at budget 70 scaled by 4 | 3.69 MiB |

One million pooled bf16 vectors are about 1.5 GB of disk, not of VRAM. Do not cache `pixel_values` or pre-pool token states for the primary arm. The live cost is the SR backward, which a disk cache does not remove.

`configs/train_ecbsr.yaml` uses LR patch 96, so the HR crop is 192×192 (36,864 pixels). 192 is divisible by 48. The Gemma 4 area for budget 70 is ~161K pixels, which is larger than 36,864. If the processor leaves the crop at 192, the soft-token count is `(192/16)² / 9 = 16`. That is arithmetic from `patch_size` and `pooling_kernel_size`. It is false if the processor upscales a small image to fill the budget. Gate 0 is what decides.

A 1920×1080 frame is 2,073,600 pixels. 1080 is not divisible by 48 (`1080/48 = 22.5`); 1920 is (`1920/48 = 40`). The processor must change at least one side. Budget 1120’s Gemma 4 area (~2.6M) can cover the pixel count and still is not a native 1080p encode. The loss is not computed on full frames.

### 6.4 Why the crop budget should be the smallest one that does not downscale

The card says higher soft-token budgets improve fine-grained understanding. That sentence is also the reason a mean-pooled retrieval embedding is a weak ×2 residual: the residual is the detail a retrieval model is trained to be partly invariant to. For a 192 crop, a budget that upscales the crop invents a second resampling; a budget that downscales it throws away the residual the student is learning. Gate 0 prints the resized size at 70 and at 280. The pre-registered budget is the smallest value in `{70, 140, 280}` whose resized canvas does not shrink the 192 crop. One value. Both SR and HR use it.

## 7. Gate 0, then stop rules

Gate 0 is a no-go filter. It is not a training run. It has not been executed.

Record, on one 192×192 crop and one 1920×1080 frame, at budgets 70 and 280 (1120 only for the full frame):

- resized `H×W`;
- `pixel_values` shape;
- count of `image_position_ids` that are not `(-1, -1)`;
- soft-token count after the 3×3 pool;
- `input_ids` length (the budget-70 doc example is 67, not 70);
- `last_hidden_state` shape and dtype;
- `torch.cuda.is_bf16_supported()` if a GPU exists.

Then one backward of arm B on a single crop, student optimizer resident, audio off, dtype from the check above. Record peak memory. If that backward does not fit, turn on gradient checkpointing and try once more. If it still does not fit, stop. Do not start arms C–E.

**No-go before training if any of these hold:**

- exp001 has no student at ≥60 FPS TensorRT FP16, 540p→1080p, batch 1;
- the processor shrinks the 192 crop at every budget that fits the backward (the embedding cannot see the ×2 residual);
- the one-crop backward does not fit in 8GB in bf16 or, if bf16 is unsupported, in fp32;
- the only way to attach the loss is to put the encoder in the ONNX graph.

**Stop after the paired runs if any of these hold:**

- the clip-bootstrap interval of B−C or of B−D on LPIPS-Alex contains 0 (B is a regularizer);
- B does not beat A on LPIPS-Alex and does not beat E on LPIPS-Alex;
- the upper end of the clip interval of the clean PSNR-Y drop exceeds the margin set from the pilot;
- LPIPS-Alex improves and tLPIPS worsens (per-frame semantic agreement with flicker);
- the exported ONNX graph hash differs from arm A’s export.

A pass is: B beats A and E on LPIPS-Alex, the intervals of B−C and B−D exclude 0 in B’s favor, PSNR-Y non-inferiority holds, tLPIPS does not worsen, and the graph hash matches. That pass is evidence about this student and this encoder. It is not evidence that the student matches a large VSR model. The recovered-fraction estimand in the prior review needs a VSR or SR teacher (arm F). Arm F is optional and has its own comparison against A. It is not folded into the embedding pass.

## 8. What remains unverified

- Apache 2.0 as a license text. The card and README front matter assert it. The `LICENSE` blob 404’d.
- Exact resize of a 192 crop and of a 1080p frame under this processor.
- Whether `image_hidden_states` is ever a 4-D tensor. The docstring and the assignment disagree; the assignment was read.
- Sliding window 1024 (card) versus 512 (`config.json`).
- 740M versus the printed 744,371,488, and 440M versus 439M.
- bf16 support and peak memory on an RTX 3070.
- SigLIP checkpoint parameter count, and the true pooling token inside `SiglipVisionModel` (modeling source not opened).
- Any quality delta. None is claimed.
