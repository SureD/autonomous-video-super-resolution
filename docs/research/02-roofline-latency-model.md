# RTX 3070 causal ×2 SR: a falsifiable latency budget

Date: 2026-10-16. **All latencies below are predictions, not measurements.** No GPU was used. This is an analytical screening model, not evidence of 60 FPS deployment. Read alongside [related work](01-related-work-check.md) and the [agenda](03-research-agenda.md).

## Hardware and timing convention

NVIDIA's [Ampere architecture whitepaper, Table 10, printed pp. 47–48](https://images.nvidia.com/aem-dam/en-zz/Solutions/geforce/ampere/pdf/NVIDIA-ampere-GA102-GPU-Architecture-Whitepaper-V1.pdf) was opened, including the table and footnotes. For the desktop RTX 3070 it lists:

| Published quantity | Dense | With structured sparsity |
|---|---:|---:|
| FP16 Tensor throughput, FP32 accumulation | 40.6 TFLOP/s | 81.3 TFLOP/s |
| FP16 Tensor throughput, FP16 accumulation | 81.3 TFLOP/s | 162.6 TFLOP/s |
| INT8 Tensor throughput | 162.6 TOP/s | 325.2 TOP/s |
| DRAM bandwidth; capacity; L2 | 448 GB/s; 8 GB; 4096 KB | — |

These are boost-clock peaks. A multiply-add counts as **two** operations throughout. Ordinary dense convolutions cannot claim sparse throughput. The actual TensorRT tactic and accumulation mode must be inspected; `--fp16` alone does not establish either peak.

At 60 FPS the service interval is 16.667 ms. Absolute, unattainable-in-general per-frame ceilings are 676.7 GFLOP (FP32 accumulation), 1355 GFLOP (FP16 accumulation), 2.71 trillion INT8 operations, and 7.467 GB DRAM traffic. Compute and traffic ceilings must both hold; neither accounts for decode, conversion, dispatch, copies, or output handling.

For screening, **allocate 12 ms to the network and reserve 4.667 ms for the rest**. This is a design assumption pending measurement. It is stricter than merely obtaining 60 network inferences/s. Report throughput and frame latency separately: overlapped pipeline stages can increase throughput without reducing source-to-output delay. Compressed-stream reordering and any buffering also belong in the latency contract.

## Architecture and accounting

Let `A=960×540=518,400`, width `C`, and depth `d` = number of internal `C→C` convolutions. The deployed graph is:

`RGB → 3×3(3,C) → [3×3(C,C)] × d → 3×3(C,12) → residual add + pixel shuffle ×2 → RGB`.

There are `d+2` convolutions, stride one, same padding; fused pointwise activations where supported. This matches the merged topology of repo ECBSR with `m=d`, or PlainStudent with `n=d`, not full SPAN attention. Current defaults are ECBSR `(16,4)` and PlainStudent `(32,6)`. Branches exist only during training. **F counts direct-convolution-equivalent operations**, assuming direct/implicit-GEMM execution. Winograd or other transformed tactics can reduce executed arithmetic, so direct-equivalent FLOPs divided by peak is not an architecture-level impossibility bound. Borders are counted as full convolution work; bias/activation arithmetic is omitted, but its traffic/launch consequences are modeled below.

For a layer `i→o`, using FP16 inputs, weights, and outputs:

```text
F_i = 18 A i o                         FLOP
Q_i = 2 A(i+o) + 18 i o               bytes
F_total = 18 A(d C² + 15 C)
Q_conv = 2 A[2 C(d+1)+15] + 18(d C²+15 C)
Q_tail = 2 A(12+3+12) = 54 A
Q_total = 2 A[2 C(d+1)+42] + 18(d C²+15 C)
```

`Q_i` assumes an input read, output write, and one weight read, with convolution reuse on chip and no explicit im2col buffer. `Q_tail` assumes one fused residual/shuffle kernel reading the 12-channel prediction and original RGB, then writing HR RGB. It is **not guaranteed TensorRT will fuse this**. Materialized repeat, add, shuffle, unfused PReLU, layout conversions, extra convolution reads, and workspace can increase traffic. Cache reuse or cross-layer fusion can reduce DRAM traffic below this tensor-materialization estimate; it is not a universal DRAM lower bound. Most feature maps already exceed the published L2 size.

Avoid early spatial upsampling. Moving `d` same-width convolutions to HR multiplies their compute and feature traffic by four. A controlled example with the same LR stem/tail and an intervening upsample/downsample detour changes `(32,6)` from 61.81 to 233.80 GFLOP, before detour costs. A practical early-upsample architecture has different stem/tail costs too. Pixel shuffle after the final 12-channel prediction keeps all expensive feature convolutions at LR; shuffle itself is a memory/layout operation, not free merely because it has zero MACs.

### Activation memory, not just parameter count

One FP16 feature map uses `2AC` bytes. At 540p, widths 8/16/24/32/48/64 consume **7.91/15.82/23.73/31.64/47.46/63.28 MiB** per map. Serial inference needs approximately two feature buffers plus live input/residual/output: a rough tensor-only allocation is `2×2AC + 2A(3+12+12)` bytes. For C=32 this is approximately **90 MiB**, excluding engine workspace, allocator pools, decode surfaces, and pipeline buffering. Allocator reuse may lower it. Even the `(64,12)` weights are under 1 MiB; weights are not the capacity issue here. Training memory is a different calculation.

At 720p input multiply feature/tensor traffic by `16/9`; at 1080p input multiply by four. C=32 then needs 126.56 MiB per feature map at 1080p. The 8 GB capacity is plausible for these inference graphs but does not prove they fit alongside TensorRT workspace and video buffers.

## Efficiency assumptions and predicted table

Use the **40.6 TFLOP/s dense FP32-accumulation** peak as the base scenario. NVIDIA's [convolution performance guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html), opened §§4.2–4.4, supports using arithmetic intensity, channel alignment, layout, and tile effects to reason about performance. Its measurements are not RTX 3070 predictions. Prefer FP16 channel counts divisible by eight for screening, then inspect actual engine padding/layouts.

These numerical efficiency ranges are **assumptions, unverified on this machine**, not values established by that guide:

| Scenario | Effective compute P | Effective bandwidth B | Traffic multiplier α | Dispatch allowance ℓ |
|---|---:|---:|---:|---:|
| Optimistic | 40% = 16.24 TFLOP/s | 80% = 358.4 GB/s | 1.0 | 3 µs/kernel |
| Nominal | 25% = 10.15 TFLOP/s | 65% = 291.2 GB/s | 1.25 | 5 µs/kernel |
| Conservative | 10% = 4.06 TFLOP/s | 50% = 224 GB/s | 2.0 | 15 µs/kernel |

For each scenario calculate **per layer**, not just the maximum of graph totals:

`T_net = Σ_i max(F_i/P, α Q_i/B) + α Q_tail/B + (d+3)ℓ`.

There are `d+2` fused convolution/activation kernels and one fused tail in this model. Separate activations increase that count. Compute/memory overlap within a layer is assumed; layers execute serially. These ranges are sensitivity cases, **not confidence intervals**, and can be exceeded in either direction. Depth-dependent fusion, poor small-channel utilization, or non-Tensor tactics can invalidate the constant efficiencies.

| C | d (total convs) | GFLOP | Q_total MB | Nominal compute-only ms | Nominal bandwidth-only ms | Predicted nominal network ms | Predicted optimistic–conservative ms | Nominal bottleneck |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 8 | 4 (6) | 3.51 | 126.50 | 0.35 | 0.54 | 0.58 | 0.37–1.27 | Bandwidth; launches may dominate |
| 16 | 4 (6) | 11.79 | 209.46 | 1.16 | 0.90 | 1.36 | 0.85–3.33 | Compute body, bandwidth stem/tail |
| 24 | 6 (8) | 35.61 | 391.98 | 3.51 | 1.68 | 3.73 | 2.33–9.24 | Compute body |
| 32 | 6 (8) | 61.81 | 508.15 | 6.09 | 2.18 | 6.32 | 3.96–15.71 | Compute body |
| 32 | 8 (10) | 80.92 | 640.90 | 7.97 | 2.75 | 8.22 | 5.14–20.45 | Compute body |
| 48 | 6 (8) | 135.71 | 740.54 | 13.37 | 3.18 | 13.63 | 8.53–33.95 | Compute body |
| 48 | 8 (10) | 178.71 | 939.69 | 17.61 | 4.03 | 17.88 | 11.18–44.57 | Compute body |
| 64 | 8 (10) | 314.72 | 1238.55 | 31.01 | 5.32 | 31.30 | 19.57–78.11 | Compute body |
| 64 | 12 (14) | 467.60 | 1769.68 | 46.07 | 7.60 | 46.39 | 29.00–115.83 | Compute body |

MB is decimal; activation allocations above use MiB. Compute-only is `ΣF/P`; bandwidth-only includes α and the tail; the nominal prediction includes layer-specific bottlenecks and dispatch, so it can exceed both columns. Interior arithmetic intensity is approximately `4.5C` FLOP/byte before α. Nominal effective balance is `P/B=34.86` FLOP/byte, or 43.57 against raw Q after α: width 8 is memory limited, while widths ≥16 have compute-limited interiors under these assumptions. A different tactic can change the classification.

### Which budgets are worth screening?

At a 12 ms network allocation, effective compute allows **48.72–194.88 GFLOP** across the assumed range, nominally 121.8 GFLOP; effective traffic service is 2.688–4.301 GB before dividing by α. A necessary compute condition **within the assumed direct-kernel model** is:

`d ≤ [0.012 P/(18A) − 15C]/C²`.

It excludes all other work, so rounding it down does not certify feasibility. Predicted sensible first screens are `(16,4)`, `(24,6)`, `(32,6)`, `(32,8)`, with `(48,6)` as a boundary case. Only the first two fit the allocated 12 ms throughout these assumed scenarios. Wider models are not physically impossible: peak throughput is above the assumed efficiencies, and an FP16-accumulation tactic could double P. Recompute the equation with that P rather than dividing total latency by two; bandwidth and dispatch stay. Precision quality must also pass.

Under nominal unchanged efficiency, `(32,6)` projects to approximately **11.20 ms** at 720p input and **25.15 ms** at 1080p input; `(16,4)` to **2.39** and **5.33 ms**, respectively. These are network-only area-scaled predictions with dispatch held fixed, not resolution-specific measurements. They motivate a separate budget/quality frontier for each resolution.

INT8 offers a 162.6 TOP/s dense peak and can halve eligible activation/weight traffic versus FP16. It cannot be assumed faster: quantize/dequantize boundaries, mixed-precision layers, padding, and memory traffic can erase gains. Report the actual quantized graph; do not equate FP16 autocast with INT8 QAT or assume a fixed speedup.

A causal RGB window `[t−2,t−1,t]` changes the stem from 3 to 9 input channels while keeping 3 RGB output channels. Additional arithmetic is `18A×6C` (1.79 GFLOP for C=32), with two LR history frames adding 5.93 MiB FP16 storage. It needs a new latency gate. A shallow d=6 stack has a 17×17 LR receptive field; larger motion is not guaranteed resolvable without alignment. A future input `t+1` adds one frame interval of lookahead and is excluded from this causal design.

## What exp001 must measure to validate or falsify this

The existing [benchmark](../../scripts/benchmark.py) uses TensorRT `--noDataTransfers --useCudaGraph`; its copy path is PyTorch and changes tensor layout. It cannot establish the ADR's decode-to-output-buffer claim. NVIDIA's opened [TensorRT best practices](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html) likewise distinguishes device timing, wall timing, transfers, environment stability, and profiling. Use documentation matching the installed TensorRT version; the opened latest page is not an installation prescription.

1. **Screen the model, then test deployment.** Add width/depth controls or export the listed graphs. Record merged topology, static shape, parameters, counted FLOP, layout, TensorRT/CUDA/driver versions, tactics/precisions, workspace limit, and graph capture on/off. Include matched plain/merged graphs and a measured SPAN-family comparator; do not substitute this table's FLOPs for its actual graph.
2. **Separate intervals.** Record CUDA-event GPU network timing, host enqueue/synchronization time, and synchronized wall time through actual TensorRT decode/preprocess/network/postprocess/output-buffer delivery. Log NVDEC/CPU decode path, codec reordering, H2D/D2H, GPU-resident output versus downloaded output, format conversions, and buffers. Measure pipeline overlap, queue delay, drops, cold start, and scene-cut/reset behavior separately. No future input frames.
3. **Repeat under sustained load.** Batch one; ≥50 warmup frames and ≥300 measured frames per run; five independent timed runs per configuration, then a ≥60 s playback run. Save every frame time and mean/p50/p95, GPU/host peak memory, clocks/power/temperature, plus missed-deadline/drop counts. Fix layout for comparisons; report captured and uncaptured results separately. Missing CUDA/TRT or failed builds must fail the gate.
4. **Identify the bottleneck.** Obtain per-layer times and, where available, profiler DRAM bytes, Tensor utilization and format/shuffle costs. Fit P, effective traffic service `B/α`, and ℓ on a declared subset of widths/depths, then predict held-out configurations and other resolutions. Timing alone cannot identify B and α separately; measured DRAM bytes are needed. More than 25% error on held-out median network timing is a proposed model-revision trigger, not a statistical confidence bound. Body time should grow approximately with `dC²` when compute limited, with `dC` when bandwidth limited; doubling width/depth sweeps discriminate them.
5. **Make a real decision.** Proposed conservative deployment gate: delivered throughput ≥60 frames/s with no drops, mean **and p95** decode-to-output-buffer latency ≤16.667 ms, at native 540p input under the recorded pipeline. Maintain network headroom based on the measured residual budget, not the arbitrary 72-network-FPS rule. A network pass with an end-to-end failure falsifies the budget allocation/pipeline assumption. A fast constant/blurred output is not a quality pass. Engine-output quality uses the same contract as the agenda.

CPU-only work can verify the formulas, count graph operations/buffers, check fusion/export equivalence, and expose extra shuffle/activation nodes. It cannot establish P, B, sustained clocks, TensorRT tactic choice, actual DRAM traffic, or the deployment pass.
