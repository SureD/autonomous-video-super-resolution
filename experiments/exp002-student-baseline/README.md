# exp002: Charbonnier student baseline

Purpose: train the provisional small ECBSR student on HR frame folders and
measure held-out PSNR-Y/SSIM-Y against bicubic using ADR 0002 and the ADR 0003 metric contract. The candidate
matches train_ecbsr.yaml; exp001 is preliminary screening; deployment latency eligibility remains unverified.

From the repository root, with existing dependencies and disjoint staged folders:

```sh
python3 scripts/run_exp002_baseline.py --config configs/exp002.yaml
```

Paths in YAML are relative to the working directory. Configure `data.root` and
`data.val_root` with separate HR-only frame trees, not prepared HR+LR parent
directories. Split by source clip; path overlap checks cannot detect duplicated
content or related clips. `train.iters` is the exact optimizer-step count. CPU is
the default; change device to cuda on the training machine. No downloads occur.
The output directory receives results.json, config_used.yaml, environment.json,
and last.pt; reruns overwrite them. Environment capture does not invoke git.

Execution passes if all iterations finish, loss/metrics are finite, and artifacts
are written. Quality passes only if both held-out mean metrics strictly exceed
the bicubic floor (`quality_pass`); short synthetic runs need not pass quality.
Real-time success requires separately measured RTX 3070 TensorRT FP16 end-to-end
60 FPS at 540p→1080p, batch 1; latency_pass remains null here. This runner uses
constant-rate Adam and no distillation, compression, or temporal loss.

No experiment results are claimed. Real datasets and GPU validation remain
pending. CPU smoke tests use temporary synthetic frame folders, not benchmarks.

`train.py` checkpoints now retain Python, NumPy, CPU/CUDA Torch RNG and epoch/batch
position. Augmentations use seed/epoch/index keys so worker prefetch does not change
samples. `epoch_len` is a deprecated ignored argument; the dataset population is
always complete, with iteration budgeting in the training loop. Resume requires
the same training configuration, including total cosine schedule length; only
output location and logging/checkpoint frequency may change. Use `--stop-after N`
to interrupt without changing that schedule. Older checkpoints without full RNG
and data position cannot provide exact resume and are rejected. CPU synthetic
resume equivalence is covered; CUDA and multiple-worker equivalence remain unverified.
