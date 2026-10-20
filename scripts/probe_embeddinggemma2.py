#!/usr/bin/env python3
"""CPU-only screening probe for EmbeddingGemma 2 as an SR teacher.

No training, GPU, or sentence-transformers dependency. Run --preflight first.
The default is two natural images, center 224 crops, bicubic x2/x4/x8.
This tests sensitivity to blur, not whether a teacher improves trained SR.

Known-working environment: torch 2.14.1+cpu, torchvision 0.29.1+cpu,
transformers 5.21.0.dev0 at TRANSFORMERS_COMMIT below, Pillow 12.3.0,
numpy 2.5.2. Install transformers from its pinned GitHub archive if a release
does not yet provide embedding_gemma2. All outputs are float32 CPU.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import time
from urllib.request import Request, urlopen

MODEL_ID = "google/embeddinggemma-2"
MODEL_REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
TRANSFORMERS_COMMIT = "8498ed425fa01bbee7d086efdf1a6634c2962aed"
WEIGHT_BYTES = 1_488_915_288
WEIGHT_SHA256 = "197a32965d4b1105faf060417baa899e193fb73cd401f42ec9295234d5553d79"


def write_json(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


def preflight(args):
    root = args.cache_dir
    root.mkdir(parents=True, exist_ok=True)
    result = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "disk_free_bytes": shutil.disk_usage(root).free,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "expected_weight_bytes": WEIGHT_BYTES,
        "reference_transformers_source_commit": TRANSFORMERS_COMMIT,
        "dependencies": {},
    }
    for package in ("torch", "torchvision", "transformers", "Pillow", "numpy", "lpips"):
        try:
            result["dependencies"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result["dependencies"][package] = None
    try:
        direct_url = importlib.metadata.distribution("transformers").read_text("direct_url.json")
        result["installed_transformers_direct_url"] = json.loads(direct_url) if direct_url else None
    except importlib.metadata.PackageNotFoundError:
        pass
    try:
        with urlopen(f"https://huggingface.co/api/models/{MODEL_ID}/revision/{MODEL_REVISION}", timeout=30) as r:
            meta = json.load(r)
        result["hub"] = {k: meta.get(k) for k in ("id", "sha", "gated", "private", "safetensors")}
        url = f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/model.safetensors"
        with urlopen(Request(url, method="HEAD"), timeout=30) as r:
            result["weight_http_status"] = r.status
            result["weight_content_length"] = int(r.headers.get("Content-Length", 0))
    except Exception as exc:
        result["network_error"] = f"{type(exc).__name__}: {exc}"
    result["natural_sample_paths"] = [str(p) for p in select_frames(args)]
    return result


def select_frames(args):
    if args.frame:
        return [Path(p) for p in args.frame]
    result = []
    for name in ("sunflower", "rush_hour", "tractor", "pedestrian_area"):
        folder = args.data_root / name / "HR"
        files = sorted(folder.glob("*.png"))
        if files:
            result.append(files[0])
        if len(result) == args.max_frames:
            break
    return result


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ranks(values):
    # Average ranks, including ties; scipy is not required.
    import numpy as np
    values = np.asarray(values, dtype=float)
    order = values.argsort(kind="stable")
    out = np.empty(len(values), dtype=float)
    begin = 0
    while begin < len(values):
        end = begin + 1
        while end < len(values) and values[order[end]] == values[order[begin]]:
            end += 1
        out[order[begin:end]] = (begin + end - 1) / 2
        begin = end
    return out


def spearman(a, b):
    import numpy as np
    a, b = ranks(a), ranks(b)
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def distances(reference, candidate):
    import numpy as np
    out = {}
    for name in ("global", "pooled_vision", "raw_patch"):
        a = reference[name].astype(np.float64)
        b = candidate[name].astype(np.float64)
        if a.shape != b.shape:
            raise ValueError(f"Unmatched geometry for {name}: {a.shape} vs {b.shape}")
        a /= np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)
        b /= np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-12)
        out[name + "_cosine_distance"] = float(np.maximum(0, 1 - (a * b).sum(-1)).mean())
    return out


def load_processor_dir(args):
    if args.local_model is not None:
        return args.local_model
    if not args.download:
        raise RuntimeError("Pass --local-model PATH or --download (after --preflight).")
    from huggingface_hub import snapshot_download
    # Processor-only screening deliberately excludes the 1.49GB weight file.
    return Path(snapshot_download(
        MODEL_ID, revision=MODEL_REVISION, cache_dir=args.cache_dir,
        allow_patterns=["*.json", "*.model", "*.jinja"],
    ))


def run_processor_probe(args, result):
    import numpy as np
    from PIL import Image
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(load_processor_dir(args), local_files_only=True)
    frames = []
    for path in select_frames(args):
        image = Image.open(path).convert("RGB")
        original_wh = list(image.size)
        side = min(args.crop, *image.size) if args.crop else None
        x, y = ((image.width - side) // 2, (image.height - side) // 2) if side else (0, 0)
        box = [x, y, x + side, y + side] if side else [0, 0, *image.size]
        image = image.crop(box)
        inputs = processor(images=[image], max_soft_tokens=args.max_soft_tokens, return_tensors="pt")
        valid = (inputs["image_position_ids"][0] >= 0).all(-1)
        a = inputs["pixel_values"][0, valid].numpy()
        frame = {"path": str(path), "original_wh": original_wh, "crop_xyxy": box,
                 "valid_raw_patches": int(valid.sum()), "degradations": []}
        for scale in args.scales:
            low = image.resize((max(1, image.width // scale), max(1, image.height // scale)), Image.Resampling.BICUBIC)
            restored = low.resize(image.size, Image.Resampling.BICUBIC)
            processed = processor(images=[restored], max_soft_tokens=args.max_soft_tokens, return_tensors="pt")
            b = processed["pixel_values"][0, valid].numpy()
            frame["degradations"].append({"scale": scale, "processor_valid_pixel_mse": float(np.mean((a - b) ** 2))})
        frames.append(frame)
    result.update(status="processor_only_complete", frames=frames,
                  limitation="No model weights loaded or feature sensitivity measured. Positive processed-pixel MSE only establishes that the degradation survived preprocessing.")


def run_probe(args, result):
    import numpy as np
    from PIL import Image
    import torch
    from transformers import AutoConfig, AutoModel, AutoProcessor
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    np.random.seed(0)
    paths = select_frames(args)
    if not paths:
        raise RuntimeError("No natural sample frames found; pass --frame PATH.")
    model_dir = args.local_model
    if model_dir is None:
        if not args.download:
            raise RuntimeError("Weights not loaded: pass --local-model PATH or --download (after --preflight).")
        if WEIGHT_BYTES > args.max_weight_gb * 1e9:
            raise RuntimeError("Pinned weight file exceeds --max-weight-gb budget.")
        if shutil.disk_usage(args.cache_dir).free < WEIGHT_BYTES * 2:
            raise RuntimeError("Need at least twice the weight size free for download/cache.")
        from huggingface_hub import snapshot_download
        model_dir = Path(snapshot_download(
            MODEL_ID, revision=MODEL_REVISION, cache_dir=args.cache_dir,
            allow_patterns=["*.json", "*.model", "*.jinja", "model.safetensors", "1_Pooling/*", "2_Normalize/*"],
        ))
    weight = model_dir / "model.safetensors"
    result["weight_file_bytes"] = weight.stat().st_size
    result["weight_sha256"] = sha256_file(weight)
    if result["weight_sha256"] != WEIGHT_SHA256:
        raise RuntimeError("Weights do not match the pinned model revision.")
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    config.audio_config = None  # image/text-only load: do not instantiate ~305M audio params
    started = time.monotonic()
    model = AutoModel.from_pretrained(
        model_dir, config=config, dtype=torch.float32, device_map=None,
        attn_implementation="sdpa", local_files_only=True,
    ).eval()
    model.requires_grad_(False)
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    result["load_seconds"] = time.monotonic() - started
    result["loaded_parameters"] = sum(p.numel() for p in model.parameters())
    result["vision_parameters"] = sum(p.numel() for p in model.vision_tower.parameters())
    result["device"] = "cpu"
    result["dtype"] = "float32"
    result["audio_disabled"] = model.audio_tower is None
    result["threads"] = args.threads
    result["token_budget"] = args.max_soft_tokens
    result["lpips"] = "not computed: optional learned metric requires extra weights; PSNR measured"
    result["protocol"] = {
        "crop": args.crop, "scales": args.scales,
        "degradation": "Pillow RGB bicubic antialiased downsample then bicubic upsample to identical HR size",
        "global": "masked mean of final 768-d text-sequence states including boundary tokens; L2 normalize",
        "pooled_vision": "final 3x3-spatially-pooled tower states, 768-d, valid tokens only, mean per-token cosine distance",
        "raw_patch": "final vision encoder layer before 3x3 pooling, 768-d, valid 16x16 patch tokens only",
        "dense_capture": "forward hooks on final vision encoder layer and vision tower during one complete image-only forward",
        "statistics": "within-image Spearman between distance to HR and negative RGB PSNR, degraded images only; 3 levels is a screening statistic",
    }
    captured = {}

    def raw_hook(_module, _inputs, output):
        captured["raw"] = output.detach()

    def pooled_hook(_module, _inputs, output):
        captured["pooled"] = output.last_hidden_state.detach()

    raw_handle = model.vision_tower.encoder.layers[-1].register_forward_hook(raw_hook)
    pool_handle = model.vision_tower.register_forward_hook(pooled_hook)

    @torch.inference_mode()
    def encode(image, reference_pixels=None):
        captured.clear()
        start = time.monotonic()
        inputs = processor(images=[image], max_soft_tokens=args.max_soft_tokens, return_tensors="pt")
        valid = (inputs["image_position_ids"][0] >= 0).all(-1)
        processed_pixels = inputs["pixel_values"][0, valid].numpy().copy()
        if not np.isfinite(processed_pixels).all():
            raise RuntimeError("Nonfinite processed image pixels")
        # This baseline is calculated BEFORE model inference.
        pixel_mse = (float(np.mean((processed_pixels - reference_pixels) ** 2))
                     if reference_pixels is not None else None)
        output = model(**inputs, return_dict=True)
        mask = inputs["attention_mask"].unsqueeze(-1)
        embedding = (output.last_hidden_state * mask).sum(1) / mask.sum(1)
        embedding = torch.nn.functional.normalize(embedding.float(), p=2, dim=-1)
        pos = inputs["image_position_ids"][0][valid]
        raw = captured["raw"]
        if raw.ndim != 3 or raw.shape[1] != len(valid):
            raise RuntimeError(f"Unexpected prepool raw hidden-state shape {tuple(raw.shape)}")
        pooled = captured["pooled"]
        expected = int(valid.sum()) // 9
        if tuple(pooled.shape) != (expected, 768):
            raise RuntimeError(f"Unexpected pooled vision geometry {tuple(pooled.shape)}; expected {(expected, 768)}")
        record = {
            "global": embedding[0].numpy().copy(),
            "pooled_vision": pooled.numpy().copy(),
            "raw_patch": raw[0, valid].numpy().copy(),
            "geometry": {
                "input_wh": list(image.size), "pixel_values_shape": list(inputs["pixel_values"].shape),
                "resized_wh": [int(pos[:, 0].max() + 1) * 16, int(pos[:, 1].max() + 1) * 16],
                "raw_grid_wh": [int(pos[:, 0].max() + 1), int(pos[:, 1].max() + 1)],
                "raw_valid_tokens": int(valid.sum()), "raw_padding_tokens": int((~valid).sum()),
                "pooled_valid_tokens": expected, "text_valid_tokens": int(mask.sum()),
                "raw_shape": list(raw.shape), "pooled_shape": list(pooled.shape),
            },
            "seconds": time.monotonic() - start,
            "processor_pixels": processed_pixels,
            "processor_valid_pixel_mse": pixel_mse,
        }
        for name in ("global", "pooled_vision", "raw_patch"):
            if not np.isfinite(record[name]).all():
                raise RuntimeError(f"Nonfinite {name} output")
        return record

    results, references = [], []
    try:
        for index, path in enumerate(paths):
            image = Image.open(path).convert("RGB")
            original_wh = list(image.size)
            crop_xyxy = [0, 0, *image.size]
            if args.crop:
                side = min(args.crop, *image.size)
                x, y = (image.width - side) // 2, (image.height - side) // 2
                crop_xyxy = [x, y, x + side, y + side]
                image = image.crop(crop_xyxy)
            reference = encode(image)
            references.append(reference)
            frame = {"path": str(path), "source_sha256": sha256_file(path), "original_wh": original_wh, "crop_xyxy": crop_xyxy,
                     "geometry": reference["geometry"], "reference_seconds": reference["seconds"], "degradations": []}
            a = np.asarray(image, dtype=np.float64)
            for scale in args.scales:
                low = image.resize((max(1, image.width // scale), max(1, image.height // scale)), Image.Resampling.BICUBIC)
                restored = low.resize(image.size, Image.Resampling.BICUBIC)
                psnr = float(10 * np.log10(255 ** 2 / np.mean((a - np.asarray(restored, dtype=np.float64)) ** 2)))
                candidate = encode(restored, reference["processor_pixels"])
                frame["degradations"].append({"scale": scale, "psnr_rgb_db": psnr,
                    **distances(reference, candidate), "processor_valid_pixel_mse": candidate["processor_valid_pixel_mse"],
                    "seconds": candidate["seconds"], "geometry": candidate["geometry"]})
            frame["spearman_vs_negative_psnr"] = {
                key: spearman([-v["psnr_rgb_db"] for v in frame["degradations"]], [v[key] for v in frame["degradations"]])
                for key in ("global_cosine_distance", "pooled_vision_cosine_distance", "raw_patch_cosine_distance")
            }
            # Feature permutation destroys dense correspondence without another model call.
            shuffled = dict(reference)
            rng = np.random.default_rng(0)
            for key in ("pooled_vision", "raw_patch"):
                shuffled[key] = reference[key][rng.permutation(len(reference[key]))]
            frame["spatially_shuffled_hr_features_control"] = distances(reference, shuffled)
            repeated = encode(image, reference["processor_pixels"])
            frame["unchanged_repeat_control"] = {**distances(reference, repeated),
                "processor_valid_pixel_mse": repeated["processor_valid_pixel_mse"], "seconds": repeated["seconds"]}
            quantized = {name: torch.from_numpy(reference[name]).to(torch.bfloat16).float().numpy()
                         for name in ("global", "pooled_vision", "raw_patch")}
            frame["bf16_cache_roundtrip_control"] = distances(reference, quantized)
            frame["all_outputs_finite"] = True
            results.append(frame)
            result["frames"] = results
            result["status"] = "running"
            write_json(args.output, result)
            print(f"Completed {path.parent.parent.name}: {frame['spearman_vs_negative_psnr']}", flush=True)
        # Cross-image controls are meaningful only for identical token geometries.
        if len(references) >= 2:
            if all(references[0][k].shape == references[1][k].shape
                   for k in ("global", "pooled_vision", "raw_patch")):
                result["different_image_hr_control"] = distances(references[0], references[1])
            else:
                result["different_image_hr_control"] = "skipped dense comparison: different token geometries"
        result["summary"] = {}
        for key in ("global_cosine_distance", "pooled_vision_cosine_distance", "raw_patch_cosine_distance"):
            correlations = [f["spearman_vs_negative_psnr"][key] for f in results]
            defined = [v for v in correlations if v is not None]
            result["summary"][key] = {"per_image_spearman": correlations,
                "median_spearman": float(np.median(defined)) if defined else None}
        result["status"] = "complete"
        result["limitation"] = "Tiny blur-only screening; no SR training, LPIPS, matched-compute DINO/CLIP/SigLIP comparison, or statistical reliability claim. A pass licenses a larger probe only."
    finally:
        raw_handle.remove()
        pool_handle.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true", help="stdlib-only metadata/dependency/disk check; no weights downloaded")
    parser.add_argument("--processor-only", action="store_true", help="test degradation survives processing without loading model weights")
    parser.add_argument("--download", action="store_true", help="allow pinned weights download within budget")
    parser.add_argument("--max-weight-gb", type=float, default=2.0)
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/embeddinggemma2-probe-cache"))
    parser.add_argument("--local-model", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("/workspace/data/prepared"))
    parser.add_argument("--frame", action="append")
    parser.add_argument("--max-frames", type=int, default=2)
    parser.add_argument("--crop", type=int, default=224, help="center crop size; 0 keeps full frame")
    parser.add_argument("--max-soft-tokens", type=int, choices=[70, 140, 280, 560, 1120], default=280)
    parser.add_argument("--scales", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("docs/research/embeddinggemma2-probe.json"))
    args = parser.parse_args()
    if args.max_frames < 1 or args.crop < 0 or args.threads < 1:
        parser.error("--max-frames/--threads must be positive and --crop must be nonnegative")
    if len(args.scales) < 2 or any(scale < 2 for scale in args.scales):
        parser.error("--scales requires at least two degradation scales, each >=2")
    result = {"status": "preflight", "preflight": preflight(args)}
    started = time.monotonic()
    try:
        if args.processor_only and not args.preflight:
            run_processor_probe(args, result)
        elif not args.preflight:
            run_probe(args, result)
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        result["total_seconds"] = time.monotonic() - started
        write_json(args.output, result)


if __name__ == "__main__":
    main()
