"""Train a frame-folder Charbonnier student and evaluate against bicubic."""
import argparse
import json
import math
import platform
from pathlib import Path

import _path  # noqa
import numpy as np
import PIL
import cv2
import torch
import yaml
from torch.utils.data import DataLoader
from vsr.data import FrameFolderSR, list_frames
from vsr.models import build
from train import charbonnier, seed_all
from eval import evaluate_model


def run(cfg, out):
    seed = cfg.get("seed", 0)
    seed_all(seed)
    torch.set_num_threads(cfg.get("threads", 1))
    d, t = cfg["data"], cfg["train"]
    train_files = {Path(p).resolve() for p in list_frames(d["root"])}
    val_files = {Path(p).resolve() for p in list_frames(d["val_root"])}
    if not val_files or train_files & val_files:
        raise ValueError("validation must be nonempty and disjoint from training")
    if t["iters"] < 1 or t["batch"] < 1:
        raise ValueError("iters and batch must be positive")
    scale = cfg.get("scale", 2)
    ds = FrameFolderSR(d["root"], scale=scale, patch=d["patch"], length=d.get("epoch_len"))
    # Fixed patches avoid variable-size batches and accidental HR/LR mismatch.
    from PIL import Image
    for path in ds.files:
        with Image.open(path) as im:
            if min(im.size) < d["patch"] * scale or d["patch"] < 1:
                raise ValueError("training frames must fit the positive configured patch")
    dl = DataLoader(ds, batch_size=t["batch"], shuffle=True, num_workers=0,
                    generator=torch.Generator().manual_seed(seed))
    device = cfg.get("device", "cpu")
    model = build(cfg["model"]["name"], **cfg["model"].get("kwargs", {})).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=t["lr"])
    model.train()
    iterator = iter(dl)
    for iteration in range(1, t["iters"] + 1):
        try:
            lr, hr = next(iterator)
        except StopIteration:
            iterator = iter(dl)
            lr, hr = next(iterator)
        loss = charbonnier(model(lr.to(device)), hr.to(device))
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite training loss")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    results = evaluate_model(model, d["val_root"], scale, cfg.get("border", 0), device)
    if not all(math.isfinite(v) for name in ("student", "bicubic") for v in results[name].values()):
        raise ValueError("nonfinite evaluation metrics")
    results.update(iterations=iteration, final_loss=loss.item(), seed=seed)
    results["quality_pass"] = all(results["student"][k] > results["bicubic"][k] for k in ("psnr_y", "ssim_y"))
    results["latency_pass"] = None
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config_used.yaml").write_text(yaml.safe_dump(cfg))
    env = dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
               pillow=PIL.__version__, opencv=cv2.__version__, yaml=yaml.__version__, seed=seed,
               device=device, platform=platform.platform(), cuda_available=torch.cuda.is_available())
    (out / "environment.json").write_text(json.dumps(env, indent=2))
    torch.save({"model": model.state_dict(), "cfg": cfg, "iter": iteration}, out / "last.pt")
    (out / "results.json").write_text(json.dumps(results, indent=2, allow_nan=False))
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/exp002.yaml")
    ap.add_argument("--out", default="experiments/exp002-student-baseline")
    a = ap.parse_args()
    print(json.dumps(run(yaml.safe_load(Path(a.config).read_text()), a.out), indent=2))


if __name__ == "__main__":
    main()
