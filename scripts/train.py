"""Train an x2 SR model with Charbonnier loss. Config YAML; logs + checkpoints to experiments/<id>/.

python scripts/train.py --config configs/train_ecbsr.yaml
python scripts/train.py --config configs/train_ecbsr.yaml --resume   (resumes from experiments/<id>/last.pt)
"""
import argparse, json, os, random, time
import _path  # noqa
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from vsr.data import FrameFolderSR, SyntheticSR
from vsr.models import build


def charbonnier(x, y, eps=1e-3):
    return torch.sqrt((x - y) ** 2 + eps * eps).mean()


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--resume", action="store_true")
    ap.add_argument("--iters", type=int, help="override total iterations")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if a.iters: cfg["train"]["iters"] = a.iters
    seed_all(cfg.get("seed", 0))
    out = os.path.join(cfg.get("exp_root", "experiments"), cfg["id"])
    os.makedirs(out, exist_ok=True)
    yaml.safe_dump(cfg, open(os.path.join(out, "config_used.yaml"), "w"))
    dev = torch.device("cuda" if torch.cuda.is_available() and cfg.get("device", "auto") != "cpu" else "cpu")
    d = cfg["data"]
    if d.get("synthetic"):
        ds = SyntheticSR(n=d.get("n", 64), scale=cfg["scale"], patch=d.get("patch", 32))
    else:
        ds = FrameFolderSR(d["root"], scale=cfg["scale"], patch=d.get("patch", 96), h264_prob=d.get("h264_prob", 0.0),
                           crf_range=tuple(d.get("crf_range", [18, 35])), length=d.get("epoch_len"))
    g = torch.Generator(); g.manual_seed(cfg.get("seed", 0))
    dl = DataLoader(ds, batch_size=cfg["train"]["batch"], shuffle=True, num_workers=d.get("workers", 0),
                    drop_last=True, generator=g, persistent_workers=d.get("workers", 0) > 0)
    model = build(cfg["model"]["name"], **cfg["model"].get("kwargs", {})).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["train"]["iters"], eta_min=cfg["train"].get("lr_min", 1e-6))
    it = 0
    ck = os.path.join(out, "last.pt")
    if a.resume and os.path.exists(ck):
        s = torch.load(ck, map_location="cpu")
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); sched.load_state_dict(s["sched"]); it = s["iter"]
        torch.set_rng_state(s["rng"]); print(f"resumed at iter {it}")
    log = open(os.path.join(out, "train_log.jsonl"), "a")
    model.train(); t0 = time.time()
    total = cfg["train"]["iters"]
    while it < total:
        for lr, hr in dl:
            lr, hr = lr.to(dev), hr.to(dev)
            loss = charbonnier(model(lr), hr)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); it += 1
            if it % cfg["train"].get("log_every", 10) == 0 or it == total:
                rec = {"iter": it, "loss": loss.item(), "lr": sched.get_last_lr()[0], "elapsed_s": round(time.time() - t0, 2)}
                print(json.dumps(rec)); log.write(json.dumps(rec) + "\n"); log.flush()
            if it % cfg["train"].get("ckpt_every", 1000) == 0 or it == total:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                            "iter": it, "rng": torch.get_rng_state(), "cfg": cfg}, ck)
            if it >= total:
                break
    print("done; checkpoint:", ck)


if __name__ == "__main__":
    main()
