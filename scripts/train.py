"""Train an x2 SR model with Charbonnier loss. Config YAML; logs + checkpoints to experiments/<id>/.

python scripts/train.py --config configs/train_ecbsr.yaml
python scripts/train.py --config configs/train_ecbsr.yaml --resume   (resumes from experiments/<id>/last.pt)
"""
import argparse, json, os, random, time
import _path  # noqa
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from vsr.data import FrameFolderSR, SyntheticSR
from vsr.models import build


def charbonnier(x, y, eps=1e-3):
    return torch.sqrt((x - y) ** 2 + eps * eps).mean()


def seed_all(s):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        if not torch.cuda.is_available():
            raise ValueError('CUDA RNG checkpoint requires CUDA for exact resume')
        torch.cuda.set_rng_state_all(state['cuda'])


class KeyedSamples(Dataset):
    """Augmentation RNG depends on seed, epoch, index, independent of worker prefetch."""
    def __init__(self, dataset, seed, epoch):
        self.dataset, self.seed, self.epoch = dataset, seed, epoch

    def __len__(self): return len(self.dataset)

    def __getitem__(self, index):
        state = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        key = (self.seed + self.epoch * len(self.dataset) + index) % (2**32)
        random.seed(key); np.random.seed(key)
        # CPU-only seed avoids touching CUDA from worker processes.
        torch.set_rng_state(torch.Generator().manual_seed(key).get_state())
        try: return self.dataset[index]
        finally:
            random.setstate(state[0]); np.random.set_state(state[1]); torch.set_rng_state(state[2])


def epoch_loader(ds, cfg, epoch, offset):
    seed = cfg.get('seed', 0)
    g = torch.Generator().manual_seed(seed + epoch)
    order = torch.randperm(len(ds), generator=g).tolist()
    batch = cfg['train']['batch']
    batches = [order[i:i+batch] for i in range(0, len(order)-batch+1, batch)]
    if not batches: raise ValueError('dataset must contain at least one full batch')
    if not 0 <= offset <= len(batches): raise ValueError('invalid resume batch offset')
    return DataLoader(KeyedSamples(ds, seed, epoch), batch_sampler=batches[offset:],
                      num_workers=cfg['data'].get('workers', 0), generator=g), len(batches)


def compatible(old, new):
    import copy
    def normalized(cfg):
        cfg = copy.deepcopy(cfg)
        for key in ('id', 'exp_root'): cfg.pop(key, None)
        for key in ('log_every', 'ckpt_every'): cfg['train'].pop(key, None)
        return cfg
    if normalized(old) != normalized(new):
        raise ValueError('resume configuration differs (including total schedule length)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--resume", action="store_true")
    ap.add_argument("--stop-after", type=int, help="stop after this iteration without changing the LR schedule")
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
    model = build(cfg["model"]["name"], **cfg["model"].get("kwargs", {})).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["train"]["iters"], eta_min=cfg["train"].get("lr_min", 1e-6))
    it = 0
    epoch, offset = 0, 0
    ck = os.path.join(out, "last.pt")
    if a.resume:
        s = torch.load(ck, map_location="cpu", weights_only=False)
        compatible(s["cfg"], cfg)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); sched.load_state_dict(s["sched"]); it = s["iter"]
        restore_rng(s["rng"])
        epoch, offset = s["data_position"]["epoch"], s["data_position"]["offset"]
        print(f"resumed at iter {it}")
    log = open(os.path.join(out, "train_log.jsonl"), "a")
    model.train(); t0 = time.time()
    total = cfg["train"]["iters"]
    stop = min(total, a.stop_after) if a.stop_after is not None else total
    if stop < 1: raise ValueError("positive stop iteration required")
    while it < stop:
        dl, n_batches = epoch_loader(ds, cfg, epoch, offset)
        for lr, hr in dl:
            lr, hr = lr.to(dev), hr.to(dev)
            loss = charbonnier(model(lr), hr)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); it += 1
            offset += 1
            if offset == n_batches:
                epoch += 1; offset = 0
            if it % cfg["train"].get("log_every", 10) == 0 or it == stop:
                rec = {"iter": it, "loss": loss.item(), "lr": sched.get_last_lr()[0], "elapsed_s": round(time.time() - t0, 2)}
                print(json.dumps(rec)); log.write(json.dumps(rec) + "\n"); log.flush()
            if it % cfg["train"].get("ckpt_every", 1000) == 0 or it == stop:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                            "iter": it, "rng": rng_state(), "cfg": cfg,
                            "data_position": dict(epoch=epoch, offset=offset)}, ck)
            if it >= stop:
                break
    print("done; checkpoint:", ck)


if __name__ == "__main__":
    main()
