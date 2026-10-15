"""Evaluate SR frames vs GT: PSNR/SSIM on Y, optional LPIPS, flow-warp temporal error (Farneback).

Either compare existing folders (--sr DIR --gt DIR), or run a model (--model NAME --ckpt PATH --lr DIR --gt DIR).
Frames are matched by sorted filename order.
"""
import argparse, json, os
import _path  # noqa
import cv2
import numpy as np
import torch
from vsr.data import list_frames, to_tensor, bicubic_down
from vsr.metrics import psnr_y, ssim_y, warp_error
from vsr.models import build


def load(p):
    return cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--sr"); ap.add_argument("--lr"); ap.add_argument("--model"); ap.add_argument("--ckpt")
    ap.add_argument("--scale", type=int, default=2); ap.add_argument("--border", type=int, default=2)
    ap.add_argument("--lpips", action="store_true"); ap.add_argument("--out")
    a = ap.parse_args()
    gts = [load(p) for p in list_frames(a.gt)]
    if a.sr:
        srs = [load(p) for p in list_frames(a.sr)]
    else:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        m = build(a.model)
        if a.ckpt:
            m.load_state_dict(torch.load(a.ckpt, map_location="cpu")["model"])
        if hasattr(m, "merge"): m.merge()
        m.to(dev).eval()
        lrs = [load(p) for p in list_frames(a.lr)] if a.lr else [bicubic_down(g, a.scale) for g in gts]
        srs = []
        with torch.no_grad():
            for l in lrs:
                y = m(to_tensor(l)[None].to(dev))[0].clamp(0, 1)
                srs.append((y.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8))
    assert len(srs) == len(gts), f"{len(srs)} SR vs {len(gts)} GT frames"
    gts = [g[: s.shape[0], : s.shape[1]] for g, s in zip(gts, srs)]
    res = {"n": len(gts),
           "psnr_y": float(np.mean([psnr_y(s, g, a.border) for s, g in zip(srs, gts)])),
           "ssim_y": float(np.mean([ssim_y(s, g, a.border) for s, g in zip(srs, gts)]))}
    if len(srs) > 1:
        # flow from GT frames; error of warped SR vs SR, and same for GT (reference floor)
        res["warp_err_sr"] = float(np.mean([warp_error(srs[i - 1], srs[i], gts[i - 1], gts[i]) for i in range(1, len(srs))]))
        res["warp_err_gt"] = float(np.mean([warp_error(gts[i - 1], gts[i]) for i in range(1, len(gts))]))
    if a.lpips:
        try:
            import lpips
            fn = lpips.LPIPS(net="alex", verbose=False)
            t = lambda im: to_tensor(im)[None] * 2 - 1
            res["lpips_alex"] = float(np.mean([fn(t(s), t(g)).item() for s, g in zip(srs, gts)]))
            if len(srs) > 1:  # tLPIPS as in TecoGAN: |LPIPS(sr_t,sr_t-1) - LPIPS(gt_t,gt_t-1)|
                res["tlpips"] = float(np.mean([abs(fn(t(srs[i]), t(srs[i-1])).item() - fn(t(gts[i]), t(gts[i-1])).item())
                                               for i in range(1, len(srs))]))
        except ImportError:
            res["lpips_alex"] = "skipped: pip install lpips"
    print(json.dumps(res, indent=2))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
