"""Evaluate explicit clip/frame identities under ADR 0003."""
import argparse
import json
from pathlib import Path
import _path  # noqa
import cv2
import numpy as np
import torch
from vsr.data import list_frames, to_tensor, bicubic_down
from vsr.metrics import psnr_y, ssim_y, warp_error
from vsr.models import build
from vsr.resize import bicubic_up, KERNEL


def load(p):
    image = cv2.imread(str(p))
    if image is None:
        raise ValueError(f"unreadable image: {p}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def frame_index(root):
    result = {}
    for path in list_frames(str(root)):
        rel = Path(path).relative_to(root)
        key = (str(rel.parent), rel.stem)
        if key in result:
            raise ValueError(f"duplicate frame identity: {key}")
        result[key] = path
    if not result:
        raise ValueError(f"empty frame folder: {root}")
    return result


def paired_index(root, gt):
    other = frame_index(root)
    if other.keys() != gt.keys():
        raise ValueError(f"frame identities differ: missing={sorted(gt.keys()-other.keys())}, unexpected={sorted(other.keys()-gt.keys())}")
    return other


def quality(srs, gts, border=0):
    if border != 0:
        raise ValueError("ADR 0003 requires no border crop")
    if not gts or len(srs) != len(gts):
        raise ValueError("nonempty, equal frame counts required")
    for sr, gt in zip(srs, gts):
        if sr.shape != gt.shape or gt.ndim != 3 or gt.shape[2] != 3 or min(gt.shape[:2]) < 11:
            raise ValueError("matching RGB shapes and at least 11 pixels required")
    return {"psnr_y": float(np.mean([psnr_y(s, g) for s, g in zip(srs, gts)])),
            "ssim_y": float(np.mean([ssim_y(s, g) for s, g in zip(srs, gts)]))}


def mean(rows):
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def report(keys, srs, gts, floors, border=0, perceptual=None):
    frames, clips = [], {}
    for i, (clip, frame) in enumerate(keys):
        row = dict(clip_id=clip, frame_id=frame, **quality([srs[i]], [gts[i]], border),
                   bicubic=quality([floors[i]], [gts[i]], border))
        if perceptual:
            row['lpips_alex'] = perceptual(srs[i], gts[i])
        frames.append(row)
        clips.setdefault(clip, []).append(i)
    clip_rows = {}
    for clip, indices in clips.items():
        values = [frames[i] for i in indices]
        row = dict(n=len(indices), **mean([{k: v[k] for k in ('psnr_y','ssim_y')} for v in values]),
                   bicubic=mean([v['bicubic'] for v in values]))
        if perceptual:
            row['lpips_alex'] = float(np.mean([v['lpips_alex'] for v in values]))
        pairs = list(zip(indices, indices[1:]))
        if pairs:
            row['warp_err_sr'] = float(np.mean([warp_error(srs[a], srs[b], gts[a], gts[b]) for a,b in pairs]))
            row['warp_err_gt'] = float(np.mean([warp_error(gts[a], gts[b]) for a,b in pairs]))
            if perceptual:
                row['tlpips'] = float(np.mean([abs(perceptual(srs[a], srs[b])-perceptual(gts[a], gts[b])) for a,b in pairs]))
        clip_rows[clip] = row
    rows = list(clip_rows.values())
    metrics = ('psnr_y','ssim_y','warp_err_sr','warp_err_gt','lpips_alex','tlpips')
    agg, std = {}, {}
    for k in metrics:
        vals = [r[k] for r in rows if k in r]
        if vals:
            agg[k] = float(np.mean(vals))
            # Identical infinite PSNRs have zero dispersion; mixed finite/inf is undefined.
            std[k] = 0.0 if all(v == vals[0] for v in vals) else float(np.std(vals))
    return dict(n=len(keys), **agg, clip_std=std, clips=clip_rows, frames=frames,
                bicubic=mean([r['bicubic'] for r in rows]),
                protocol=dict(y='BT.709 full range; peak 255', border=0, aggregation='equal clip mean; population std',
                              ssim='11x11 sigma 1.5 valid centers'))


def infer(model, lr, device):
    with torch.no_grad():
        y = model(to_tensor(lr)[None].to(device))[0].clamp(0,1)
    return (y.permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)


def generated_lr(gt, scale, modcrop=False):
    if gt.shape[0] % scale or gt.shape[1] % scale:
        if not modcrop:
            raise ValueError('GT dimensions require explicit --modcrop')
        gt = gt[:gt.shape[0]//scale*scale, :gt.shape[1]//scale*scale]
    return gt, bicubic_down(gt, scale)


def evaluate_model(model, gt_root, scale=2, border=0, device='cpu'):
    model.eval()
    index = frame_index(gt_root); keys = sorted(index)
    gts, srs, floors = [], [], []
    for key in keys:
        gt, lr = generated_lr(load(index[key]), scale)
        gts.append(gt); srs.append(infer(model, lr, device)); floors.append(bicubic_up(lr, scale))
    result = report(keys, srs, gts, floors, border)
    result['student'] = {k: result[k] for k in ('psnr_y','ssim_y')}
    result['protocol'].update(downsample=KERNEL, scale=scale, modcrop=False)
    return result


def checkpoint_model(path=None, name=None, scale=None, allow_random=False):
    if path:
        ck = torch.load(path, map_location='cpu', weights_only=False)
        cfg = ck['cfg']; mc = cfg['model']; actual_scale = cfg.get('scale', 2)
        if (name and name != mc['name']) or (scale is not None and scale != actual_scale):
            raise ValueError('model/scale override conflicts with checkpoint cfg')
        kwargs = dict(mc.get('kwargs', {}))
        if kwargs.get('scale', actual_scale) != actual_scale:
            raise ValueError('checkpoint scale metadata conflicts')
        kwargs['scale'] = actual_scale
        model = build(mc['name'], **kwargs); model.load_state_dict(ck['model'])
    else:
        if not allow_random or not name:
            raise ValueError('model evaluation requires --ckpt or --allow-random-weights with --model')
        actual_scale = scale or 2
        model = build(name, scale=actual_scale)
    return model, actual_scale


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gt', required=True)
    for arg in ('sr','lr','model','ckpt','out'): ap.add_argument('--'+arg)
    ap.add_argument('--scale', type=int); ap.add_argument('--border', type=int, default=0)
    ap.add_argument('--modcrop', action='store_true'); ap.add_argument('--allow-random-weights', action='store_true')
    ap.add_argument('--lpips', action='store_true')
    a = ap.parse_args()
    if a.sr and (a.ckpt or a.model or a.allow_random_weights or a.modcrop):
        ap.error('--sr cannot combine with model options or modcrop')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = None
    scale = a.scale or 2
    if not a.sr:
        model, scale = checkpoint_model(a.ckpt, a.model, a.scale, a.allow_random_weights)
        model.to(device).eval()
    if scale < 1: raise ValueError('positive scale required')
    gt_index = frame_index(a.gt); keys = sorted(gt_index)
    sr_index = paired_index(a.sr, gt_index) if a.sr else None
    lr_index = paired_index(a.lr, gt_index) if a.lr else None
    gts, srs, floors = [], [], []
    for key in keys:
        gt = load(gt_index[key])
        if lr_index:
            lr = load(lr_index[key])
            if gt.shape[:2] != (lr.shape[0]*scale, lr.shape[1]*scale):
                raise ValueError(f'LR/GT shape mismatch: {key}')
        else:
            gt, lr = generated_lr(gt, scale, a.modcrop)
        sr = load(sr_index[key]) if sr_index else infer(model, lr, device)
        gts.append(gt); srs.append(sr); floors.append(bicubic_up(lr, scale))
    perceptual = None
    if a.lpips:
        import lpips
        fn = lpips.LPIPS(net='alex', verbose=False).eval()
        def perceptual(s, g):
            with torch.no_grad():
                return fn(to_tensor(s)[None]*2-1, to_tensor(g)[None]*2-1).item()
    res = report(keys, srs, gts, floors, a.border, perceptual)
    res['protocol'].update(scale=scale, modcrop=a.modcrop, downsample='provided LR' if a.lr else KERNEL,
                           random_weights=bool(a.allow_random_weights and not a.ckpt))
    print(json.dumps(res, indent=2))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=2))


if __name__ == '__main__': main()
