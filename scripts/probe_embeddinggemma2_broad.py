#!/usr/bin/env python3
"""CPU-only, resumable broader EG2 probe. Downloads are a separate preflight step.

Run stages prepare, metrics, eg2, random, dino, summarize (or all).
Model inference is local_files_only; LPIPS loads an explicitly local AlexNet.
The random control replaces only EG2's vision tower, retaining its frozen
projection/text stack so each global/dense comparison changes the same factor.
"""
from __future__ import annotations
import argparse
import copy
import gc
import fcntl
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time

# Set before importing any model library. No CUDA execution or implicit HF downloads.
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import numpy as np
from PIL import Image, ImageFilter
import torch
from probe_embeddinggemma2 import MODEL_REVISION, WEIGHT_SHA256, sha256_file

CLIPS = ['vid4_calendar', 'vid4_city', 'vid4_foliage', 'vid4_walk',
         'sunflower', 'rush_hour', 'bbb_t120', 'sintel_trailer']
LEVELS = ['bicubic_x2', 'bicubic_x4', 'bicubic_x8', 'h264_crf23', 'h264_crf33']
SENSITIVITY = ['shift_right1', 'sharpen_hr', 'sharpen_x2']
VARIANTS = ['hr'] + LEVELS + SENSITIVITY
VIEWS = ['crop192', 'native']


def write_json(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".json.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def read_json(path):
    return json.loads(path.read_text())


def run(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


def prepare(args):
    started = time.monotonic()
    manifest = dict(utc=datetime.now(timezone.utc).isoformat(), clips=[], samples=[],
                    codec_commands=[], ffmpeg=subprocess.check_output(['ffmpeg', '-version'], text=True).splitlines()[0])
    for clip in CLIPS:
        folder = args.data_root / clip
        files = sorted((folder / 'HR').glob('*.png'))
        if len(files) < 5:
            raise RuntimeError(f'Need >=5 HR frames in {folder}')
        indices = np.linspace(0, len(files) - 1, 5).round().astype(int).tolist()
        meta = read_json(folder / 'meta.json')
        fps = meta.get('fps', 25)
        clipdir = args.cache / 'codec' / clip
        clipdir.mkdir(parents=True, exist_ok=True)
        manifest['clips'].append(dict(clip=clip, available_frames=len(files), sampled_indices_zero_based=indices,
                                      fps=fps, meta=meta))
        for crf in [23, 33]:
            video = clipdir / f'crf{crf}.mp4'
            encode = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-framerate', str(fps),
                      '-start_number', str(int(files[0].stem)), '-i', str(folder / 'HR' / '%05d.png'),
                      '-frames:v', str(len(files)), '-c:v', 'libx264', '-preset', 'medium', '-crf', str(crf),
                      '-pix_fmt', 'yuv420p', '-threads', str(args.threads), str(video)]
            manifest['codec_commands'].append(encode)
            if not video.exists():
                run(encode)
            select = '+'.join(f'eq(n\\,{i})' for i in indices)
            decode = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-threads', str(args.threads),
                      '-i', str(video), '-vf', f'select={select}', '-fps_mode', 'vfr',
                      '-threads', str(args.threads), str(clipdir / f'crf{crf}_%02d.png')]
            manifest['codec_commands'].append(decode)
            if not all((clipdir / f'crf{crf}_{j+1:02d}.png').exists() for j in range(5)):
                run(decode)
            manifest['clips'][-1][f'codec_crf{crf}'] = dict(sha256=sha256_file(video), bytes=video.stat().st_size)
        for sample_i, index in enumerate(indices):
            source = files[index]
            hr = Image.open(source).convert('RGB')
            images = {'hr': hr}
            for scale in [2, 4, 8]:
                low = hr.resize((hr.width // scale, hr.height // scale), Image.Resampling.BICUBIC)
                images[f'bicubic_x{scale}'] = low.resize(hr.size, Image.Resampling.BICUBIC)
            for crf in [23, 33]:
                images[f'h264_crf{crf}'] = Image.open(clipdir / f'crf{crf}_{sample_i+1:02d}.png').convert('RGB')
            shifted = np.asarray(hr).copy()
            shifted[:, 1:] = shifted[:, :-1].copy()  # right shift with replicated left edge, no wrapping
            images['shift_right1'] = Image.fromarray(shifted)
            for kind, base in [('sharpen_hr', 'hr'), ('sharpen_x2', 'bicubic_x2')]:
                images[kind] = images[base].filter(ImageFilter.UnsharpMask(radius=1, percent=50, threshold=0))
            x, y = (hr.width - 192) // 2, (hr.height - 192) // 2
            box = [x, y, x + 192, y + 192]
            sample = dict(id=f'{clip}/{source.stem}', clip=clip, frame=source.stem,
                          source_path=str(source), source_sha256=sha256_file(source), native_wh=list(hr.size), crop_xyxy=box,
                          images={})
            for view in VIEWS:
                outdir = args.cache / 'images' / view / clip / source.stem
                outdir.mkdir(parents=True, exist_ok=True)
                sample['images'][view] = {}
                for kind, image in images.items():
                    out = outdir / f'{kind}.png'
                    if not out.exists():
                        (image.crop(box) if view == 'crop192' else image).save(out)
                    sample['images'][view][kind] = str(out)
            manifest['samples'].append(sample)
        write_json(args.output / 'embeddinggemma2-broad-manifest.json', manifest)
        print(f'Prepared {clip}, 5 samples, two views, 9 variants', flush=True)
    manifest['prepare_seconds'] = time.monotonic() - started
    write_json(args.output / 'embeddinggemma2-broad-manifest.json', manifest)


def progress_file(args, stage):
    return args.output / f'embeddinggemma2-broad-{stage}.json'


def open_progress(args, stage):
    p = progress_file(args, stage)
    if p.exists():
        return read_json(p)
    return dict(status='running', stage=stage, records=[], cpu_threads=args.threads, device='cpu', dtype='float32',
                utc=datetime.now(timezone.utc).isoformat(), versions={p:importlib.metadata.version(p) for p in
                     ['torch', 'torchvision', 'transformers', 'numpy', 'Pillow', 'lpips', 'scipy']})


def sample_views(args):
    for sample in read_json(args.output / 'embeddinggemma2-broad-manifest.json')['samples']:
        for view in VIEWS:
            yield sample, view


def load_images(sample, view):
    return {kind: Image.open(path).convert('RGB') for kind, path in sample['images'][view].items()}


def metric_stage(args):
    import lpips
    result = open_progress(args, 'metrics')
    weights = args.alexnet
    if not weights.exists():
        raise RuntimeError('Explicit local AlexNet weights unavailable; no automatic download permitted')
    # pnet_rand avoids torchvision's download path; immediately replace its entire trunk.
    metric = lpips.LPIPS(net='alex', version='0.1', pnet_rand=True, verbose=False).eval().requires_grad_(False)
    state = torch.load(weights, map_location='cpu', weights_only=True)
    slices = [metric.net.slice1, metric.net.slice2, metric.net.slice3, metric.net.slice4, metric.net.slice5]
    for block in slices:
        for name, layer in block.named_children():
            if isinstance(layer, torch.nn.Conv2d):
                layer.load_state_dict({k:state[f'features.{name}.{k}'] for k in ['weight', 'bias']})
    result['lpips'] = dict(package='0.1.4', version='0.1', net='alex', pretrained_trunk=True,
                           alexnet_sha256=sha256_file(weights),
                           calibration_sha256=sha256_file(Path(lpips.__file__).parent / 'weights/v0.1/alex.pth'),
                           input='RGB uint8 -> float32 [-1,1], full view, no resize or border crop')
    result['psnr_y'] = 'BT.601 limited range Y = 16 + [65.481,128.553,24.966].RGB/255; peak 255; border 0'
    done = {(r['id'], r['view']) for r in result['records']}
    def tensor(image):
        return torch.from_numpy(np.asarray(image).copy()).permute(2,0,1).unsqueeze(0).float() / 127.5 - 1
    with torch.inference_mode():
        for sample, view in sample_views(args):
            if (sample['id'], view) in done:
                continue
            start = time.monotonic()
            images = load_images(sample, view)
            a = np.asarray(images['hr'], dtype=np.float64)
            ay = a @ np.array([65.481,128.553,24.966]) / 255 + 16
            target = tensor(images['hr'])
            rows = {}
            for kind in LEVELS + SENSITIVITY:
                b = np.asarray(images[kind], dtype=np.float64)
                by = b @ np.array([65.481,128.553,24.966]) / 255 + 16
                mse = np.mean((ay-by)**2)
                rows[kind] = dict(psnr_y_db=float(10*np.log10(255**2/mse)),
                                  lpips=float(metric(target, tensor(images[kind])).item()))
            result['records'].append(dict(id=sample['id'], clip=sample['clip'], view=view, values=rows,
                                          seconds=time.monotonic()-start))
            write_json(progress_file(args, 'metrics'), result)
            print(f'Metrics {len(result["records"])}/80 {sample["id"]} {view} {time.monotonic()-start:.2f}s', flush=True)
    result['status'] = 'complete'
    write_json(progress_file(args, 'metrics'), result)


def cosine(a, b):
    if a.shape != b.shape:
        raise RuntimeError(f'Unmatched geometry {a.shape} vs {b.shape}')
    # Float64 normalization, matching the original screen's numerical convention.
    a, b = a.astype(np.float64), b.astype(np.float64)
    a /= np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)
    b /= np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-12)
    return float(np.maximum(0, 1-(a*b).sum(-1)).mean())


class EG2:
    def __init__(self, args, random=False):
        from transformers import AutoConfig, AutoModel, AutoProcessor
        from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionModel
        from transformers.utils import logging
        logging.set_verbosity_error()
        if sha256_file(args.eg2 / 'model.safetensors') != WEIGHT_SHA256:
            raise RuntimeError('EG2 checkpoint SHA256 mismatch')
        cfg = AutoConfig.from_pretrained(args.eg2, local_files_only=True)
        cfg.audio_config = None
        self.model = AutoModel.from_pretrained(args.eg2, config=cfg, local_files_only=True,
                            dtype=torch.float32, attn_implementation='sdpa').eval().requires_grad_(False)
        self.processor = AutoProcessor.from_pretrained(args.eg2, local_files_only=True)
        if random:
            torch.manual_seed(1729)
            vcfg = copy.deepcopy(self.model.vision_tower.config)
            vcfg.dtype = torch.float32
            self.model.vision_tower = Gemma4VisionModel(vcfg).float().eval().requires_grad_(False)
        self.captured = {}
        self.handles = [self.model.vision_tower.encoder.layers[-1].register_forward_hook(self.raw_hook),
                        self.model.vision_tower.register_forward_hook(self.pool_hook)]
        self.metadata = dict(revision=MODEL_REVISION, sha256=WEIGHT_SHA256, random_seed=1729 if random else None,
                             random_control='Only vision tower reinitialized; same pretrained projection/text stack' if random else None,
                             parameters=sum(p.numel() for p in self.model.parameters()),
                             vision_parameters=sum(p.numel() for p in self.model.vision_tower.parameters()),
                             token_budget=280, audio_disabled=self.model.audio_tower is None)

    def close(self):
        self.captured.clear()
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def raw_hook(self, module, inputs, output):
        self.captured['raw'] = output.detach()

    def pool_hook(self, module, inputs, output):
        self.captured['pooled'] = output.last_hidden_state.detach()

    @torch.inference_mode()
    def __call__(self, image):
        inputs = self.processor(images=[image], max_soft_tokens=280, return_tensors='pt')
        valid = (inputs['image_position_ids'][0] >= 0).all(-1)
        pos = inputs['image_position_ids'][0, valid]
        grid = [int(pos[:,0].max()+1), int(pos[:,1].max()+1)]
        output = self.model(**inputs, return_dict=True)
        mask = inputs['attention_mask'].unsqueeze(-1)
        global_ = (output.last_hidden_state*mask).sum(1)/mask.sum(1)
        raw, pooled = self.captured['raw'], self.captured['pooled']
        assert raw.ndim == 3 and raw.shape[1] == len(valid)
        assert pooled.shape == (int(valid.sum())//9, 768)
        features = dict(global_=global_[0].numpy().copy(), pooled=pooled.numpy().copy(), raw=raw[0,valid].numpy().copy())
        return features, dict(input_wh=list(image.size), resized_wh=[16*x for x in grid], grid_wh=grid,
                              valid_raw_tokens=int(valid.sum()), pooled_tokens=len(pooled),
                              padded_tokens=int((~valid).sum()), sequence_tokens=int(mask.sum()))


class DINO:
    def __init__(self, args):
        from transformers import AutoModel, AutoProcessor
        from transformers.utils import logging
        logging.set_verbosity_error()
        self.model = AutoModel.from_pretrained(args.dino, local_files_only=True, dtype=torch.float32,
                         attn_implementation='sdpa').eval().requires_grad_(False)
        self.eg_processor = AutoProcessor.from_pretrained(args.eg2, local_files_only=True)
        self.metadata = dict(checkpoint='facebook/dinov2-base', sha256=sha256_file(args.dino/'model.safetensors'),
                             patch_size=14, tokens='last_hidden_state[:,1:], post-layernorm, CLS excluded',
                             preprocessing='full image bicubic/antialias resize to EG2 grid times 14; identical patch grid count, ImageNet normalization; no center crop',
                             parameters=sum(p.numel() for p in self.model.parameters()))
        self.mean = torch.tensor([.485,.456,.406])[:,None,None]
        self.std = torch.tensor([.229,.224,.225])[:,None,None]

    @torch.inference_mode()
    def __call__(self, image):
        # Match EG2 raw spatial token grid rather than DINO's default central 224 crop.
        # This preserves the complete native view and matches raw token count, not compute.
        from torchvision.transforms.functional import resize
        from torchvision.transforms import InterpolationMode
        inputs = self.eg_processor(images=[image], max_soft_tokens=280, return_tensors='pt')
        valid = (inputs['image_position_ids'][0]>=0).all(-1)
        pos = inputs['image_position_ids'][0,valid]
        gw, gh = int(pos[:,0].max()+1), int(pos[:,1].max()+1)
        x = torch.from_numpy(np.asarray(image).copy()).permute(2,0,1).float()/255
        x = resize(x, [gh*14, gw*14], interpolation=InterpolationMode.BICUBIC, antialias=True)
        x = ((x-self.mean)/self.std).unsqueeze(0)
        out = self.model(pixel_values=x, return_dict=True).last_hidden_state[:,1:]
        assert out.shape == (1,gw*gh,768)
        return {'patch':out[0].numpy().copy()}, dict(input_wh=list(image.size), resized_wh=[gw*14,gh*14],
                                                    grid_wh=[gw,gh], valid_raw_tokens=gw*gh)


def encoder_stage(args, stage):
    started = time.monotonic()
    result = open_progress(args, stage)
    if result['status'] == 'complete' and len(result['records']) == 80:
        print(f'{stage}: all 80 frame/view cases already complete', flush=True)
        return
    encoder = DINO(args) if stage == 'dino' else EG2(args, random=stage=='random')
    result['model'] = encoder.metadata
    result['load_seconds'] = time.monotonic()-started
    done = {(r['id'],r['view']) for r in result['records']}
    for sample, view in sample_views(args):
        if (sample['id'],view) in done:
            continue
        start = time.monotonic()
        images = load_images(sample, view)
        ref, geometry = encoder(images['hr'])
        if not all(np.isfinite(v).all() for v in ref.values()):
            raise RuntimeError('Nonfinite reference features')
        repeats = {}
        if sample['id'].endswith('/00001') and sample['clip'] == CLIPS[0]:
            repeated, _ = encoder(images['hr'])
            repeats = {k:cosine(ref[k],repeated[k]) for k in ref}
        rows = {}
        for kind in LEVELS+SENSITIVITY:
            candidate, cand_geometry = encoder(images[kind])
            assert cand_geometry == geometry
            if not all(np.isfinite(v).all() for v in candidate.values()):
                raise RuntimeError('Nonfinite candidate features')
            rows[kind] = {k:cosine(ref[k],candidate[k]) for k in ref}
        result['records'].append(dict(id=sample['id'],clip=sample['clip'],view=view,geometry=geometry,
                                      unchanged_repeat=repeats, values=rows, seconds=time.monotonic()-start))
        write_json(progress_file(args, stage),result)
        print(f'{stage} {len(result["records"])}/80 {sample["id"]} {view} {time.monotonic()-start:.2f}s',flush=True)
    result['status']='complete'
    result['total_stage_seconds']=time.monotonic()-started
    write_json(progress_file(args, stage),result)
    if isinstance(encoder, EG2):
        encoder.close()
    del encoder
    gc.collect()


def corr(x,y):
    from scipy.stats import rankdata, kendalltau
    a,b=rankdata(x),rankdata(y)
    if a.std()==0 or b.std()==0:
        return [None,None]
    a, b = a-a.mean(), b-b.mean()
    rho = float(np.dot(a,b) / np.sqrt(np.dot(a,a)*np.dot(b,b)))
    return [rho,float(kendalltau(x,y,variant='b').statistic)]


def estimate(point, boots):
    boots=np.asarray(boots,dtype=float)
    return dict(spearman=dict(value=point[0],ci95=np.nanpercentile(boots[:,0],[2.5,97.5]).tolist()),
                kendall=dict(value=point[1],ci95=np.nanpercentile(boots[:,1],[2.5,97.5]).tolist()))


def summarize(args):
    from scipy.stats import spearmanr
    metrics=read_json(progress_file(args,'metrics'))
    stages={s:read_json(progress_file(args,s)) for s in ['eg2','random','dino']}
    assert len(metrics['records'])==80 and all(len(s['records'])==80 for s in stages.values())
    joined={}
    for r in metrics['records']:
        for level,values in r['values'].items():
            joined[(r['id'],r['view'],level)]=dict(id=r['id'],clip=r['clip'],view=r['view'],level=level,**values)
    for stage,result in stages.items():
        for r in result['records']:
            for level,values in r['values'].items():
                joined[(r['id'],r['view'],level)].update({f'{stage}_{k}':v for k,v in values.items()})
    rows=list(joined.values())
    names=['eg2_global_','eg2_pooled','eg2_raw','dino_patch','random_global_','random_pooled','random_raw']
    rng=np.random.default_rng(20261017)
    draws=rng.integers(0,len(CLIPS),size=(args.bootstrap,len(CLIPS)))
    output=dict(utc=datetime.now(timezone.utc).isoformat(),n_source_frames=40,n_clips=8,views=VIEWS,
                n_levels=5,n_degradation_pairs_per_view=200,bootstrap_replicates=args.bootstrap,
                bootstrap_seed=20261017,bootstrap='Paired percentile cluster bootstrap: sample 8 source clips with replacement, retain all 5 frames and levels; identical draws for pretrained/control differences. CIs are descriptive, not multiplicity adjusted.',
                correlations=[],paired_control_deltas=[],sensitivity=[],monotonicity=[],aggregate_metrics=[],rows=rows)
    groups={'all_levels':LEVELS,'bicubic_only':LEVELS[:3],'codec_only':LEVELS[3:],**{k:[k] for k in LEVELS}}
    correlation_boots={}
    for view in VIEWS:
        for group,levels in groups.items():
            data=[r for r in rows if r['view']==view and r['level'] in levels]
            clipindices=[np.array([i for i,r in enumerate(data) if r['clip']==clip]) for clip in CLIPS]
            bootindices=[np.concatenate([clipindices[c] for c in draw]) for draw in draws]
            for target in ['lpips','negative_psnr_y']:
                y=np.array([r['lpips'] if target=='lpips' else -r['psnr_y_db'] for r in data])
                cached={}
                for name in names:
                    x=np.array([r[name] for r in data])
                    point=corr(x,y)
                    boots=[corr(x[i],y[i]) for i in bootindices]
                    cached[name]=(point,np.array(boots))
                    correlation_boots[(view,group,target,name)]=np.array(boots)
                    output['correlations'].append(dict(view=view,group=group,target=target,encoder=name,n=len(data),
                                                       **estimate(point,boots)))
                for name in names[:3]:
                    control=name.replace('eg2','random')
                    point=np.array(cached[name][0])-np.array(cached[control][0])
                    boots=cached[name][1]-cached[control][1]
                    output['paired_control_deltas'].append(dict(view=view,group=group,target=target,encoder=name,control=control,
                                                                **estimate(point,boots)))
            print(f'Summarized {view} {group}',flush=True)
        # Mean of within-level correlations; excludes the apparent signal from mixing severities.
        macro_cache={}
        for target in ['lpips','negative_psnr_y']:
            for name in names:
                relevant=[c for c in output['correlations'] if c['view']==view and c['target']==target and c['encoder']==name and c['group'] in LEVELS]
                points=np.mean([[r['spearman']['value'],r['kendall']['value']] for r in relevant],axis=0)
                # Same clip draws as the per-level statistics: reuse those exact
                # recomputed correlations rather than bootstrap a CI endpoint.
                boots=np.mean([correlation_boots[(view,level,target,name)] for level in LEVELS],axis=0)
                macro_cache[(target,name)]=(points,np.array(boots))
                output['correlations'].append(dict(view=view,group='within_level_macro',target=target,encoder=name,n=40,**estimate(points,boots)))
            for name in names[:3]:
                control=name.replace('eg2','random')
                point=macro_cache[(target,name)][0]-macro_cache[(target,control)][0]
                boots=macro_cache[(target,name)][1]-macro_cache[(target,control)][1]
                output['paired_control_deltas'].append(dict(view=view,group='within_level_macro',target=target,encoder=name,control=control,**estimate(point,boots)))
        for target in ['lpips','negative_psnr_y']:
            cache={}
            samples=read_json(args.output/'embeddinggemma2-broad-manifest.json')['samples']
            for name in names:
                fc=[]
                for sample in samples:
                    data=[joined[(sample['id'],view,level)] for level in LEVELS]
                    y=[r['lpips'] if target=='lpips' else -r['psnr_y_db'] for r in data]
                    fc.append(corr([r[name] for r in data],y))
                point=np.mean(fc,axis=0)
                boots=np.array([np.mean([fc[i] for c in draw for i,s in enumerate(samples) if s['clip']==CLIPS[c]],axis=0) for draw in draws])
                cache[name]=(point,boots)
                output['correlations'].append(dict(view=view,group='within_frame_across_levels',target=target,encoder=name,n=40,**estimate(point,boots)))
            for name in names[:3]:
                control=name.replace('eg2','random')
                point=cache[name][0]-cache[control][0]
                boots=cache[name][1]-cache[control][1]
                output['paired_control_deltas'].append(dict(view=view,group='within_frame_across_levels',target=target,encoder=name,control=control,**estimate(point,boots)))
        for name in names:
            for subset,levels in [('bicubic',LEVELS[:3]),('all',LEVELS)]:
                framevalues=[]
                for sample in read_json(args.output/'embeddinggemma2-broad-manifest.json')['samples']:
                    d=[joined[(sample['id'],view,level)] for level in levels]
                    framevalues.append(dict(clip=sample['clip'],rho=float(spearmanr([r[name] for r in d],[-r['psnr_y_db'] for r in d]).statistic),
                                            strict_order=bool(all(d[i][name]<d[i+1][name] for i in range(len(d)-1))) if subset=='bicubic' else None))
                med=float(np.median([f['rho'] for f in framevalues]))
                bm=[float(np.median([f['rho'] for c in draw for f in framevalues if f['clip']==CLIPS[c]])) for draw in draws]
                output['monotonicity'].append(dict(view=view,encoder=name,subset=subset,median_per_frame_spearman=med,
                            ci95=np.percentile(bm,[2.5,97.5]).tolist(),
                            strict_bicubic_fraction=float(np.mean([f['strict_order'] for f in framevalues])) if subset=='bicubic' else None))
            for kind in SENSITIVITY:
                data=[r for r in rows if r['view']==view and r['level']==kind]
                ratios=[r[name]/joined[(r['id'],view,'bicubic_x2')][name] for r in data]
                lp=[r['lpips']/joined[(r['id'],view,'bicubic_x2')]['lpips'] for r in data]
                pp=[r['psnr_y_db']-joined[(r['id'],view,'bicubic_x2')]['psnr_y_db'] for r in data]
                bm=[np.median([ratios[i] for c in draw for i,r in enumerate(data) if r['clip']==CLIPS[c]]) for draw in draws]
                # Check sign agreement with each independent fidelity metric for x2 sharpening.
                agrees={target:float(np.mean([(r[name]-joined[(r['id'],view,'bicubic_x2')][name]) *
                    ((r['lpips']-joined[(r['id'],view,'bicubic_x2')]['lpips']) if target=='lpips' else
                     (joined[(r['id'],view,'bicubic_x2')]['psnr_y_db']-r['psnr_y_db'])) > 0 for r in data])) for target in ['lpips','negative_psnr_y']}
                output['sensitivity'].append(dict(view=view,encoder=name,kind=kind,median_distance_ratio_to_x2=float(np.median(ratios)),
                    ratio_ci95=np.percentile(bm,[2.5,97.5]).tolist(),fraction_distance_above_x2=float(np.mean(np.array(ratios)>1)),
                    median_lpips_ratio_to_x2=float(np.median(lp)),median_psnr_y_delta_from_x2=float(np.median(pp)),
                    direction_agreement_vs_x2=agrees))
        for level in LEVELS+SENSITIVITY:
            data=[r for r in rows if r['view']==view and r['level']==level]
            output['aggregate_metrics'].append(dict(view=view,level=level,n=len(data),mean_psnr_y_db=float(np.mean([r['psnr_y_db'] for r in data])),
                mean_lpips=float(np.mean([r['lpips'] for r in data])),mean_distances={name:float(np.mean([r[name] for r in data])) for name in names}))
    write_json(args.output/'embeddinggemma2-broad-summary.json',output)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['all','prepare','metrics','eg2','random','dino','summarize'],default='all')
    p.add_argument('--data-root',type=Path,default=Path('/workspace/data/prepared'))
    p.add_argument('--cache',type=Path,default=Path('/tmp/eg2-broad'))
    p.add_argument('--output',type=Path,default=Path('docs/research'))
    p.add_argument('--eg2',type=Path,default=Path('/tmp/eg2-model'))
    p.add_argument('--dino',type=Path,default=Path('/tmp/dinov2-base'))
    p.add_argument('--alexnet',type=Path,default=Path('/tmp/eg2-torch/hub/checkpoints/alexnet-owt-7be5be79.pth'))
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--bootstrap',type=int,default=2000)
    args=p.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    args.output.mkdir(parents=True,exist_ok=True)
    args.cache.mkdir(parents=True,exist_ok=True)
    stages=['prepare','metrics','eg2','random','dino','summarize'] if args.stage=='all' else [args.stage]
    for stage in stages:
        if stage=='prepare': prepare(args)
        elif stage=='metrics': metric_stage(args)
        elif stage=='summarize': summarize(args)
        else:
            # Serialize same-stage writers, including independent resumable invocations.
            with (args.cache / f'{stage}.lock').open('w') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                encoder_stage(args,stage)

if __name__=='__main__':
    main()
