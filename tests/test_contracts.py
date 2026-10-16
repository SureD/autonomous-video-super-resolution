import sys
from pathlib import Path
import numpy as np
import pytest
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / p) for p in ('src', 'scripts')]
from vsr.metrics import rgb_to_y, psnr_y, ssim_y


def test_metric_numerical_contract():
    colors = np.array([[[0,0,0], [255,255,255], [255,0,0], [0,255,0], [0,0,255]]])
    np.testing.assert_allclose(rgb_to_y(colors), [[0,255,54.213,182.376,18.411]], atol=1e-12)
    a = np.zeros((16,16,3), np.uint8); b = np.full_like(a, 10)
    assert psnr_y(a,b) == pytest.approx(28.130803608679106)
    assert ssim_y(a,b) == pytest.approx(6.5025 / 106.5025)
    assert ssim_y(a,a) == pytest.approx(1)
    with pytest.raises(ValueError, match='no border crop'): psnr_y(a,b,border=2)
    with pytest.raises(ValueError, match='no border crop'): ssim_y(a,b,border=2)
    b[:] = 0; b[0] = 10
    assert psnr_y(a,b) == pytest.approx(40.17200343523835)


def write_frame(root, name, value=0, shape=(16,16,3)):
    import cv2
    path = root / name; path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), np.full(shape, value, np.uint8))


def test_identity_pairing(tmp_path):
    from eval import frame_index, paired_index
    gt, sr = tmp_path/'gt', tmp_path/'sr'
    write_frame(gt, 'a/001.png'); write_frame(sr, 'a/002.png')
    with pytest.raises(ValueError, match='identities differ'): paired_index(sr, frame_index(gt))
    write_frame(gt, 'a/001.bmp')
    with pytest.raises(ValueError, match='duplicate'): frame_index(gt)


def test_clip_means_and_temporal_boundaries(monkeypatch):
    import eval as ev
    calls = []
    monkeypatch.setattr(ev, 'warp_error', lambda *args: calls.append(args) or 1.)
    gts = [np.zeros((16,16,3),np.uint8) for _ in range(3)]
    srs = [np.full_like(gts[0], v) for v in (10,10,20)]
    res = ev.report([('a','1'),('a','2'),('b','1')], srs, gts, srs)
    assert len(calls) == 2  # one within-clip pair, SR and GT
    assert res['psnr_y'] == pytest.approx((28.130803608679106+22.11020369539948)/2)
    assert res['clips']['a']['n'] == 2 and len(res['frames']) == 3
    with pytest.raises(ValueError, match='matching RGB'):
        ev.quality([srs[0][:-1]], [gts[0]])


def test_checkpoint_architecture_and_random_guard(tmp_path):
    import torch
    from vsr.models import build
    from eval import checkpoint_model, generated_lr
    cfg = dict(scale=2, model=dict(name='ecbsr', kwargs=dict(m=2,ch=8)))
    model = build('ecbsr', m=2,ch=8)
    path = tmp_path/'ck.pt'; torch.save(dict(cfg=cfg, model=model.state_dict()),path)
    loaded, scale = checkpoint_model(path)
    assert scale == 2
    for k,v in model.state_dict().items(): assert torch.equal(v,loaded.state_dict()[k])
    with pytest.raises(ValueError, match='conflicts'): checkpoint_model(path, scale=3)
    with pytest.raises(ValueError, match='requires'): checkpoint_model(name='ecbsr')
    assert checkpoint_model(name='espcn', allow_random=True)[1] == 2
    with pytest.raises(ValueError, match='modcrop'): generated_lr(np.zeros((17,16,3),np.uint8),2)


def test_epoch_len_cannot_exclude_files(tmp_path):
    from vsr.data import FrameFolderSR
    for i in range(5): write_frame(tmp_path, f'{i}.png')
    ds = FrameFolderSR(str(tmp_path), length=2)
    assert len(ds) == 5
    from train import epoch_loader
    cfg = dict(seed=0, data={}, train=dict(batch=1))
    loader, n = epoch_loader(ds,cfg,0,0)
    assert n == 5
    assert sorted(i for batch in loader.batch_sampler for i in batch) == list(range(5))


def test_timing_summary_keeps_wall_and_event_samples():
    from benchmark import summarize_samples
    samples = [dict(wall_ms=v,cuda_event_ms=v/2) for v in (4,1,3,2)]
    result = summarize_samples(samples)
    assert result['wall']['mean_ms'] == 2.5
    assert result['wall']['p95_ms'] == 4
    assert result['cuda_event']['mean_ms'] == 1.25
    assert result['samples'] == samples
    assert result['layout'] == 'contiguous NCHW'


def test_all_rng_roundtrip():
    import random
    import torch
    from train import rng_state, restore_rng, seed_all
    seed_all(12); state = rng_state()
    expected = (random.random(), np.random.rand(), torch.rand(4))
    restore_rng(state)
    assert random.random() == expected[0]
    assert np.random.rand() == expected[1]
    assert torch.equal(torch.rand(4),expected[2])


def test_cpu_resume_equivalence(tmp_path):
    import subprocess
    import torch
    import yaml
    cfg = dict(id='full', exp_root=str(tmp_path), seed=7, device='cpu', scale=2,
               model=dict(name='espcn',kwargs={}), data=dict(synthetic=True,n=6,patch=8,workers=0),
               train=dict(iters=5,batch=2,lr=.001,ckpt_every=1,log_every=1))
    script = str(Path(__file__).resolve().parents[1]/'scripts/train.py')
    config = tmp_path/'config.yaml'
    def run(*args):
        config.write_text(yaml.safe_dump(cfg))
        import os
        env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        r = subprocess.run([sys.executable,script,'--config',str(config),*args],capture_output=True,text=True,env=env)
        assert r.returncode == 0, r.stderr
    run(); cfg['id'] = 'resumed'
    run('--stop-after','2'); run('--resume')
    full = torch.load(tmp_path/'full/last.pt', weights_only=False)
    resumed = torch.load(tmp_path/'resumed/last.pt', weights_only=False)
    def equal(a,b):
        if isinstance(a, torch.Tensor): assert torch.equal(a,b)
        elif isinstance(a, np.ndarray): np.testing.assert_array_equal(a,b)
        elif isinstance(a,dict):
            assert a.keys() == b.keys()
            for k in a: equal(a[k],b[k])
        elif isinstance(a,(tuple,list)):
            assert len(a) == len(b)
            for x,y in zip(a,b): equal(x,y)
        else: assert a == b
    for key in ('model','opt','sched','rng','data_position','iter'): equal(full[key], resumed[key])
    cfg['train']['lr'] = .002
    from train import compatible
    with pytest.raises(ValueError, match='configuration differs'): compatible(full['cfg'],cfg)
