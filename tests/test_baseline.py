"""Regression coverage for the canonical degradation and baseline runner."""
import sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from vsr.resize import bicubic_down


def test_resize_matches_pillow():
    hr = np.random.default_rng(0).integers(0, 256, (35, 39, 3), dtype=np.uint8)
    expected = np.array(Image.fromarray(hr[:34, :38]).resize((19, 17), Image.Resampling.BICUBIC))
    np.testing.assert_array_equal(bicubic_down(hr, 2), expected)


def test_train_prepare_eval_lr_identical(tmp_path, monkeypatch):
    import torch
    import eval as evaluation
    from vsr.data import FrameFolderSR
    from prepare_data import _down_one
    hr = np.random.default_rng(1).integers(0, 256, (36, 40, 3), dtype=np.uint8)
    folder = tmp_path / 'hr'; folder.mkdir()
    source = folder / '00001.png'; Image.fromarray(hr).save(source)
    lr, _ = FrameFolderSR(str(folder), patch=0, augment=False)[0]
    _down_one((str(source), str(tmp_path / 'lr.png'), 2, 'pil'))
    expected = np.array(Image.open(tmp_path / 'lr.png'))
    np.testing.assert_array_equal((lr.numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8), expected)
    seen = []
    original = evaluation.bicubic_down
    def capture(img, scale):
        result = original(img, scale); seen.append(result); return result
    monkeypatch.setattr(evaluation, 'bicubic_down', capture)
    evaluation.evaluate_model(torch.nn.Upsample(scale_factor=2, mode='nearest'), str(folder))
    np.testing.assert_array_equal(seen[0], expected)


def test_exp002_tiny(tmp_path):
    from run_exp002_baseline import run
    import json
    for split in ('train', 'val'):
        folder = tmp_path / split; folder.mkdir()
        hr = np.random.default_rng(2 if split == 'train' else 3).integers(0, 256, (32, 32, 3), dtype=np.uint8)
        Image.fromarray(hr).save(folder / '00001.png')
    cfg = dict(seed=0, scale=2, device='cpu', model=dict(name='ecbsr', kwargs=dict(m=1, ch=4)),
               data=dict(root=str(tmp_path / 'train'), val_root=str(tmp_path / 'val'), patch=16),
               train=dict(iters=2, batch=1, lr=0.0005))
    out = tmp_path / 'result'
    result = run(cfg, out)
    assert result['iterations'] == 2 and result['n'] == 1
    assert set(result['bicubic']) == {'psnr_y', 'ssim_y'}
    assert result['latency_pass'] is None
    assert json.loads((out / 'environment.json').read_text())['seed'] == 0
    assert all((out / name).exists() for name in ('results.json', 'config_used.yaml', 'environment.json', 'last.pt'))


def test_eval_cli_includes_floor(tmp_path):
    import json
    import subprocess
    folder = tmp_path / 'frames'; folder.mkdir()
    hr = np.random.default_rng(4).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    Image.fromarray(hr).save(folder / '00001.png')
    out = tmp_path / 'eval.json'
    proc = subprocess.run([sys.executable, str(ROOT / 'scripts/eval.py'),
                           '--gt', str(folder), '--sr', str(folder), '--out', str(out)],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    result = json.loads(out.read_text())
    assert np.isfinite(result['bicubic']['psnr_y'])
    assert np.isfinite(result['bicubic']['ssim_y'])


def test_eval_rejects_invalid_shapes():
    import pytest
    from eval import quality
    with pytest.raises(ValueError):
        quality([np.zeros((10, 10, 3), np.uint8)], [np.zeros((10, 10, 3), np.uint8)])
