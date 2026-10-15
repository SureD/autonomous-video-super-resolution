"""CPU smoke tests: forward shapes, reparam equivalence, ONNX export, 20-iter training on synthetic data."""
import os, subprocess, sys
import pytest
import torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from vsr.models import build, ECB, ECBSR, MODELS  # noqa


@pytest.mark.parametrize("name", list(MODELS))
def test_forward_shape(name):
    m = build(name).eval()
    with torch.no_grad():
        y = m(torch.rand(1, 3, 24, 40))
    assert y.shape == (1, 3, 48, 80)


def test_ecb_block_equivalence():
    torch.manual_seed(0)
    for cin, cout in [(3, 16), (16, 16)]:
        b = ECB(cin, cout).eval()
        with torch.no_grad():  # perturb mask scales so edge branches are non-trivial
            for br in b.branches:
                if hasattr(br, "scale"): br.scale.normal_(0, 0.5); br.bias.normal_(0, 0.5)
        x = torch.randn(2, cin, 17, 23)
        with torch.no_grad():
            y0 = b(x); b.merge(); y1 = b(x)
        d = (y0 - y1).abs().max().item()
        print(f"ECB {cin}->{cout} max abs diff {d:.3e}")
        assert d < 1e-4


def test_ecbsr_net_equivalence():
    torch.manual_seed(1)
    m = ECBSR().eval(); x = torch.rand(1, 3, 32, 32)
    with torch.no_grad():
        y0 = m(x); m.merge(); y1 = m(x)
    d = (y0 - y1).abs().max().item()
    print(f"ECBSR net max abs diff {d:.3e}")
    assert d < 1e-4
    assert not any(isinstance(mod, torch.nn.ModuleList) for mod in m.modules())


@pytest.mark.parametrize("name", ["espcn", "fsrcnn", "ecbsr", "plain"])
def test_onnx_export(tmp_path, name):
    from benchmark import export_onnx
    m = build(name)
    if hasattr(m, "merge"): m.merge()
    p = export_onnx(m, str(tmp_path / f"{name}.onnx"), 54, 96)
    import onnx
    onnx.checker.check_model(onnx.load(p))
    try:
        import onnxruntime as ort
        import numpy as np
        x = torch.rand(1, 3, 54, 96)
        ref = m.eval()(x).detach().numpy()
        out = ort.InferenceSession(p, providers=["CPUExecutionProvider"]).run(None, {"lr": x.numpy()})[0]
        assert np.abs(out - ref).max() < 1e-4
    except ImportError:
        pass


def test_train_20_iters():
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "train.py"), "--config",
                        os.path.join(ROOT, "configs", "smoke_synthetic.yaml")], cwd=ROOT, capture_output=True, text=True)
    print(r.stdout[-2000:], r.stderr[-2000:])
    assert r.returncode == 0
    assert '"iter": 20' in r.stdout
    assert os.path.exists(os.path.join(ROOT, "experiments", "_smoke", "smoke-synthetic", "last.pt"))
