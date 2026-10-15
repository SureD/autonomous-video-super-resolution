"""Small real-time x2 SR models: ESPCN, FSRCNN, ECBSR-style reparam net, plain-conv student."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ESPCN(nn.Module):
    """Shi et al. 2016 (arXiv 1609.05158), RGB variant."""
    def __init__(self, scale=2, c=3):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(c, 64, 5, padding=2), nn.Tanh(),
            nn.Conv2d(64, 32, 3, padding=1), nn.Tanh(),
            nn.Conv2d(32, c * scale * scale, 3, padding=1))
        self.ps = nn.PixelShuffle(scale)

    def forward(self, x):
        return self.ps(self.body(x))


class FSRCNN(nn.Module):
    """Dong et al. 2016 (arXiv 1608.00367), d=56 s=12 m=4, RGB."""
    def __init__(self, scale=2, c=3, d=56, s=12, m=4):
        super().__init__()
        layers = [nn.Conv2d(c, d, 5, padding=2), nn.PReLU(d), nn.Conv2d(d, s, 1), nn.PReLU(s)]
        for _ in range(m):
            layers += [nn.Conv2d(s, s, 3, padding=1), nn.PReLU(s)]
        layers += [nn.Conv2d(s, d, 1), nn.PReLU(d)]
        self.body = nn.Sequential(*layers)
        self.up = nn.ConvTranspose2d(d, c, 9, stride=scale, padding=4, output_padding=scale - 1)

    def forward(self, x):
        return self.up(self.body(x))


def _pad_with_bias(y, b):
    y = F.pad(y, (1, 1, 1, 1))
    b = b.view(1, -1, 1, 1)
    y[:, :, 0:1, :] = b; y[:, :, -1:, :] = b
    y[:, :, :, 0:1] = b; y[:, :, :, -1:] = b
    return y


class SeqConv(nn.Module):
    """1x1 conv followed by 3x3 conv (learned) or fixed edge mask (sobelx/sobely/laplacian)."""
    def __init__(self, cin, cout, kind, mult=2):
        super().__init__()
        self.kind, self.cout = kind, cout
        if kind == "conv1x1-conv3x3":
            mid = int(cout * mult)
            self.k0 = nn.Conv2d(cin, mid, 1)
            self.k1 = nn.Conv2d(mid, cout, 3)
        else:
            self.k0 = nn.Conv2d(cin, cout, 1)
            self.scale = nn.Parameter(torch.randn(cout, 1, 1, 1) * 1e-3)
            self.bias = nn.Parameter(torch.randn(cout) * 1e-3)
            m = {"sobelx": [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                 "sobely": [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                 "laplacian": [[0, 1, 0], [1, -4, 1], [0, 1, 0]]}[kind]
            self.register_buffer("mask", torch.tensor(m, dtype=torch.float32).view(1, 1, 3, 3).repeat(cout, 1, 1, 1))

    def forward(self, x):
        y = _pad_with_bias(self.k0(x), self.k0.bias)
        if self.kind == "conv1x1-conv3x3":
            return self.k1(y)
        return F.conv2d(y, self.scale * self.mask, self.bias, groups=self.cout)

    def rep_params(self):
        w0, b0 = self.k0.weight[:, :, 0, 0], self.k0.bias  # [mid,in]
        if self.kind == "conv1x1-conv3x3":
            w1, b1 = self.k1.weight, self.k1.bias          # [out,mid,3,3]
        else:
            w1 = torch.diag_embed((self.scale * self.mask).squeeze(1).permute(1, 2, 0)).permute(2, 3, 0, 1)
            b1 = self.bias
        w = torch.einsum("omhw,mi->oihw", w1, w0)
        b = w1.sum((2, 3)) @ b0 + b1
        return w, b


class ECB(nn.Module):
    """ECBSR-style edge-oriented conv block (train-time multi-branch; merge() -> single 3x3)."""
    def __init__(self, cin, cout, mult=2.0, act=True, identity=True):
        super().__init__()
        self.cin, self.cout = cin, cout
        self.identity = identity and cin == cout
        self.conv3 = nn.Conv2d(cin, cout, 3, padding=1)
        self.branches = nn.ModuleList([SeqConv(cin, cout, k, mult) for k in
                                       ("conv1x1-conv3x3", "sobelx", "sobely", "laplacian")])
        self.act = nn.PReLU(cout) if act else nn.Identity()
        self.merged = None

    def forward(self, x):
        if self.merged is not None:
            return self.act(self.merged(x))
        y = self.conv3(x) + sum(b(x) for b in self.branches)
        if self.identity:
            y = y + x
        return self.act(y)

    @torch.no_grad()
    def merge(self):
        w, b = self.conv3.weight.clone(), self.conv3.bias.clone()
        for br in self.branches:
            wi, bi = br.rep_params(); w += wi; b += bi
        if self.identity:
            for i in range(self.cout):
                w[i, i, 1, 1] += 1.0
        conv = nn.Conv2d(self.cin, self.cout, 3, padding=1).to(w.device)
        conv.weight.copy_(w); conv.bias.copy_(b)
        self.merged = conv
        del self.conv3, self.branches


class ECBSR(nn.Module):
    """ECBSR-style x2 net. reparam=False builds the merged topology (plain 3x3) for H2's control arm."""
    def __init__(self, scale=2, c=3, m=4, ch=16, reparam=True):
        super().__init__()
        def blk(i, o, act=True):
            return ECB(i, o, act=act) if reparam else nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.PReLU(o) if act else nn.Identity())
        self.body = nn.Sequential(blk(c, ch), *[blk(ch, ch) for _ in range(m)], blk(ch, c * scale * scale, act=False))
        self.ps = nn.PixelShuffle(scale)
        self.scale = scale

    def forward(self, x):
        y = self.body(x) + x.repeat_interleave(self.scale * self.scale, dim=1)  # nearest residual, as ECBSR
        return self.ps(y)

    def merge(self):
        for mod in self.modules():
            if isinstance(mod, ECB) and mod.merged is None:
                mod.merge()
        return self


class PlainStudent(nn.Module):
    """Plain-conv x2 student (SPAN-like budget, no attention): conv-ReLU stack + global bilinear-free residual."""
    def __init__(self, scale=2, c=3, ch=32, n=6):
        super().__init__()
        layers = [nn.Conv2d(c, ch, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(n):
            layers += [nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(ch, c * scale * scale, 3, padding=1)]
        self.body = nn.Sequential(*layers)
        self.ps = nn.PixelShuffle(scale)
        self.scale = scale

    def forward(self, x):
        return self.ps(self.body(x) + x.repeat_interleave(self.scale * self.scale, dim=1))


MODELS = {"espcn": ESPCN, "fsrcnn": FSRCNN, "ecbsr": ECBSR,
          "ecbsr_plain": lambda **k: ECBSR(reparam=False, **k), "plain": PlainStudent}


def build(name, **kw):
    m = MODELS[name](**kw)
    return m
