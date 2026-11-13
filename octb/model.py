"""FFC-UNet: residual U-Net whose deeper stages use Fast Fourier Convolution
(global receptive field, same idea as the FFC branch of Y-Net). This is a
self-contained re-implementation, NOT the official Y-Net code.

Output: (B, C, H, W) logits; softmax is taken along H (dim=2) per column.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def gn(c):
    return nn.GroupNorm(math.gcd(8, c), c)


class ConvNormAct(nn.Sequential):
    def __init__(self, cin, cout, k=3, s=1):
        super().__init__(nn.Conv2d(cin, cout, k, s, k // 2, bias=False), gn(cout), nn.ReLU(inplace=True))


class ResBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.c1 = ConvNormAct(cin, cout)
        self.c2 = nn.Sequential(nn.Conv2d(cout, cout, 3, 1, 1, bias=False), gn(cout))
        self.skip = nn.Identity() if cin == cout else nn.Sequential(nn.Conv2d(cin, cout, 1, bias=False), gn(cout))

    def forward(self, x):
        return F.relu(self.c2(self.c1(x)) + self.skip(x), inplace=True)


class FourierUnit(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Conv2d(cin * 2, cout * 2, 1, bias=False)
        self.norm = gn(cout * 2)

    def forward(self, x):
        H, W = x.shape[-2:]
        # cuFFT has no bf16 path -> always run the spectral part in fp32
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            f = torch.fft.rfft2(x, norm="ortho")
            f = torch.cat([f.real, f.imag], 1)
            f = F.relu(self.norm(self.conv(f)), inplace=True)
            re, im = f.chunk(2, 1)
            out = torch.fft.irfft2(torch.complex(re, im), s=(H, W), norm="ortho")
        return out


class SpectralTransform(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        mid = max(cout // 2, 1)
        self.c1 = ConvNormAct(cin, mid, k=1)
        self.fu = FourierUnit(mid, mid)
        self.c2 = nn.Conv2d(mid, cout, 1, bias=False)

    def forward(self, x):
        x = self.c1(x)
        return self.c2(x + self.fu(x).to(x.dtype))


class FFC(nn.Module):
    def __init__(self, ch, ratio_g=0.5):
        super().__init__()
        self.cg = int(ch * ratio_g)
        self.cl = ch - self.cg
        cl, cg = self.cl, self.cg
        self.l2l = nn.Conv2d(cl, cl, 3, 1, 1, bias=False)
        self.g2l = nn.Conv2d(cg, cl, 3, 1, 1, bias=False)
        self.l2g = nn.Conv2d(cl, cg, 3, 1, 1, bias=False)
        self.g2g = SpectralTransform(cg, cg)
        self.nl, self.ng = gn(cl), gn(cg)

    def forward(self, x):
        xl, xg = x[:, :self.cl], x[:, self.cl:]
        yl = self.l2l(xl) + self.g2l(xg)
        yg = self.l2g(xl) + self.g2g(xg).to(yl.dtype)
        return torch.cat([F.relu(self.nl(yl), inplace=True), F.relu(self.ng(yg), inplace=True)], 1)


class FFCResBlock(nn.Module):
    def __init__(self, ch, ratio_g=0.5):
        super().__init__()
        self.f1 = FFC(ch, ratio_g)
        self.f2 = FFC(ch, ratio_g)

    def forward(self, x):
        return x + self.f2(self.f1(x))


class Up(nn.Module):
    def __init__(self, cin, cskip, cout):
        super().__init__()
        self.reduce = ConvNormAct(cin, cout, k=1)
        self.block = ResBlock(cout + cskip, cout)

    def forward(self, x, skip):
        x = F.interpolate(self.reduce(x), size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([x, skip], 1))


class FFCUNet(nn.Module):
    def __init__(self, in_ch=1, n_out=11, base=32, depth=5, ffc_from=2, ratio_g=0.5):
        super().__init__()
        ch = [base * 2 ** i for i in range(depth)]
        self.mult = 2 ** (depth - 1)
        self.stem = nn.Sequential(ConvNormAct(in_ch, ch[0]), ResBlock(ch[0], ch[0]))
        self.downs = nn.ModuleList()
        self.encs = nn.ModuleList()
        for i in range(1, depth):
            self.downs.append(ConvNormAct(ch[i - 1], ch[i], k=3, s=2))
            self.encs.append(FFCResBlock(ch[i], ratio_g) if i >= ffc_from else ResBlock(ch[i], ch[i]))
        self.bottleneck = FFCResBlock(ch[-1], ratio_g)
        self.ups = nn.ModuleList([Up(ch[i + 1], ch[i], ch[i]) for i in reversed(range(depth - 1))])
        self.head = nn.Conv2d(ch[0], n_out, 1)

    def forward(self, x):
        H, W = x.shape[-2:]
        ph = (-H) % self.mult
        pw = (-W) % self.mult
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="replicate")
        skips = []
        h = self.stem(x)
        for down, enc in zip(self.downs, self.encs):
            skips.append(h)
            h = enc(down(h))
        h = self.bottleneck(h)
        for up, s in zip(self.ups, reversed(skips)):
            h = up(h, s)
        return self.head(h)[..., :H, :W]


def build_model(arch="ffc_unet", n_out=11, base=32, depth=5, ffc_from=2, **_):
    if arch == "ffc_unet":
        return FFCUNet(1, n_out, base=base, depth=depth, ffc_from=ffc_from)
    if arch == "ynet_official":
        raise NotImplementedError(
            "Official Y-Net adapter not written yet: paste get_model() and the "
            "y_net_gen_ffc class (__init__ + forward) from ynet_official/ynet/networks.py.")
    raise ValueError(f"unknown arch {arch}")


def load_checkpoint(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    a = ck["args"]
    model = build_model(a["arch"], n_out=a.get("n_out", 11), base=a["base"], depth=a["depth"],
                        ffc_from=a["ffc_from"])
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    return model, ck
