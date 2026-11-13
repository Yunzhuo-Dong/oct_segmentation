"""Column-wise boundary losses.

E0: vertical soft-target cross-entropy on annotated columns only.
E1: + anatomical order penalty + weak continuity, both applied only where the
    network is confident (local mass gate), so a boundary that genuinely
    disappears (flat / bimodal distribution) is NOT forced into existence.
"""
import torch
import torch.nn.functional as F


def column_ce(logits, target, valid, balance="column"):
    logp = F.log_softmax(logits.float(), dim=2)
    ce = -(target * logp).sum(2)  # B,C,W
    v = valid.float()
    if balance == "column":
        return (ce * v).sum() / v.sum().clamp_min(1.0)
    n_c = v.sum((0, 2))
    per_c = (ce * v).sum((0, 2)) / n_c.clamp_min(1.0)
    present = n_c > 0
    return per_c[present].mean() if present.any() else (ce * 0).sum()


def decode(logits, win=5):
    """Local soft-argmax around the mode.

    Returns yhat (B,C,W, differentiable), mass (B,C,W, detached: probability
    within +-win px of the mode, used as confidence), probs (B,C,H,W).
    """
    p = torch.softmax(logits.float(), dim=2)
    H = p.shape[2]
    mode = p.argmax(2)  # B,C,W
    off = torch.arange(-win, win + 1, device=p.device).view(1, 1, -1, 1)
    idx = mode.unsqueeze(2) + off  # B,C,K,W
    inb = (idx >= 0) & (idx < H)
    idx_c = idx.clamp(0, H - 1)
    pw = torch.gather(p, 2, idx_c) * inb
    mass = pw.sum(2)
    yhat = (pw * idx_c.float()).sum(2) / mass.clamp_min(1e-8)
    return yhat, mass.detach(), p


def order_loss(yhat, conf, pairs, thr=0.7, colmask=None):
    """pairs: list of (a, b) meaning y_a <= y_b (a above b in array coords).
    relu(y_a - y_b): crossing is penalised, touching (zero thickness) is not."""
    if not pairs:
        return yhat.sum() * 0, 0.0
    gate = (conf > thr).float()
    a = torch.tensor([p[0] for p in pairs], device=yhat.device)
    b = torch.tensor([p[1] for p in pairs], device=yhat.device)
    g = gate[:, a] * gate[:, b]
    if colmask is not None:
        g = g * colmask.unsqueeze(1)
    viol = F.relu(yhat[:, a] - yhat[:, b])
    n = g.sum()
    return (viol * g).sum() / n.clamp_min(1.0), float(n.item()) / max(g.numel(), 1)


def smooth_loss(yhat, conf, channels, thr=0.7, delta=2.0, colmask=None):
    if not channels or yhat.shape[-1] < 3:
        return yhat.sum() * 0
    ch = torch.tensor(channels, device=yhat.device)
    y = yhat[:, ch]
    g = (conf[:, ch] > thr).float()
    if colmask is not None:
        g = g * colmask.unsqueeze(1)
    d2 = y[..., 2:] - 2 * y[..., 1:-1] + y[..., :-2]
    gg = g[..., 2:] * g[..., 1:-1] * g[..., :-2]
    h = F.huber_loss(d2, torch.zeros_like(d2), delta=delta, reduction="none")
    return (h * gg).sum() / gg.sum().clamp_min(1.0)
