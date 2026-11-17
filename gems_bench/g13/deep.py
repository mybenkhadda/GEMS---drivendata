"""Optional tier-2 / tier-3 network: compact residual multitask U-Net (sections 22, 24). Requires torch.

Heads (supervision derived only from the permitted training faults of each unit):

1. buffered fault probability (within 100 m)            BCE + soft Tversky (alpha 0.2, beta 0.8)
2. centre-line probability (skeleton)                    BCE
3. local axial orientation (cos 2 theta, sin 2 theta)    1 - cos(2 dtheta), only within 300 m of a fault
4. distance-to-fault proxy (km, clipped at 3)            smooth L1, only within 3 km
5. junction likelihood (skeleton branch points, dilated) BCE

Patches are drawn so that the whole receptive field lies inside the unit's training mask (integral-image check),
so no input window of a training patch overlaps validation geography. Inference: Hann-blended overlapping tiles.
If multitask learning does not improve held-out DTI over the buffered-probability-only variant, the simpler model
is retained (decided by the scorecard, not here).
"""

from __future__ import annotations

import time

import numpy as np
from scipy import ndimage


def available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def targets(labels: np.ndarray) -> dict[str, np.ndarray]:
    from ..geology.skeleton import branch_points, skeletonize
    from .signal import structure_tensor

    d = (
        ndimage.distance_transform_edt(~labels).astype(np.float32)
        if labels.any()
        else np.full(labels.shape, 1e3, np.float32)
    )
    sk = skeletonize(labels)
    _, th, _ = structure_tensor(labels.astype(np.float32), 1.0, 3.0)
    junc = ndimage.binary_dilation(branch_points(sk), iterations=2) if sk.any() else sk
    return {
        "buffer": (d <= 1).astype(np.float32),
        "center": sk.astype(np.float32),
        "cos2": np.cos(2 * th).astype(np.float32),
        "sin2": np.sin(2 * th).astype(np.float32),
        "orient_w": (d <= 3).astype(np.float32),
        "dist": np.minimum(d * 0.1, 3.0).astype(np.float32),
        "dist_w": (d <= 30).astype(np.float32),
        "junction": junc.astype(np.float32),
    }


def patch_origins(train: np.ndarray, tile: int, rng, n: int, labels: np.ndarray, pos_frac: float = 0.5) -> np.ndarray:
    """Top-left corners of tiles lying entirely inside ``train``; ``pos_frac`` of them centred near faults."""
    ii = np.pad(np.cumsum(np.cumsum(train.astype(np.int64), 0), 1), ((1, 0), (1, 0)))
    h, w = train.shape
    r = np.arange(0, h - tile + 1)
    c = np.arange(0, w - tile + 1)
    R, C = np.meshgrid(r, c, indexing="ij")
    full = (ii[R + tile, C + tile] - ii[R, C + tile] - ii[R + tile, C] + ii[R, C]) == tile * tile
    ok = np.argwhere(full)
    if not len(ok):
        raise ValueError("no training tile fits inside the training mask; reduce deep_tile")
    lab_ii = np.pad(np.cumsum(np.cumsum(labels.astype(np.int64), 0), 1), ((1, 0), (1, 0)))
    has = (
        lab_ii[ok[:, 0] + tile, ok[:, 1] + tile]
        - lab_ii[ok[:, 0], ok[:, 1] + tile]
        - lab_ii[ok[:, 0] + tile, ok[:, 1]]
        + lab_ii[ok[:, 0], ok[:, 1]]
    ) > 0
    npos = int(n * pos_frac)
    pos, neg = ok[has], ok[~has]
    a = pos[rng.integers(0, len(pos), npos)] if len(pos) else ok[rng.integers(0, len(ok), npos)]
    b = neg[rng.integers(0, len(neg), n - npos)] if len(neg) else ok[rng.integers(0, len(ok), n - npos)]
    return np.vstack([a, b])


def build_net(in_ch: int, width: int = 32, multitask: bool = True):
    import torch
    from torch import nn

    class Res(nn.Module):
        def __init__(self, ci, co):
            super().__init__()
            self.a = nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1),
                nn.BatchNorm2d(co),
                nn.GELU(),
                nn.Conv2d(co, co, 3, padding=1),
                nn.BatchNorm2d(co),
            )
            self.s = nn.Conv2d(ci, co, 1) if ci != co else nn.Identity()
            self.act = nn.GELU()

        def forward(self, x):
            return self.act(self.a(x) + self.s(x))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            w = width
            self.e1, self.e2, self.e3 = Res(in_ch, w), Res(w, 2 * w), Res(2 * w, 4 * w)
            # dilated bottleneck: along-strike context of ~5 km at 1/4 resolution
            self.b = nn.Sequential(
                nn.Conv2d(4 * w, 4 * w, 3, padding=2, dilation=2),
                nn.GELU(),
                nn.Conv2d(4 * w, 4 * w, 3, padding=4, dilation=4),
                nn.GELU(),
            )
            self.u2, self.d2 = nn.ConvTranspose2d(4 * w, 2 * w, 2, 2), Res(4 * w, 2 * w)
            self.u1, self.d1 = nn.ConvTranspose2d(2 * w, w, 2, 2), Res(2 * w, w)
            self.pool = nn.MaxPool2d(2)
            self.n_out = 6 if multitask else 1
            self.head = nn.Conv2d(w, self.n_out, 1)

        def forward(self, x):
            e1 = self.e1(x)
            e2 = self.e2(self.pool(e1))
            e3 = self.b(self.e3(self.pool(e2)))
            d2 = self.d2(torch.cat([self.u2(e3), e2], 1))
            d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
            return self.head(d1)

    return Net()


def loss_fn(out, tg, multitask: bool = True):
    import torch
    import torch.nn.functional as F

    p = torch.sigmoid(out[:, 0])
    y = tg["buffer"]
    bce = F.binary_cross_entropy_with_logits(out[:, 0], y, pos_weight=torch.tensor(5.0, device=out.device))
    tp = (p * y).sum()
    tv = 1 - (tp + 1) / (tp + 0.2 * (p * (1 - y)).sum() + 0.8 * ((1 - p) * y).sum() + 1)
    loss = bce + tv
    if multitask and out.shape[1] >= 6:
        loss = loss + 0.5 * F.binary_cross_entropy_with_logits(
            out[:, 1], tg["center"], pos_weight=torch.tensor(10.0, device=out.device)
        )
        c, s = torch.tanh(out[:, 2]), torch.tanh(out[:, 3])
        norm = torch.sqrt(c * c + s * s + 1e-6)
        cosd = (c * tg["cos2"] + s * tg["sin2"]) / norm
        ow = tg["orient_w"]
        loss = loss + 0.2 * ((1 - cosd) * ow).sum() / (ow.sum() + 1)
        dw = tg["dist_w"]
        loss = loss + 0.2 * (F.smooth_l1_loss(F.softplus(out[:, 4]), tg["dist"], reduction="none") * dw).sum() / (
            dw.sum() + 1
        )
        loss = loss + 0.2 * F.binary_cross_entropy_with_logits(out[:, 5], tg["junction"])
    return loss


def standardise(stack: np.ndarray, train: np.ndarray) -> tuple[np.ndarray, list]:
    """Per-channel median / IQR fitted on training pixels only; NaN -> 0 after scaling."""
    stats = []
    for i in range(stack.shape[0]):
        v = stack[i][train & np.isfinite(stack[i])]
        med = float(np.median(v)) if v.size else 0.0
        iqr = float(np.subtract(*np.percentile(v, [75, 25]))) if v.size else 1.0
        stats.append((med, iqr or 1.0))
        stack[i] = np.nan_to_num((stack[i] - med) / (iqr or 1.0), nan=0.0)
    return np.clip(stack, -20, 20).astype(np.float32), stats


def train_unit(
    stack: np.ndarray,
    labels: np.ndarray,
    train: np.ndarray,
    cfg,
    seed: int,
    *,
    multitask: bool = True,
    steps_per_epoch: int = 150,
    batch: int = 8,
    log=print,
):
    import torch

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tile = cfg.deep_tile
    net = build_net(stack.shape[0], multitask=multitask).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=cfg.deep_epochs * steps_per_epoch)
    tg = targets(labels)
    origins = patch_origins(train, tile, rng, cfg.deep_epochs * steps_per_epoch * batch, labels)
    t0 = time.perf_counter()
    net.train()
    for step in range(cfg.deep_epochs * steps_per_epoch):
        o = origins[step * batch : (step + 1) * batch]
        X = np.stack([stack[:, r : r + tile, c : c + tile] for r, c in o])
        T = {k: torch.from_numpy(np.stack([v[r : r + tile, c : c + tile] for r, c in o])).to(dev) for k, v in tg.items()}
        k = int(rng.integers(4))  # 90-degree rotations: orientation targets rotate with it
        Xt = torch.from_numpy(X).to(dev)
        if k:
            Xt = torch.rot90(Xt, k, (2, 3))
            T = {n: torch.rot90(v, k, (1, 2)) for n, v in T.items()}
            if k % 2:
                T["cos2"], T["sin2"] = -T["cos2"], -T["sin2"]
        loss = loss_fn(net(Xt), T, multitask)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % steps_per_epoch == 0:
            log(f"[deep] step {step} loss {float(loss):.4f} ({time.perf_counter() - t0:.0f}s)")
    return net


def predict(net, stack: np.ndarray, mask: np.ndarray, tile: int = 256, overlap: int = 64) -> np.ndarray:
    import torch

    dev = next(net.parameters()).device
    net.eval()
    h, w = mask.shape
    acc = np.zeros((h, w), np.float32)
    wsum = np.zeros((h, w), np.float32)
    hann = np.outer(np.hanning(tile), np.hanning(tile)).astype(np.float32) + 1e-3
    step = tile - overlap
    rows = list(range(0, max(h - tile, 0) + 1, step)) + ([h - tile] if h > tile else [])
    cols = list(range(0, max(w - tile, 0) + 1, step)) + ([w - tile] if w > tile else [])
    with torch.no_grad():
        for r in sorted(set(rows)):
            for c in sorted(set(cols)):
                if not mask[r : r + tile, c : c + tile].any():
                    continue
                x = torch.from_numpy(stack[None, :, r : r + tile, c : c + tile]).to(dev)
                p = torch.sigmoid(net(x)[0, 0]).cpu().numpy()
                acc[r : r + tile, c : c + tile] += p * hann
                wsum[r : r + tile, c : c + tile] += hann
    out = np.where(wsum > 0, acc / np.maximum(wsum, 1e-9), np.nan).astype(np.float32)
    out[~mask] = np.nan
    return out
