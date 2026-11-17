"""Multi-resolution local-regional-global dataset (Phase 4) and geologically coherent augmentation (Phase 7).

Geometry (defaults, all configurable)::

    local     100 m x 128 px = 12.8 km     level factor 1
    regional  200 m x 256 px = 51.2 km     level factor 2
    global    800 m x 256 px = 204.8 km    level factor 8

Window centres lie on the lattice of the coarsest factor, so the three windows share one exact geographic centre and
the local window is exactly the central (local_extent / view_extent) fraction of each coarser view (1/4 and 1/16):
no sub-pixel resampling, and ``model.align_center`` can crop it exactly. Coarser levels are NaN-aware block means of
the 100 m stack (aggregates, not new information). Outside the raster or the data footprint: values 0 and validity 0.

Known-fault context channels are computed per unit from the unit's *permitted* map and pyramided the same way.
Targets come only from the unit's visible labels; the loss mask is the unit's training region (inputs from other
geography may appear in a window, their labels never do).

UNTESTED: written without execution.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .contract import AXIAL_COS, AXIAL_SIN, SCALAR, VEC_X, VEC_Y, ChannelStack, UnitNormalizer, block_fraction
from .contract import block_nanmean, known_context

LEVEL_NAMES = ("local", "regional", "global")


# =============================================================================== targets
def unit_targets(labels: np.ndarray, kernel_radius_px: float = 3.0) -> dict[str, np.ndarray]:
    """Full-grid targets from VISIBLE labels: trace, DTI-kernel soft target, distance (km), axial orientation, endpoint
    and junction heat-maps."""
    from ..g13 import signal as S
    from ..geology.skeleton import branch_points, endpoints, skeletonize

    lab = np.asarray(labels, bool)
    d = ndimage.distance_transform_edt(~lab).astype(np.float32) if lab.any() else np.full(lab.shape, 1e4, np.float32)
    _, th, _ = S.structure_tensor(lab.astype(np.float32), 1.0, 3.0)
    sk = skeletonize(lab) if lab.any() else lab
    ep = ndimage.binary_dilation(endpoints(sk), iterations=1) if lab.any() else lab
    jn = ndimage.binary_dilation(branch_points(sk), iterations=1) if lab.any() else lab
    return {"trace": lab.astype(np.float32), "soft": np.clip(1 - d / kernel_radius_px, 0, 1).astype(np.float32),
            "dist_px": d, "cos2": np.cos(2 * th).astype(np.float32), "sin2": np.sin(2 * th).astype(np.float32),
            "endpoint": ep.astype(np.float32), "junction": jn.astype(np.float32)}


# =============================================================================== per-unit inputs
class UnitInputs:
    """Stack + per-unit known context + per-unit normalisation, at every pyramid level."""

    def __init__(self, stack: ChannelStack, data, unit, cfg, seed: int = 0):
        self.stack, self.data, self.unit, self.cfg = stack, data, unit, cfg
        self.factors = stack.factors
        self.channels = stack.channels
        self.known_names = [c["name"] for c in stack.manifest.get("known_channels", [])] if cfg.data.known_context \
            and "known_context" in cfg.data.modalities else []
        self.kinds = [c.kind for c in self.channels] + [k["kind"] for k in stack.manifest.get("known_channels", [])
                                                       if k["name"] in self.known_names]
        self.modalities = list(stack.modalities) + (["known_context"] if self.known_names else [])
        self.known = {}
        if self.known_names:
            kc = known_context(unit.permitted_known, data.valid)
            for f in self.factors:
                self.known[f] = np.stack([block_nanmean(kc[n], f) for n in self.known_names]).astype(np.float32)
            vf = {f: block_fraction(data.valid, f) for f in self.factors}
            self.known_mask = vf
        self.norm = self._fit_norm(seed)

    @property
    def n_channels(self) -> int:
        return len(self.channels) + len(self.known_names)

    def channel_names(self) -> list[str]:
        return [c.name for c in self.channels] + self.known_names

    def modality_index(self) -> dict[str, list[int]]:
        out: dict[str, list[int]] = {}
        for i, c in enumerate(self.channels):
            out.setdefault(c.modality, []).append(i)
        for j in range(len(self.known_names)):
            out.setdefault("known_context", []).append(len(self.channels) + j)
        return out

    def _fit_norm(self, seed: int) -> UnitNormalizer:
        """Robust stats from the unit's training pixels only (never its inner validation or held-out zone)."""
        rng = np.random.default_rng(seed)
        ids = np.flatnonzero(self.unit.train.ravel())
        ids = np.sort(rng.choice(ids, min(200_000, ids.size), replace=False)) if ids.size else ids
        a = self.stack.level(1)
        cols = [np.asarray(a[i].reshape(-1)[ids], np.float32) for i in range(len(self.channels))]
        if self.known_names:
            cols += [self.known[1][j].reshape(-1)[ids] for j in range(len(self.known_names))]
        sample = np.stack(cols, 1) if cols else np.zeros((0, 0), np.float32)
        return UnitNormalizer.fit(sample, self.kinds, {"unit": self.unit.key, "n_pixels": int(ids.size),
                                                       "mask": "unit.train"})

    # ------------------------------------------------------------------ windows
    def window(self, level: int, top: int, left: int, size: int) -> tuple[np.ndarray, np.ndarray]:
        """(C, size, size) normalised values (NaN -> 0) and (M, size, size) availability in [0, 1] of one level,
        with zero padding outside the raster."""
        a = self.stack.level(level)
        mk = self.stack.masks(level)
        C, H, W = a.shape
        x = np.full((self.n_channels, size, size), np.nan, np.float32)
        m = np.zeros((len(self.modalities), size, size), np.float32)
        r0, c0 = max(top, 0), max(left, 0)
        r1, c1 = min(top + size, H), min(left + size, W)
        if r1 > r0 and c1 > c0:
            dr, dc = r0 - top, c0 - left
            x[:C, dr:dr + r1 - r0, dc:dc + c1 - c0] = a[:, r0:r1, c0:c1]
            m[:mk.shape[0], dr:dr + r1 - r0, dc:dc + c1 - c0] = mk[:, r0:r1, c0:c1] / 255.0
            if self.known_names:
                x[C:, dr:dr + r1 - r0, dc:dc + c1 - c0] = self.known[level][:, r0:r1, c0:c1]
                m[-1, dr:dr + r1 - r0, dc:dc + c1 - c0] = self.known_mask[level][r0:r1, c0:c1]
        x = self.norm(x)
        return np.nan_to_num(x, nan=0.0), m


@dataclass
class Sample:
    unit: str
    top: int          # local window top-left (level-1 pixels)
    left: int
    positive: bool


def lattice_centres(train: np.ndarray, labels: np.ndarray, local_px: int, step: int, min_frac: float,
                    strict_context: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Admissible local windows (top, left) on the centre lattice: at least ``min_frac`` of the window in ``train``.
    Returns (all admissible, positive admissible = window contains a visible label inside train)."""
    H, W = train.shape
    half = local_px // 2
    ii = np.pad(np.cumsum(np.cumsum(train.astype(np.int64), 0), 1), ((1, 0), (1, 0)))
    li = np.pad(np.cumsum(np.cumsum((labels & train).astype(np.int64), 0), 1), ((1, 0), (1, 0)))
    cr = np.arange(step, H, step)
    cc = np.arange(step, W, step)
    R, Cc = np.meshgrid(cr, cc, indexing="ij")
    top, left = R - half, Cc - half
    t0, l0 = np.clip(top, 0, H), np.clip(left, 0, W)
    t1, l1 = np.clip(top + local_px, 0, H), np.clip(left + local_px, 0, W)

    def box(I):
        return I[t1, l1] - I[t0, l1] - I[t1, l0] + I[t0, l0]

    frac = box(ii) / float(local_px * local_px)
    ok = frac >= min_frac
    if strict_context is not None:
        ok &= ~strict_context[np.clip(R, 0, H - 1), np.clip(Cc, 0, W - 1)]
    pos = ok & (box(li) > 0)
    allw = np.c_[top[ok], left[ok]]
    posw = np.c_[top[pos], left[pos]]
    return allw, posw


class MultiScaleDataset:
    """Map-style dataset (usable directly by ``torch.utils.data.DataLoader``) of aligned three-view samples."""

    def __init__(self, inputs: UnitInputs, targets: dict, loss_region: np.ndarray, ignore: np.ndarray, cfg,
                 n_samples: int, seed: int, augment: bool, positive_frac: float | None = None,
                 strict_context: np.ndarray | None = None):
        self.inp, self.tg, self.cfg = inputs, targets, cfg
        d = cfg.data
        self.L = d.local.size_px
        self.views = [(1, d.local.size_px),
                      (int(round(d.regional.resolution_m / d.local.resolution_m)), d.regional.size_px),
                      (int(round(d.global_.resolution_m / d.local.resolution_m)), d.global_.size_px)]
        step = max(f for f, _ in self.views)
        self.loss_region = loss_region & inputs.data.valid
        self.ignore = ignore
        self.neg_far = targets["dist_px"] > cfg.loss.reliable_negative_px
        allw, posw = lattice_centres(self.loss_region, targets["trace"] > 0, self.L, step, d.min_train_frac,
                                     strict_context)
        if not len(allw):
            raise ValueError(f"{inputs.unit.key}: no admissible local window; lower data.min_train_frac")
        rng = np.random.default_rng(seed)
        pf = d.positive_center_frac if positive_frac is None else positive_frac
        n_pos = int(round(n_samples * pf)) if len(posw) else 0
        a = posw[rng.integers(0, len(posw), n_pos)] if n_pos else np.empty((0, 2), int)
        b = allw[rng.integers(0, len(allw), n_samples - n_pos)]
        win = np.vstack([a, b])
        self.samples = [Sample(inputs.unit.key, int(t), int(l), i < n_pos) for i, (t, l) in enumerate(win)]
        self.augment = augment
        self.seed = seed
        self.kinds = inputs.kinds

    def __len__(self) -> int:
        return len(self.samples)

    def view_tops(self, top: int, left: int) -> list[tuple[int, int, int, int]]:
        """(level factor, top, left, size) of the three views sharing the local window's centre."""
        cy, cx = top + self.L // 2, left + self.L // 2        # level-1 pixel-edge coordinates of the centre
        out = []
        for f, S in self.views:
            out.append((f, cy // f - S // 2, cx // f - S // 2, S))
        return out

    def __getitem__(self, i: int) -> dict:
        s = self.samples[i]
        rng = np.random.default_rng((self.seed, i))
        item = {}
        for name, (f, t, l, S) in zip(LEVEL_NAMES, self.view_tops(s.top, s.left)):
            x, m = self.inp.window(f, t, l, S)
            item[name], item[f"{name}_mask"] = x, m
        sl = (slice(s.top, s.top + self.L), slice(s.left, s.left + self.L))
        tg = {k: _crop(v, sl, self.L) for k, v in self.tg.items()}
        region = _crop(self.loss_region.astype(np.float32), sl, self.L)
        ign = _crop(self.ignore.astype(np.float32), sl, self.L)
        neg_far = _crop(self.neg_far.astype(np.float32), sl, self.L)
        tg["loss_mask"] = region
        tg["neg_weight"] = region * (1 - ign) * (tg["dist_px"] > self.cfg.loss.ignore_ring_px)
        tg["reliable_neg"] = region * (1 - ign) * neg_far
        tg["orient_w"] = region * (tg["dist_px"] <= 3)
        tg["has_pos"] = np.float32((tg["trace"] * region).sum() > 0)
        if self.augment and self.cfg.data.augment:
            k, flip = int(rng.integers(4)), bool(rng.integers(2))
            for name in LEVEL_NAMES:
                item[name] = transform_channels(item[name], self.kinds, k, flip)
                item[f"{name}_mask"] = transform_scalars(item[f"{name}_mask"], k, flip)
            for kk in list(tg):
                if kk in ("cos2", "sin2", "has_pos"):
                    continue
                tg[kk] = transform_scalars(tg[kk][None], k, flip)[0]
            c2, s2 = transform_scalars(tg["cos2"][None], k, flip)[0], transform_scalars(tg["sin2"][None], k, flip)[0]
            tg["cos2"], tg["sin2"] = axial_update(c2, s2, k, flip)
            self._radiometric(item, rng)
        item["targets"] = tg
        item["meta"] = {"unit": s.unit, "top": s.top, "left": s.left, "positive": s.positive}
        return item

    def _radiometric(self, item: dict, rng) -> None:
        """Mild noise on scalar channels; whole-modality dropout; geographically consistent sensor dropout."""
        d = self.cfg.data
        mods = self.inp.modality_index()
        scal = np.array([k == SCALAR for k in self.kinds])
        for name in LEVEL_NAMES:
            if d.noise_std > 0:
                item[name][scal] += rng.normal(0, d.noise_std, item[name][scal].shape).astype(np.float32)
        names = [m for m in mods if m != "known_context"]
        if names and rng.random() < d.modality_dropout:
            m = names[int(rng.integers(len(names)))]
            j = self.inp.modalities.index(m)
            for name in LEVEL_NAMES:
                item[name][mods[m]] = 0.0
                item[f"{name}_mask"][j] = 0.0
        if names and rng.random() < d.sensor_dropout:
            m = names[int(rng.integers(len(names)))]
            j = self.inp.modalities.index(m)
            # a rectangle in level-1 coordinates relative to the shared centre, applied to every view
            h = int(rng.integers(8, self.L // 2))
            w = int(rng.integers(8, self.L // 2))
            y0 = int(rng.integers(-self.L // 2, self.L // 2 - h))
            x0 = int(rng.integers(-self.L // 2, self.L // 2 - w))
            for name, (f, S) in zip(LEVEL_NAMES, self.views):
                cy = cx = S // 2
                ys, xs = cy + y0 // f, cx + x0 // f
                ye, xe = cy + (y0 + h) // f + 1, cx + (x0 + w) // f + 1
                item[name][mods[m], max(ys, 0):ye, max(xs, 0):xe] = 0.0
                item[f"{name}_mask"][j, max(ys, 0):ye, max(xs, 0):xe] = 0.0


def _crop(a: np.ndarray, sl, size: int) -> np.ndarray:
    H, W = a.shape
    out = np.zeros((size, size), np.float32)
    r0, c0 = max(sl[0].start, 0), max(sl[1].start, 0)
    r1, c1 = min(sl[0].stop, H), min(sl[1].stop, W)
    if r1 > r0 and c1 > c0:
        out[r0 - sl[0].start:r1 - sl[0].start, c0 - sl[1].start:c1 - sl[1].start] = a[r0:r1, c0:c1]
    return np.nan_to_num(out)


# =============================================================================== augmentation
def transform_scalars(x: np.ndarray, k: int, flip: bool) -> np.ndarray:
    """Rotate (k x 90 deg counter-clockwise in map view) then flip east-west. x: (C, h, w)."""
    y = np.rot90(x, k, axes=(-2, -1))
    if flip:
        y = y[..., ::-1]
    return np.ascontiguousarray(y)


def axial_update(c2: np.ndarray, s2: np.ndarray, k: int, flip: bool) -> tuple[np.ndarray, np.ndarray]:
    """Axial (cos 2t, sin 2t) after k CCW quarter turns (t -> t + k pi/2: both signs flip when k is odd) and an
    east-west flip (t -> pi - t: sin 2t changes sign)."""
    if k % 2:
        c2, s2 = -c2, -s2
    if flip:
        s2 = -s2
    return c2, s2


def vector_update(vx: np.ndarray, vy: np.ndarray, k: int, flip: bool) -> tuple[np.ndarray, np.ndarray]:
    """Geographic vector (east, north) after k CCW quarter turns ((x, y) -> (-y, x)) and an east-west flip."""
    for _ in range(k % 4):
        vx, vy = -vy, vx
    if flip:
        vx = -vx
    return vx, vy


def transform_channels(x: np.ndarray, kinds: list[str], k: int, flip: bool) -> np.ndarray:
    """Spatial transform of every channel plus the value transform of directional channels.
    Axial channels must come as (cos, sin) pairs and vectors as (x, y) pairs, in that order (schema guarantees it)."""
    y = transform_scalars(x, k, flip)
    i = 0
    while i < len(kinds):
        kd = kinds[i]
        if kd == AXIAL_COS:
            assert kinds[i + 1] == AXIAL_SIN, "axial channels must be adjacent (cos, sin)"
            y[i], y[i + 1] = axial_update(y[i], y[i + 1], k, flip)
            i += 2
        elif kd == VEC_X:
            assert kinds[i + 1] == VEC_Y, "vector channels must be adjacent (x, y)"
            y[i], y[i + 1] = vector_update(y[i], y[i + 1], k, flip)
            i += 2
        else:
            i += 1
    return y


def collate(batch: list[dict]):
    """Stack numpy samples into torch tensors (local / regional / global, masks, targets)."""
    import torch

    out = {}
    for key in ("local", "regional", "global", "local_mask", "regional_mask", "global_mask"):
        out[key] = torch.from_numpy(np.stack([b[key] for b in batch]))
    out["targets"] = {k: torch.from_numpy(np.stack([np.asarray(b["targets"][k], np.float32) for b in batch]))
                      for k in batch[0]["targets"]}
    out["meta"] = [b["meta"] for b in batch]
    return out
