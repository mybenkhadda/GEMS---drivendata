"""Section 04: training units, input cube, fold-wise scaling, targets, patch batches and the CV-A..F manifests.

Training units (``training_units``)
    ``geo_f<k>``   CV-A: patches fully inside fold k's inner-training zone (PatchSampler); checkpoint selected on the
                   inner-validation zone (binary top-frac DTI, no map context); prediction = fold k's validation zone.
    ``disc_s<d>``  CV-B: patches anywhere valid; withheld systems + ignore ring get weight 0 (never negatives) and are
                   never shown as map context; fixed epochs = median best epoch of the geo units; prediction = the
                   split's evaluation zone, with and without the retained-map context.
    ``full``       optional (``inference.source = full_model``): every label, fixed epochs, full-region inference.

Inputs
    ``cube``    (C, H, W) float16 memory-mapped stack of the LOCAL channels (raw, structural, spectral, DEM) + a
                footprint channel, stored once in ``cache/m9`` as the store's robust z-values (clipped at +-8);
    ``pooled``  (C + C_ctx, H/f, W/f) average-pooled local + regional-context channels for the low-resolution branch.
    Scaling: a per-unit robust affine (median / IQR) fitted on a random sample of the unit's PERMITTED pixels only
    (``FoldScaler``), applied per batch; NaN -> 0 after scaling. The full cube never goes to VRAM: patches stream.

Targets (per unit, from VISIBLE labels only = labels inside the permitted geography minus ignored systems)
    seg = visible faults; kern = 300 m kernel target; (cos 2t, sin 2t, valid) orientation targets (no junctions, no
    pieces < 1 km, coherent); endpoint / junction Gaussian heatmaps with a topology-valid mask (pixels near the
    permitted-zone boundary are excluded: endpoints there are artefacts of the split); skeleton (L3).
    Weights: positives 1; unlabeled background ``pu.unlabeled_weight`` x label-free background weight (strong
    label-free structure is down-weighted: possibly an unmapped fault); ignored pixels 0.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import ndimage

CLIP_Z = 8.0


# =============================================================================== units
@dataclass
class TrainUnit:
    name: str
    kind: str                              # cv | disc | full
    zone: int
    allowed: np.ndarray = field(repr=False)        # patches must lie inside (bool, full grid)
    weight_mask: np.ndarray = field(repr=False)    # pixels that may carry loss
    visible: np.ndarray = field(repr=False)        # labels the unit may see (targets and map context)
    inner: np.ndarray | None = field(default=None, repr=False)   # inner-validation zone (cv only)
    predict: np.ndarray | None = field(default=None, repr=False)  # held-out prediction zone

    @property
    def fixed_epochs(self) -> bool:
        return self.kind != "cv"


def training_units(m9, include_full: bool | None = None) -> list[TrainUnit]:
    ctx = m9.ctx
    valid = np.asarray(ctx.data.valid, bool)
    lab = np.asarray(ctx.data.labels, bool)
    out = []
    for k in m9.train_folds():
        allowed = ctx.folds.inner_train(k) & valid
        out.append(TrainUnit(f"geo_f{k}", "cv", int(k), allowed, allowed.copy(), lab & allowed,
                             ctx.folds.inner_val(k) & valid, ctx.folds.val(k) & valid))
    for d in m9.train_splits():
        ign = ctx.splits.ignore(d)
        out.append(TrainUnit(f"disc_s{d}", "disc", int(d), valid.copy(), valid & ~ign, lab & ~ign, None,
                             ctx.splits.eval_zone(d) & valid))
    if include_full if include_full is not None else m9.cfg["inference"]["source"] == "full_model":
        out.append(TrainUnit("full", "full", 0, valid.copy(), valid.copy(), lab.copy(), None, valid.copy()))
    return out


# =============================================================================== input cube
def channel_order(groups: dict) -> tuple[list[str], list[str]]:
    """(local channels in cube order, context-only channels)."""
    local = list(dict.fromkeys(groups["raw"] + groups["struct"] + groups["spectral"] + groups["dem"]))
    ctx_only = [c for c in groups["context"] if c not in set(local)]
    return local, ctx_only


def cube_key(local: list[str], ctx_only: list[str], factor: int) -> str:
    return hashlib.sha1(json.dumps([local, ctx_only, int(factor)]).encode()).hexdigest()[:12]


def build_cube(m9, groups: dict) -> dict:
    """Memory-mapped local cube + pooled cube in the M9 cache (built once per channel list); returns paths + meta."""
    from ..artifacts.atomic import atomic_write_json
    from .features import read

    local, ctx_only = channel_order(groups)
    f = int(m9.cfg["data"]["context_factor"])
    key = cube_key(local, ctx_only, f)
    root = m9.cache
    p_loc, p_pool, p_meta = root / f"cube_{key}.npy", root / f"pooled_{key}.npy", root / f"cube_{key}.json"
    valid = np.asarray(m9.ctx.data.valid, bool)
    H, W = valid.shape
    if p_meta.exists() and p_loc.exists() and p_pool.exists():
        return {**json.loads(p_meta.read_text(encoding="utf-8")), "cube": p_loc, "pooled": p_pool}
    names = local + ["m9:footprint"]
    mm = np.lib.format.open_memmap(str(p_loc) + ".partial", mode="w+", dtype=np.float16, shape=(len(names), H, W))
    hp, wp = H // f, W // f
    pool_names = names + ctx_only
    pm = np.lib.format.open_memmap(str(p_pool) + ".partial", mode="w+", dtype=np.float16,
                                   shape=(len(pool_names), hp, wp))
    for j, n in enumerate(pool_names):
        a = valid.astype(np.float32) if n == "m9:footprint" else np.clip(np.asarray(read(m9, n), np.float32), -CLIP_Z, CLIP_Z)
        if j < len(names):
            mm[j] = a.astype(np.float16)
        b = np.nan_to_num(a[:hp * f, :wp * f]).reshape(hp, f, wp, f)
        cnt = np.isfinite(a[:hp * f, :wp * f]).reshape(hp, f, wp, f).sum((1, 3))
        pm[j] = (b.sum((1, 3)) / np.maximum(cnt, 1)).astype(np.float16)
    mm.flush()
    pm.flush()
    del mm, pm
    import os

    os.replace(str(p_loc) + ".partial", p_loc)
    os.replace(str(p_pool) + ".partial", p_pool)
    meta = {"key": key, "local": names, "context_only": ctx_only, "pooled": pool_names, "factor": f,
            "shape": [H, W], "dtype": "float16", "clip_z": CLIP_Z}
    atomic_write_json(p_meta, meta)
    return {**meta, "cube": p_loc, "pooled": p_pool}


def open_cube(meta: dict, in_ram: bool = True):
    """(local cube, pooled cube) as arrays (RAM copy when it fits comfortably, else memory-mapped)."""
    loc = np.load(meta["cube"], mmap_mode="r")
    pool = np.load(meta["pooled"], mmap_mode="r")
    if in_ram:
        try:
            import psutil

            if loc.nbytes < 0.4 * psutil.virtual_memory().available:
                loc = np.ascontiguousarray(loc)
        except Exception:
            pass
        pool = np.ascontiguousarray(pool)
    return loc, pool


# =============================================================================== scaling
def _robust(v: np.ndarray) -> tuple[float, float]:
    """(median, IQR / 1.349); near-constant / discrete channels (IQR ~ 0) fall back to the std, then 1."""
    q25, q50, q75 = np.percentile(v, [25, 50, 75])
    s = (q75 - q25) / 1.349
    if s < 1e-3:
        s = float(np.std(v)) if np.std(v) >= 1e-3 else 1.0
    return float(q50), float(s)


class FoldScaler:
    """Robust per-channel affine fitted on PERMITTED pixels of one training unit (no other geography)."""

    def __init__(self, center: np.ndarray, scale: np.ndarray, clip: float, names: list[str], n: int):
        self.center, self.scale, self.clip, self.names, self.n = center, scale, clip, names, n

    @classmethod
    def fit(cls, cube: np.ndarray, names: list[str], allowed: np.ndarray, n: int, clip: float, seed: int):
        idx = np.flatnonzero(allowed.ravel())
        if idx.size > n:
            idx = np.sort(np.random.default_rng(seed).choice(idx, n, replace=False))
        flat = cube.reshape(cube.shape[0], -1)
        c, s = np.zeros(len(names), np.float32), np.ones(len(names), np.float32)
        for j, nm in enumerate(names):
            if nm == "m9:footprint":
                continue
            v = np.asarray(flat[j, idx], np.float32)
            v = v[np.isfinite(v)]
            if v.size < 100:
                continue
            c[j], s[j] = _robust(v)
        return cls(c, s, float(clip), list(names), int(idx.size))

    def apply(self, x: np.ndarray) -> np.ndarray:
        """x: (..., C, h, w) float -> scaled float32, NaN -> 0."""
        sh = (-1, 1, 1)
        y = (np.asarray(x, np.float32) - self.center.reshape(sh)) / self.scale.reshape(sh)
        return np.clip(np.nan_to_num(y, nan=0.0), -self.clip, self.clip)

    def to_json(self) -> dict:
        return {"kind": "robust_median_iqr", "clip": self.clip, "n_sample": self.n, "names": self.names,
                "center": [float(x) for x in self.center], "scale": [float(x) for x in self.scale],
                "hash": hashlib.sha1(np.r_[self.center, self.scale].tobytes()).hexdigest()[:12]}

    @classmethod
    def from_json(cls, d: dict) -> "FoldScaler":
        return cls(np.asarray(d["center"], np.float32), np.asarray(d["scale"], np.float32), float(d["clip"]),
                   list(d["names"]), int(d["n_sample"]))


def pooled_scaler(sc: FoldScaler, pool_names: list[str], pooled: np.ndarray, allowed_pooled: np.ndarray,
                  seed: int) -> FoldScaler:
    """Scaler of the pooled cube: local channels reuse the unit scaler; context-only channels are fitted on the
    pooled permitted cells."""
    c = np.zeros(len(pool_names), np.float32)
    s = np.ones(len(pool_names), np.float32)
    pos = {n: i for i, n in enumerate(sc.names)}
    flat = pooled.reshape(pooled.shape[0], -1)
    idx = np.flatnonzero(allowed_pooled.ravel())
    if idx.size > 200_000:
        idx = np.sort(np.random.default_rng(seed).choice(idx, 200_000, replace=False))
    for j, n in enumerate(pool_names):
        if n in pos:
            c[j], s[j] = sc.center[pos[n]], sc.scale[pos[n]]
            continue
        v = np.asarray(flat[j, idx], np.float32)
        v = v[np.isfinite(v)]
        if v.size >= 100:
            c[j], s[j] = _robust(v)
    return FoldScaler(c, s, sc.clip, list(pool_names), sc.n)


# =============================================================================== targets
def gaussian_points(mask: np.ndarray, sigma_px: float) -> np.ndarray:
    """Peak-normalised Gaussian heatmap around the True pixels of ``mask``."""
    if not mask.any():
        return np.zeros(mask.shape, np.float32)
    d = ndimage.distance_transform_edt(~mask)
    return np.exp(-(d ** 2) / (2 * sigma_px ** 2)).astype(np.float32) * (d <= 3 * sigma_px)


def build_targets(m9, tu: TrainUnit) -> dict:
    """Full-grid targets of one unit (float16 where possible)."""
    from ..cnn import kernel_target
    from ..geology.skeleton import branch_points, component_length_km, endpoints, orientation_targets, skeletonize

    dc = m9.cfg["data"]
    grid = m9.ctx.data.grid
    vis = tu.visible
    kern = kernel_target(vis, grid, float(dc["kernel_radius_m"]))
    oc = dc["orientation"]
    ot = orientation_targets(vis, grid, kern=kern, scale_km=float(oc["scale_km"]),
                             min_coherence=float(oc["min_coherence"]), junction_px=float(oc["junction_px"]),
                             min_length_km=float(oc["min_length_km"]))
    sk = skeletonize(vis)
    tc = dc["topology"]
    long_ok = component_length_km(sk, grid.pixel_km) >= float(tc["min_length_km"])
    ep = gaussian_points(endpoints(sk) & long_ok, float(tc["sigma_px"]))
    jn = gaussian_points(branch_points(sk), float(tc["sigma_px"]))
    edge = ndimage.distance_transform_edt(tu.allowed & np.asarray(m9.ctx.data.valid, bool))
    topo_valid = (edge > float(tc["boundary_margin_px"])).astype(np.float32)
    w = tu.weight_mask.astype(np.float32)
    pu = dc["pu"]
    unl = float(pu["unlabeled_weight"])
    if pu.get("use_bg_weights"):
        bg = m9.cache_get(("bg_weights", pu["bg_evidence"]), lambda: _bg_weights(m9))
        unl_w = unl * bg
    else:
        unl_w = np.full(w.shape, unl, np.float32)
    w = np.where(vis, w, w * unl_w).astype(np.float32)
    sysmap = np.where(vis, np.asarray(m9.ctx.splits.system_map), -1).astype(np.int32)
    return {"seg": vis.astype(np.uint8), "kern": kern.astype(np.float16), "ocos": ot["ocos"].astype(np.float16),
            "osin": ot["osin"].astype(np.float16), "ovalid": ot["ovalid"].astype(np.uint8),
            "ep": ep.astype(np.float16), "jn": jn.astype(np.float16), "topo_valid": topo_valid.astype(np.uint8),
            "skel": sk.astype(np.uint8), "weight": w.astype(np.float16), "sysmap": sysmap}


def _bg_weights(m9) -> np.ndarray:
    from ..losses.pu import background_weights, resolve_evidence

    pu = m9.cfg["data"]["pu"]
    valid = np.asarray(m9.ctx.data.valid, bool)
    try:
        ev = resolve_evidence(m9.ctx, pu["bg_evidence"])
    except Exception as e:                                # missing evidence never blocks training
        m9.say(f"background weights disabled ({type(e).__name__}: {e})")
        return np.ones(valid.shape, np.float32)
    return background_weights(np.asarray(ev, np.float32), valid, lam=float(pu["bg_lambda"]), q0=float(pu["bg_q0"]),
                              w_min=float(pu["bg_min_weight"]), seed=int(m9.cfg["seed"]))


# =============================================================================== batches
def context_crop(pooled: np.ndarray, psc: FoldScaler, r0: int, c0: int, P: int, f: int,
                 mask: np.ndarray | None = None) -> np.ndarray:
    """(C_pool, P, P) low-resolution context centred on the local patch at (r0, c0) (zero outside the grid; zero
    outside ``mask`` when given: strict training context)."""
    hp, wp = pooled.shape[1:]
    rc, cc = (r0 + P // 2) // f, (c0 + P // 2) // f
    a0, b0 = rc - P // 2, cc - P // 2
    out = np.zeros((pooled.shape[0], P, P), np.float32)
    ra, rb, ca, cb = max(a0, 0), min(a0 + P, hp), max(b0, 0), min(b0 + P, wp)
    if ra < rb and ca < cb:
        blk = psc.apply(pooled[:, ra:rb, ca:cb])
        if mask is not None:
            blk = blk * mask[ra:rb, ca:cb][None]
        out[:, ra - a0:rb - a0, ca - b0:cb - b0] = blk
    return out


class BatchBuilder:
    """Random training patches of one unit: inputs (local + low-res context + map context) and every target."""

    def __init__(self, m9, tu: TrainUnit, vcfg: dict, cube: np.ndarray, pooled: np.ndarray, sc: FoldScaler,
                 psc: FoldScaler, targets: dict, rng: np.random.Generator):
        from ..cnn import PatchSampler
        from .corruption import Corruptor

        self.cfg, self.tu, self.cube, self.pooled, self.sc, self.psc, self.t, self.rng = vcfg, tu, cube, pooled, sc, psc, \
            targets, rng
        dc = vcfg["data"]
        self.P, self.f = int(dc["patch_px"]), int(dc["context_factor"])
        valid = np.asarray(m9.ctx.data.valid, bool)
        self.sampler = PatchSampler(tu.allowed, valid, tu.visible, self.P, float(dc["min_valid_frac"]), rng)
        self.strict = bool(dc["context_strict"])
        hp, wp = pooled.shape[1:]
        al = tu.allowed[:hp * self.f, :wp * self.f].reshape(hp, self.f, wp, self.f).mean((1, 3))
        self.allowed_pooled = al > 0.5
        self.corr = Corruptor(vcfg["corruption"], rng)
        self.pos_frac = float(dc["pos_patch_frac"])
        self.rot = bool(dc["rot180"])
        self.stats: dict = {}

    def context_crop(self, r0: int, c0: int, rot: bool = False) -> np.ndarray:
        out = context_crop(self.pooled, self.psc, r0, c0, self.P, self.f,
                           self.allowed_pooled if self.strict else None)
        return out[:, ::-1, ::-1] if rot else out

    def sample(self):
        P = self.P
        rc = self.sampler.draw(self.rng.random() < self.pos_frac) or self.sampler.draw(False, 1000)
        if rc is None:
            raise RuntimeError(f"{self.tu.name}: no admissible patch (zone too fragmented for {P} px patches)")
        r0, c0 = rc
        sl = (slice(r0, r0 + P), slice(c0, c0 + P))
        x = self.sc.apply(self.cube[:, sl[0], sl[1]])
        t = {k: np.asarray(v[sl], np.float32) for k, v in self.t.items() if k != "sysmap"}
        ctxmap, removed, ops, mode = self.corr(t["seg"] > 0, np.asarray(self.t["sysmap"][sl]))
        for o in ops + [f"mode_{mode}"]:
            self.stats[o] = self.stats.get(o, 0) + 1
        rot = self.rot and self.rng.random() < 0.5
        xc = self.context_crop(r0, c0, rot)
        if rot:
            x = x[:, ::-1, ::-1]
            t = {k: v[::-1, ::-1] for k, v in t.items()}
            ctxmap, removed = ctxmap[::-1, ::-1], removed[::-1, ::-1]
        flag = np.full((P, P), 0.0 if mode == "none" else 1.0, np.float32)   # 1 = a map is given (maybe empty here)
        return x, xc, np.stack([ctxmap.astype(np.float32), flag]), removed, t

    def batch(self, n: int) -> dict:
        import torch

        cc = self.cfg["corruption"]
        xs, xcs, ms, rs, ts = [], [], [], [], []
        for _ in range(n):
            x, xc, m, rem, t = self.sample()
            xs.append(np.ascontiguousarray(x))
            xcs.append(np.ascontiguousarray(xc))
            ms.append(np.ascontiguousarray(m))
            rs.append(np.ascontiguousarray(rem))
            ts.append(t)
        T = {k: np.stack([np.ascontiguousarray(t[k]) for t in ts])[:, None] for k in ts[0]}
        rem = np.stack(rs)[:, None].astype(np.float32)
        shown = np.stack(ms)[:, :1] * T["seg"]
        w = T["weight"] * np.where(shown > 0, float(cc["context_pixel_weight"]), 1.0)
        rec_zone = (ndimage.maximum_filter(rem, size=(1, 1, 7, 7)) > 0).astype(np.float32) * (T["weight"] > 0)
        out = {"x": np.stack(xs), "xc": np.stack(xcs), "m": np.stack(ms), "w": w.astype(np.float32),
               "rec": rem, "rec_zone": rec_zone, **{k: v.astype(np.float32) for k, v in T.items() if k != "weight"}}
        return {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in out.items()}


# =============================================================================== CV manifests (section 04)
def cv_manifests(m9) -> dict[str, pd.DataFrame]:
    """CV-A geographic folds, CV-B discovery splits, CV-D length strata, CV-F proximity bands (descriptive)."""
    ctx = m9.ctx
    lab = np.asarray(ctx.data.labels, bool)
    valid = np.asarray(ctx.data.valid, bool)
    A = []
    for k in ctx.fold_ids:
        A.append({"fold": int(k), "trained_in_profile": int(k) in m9.train_folds(),
                  "train_px": int((ctx.folds.inner_train(k) & valid).sum()),
                  "inner_val_px": int((ctx.folds.inner_val(k) & valid).sum()), "val_px": int((ctx.folds.val(k) & valid).sum()),
                  "buffer_px": int(ctx.folds.buffer(k).sum()), "train_fault_px": int((ctx.folds.inner_train(k) & lab).sum()),
                  "val_fault_px": int((ctx.folds.val(k) & lab).sum())})
    systems = ctx.splits.systems.set_index("system_id")
    B, D = [], []
    bins = [0.0] + list(m9.cfg["data"]["length_bins_km"]) + [np.inf]
    for d in range(ctx.splits.n_splits):
        G = ctx.splits.withheld(d)
        ids = np.unique(np.asarray(ctx.splits.system_map)[G])
        ids = ids[ids >= 0]
        km = systems.trace_km.reindex(ids)
        B.append({"split": d, "trained_in_profile": d in m9.train_splits(), "withheld_systems": int(len(ids)),
                  "withheld_km": float(km.sum()), "withheld_px": int(G.sum()),
                  "ignore_px": int(ctx.splits.ignore(d).sum()), "eval_zone_px": int((ctx.splits.eval_zone(d) & valid).sum())})
        cat = pd.cut(km, bins, right=False)
        for b, n in cat.value_counts().sort_index().items():
            D.append({"task": "disc_official", "zone": d, "length_bin_km": str(b), "n_systems": int(n)})
    F = []
    for u in m9.units_of(m9.prim):
        if u.known is None:
            continue
        from ..m7.protocol import distance_km

        dk = distance_km(u.known)
        G = u.eval_truth()
        b1, b2 = m9.sh["evaluation"]["proximity_bands_km"]
        for lab_, m in ((f"<{b1:g} km", dk < b1), (f"{b1:g}-{b2:g} km", (dk >= b1) & (dk < b2)), (f">{b2:g} km", dk >= b2)):
            F.append({"zone": u.zone, "band": lab_, "truth_px": int((G & m).sum()), "share": float((G & m).sum() / max(G.sum(), 1))})
    return {"cv_a_geographic": pd.DataFrame(A), "cv_b_discovery": pd.DataFrame(B), "cv_d_length_strata": pd.DataFrame(D),
            "cv_f_proximity": pd.DataFrame(F)}


def geological_domains(m9) -> tuple[np.ndarray, pd.DataFrame]:
    """CV-E: label-free domains = k-means of 20 km block descriptors (terrain roughness, magnetic / gravity gradient
    energy, basement depth). Returns (domain id raster, block table)."""
    from sklearn.cluster import KMeans

    data = m9.ctx.data
    valid = np.asarray(data.valid, bool)
    bk = max(int(round(float(m9.cfg["data"]["domains"]["block_km"]) / 0.1)), 10)
    H, W = valid.shape
    desc = {}
    for nm, band, op in (("terrain_roughness", "det_elev_slope", "std"), ("magnetic_gradient", "rtp", "grad"),
                         ("gravity_gradient", "iso_grav_anom", "grad"), ("basement_depth", "depth_to_base_surf", "median")):
        if band not in data.band_names:
            continue
        a = data.read_band(band)
        if op == "grad":
            g = np.nan_to_num(ndimage.gaussian_filter(np.nan_to_num(a, nan=float(np.nanmedian(a))), 3.0))
            gy, gx = np.gradient(g)
            a = np.where(np.isfinite(a), np.hypot(gx, gy), np.nan)
        desc[nm] = (a, op)
    rows = []
    for r0 in range(0, H, bk):
        for c0 in range(0, W, bk):
            v = valid[r0:r0 + bk, c0:c0 + bk]
            if v.mean() < 0.25:
                continue
            row = {"r0": r0, "c0": c0}
            for nm, (a, op) in desc.items():
                x = a[r0:r0 + bk, c0:c0 + bk][v]
                x = x[np.isfinite(x)]
                row[nm] = float(np.std(x) if op == "std" else np.median(x)) if x.size else np.nan
            rows.append(row)
    T = pd.DataFrame(rows)
    cols = [c for c in desc if c in T]
    Z = T[cols].apply(lambda s: (s - s.median()) / (s.quantile(0.75) - s.quantile(0.25) + 1e-9)).fillna(0.0)
    k = int(m9.cfg["data"]["domains"]["k"])
    T["domain"] = KMeans(n_clusters=k, n_init=10, random_state=int(m9.cfg["seed"])).fit_predict(Z.to_numpy())
    dom = np.full((H, W), -1, np.int16)
    for r in T.itertuples():
        dom[r.r0:r.r0 + bk, r.c0:r.c0 + bk] = r.domain
    dom[~valid] = -1
    return dom, T
