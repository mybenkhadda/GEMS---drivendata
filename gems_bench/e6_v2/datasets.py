"""Normalised model inputs on the padded 100 m grid, per-unit training targets and the patch sampler.

``datasets/`` (Drive, persistent):
  terrain.npy (Ct, Hp, Wp) f16   field.npy (Cf, Hp, Wp) f16   dem.npy (Cd, Hp, Wp) f16
  dem_coverage.npy (Hp, Wp) f32  context.npy (Ct+Cf, Hp/8, Wp/8) f32 (8 x 8 mean of terrain + field)
  labels.npy / valid.npy / template_valid.npy (Hp, Wp) u1   dataset_schema.json   manifest.json
Hp, Wp = H, W rounded up to a multiple of ``model.context_pool``; padding is zero / invalid.
DEM rule: wherever any DEM statistic is NaN the coverage is forced to 0 (the cell then falls back exactly to the
reference model) and the DEM inputs are 0; the counts are written to dataset_schema.json.
"""
from __future__ import annotations

import logging
import shutil

import numpy as np
from scipy import ndimage

from . import augmentation
from .cache import (MissingArtifact, atomic_save_npy, atomic_write_json, load_npy, manifest_status, read_json,
                    write_manifest)
from .config import module_hash, stable_hash
from .contracts import DEM_CHANNELS, schema, schema_hash
from .normalization import apply as norm_apply
from .paths import E6V2Paths

log = logging.getLogger("e6v2.datasets")
ARRAYS = ("terrain", "field", "dem", "dem_coverage", "context", "labels", "valid", "template_valid")


def _pad(a: np.ndarray, Hp: int, Wp: int, value=0):
    h, w = a.shape[-2:]
    pad = [(0, 0)] * (a.ndim - 2) + [(0, Hp - h), (0, Wp - w)]
    return np.pad(a, pad, constant_values=value)


def datasets_fingerprint(paths: E6V2Paths, cfg: dict) -> str:
    fps = []
    for man in (paths.terrain / "manifest.json", paths.potential / "manifest.json",
                paths.dem_dir("manifests") / "dem_manifest.json", paths.base / "manifest.json"):
        m = read_json(man)
        if m is None:
            raise MissingArtifact(f"{paths.rel(man)} missing - run sections 04-06 first")
        fps.append(m["fingerprint"])
    return stable_hash([fps, schema_hash(cfg), cfg["model"]["context_pool"], module_hash("datasets", "normalization")])


def prepare_datasets(paths: E6V2Paths, cfg: dict, force: bool = False) -> dict:
    from .geophysics import load_base

    man = paths.datasets / "manifest.json"
    fp = datasets_fingerprint(paths, cfg)
    if not force:
        st, why = manifest_status(man, fp, paths.root)
        if st == "complete":
            return read_json(man)
        log.info("datasets: %s (%s)", st, why)
    base = load_base(paths)
    H, W = base["valid"].shape
    pool = int(cfg["model"]["context_pool"])
    Hp, Wp = -(-H // pool) * pool, -(-W // pool) * pool
    sch = schema(cfg)
    g = sch["groups"]
    norm = {**read_json(paths.terrain / "normalization.json"), **read_json(paths.potential / "normalization.json"),
            **read_json(paths.dem_dir("features_100m") / "normalization.json")}
    valid = base["valid"]
    outs = []

    def build(names, sources) -> np.ndarray:
        out = np.zeros((len(names), Hp, Wp), np.float16)
        i = 0
        for src_path, src_names in sources:
            arr = load_npy(src_path, mmap=True)
            if arr.shape[0] != len(src_names):
                raise RuntimeError(f"{paths.rel(src_path)} has {arr.shape[0]} channels, schema {len(src_names)}")
            for j, n in enumerate(src_names):
                if n != names[i]:
                    raise RuntimeError(f"channel order: {n} != {names[i]}")
                x = norm_apply(np.asarray(arr[j]), norm[n])
                x[~valid] = 0
                out[i, :H, :W] = x
                i += 1
        return out

    terrain = build(g["terrain"], [(paths.terrain / "feature_stack.npy", g["terrain"])])
    pm = read_json(paths.potential / "manifest.json")["channels"]
    field = build(g["field"], [(paths.potential / "magnetic_features.npy", pm["magnetic"]),
                               (paths.potential / "gravity_features.npy", pm["gravity"]),
                               (paths.potential / "subsurface_features.npy", pm["subsurface"])])
    dem_raw = load_npy(paths.dem_dir("features_100m") / "dem_features_100m.npy", mmap=True)
    cov = np.array(load_npy(paths.dem_dir("coverage_100m") / "dem_coverage_100m.npy"), np.float32)
    any_nan = np.zeros((H, W), bool)
    for j in range(dem_raw.shape[0]):
        any_nan |= ~np.isfinite(np.asarray(dem_raw[j]))
    forced = int(((cov > 0) & any_nan).sum())
    cov[any_nan | ~valid] = 0.0
    dem = build(list(DEM_CHANNELS), [(paths.dem_dir("features_100m") / "dem_features_100m.npy", list(DEM_CHANNELS))])
    dem[:, :H, :W][:, cov == 0] = 0
    context = np.stack([ch.astype(np.float32).reshape(Hp // pool, pool, Wp // pool, pool).mean(axis=(1, 3))
                        for grp in (terrain, field) for ch in grp]).astype(np.float32)
    d = paths.datasets
    arrays = {"terrain": terrain, "field": field, "dem": dem, "dem_coverage": _pad(cov, Hp, Wp, 0.0),
              "context": context, "labels": _pad(base["labels"] & valid, Hp, Wp, False).astype(np.uint8),
              "valid": _pad(valid, Hp, Wp, False).astype(np.uint8),
              "template_valid": _pad(base["template_valid"], Hp, Wp, False).astype(np.uint8)}
    for name, a in arrays.items():
        outs.append(atomic_save_npy(d / f"{name}.npy", a))
    info = {**sch, "schema_hash": schema_hash(cfg), "grid_shape": [H, W], "padded_shape": [Hp, Wp],
            "grid": base["meta"], "dem_cells_forced_uncovered": forced,
            "dem_cells_covered": int((cov > 0).sum()), "normalization": {k: norm[k] for k in
                                                                          g["terrain"] + g["field"] + g["dem"]},
            "normalization_hash": stable_hash({k: norm[k] for k in g["terrain"] + g["field"] + g["dem"]})}
    outs.append(atomic_write_json(d / "dataset_schema.json", info))
    return write_manifest(man, stage="datasets", fingerprint=fp, outputs=outs, root=paths.root,
                          extra={"schema_hash": info["schema_hash"], "normalization_hash": info["normalization_hash"],
                                 "dem_cells_forced_uncovered": forced})


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
class E6Data:
    """All dataset arrays in RAM (about 2.5 GB), staged to local scratch first (Drive reads are slow)."""

    def __init__(self, paths: E6V2Paths, cfg: dict):
        man = read_json(paths.datasets / "manifest.json")
        if man is None:
            raise MissingArtifact("datasets/manifest.json missing - run section 07")
        st, why = manifest_status(paths.datasets / "manifest.json", man["fingerprint"], paths.root)
        if st != "complete":
            raise MissingArtifact(f"datasets incomplete ({why}) - re-run section 07")
        local = paths.scratch_root / "datasets"
        local.mkdir(parents=True, exist_ok=True)
        stamp = local / "fingerprint.txt"
        if not stamp.exists() or stamp.read_text().strip() != man["fingerprint"]:
            for n in ARRAYS:
                shutil.copyfile(paths.datasets / f"{n}.npy", local / f"{n}.npy")
            stamp.write_text(man["fingerprint"])
        for n in ARRAYS:
            src = paths.datasets / f"{n}.npy"
            if (local / f"{n}.npy").stat().st_size != src.stat().st_size:
                shutil.copyfile(src, local / f"{n}.npy")
            setattr(self, n, load_npy(local / f"{n}.npy"))
        self.labels = self.labels.astype(bool)
        self.valid = self.valid.astype(bool)
        self.template_valid = self.template_valid.astype(bool)
        self.schema = read_json(paths.datasets / "dataset_schema.json")
        self.fingerprint = man["fingerprint"]
        self.H, self.W = self.schema["grid_shape"]
        self.Hp, self.Wp = self.schema["padded_shape"]
        self.pool = int(self.schema["pool"])
        self.dem_axial = [tuple(p) for p in self.schema["axial_pairs"]["dem"]]

    def inputs(self, r0: int, c0: int, P: int, ext: int) -> dict:
        """Input crops for a P x P patch at (r0, c0) (multiples of ``pool``) and its pooled context crop."""
        sl = (slice(r0, r0 + P), slice(c0, c0 + P))
        pc = P // self.pool
        cs = pc * ext
        off = (cs - pc) // 2
        ctx = crop_boundless(self.context, r0 // self.pool - off, c0 // self.pool - off, cs, cs)
        return {"terrain": self.terrain[(slice(None),) + sl], "field": self.field[(slice(None),) + sl],
                "dem": self.dem[(slice(None),) + sl], "cov": self.dem_coverage[sl][None], "ctx": ctx}


def crop_boundless(a: np.ndarray, r0: int, c0: int, h: int, w: int) -> np.ndarray:
    out = np.zeros(a.shape[:-2] + (h, w), a.dtype)
    H, W = a.shape[-2:]
    rs, cs = max(r0, 0), max(c0, 0)
    re, ce = min(r0 + h, H), min(c0 + w, W)
    if re > rs and ce > cs:
        out[..., rs - r0:re - r0, cs - c0:ce - c0] = a[..., rs:re, cs:ce]
    return out


# ---------------------------------------------------------------------------
# per-unit targets
# ---------------------------------------------------------------------------
def label_strike(visible: np.ndarray, sigma_px: float = 1.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Doubled-angle strike (cos 2t, sin 2t) and coherence of the mapped traces (same convention as the DEM)."""
    v = ndimage.gaussian_filter(visible.astype(np.float32), sigma_px)
    gx = ndimage.sobel(v, axis=1)
    gy = ndimage.sobel(v, axis=0)
    jxx = ndimage.gaussian_filter(gx * gx, sigma_px)
    jxy = ndimage.gaussian_filter(gx * gy, sigma_px)
    jyy = ndimage.gaussian_filter(gy * gy, sigma_px)
    rr = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2)
    inv = 1.0 / np.maximum(rr, 1e-12)
    coh = (rr / (jxx + jyy + 1e-12)) ** 2
    return (-(jxx - jyy) * inv).astype(np.float32), (-2 * jxy * inv).astype(np.float32), coh.astype(np.float32)


class UnitTargets:
    """Weights and targets of one unit on the padded grid."""

    def __init__(self, data: E6Data, masks: dict, cfg: dict):
        t = cfg["training"]
        px = float(data.schema["grid"]["pixel_m"])
        Hp, Wp, H, W = data.Hp, data.Wp, data.H, data.W
        P = lambda m: _pad(m, Hp, Wp, False) if m.shape != (Hp, Wp) else m  # noqa: E731
        self.train = P(masks["train"])
        self.forbidden = P(masks["forbidden"])
        vis = P(masks["visible"])
        ign = P(masks["ignore"])
        R = float(t["loss"]["radius_m"])
        d_px, idx = ndimage.distance_transform_edt(~vis, return_indices=True) if vis.any() else \
            (np.full((Hp, Wp), np.inf), None)
        dist = (d_px * px).astype(np.float32)
        near = (dist > 0) & (dist <= float(t["ignore_m"]))
        w = self.train & ~ign & ~near
        self.pos = vis.astype(np.uint8)
        self.weight = w.astype(np.float32)
        self.soft = np.clip(1.0 - dist / R, 0.0, 1.0).astype(np.float16)
        self.prox = (np.minimum(dist, R) / R).astype(np.float16)
        self.prox_w = (self.train & ~ign & (dist <= float(t["proximity_neighborhood_m"]))).astype(np.uint8)
        if idx is not None:
            c2, s2, coh = label_strike(vis)
            near_c2, near_s2, near_coh = (a[idx[0], idx[1]] for a in (c2, s2, coh))
            self.orient = np.stack([near_c2, near_s2]).astype(np.float16)
            self.orient_w = (self.train & ~ign & (dist <= R) &
                             (near_coh >= float(t["orientation_min_coherence"]))).astype(np.uint8)
        else:
            self.orient = np.zeros((2, Hp, Wp), np.float16)
            self.orient_w = np.zeros((Hp, Wp), np.uint8)
        n_pos = float((vis & w).sum())
        n_w = float(w.sum())
        share = n_pos / max(n_w, 1.0)
        self.prior_observed = share
        self.prior = float(np.clip(share * float(t["prior_inflation"]), t["prior_min"], t["prior_max"]))
        self.label_density = float(vis.sum()) / max(float((self.train).sum()), 1.0)
        self.stats = {"train_px": int(self.train.sum()), "weighted_px": int(n_w), "visible_pos_px": int(vis.sum()),
                      "prior_observed": share, "prior": self.prior, "label_density": self.label_density,
                      "ignored_px": int((self.train & (ign | near)).sum())}

    def crops(self, r0, c0, P) -> dict:
        sl = (slice(r0, r0 + P), slice(c0, c0 + P))
        return {"pos": self.pos[sl][None], "weight": self.weight[sl][None], "soft": self.soft[sl][None],
                "prox": self.prox[sl][None], "prox_w": self.prox_w[sl][None],
                "orient": self.orient[(slice(None),) + sl], "orient_w": self.orient_w[sl][None]}


def _window_sums(mask: np.ndarray, P: int, step: int) -> np.ndarray:
    S = np.zeros((mask.shape[0] + 1, mask.shape[1] + 1), np.float64)
    S[1:, 1:] = mask.astype(np.float64).cumsum(0).cumsum(1)
    r = np.arange(0, mask.shape[0] - P + 1, step)
    c = np.arange(0, mask.shape[1] - P + 1, step)
    return S[r[:, None] + P, c[None] + P] - S[r[:, None], c[None] + P] - S[r[:, None] + P, c[None]] + S[r[:, None], c[None]]


class PatchSampler:
    """Patch positions on the ``pool`` grid lying entirely outside the unit's forbidden region."""

    def __init__(self, data: E6Data, tg: UnitTargets, cfg: dict, seed: int):
        t = cfg["training"]
        self.data, self.tg, self.cfg = data, tg, cfg
        self.P, self.step = int(t["patch_px"]), data.pool
        self.ext = int(cfg["model"]["context_extent"])
        self.rng = np.random.default_rng(seed)
        P, A = self.P, float(self.P * self.P)
        forb = _window_sums(tg.forbidden, P, self.step)
        trn = _window_sums(tg.train, P, self.step) / A
        pos = _window_sums(tg.pos.astype(bool) & (tg.weight > 0), P, self.step)
        cov = _window_sums(data.dem_coverage, P, self.step) / A
        ok = (forb == 0) & (trn >= float(t["min_valid_frac"]))
        rr, cc = np.nonzero(ok)
        self.all = np.stack([rr, cc], 1) * self.step
        rr, cc = np.nonzero(ok & (pos > 0))
        self.with_pos = np.stack([rr, cc], 1) * self.step
        rr, cc = np.nonzero(ok & (cov >= float(t["dem_patch_min_cov"])))
        self.with_dem = np.stack([rr, cc], 1) * self.step
        self.stats = {"candidates": len(self.all), "with_positives": len(self.with_pos), "with_dem": len(self.with_dem)}
        if not len(self.all):
            raise RuntimeError("no admissible training patch for this unit (forbidden region / min_valid_frac)")

    def get_state(self):
        return self.rng.bit_generator.state

    def set_state(self, s):
        self.rng.bit_generator.state = s

    def _pick(self, dem_mode: bool) -> tuple[int, int]:
        t = self.cfg["training"]
        if dem_mode and len(self.with_dem) and self.rng.random() < float(t["dem_patch_frac"]):
            pool = self.with_dem
        elif len(self.with_pos) and self.rng.random() < float(t["pos_patch_frac"]):
            pool = self.with_pos
        else:
            pool = self.all
        r, c = pool[self.rng.integers(len(pool))]
        return int(r), int(c)

    def batch(self, n: int, dem_mode: bool, dem_dropout: float) -> dict:
        items = []
        for _ in range(n):
            r0, c0 = self._pick(dem_mode)
            x = self.data.inputs(r0, c0, self.P, self.ext)
            y = self.tg.crops(r0, c0, self.P)
            k, flip = augmentation.draw(self.rng, self.cfg)
            out = {"terrain": augmentation.apply(x["terrain"], k, flip),
                   "field": augmentation.apply(x["field"], k, flip),
                   "dem": augmentation.apply(x["dem"], k, flip, self.data.dem_axial),
                   "cov": augmentation.apply(x["cov"], k, flip),
                   "ctx": augmentation.apply(x["ctx"], k, flip)}
            for key, v in y.items():
                out[key] = augmentation.apply(v, k, flip, [(0, 1)] if key == "orient" else ())
            out["gate_cov"] = out["cov"]
            if dem_dropout > 0 and self.rng.random() < dem_dropout:
                out["gate_cov"] = np.zeros_like(out["cov"])
                out["dem"] = np.zeros_like(out["dem"])
            items.append(out)
        return {k: np.stack([it[k] for it in items]).astype(np.float32) for k in items[0]}
