"""G14 data contract: modality registry, ordered channel schema, availability masks, permitted known-fault context,
train-only normalisation, provenance manifest and a contract checker.

Sources of channels (no channel is invented; absent sources are skipped and recorded)
-----------------------------------------------------------------------------------
``raw``    GeoDAWN bands through ``gems_bench.g13.contracts.GeoData`` (canonical band names, grid-checked)
``g13``    label-free features of the G13 feature store (``data.g13_features_dir``), reused, not recomputed
``g14``    label-free composites defined here (aspect vector, curvature, roughness, MSDI / GSDI / TLI / CSCI,
           conductivity-depth alignment). The four index names come from the G14 brief and have no external
           definition; their formulas below are G14 definitions, documented in docs/g14/DATA_CONTRACT.md.
``known``  geometry of the *permitted* known-fault map of a unit (``gems_bench.p6.stack.known_features``):
           geo fold k -> faults outside fold k and its buffer; discovery split d -> retained faults;
           full region / submission -> every mapped fault (``existing_faults.tif`` is a test-time input).

Normalisation
-------------
Stored channels carry a label-free global affine pre-scaling (median / IQR of all valid pixels, needed so float16
storage does not overflow on e.g. earthquake densities). The *model* sees per-unit robust standardisation fitted on
that unit's training pixels only (``UnitNormalizer``). An affine map composed with a train-fitted affine map is a
train-fitted affine map, so no held-out statistic reaches the model.

UNTESTED: written without execution.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

from . import SCHEMA_VERSION

# transform kinds under rotations / flips (multiscale.augment)
SCALAR, AXIAL_COS, AXIAL_SIN, VEC_X, VEC_Y = "scalar", "axial_cos", "axial_sin", "vec_x", "vec_y"
FLIP_UNSAFE_RAW = {"tmi", "mag_anom", "tmi_vg", "tmi_hg"}          # asymmetric total-field products (inclination ~63)


@dataclass
class Channel:
    name: str
    modality: str
    source: str            # raw | g13 | g14 | known
    kind: str = SCALAR
    resample: str = "nanmean"
    flip_safe: bool = True
    formula: str = ""
    native_resolution_m: float = 100.0


# G13 feature names per modality (used only if present in the G13 registry)
G13_CHANNELS = {
    "magnetic": ["magnetic:rtp:thd", "magnetic:rtp:vd", "magnetic:rtp:tilt", "magnetic:rtp:tdx", "magnetic:rtp:asig",
                 "magnetic:rtp:nss", "magnetic:rtp:uc2000_thd", "magnetic:rtp:coh_s800", "magnetic:rtp:coh_s1600",
                 "magnetic:rtp:phase_congruency", "magnetic:rtp:xscale_consistency",
                 ("magnetic:rtp:pf_strike_cos2", AXIAL_COS), ("magnetic:rtp:pf_strike_sin2", AXIAL_SIN)],
    "gravity": ["gravity:iso_grav_anom:thd", "gravity:iso_grav_anom:tilt", "gravity:iso_grav_anom:tdx",
                "gravity:iso_grav_anom:uc2000_thd", "gravity:iso_grav_anom:coh_s800",
                "gravity:iso_grav_anom:phase_congruency",
                ("gravity:iso_grav_anom:pf_strike_cos2", AXIAL_COS), ("gravity:iso_grav_anom:pf_strike_sin2", AXIAL_SIN)],
    "terrain": ["terrain:det_elev:slope_break_s400", "terrain:det_elev:slope_break_s800",
                "terrain:det_elev:valley_s800", "terrain:det_elev:ridge_s800", "terrain:det_elev:relief_aniso_s800",
                "terrain:det_elev:tpi_s5", "terrain:det_elev:tpi_s20", "terrain:det_elev:lstd_s20"],
    "conductivity": ["conductivity:cond_surf:grad_s800", "conductivity:cond_surf:coh_s800"],
    "depth": ["depth:depth_to_base_surf:grad_s800"],
    "structural": ["structural:rtp:orient_persistence", "structural:iso_grav_anom:orient_persistence",
                   "structural:det_elev:orient_persistence", "structural:rtp:across_asym_d1000",
                   "structural:iso_grav_anom:across_asym_d1000", "structural:rtp:char_scale",
                   "cross_modal:magn_grav:axial_alignment", "cross_modal:magn_grav:joint_edge",
                   "cross_modal:all:agreement_frac", "cross_modal:terr_grav:joint_edge"],
}
RAW_BY_MODALITY = {
    "magnetic": ["rtp", "tmi_hg", "tmi_vg", "tmi", "mag_anom", "tc"],
    "gravity": ["iso_grav_anom", "iso_grav_anom_hg", "iso_grav_anom_vg", "iso_grav_anom_slope"],
    "terrain": ["det_elev", "det_elev_slope"],
    "conductivity": ["cond_surf"],
    "depth": ["depth_to_base_surf"],
    "geodesy": ["geod_shearrate", "geod_dilaterate", "geod_2ndinv"],
    "seismicity": ["deq_n100a15", "ieq_n100a15"],
}
AVAILABILITY_BAND = {"magnetic": "rtp", "gravity": "iso_grav_anom", "terrain": "det_elev", "conductivity": "cond_surf",
                     "depth": "depth_to_base_surf", "geodesy": "geod_shearrate", "seismicity": "deq_n100a15"}
KNOWN_CHANNELS = [Channel("known:dist_km", "known_context", "known", formula="min(EDT to permitted known faults, 20 km)"),
                  Channel("known:density", "known_context", "known", formula="known-pixel fraction in a 4 km box"),
                  Channel("known:tip_dist_km", "known_context", "known", formula="distance to nearest trace tip"),
                  Channel("known:tip_along_cos", "known_context", "known",
                          formula="cos(tip->pixel, trace direction at tip), 0 beyond 5 km"),
                  Channel("known:strike_cos2", "known_context", "known", AXIAL_COS,
                          formula="cos 2theta of the nearest permitted known trace"),
                  Channel("known:strike_sin2", "known_context", "known", AXIAL_SIN,
                          formula="sin 2theta of the nearest permitted known trace")]
G14_FORMULAS = {
    "g14:aspect_x": "east component of the unit downslope vector of det_elev (Gaussian sigma 2 px)",
    "g14:aspect_y": "north component of the unit downslope vector of det_elev",
    "g14:curvature": "Laplacian of Gaussian-smoothed det_elev (sigma 2 px), per 100 m^2",
    "g14:roughness": "std of det_elev residual (det_elev - G_4px det_elev) in a 5 px window",
    "g14:msdi": "MSDI := rank(magnetic tdx) * coh_s800(rtp) * orient_persistence(rtp)  [G14 definition]",
    "g14:gsdi": "GSDI := rank(gravity tdx) * coh_s800(iso_grav_anom) * orient_persistence(iso_grav)  [G14 definition]",
    "g14:tli": "TLI := rank(slope_break_s800) * relief_aniso_s800 * orient_persistence(det_elev)  [G14 definition]",
    "g14:csci": "CSCI := rank(cond grad_s800) * coh_s800(cond_surf)  [G14 definition]",
    "g14:cond_depth_alignment": "cos^2(dtheta) * min(coh) of cond_surf and depth_to_base_surf structure tensors (800 m)",
}
G14_DEPENDENCIES = {
    "g14:msdi": ("magnetic:rtp:tdx", "magnetic:rtp:coh_s800", "structural:rtp:orient_persistence"),
    "g14:gsdi": ("gravity:iso_grav_anom:tdx", "gravity:iso_grav_anom:coh_s800",
                 "structural:iso_grav_anom:orient_persistence"),
    "g14:tli": ("terrain:det_elev:slope_break_s800", "terrain:det_elev:relief_aniso_s800",
                "structural:det_elev:orient_persistence"),
    "g14:csci": ("conductivity:cond_surf:grad_s800", "conductivity:cond_surf:coh_s800"),
}
G14_RAW_DEPENDENCIES = {"g14:aspect_x": ("det_elev",), "g14:aspect_y": ("det_elev",), "g14:curvature": ("det_elev",),
                        "g14:roughness": ("det_elev",), "g14:cond_depth_alignment": ("cond_surf", "depth_to_base_surf")}
G14_MODALITY = {"g14:aspect_x": "terrain", "g14:aspect_y": "terrain", "g14:curvature": "terrain",
                "g14:roughness": "terrain", "g14:msdi": "magnetic", "g14:gsdi": "gravity", "g14:tli": "terrain",
                "g14:csci": "conductivity", "g14:cond_depth_alignment": "conductivity"}


# =============================================================================== schema
def build_schema(data, cfg, g13_store=None) -> tuple[list[Channel], pd.DataFrame]:
    """Ordered channel list for ``cfg.data`` and a table of every candidate with its inclusion decision."""
    d = cfg.data
    mods = list(d.modalities)
    chans, log = [], []

    def add(ch: Channel, ok: bool, why: str):
        log.append({"name": ch.name, "modality": ch.modality, "source": ch.source, "included": ok, "reason": why})
        if ok:
            chans.append(ch)

    for m in mods:
        for b in RAW_BY_MODALITY.get(m, []):
            safe = b not in FLIP_UNSAFE_RAW
            ch = Channel(f"raw:{b}", m, "raw", flip_safe=safe, formula="GeoDAWN band")
            if not data.has(b):
                add(ch, False, "band absent")
            elif not safe and not d.include_flip_unsafe:
                add(ch, False, "flip-unsafe total-field product excluded (include_flip_unsafe=False)")
            else:
                add(ch, True, "raw band")
    if not d.raw_only:
        reg = set(g13_store.registry.name) if g13_store is not None and len(g13_store.registry) else set()
        for m in mods:
            for item in G13_CHANNELS.get(m, []):
                nm, kind = (item if isinstance(item, tuple) else (item, SCALAR))
                ch = Channel(nm, m, "g13", kind, formula="G13 feature store (label-free)")
                add(ch, nm in reg, "present in G13 registry" if nm in reg else "G13 feature not built")
        for nm, m in G14_MODALITY.items():
            if m in mods:
                kind = VEC_X if nm == "g14:aspect_x" else VEC_Y if nm == "g14:aspect_y" else SCALAR
                deps = [x for x in G14_DEPENDENCIES.get(nm, ()) if x]
                ok_dep = all(x in reg for x in deps)
                raw_dep = all(data.has(b) for b in G14_RAW_DEPENDENCIES.get(nm, ()))
                add(Channel(nm, m, "g14", kind, formula=G14_FORMULAS[nm]), ok_dep and raw_dep,
                    "G14 composite (label-free)" if ok_dep and raw_dep else f"missing inputs {deps}")
    if d.known_context and "known_context" in mods:
        for ch in KNOWN_CHANNELS:
            add(ch, True, "permitted known-fault geometry (per unit)")
    return chans, pd.DataFrame(log)


def schema_hash(chans: list[Channel]) -> str:
    s = json.dumps([SCHEMA_VERSION] + [(c.name, c.kind, c.source) for c in chans])
    return hashlib.sha256(s.encode()).hexdigest()[:12]


def modality_groups(chans: list[Channel]) -> dict[str, list[int]]:
    """modality -> channel indices (ordered as in the schema)."""
    out: dict[str, list[int]] = {}
    for i, c in enumerate(chans):
        out.setdefault(c.modality, []).append(i)
    return out


# =============================================================================== label-free G14 composites
def _rank(a, valid, n_ref=400_000, seed=0):
    from ..g13.features import _rank01

    return _rank01(a, valid, n_ref, seed)


def compute_g14_channel(name: str, data, g13_store) -> np.ndarray:
    from ..g13 import signal as S

    v = data.valid

    def g13(nm):
        return g13_store.physical(nm) if g13_store is not None and g13_store.has(nm) else None

    def ones():
        return np.ones(data.shape, np.float32)

    if name in ("g14:aspect_x", "g14:aspect_y", "g14:curvature", "g14:roughness"):
        z = data.band("det_elev")
        vz = v & np.isfinite(z)
        f = S.fill_nodata(z, vz)
        if name in ("g14:aspect_x", "g14:aspect_y"):
            gx, gy = S.gaussian_gradient(f, 2.0)
            n = np.hypot(gx, gy) + 1e-9
            a = -gx / n if name == "g14:aspect_x" else -gy / n
        elif name == "g14:curvature":
            a = ndimage.gaussian_laplace(f, 2.0)
        else:
            res = f - ndimage.gaussian_filter(f, 4.0)
            m = ndimage.uniform_filter(res, 5)
            a = np.sqrt(np.maximum(ndimage.uniform_filter(res * res, 5) - m * m, 0))
        return S.remask(a, vz)
    if name == "g14:cond_depth_alignment":
        out = []
        for b in ("cond_surf", "depth_to_base_surf"):
            x = data.band(b)
            vb = v & np.isfinite(x)
            coh, th, _ = S.structure_tensor(S.fill_nodata(x, vb), 4.0, 4.0)
            out.append((coh, th, vb))
        (c1, t1, v1), (c2, t2, v2) = out
        return S.remask(np.cos(S.axial_diff(t1, t2)) ** 2 * np.minimum(c1, c2), v1 & v2)
    spec = {"g14:msdi": ("magnetic:rtp:tdx", "magnetic:rtp:coh_s800", "structural:rtp:orient_persistence"),
            "g14:gsdi": ("gravity:iso_grav_anom:tdx", "gravity:iso_grav_anom:coh_s800",
                         "structural:iso_grav_anom:orient_persistence"),
            "g14:tli": ("terrain:det_elev:slope_break_s800", "terrain:det_elev:relief_aniso_s800",
                        "structural:det_elev:orient_persistence"),
            "g14:csci": ("conductivity:cond_surf:grad_s800", "conductivity:cond_surf:coh_s800", None)}
    if name in spec:
        a_nm, b_nm, c_nm = spec[name]
        a, b = g13(a_nm), g13(b_nm)
        if a is None or b is None:
            return np.full(data.shape, np.nan, np.float32)
        c = g13(c_nm) if c_nm else ones()
        c = ones() if c is None else c
        return S.remask(_rank(a, v) * np.clip(b, 0, 1) * np.clip(c, 0, 1), v & np.isfinite(a))
    raise KeyError(name)


# =============================================================================== known-fault context
def known_context(permitted: np.ndarray, valid: np.ndarray) -> dict[str, np.ndarray]:
    """Permitted-known geometry (P6 ``known_features`` + strike of the nearest trace)."""
    from ..g13 import signal as S
    from ..p6.stack import known_features

    k = known_features(permitted, valid)
    out = {"known:dist_km": k["known_dist_km"], "known:density": k["known_density"],
           "known:tip_dist_km": k["tip_dist_km"], "known:tip_along_cos": k["tip_along_cos"]}
    if permitted.any():
        _, th, _ = S.structure_tensor(permitted.astype(np.float32), 1.0, 3.0)
        _, (ir, ic) = ndimage.distance_transform_edt(~permitted, return_indices=True)
        t = th[ir, ic]
        out["known:strike_cos2"], out["known:strike_sin2"] = np.cos(2 * t).astype(np.float32), np.sin(2 * t).astype(np.float32)
    else:
        out["known:strike_cos2"] = np.zeros(permitted.shape, np.float32)
        out["known:strike_sin2"] = np.zeros(permitted.shape, np.float32)
    for kk in out:
        out[kk] = np.where(valid, out[kk], np.nan).astype(np.float32)
    return out


# =============================================================================== the stored stack
class ChannelStack:
    """``<root>/stack_l<f>.npy`` (C, H/f, W/f) float16 for f in (1, regional factor, global factor), availability
    masks ``mask_l<f>.npy`` (M, H/f, W/f) uint8 (fraction valid x 255), ``manifest.json``.

    Known-context channels are NOT stored here (they depend on the unit); ``multiscale.UnitInputs`` adds them."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text()) if (self.root / "manifest.json").exists() \
            else None
        self._mm: dict = {}

    @property
    def channels(self) -> list[Channel]:
        return [Channel(**c) for c in self.manifest["channels"]]

    @property
    def modalities(self) -> list[str]:
        return list(self.manifest["mask_modalities"])

    @property
    def factors(self) -> list[int]:
        return [int(f) for f in self.manifest["factors"]]

    def level(self, f: int) -> np.ndarray:
        key = f"stack_l{f}"
        if key not in self._mm:
            self._mm[key] = np.load(self.root / f"{key}.npy", mmap_mode="r")
        return self._mm[key]

    def masks(self, f: int) -> np.ndarray:
        key = f"mask_l{f}"
        if key not in self._mm:
            self._mm[key] = np.load(self.root / f"{key}.npy", mmap_mode="r")
        return self._mm[key]

    def exists(self) -> bool:
        return self.manifest is not None and all((self.root / f"stack_l{f}.npy").exists() for f in self.factors)


def block_nanmean(a: np.ndarray, f: int) -> np.ndarray:
    """NaN-aware f x f block mean (output shape ceil(H/f), ceil(W/f)); all-NaN blocks stay NaN."""
    if f == 1:
        return np.asarray(a, np.float32)
    h, w = a.shape
    H, W = -(-h // f) * f, -(-w // f) * f
    p = np.full((H, W), np.nan, np.float32)
    p[:h, :w] = a
    b = p.reshape(H // f, f, W // f, f)
    fin = np.isfinite(b)
    s = np.where(fin, b, 0).sum((1, 3))
    n = fin.sum((1, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, s / np.maximum(n, 1), np.nan).astype(np.float32)


def block_fraction(m: np.ndarray, f: int) -> np.ndarray:
    if f == 1:
        return np.asarray(m, np.float32)
    h, w = m.shape
    H, W = -(-h // f) * f, -(-w // f) * f
    p = np.zeros((H, W), np.float32)
    p[:h, :w] = m
    return p.reshape(H // f, f, W // f, f).mean((1, 3)).astype(np.float32)


def build_stack(data, cfg, root: str | Path, g13_store=None, log=print) -> ChannelStack:
    """Write the label-free channel stack and its pyramids. Reuses an existing stack with the same schema hash.

    Peak RAM: one full-resolution channel plus the output memmaps (never the whole stack in memory)."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    chans, decisions = build_schema(data, cfg, g13_store)
    stored = [c for c in chans if c.source != "known"]
    sh = schema_hash(chans)
    factors = [1, int(round(cfg.data.regional.resolution_m / cfg.data.local.resolution_m)),
               int(round(cfg.data.global_.resolution_m / cfg.data.local.resolution_m))]
    old = ChannelStack(root)
    if old.exists() and old.manifest.get("schema_hash") == sh:
        log(f"[contract] stack reused ({len(stored)} stored channels, schema {sh})")
        return old
    mods = [m for m in cfg.data.modalities if m != "known_context"]
    shapes = {f: (-(-data.shape[0] // f), -(-data.shape[1] // f)) for f in factors}
    outs = {f: np.lib.format.open_memmap(root / f"stack_l{f}.partial.npy", mode="w+", dtype=cfg.data.store_dtype,
                                         shape=(len(stored), *shapes[f])) for f in factors}
    scale, t0 = [], time.perf_counter()
    for i, c in enumerate(stored):
        if c.source == "raw":
            a = data.band(c.name[4:])
        elif c.source == "g13":
            a = g13_store.physical(c.name)
        else:
            a = compute_g14_channel(c.name, data, g13_store)
        a = np.where(data.valid, a, np.nan).astype(np.float32)
        if c.kind == SCALAR:
            fin = a[np.isfinite(a)]
            sub = fin[:: max(fin.size // 500_000, 1)] if fin.size else fin
            med = float(np.median(sub)) if sub.size else 0.0
            iqr = float(np.subtract(*np.percentile(sub, [75, 25]))) if sub.size else 1.0
            iqr = iqr or float(np.std(sub)) or 1.0
            a = np.clip((a - med) / iqr, -1e4, 1e4)
        else:
            med, iqr = 0.0, 1.0                          # unit-vector / cos-sin components keep their range
        scale.append({"name": c.name, "prescale_median": med, "prescale_iqr": iqr})
        for f in factors:
            outs[f][i] = block_nanmean(a, f).astype(cfg.data.store_dtype)
        log(f"[contract] {i + 1}/{len(stored)} {c.name}")
    for f in factors:
        outs[f].flush()
        del outs[f]
    masks = {}
    for f in factors:
        mm = np.zeros((len(mods), *shapes[f]), np.uint8)
        for j, m in enumerate(mods):
            b = AVAILABILITY_BAND.get(m)
            ok = data.valid & np.isfinite(data.band(b)) if b and data.has(b) else data.valid.copy()
            mm[j] = np.round(block_fraction(ok, f) * 255).astype(np.uint8)
        masks[f] = mm
    for f in factors:
        (root / f"stack_l{f}.partial.npy").replace(root / f"stack_l{f}.npy")
        np.save(root / f"mask_l{f}.npy", masks[f])
    t = data.transform
    manifest = {"schema_version": SCHEMA_VERSION, "schema_hash": sh, "created_unix": time.time(),
                "seconds": time.perf_counter() - t0, "channels": [asdict(c) for c in stored],
                "known_channels": [asdict(c) for c in chans if c.source == "known"], "mask_modalities": mods,
                "factors": factors, "shape": list(data.shape), "crs": str(data.crs), "transform": list(t)[:6],
                "level_transforms": {str(f): [t.a * f, t.b, t.c, t.d, t.e * f, t.f] for f in factors},
                "prescale": scale, "dtype": cfg.data.store_dtype, "raw_path": data.raw_path,
                "g13_store": str(g13_store.root) if g13_store is not None else None,
                "label_free": True, "resampling": "nan-aware block mean of 100 m values; masks = valid fraction; "
                "coarse levels are aggregates of 100 m data, not higher-resolution information"}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    decisions.to_csv(root / "channel_decisions.csv", index=False)
    return ChannelStack(root)


# =============================================================================== train-only normalisation
class UnitNormalizer:
    """Per-channel robust (median, IQR) fitted on the training pixels of ONE unit; axial / vector channels pass."""

    def __init__(self, med: np.ndarray, iqr: np.ndarray, kinds: list[str], fitted_on: dict):
        self.med, self.iqr, self.kinds, self.fitted_on = med, iqr, kinds, fitted_on

    @classmethod
    def fit(cls, sample: np.ndarray, kinds: list[str], fitted_on: dict) -> "UnitNormalizer":
        """``sample``: (n_pixels, C) values drawn from the unit's training mask only."""
        med = np.nanmedian(sample, 0)
        q75, q25 = np.nanpercentile(sample, 75, 0), np.nanpercentile(sample, 25, 0)
        iqr = np.where((q75 - q25) > 1e-6, q75 - q25, 1.0)
        med = np.where(np.isfinite(med), med, 0.0)
        for i, k in enumerate(kinds):
            if k != SCALAR:
                med[i], iqr[i] = 0.0, 1.0
        return cls(med.astype(np.float32), iqr.astype(np.float32), kinds, fitted_on)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """x: (C, h, w) -> standardised, clipped to [-10, 10]; NaN preserved (the dataset turns NaN into 0 + mask)."""
        y = (x - self.med[:, None, None]) / self.iqr[:, None, None]
        return np.clip(y, -10, 10).astype(np.float32)

    def to_json(self) -> dict:
        return {"median": self.med.tolist(), "iqr": self.iqr.tolist(), "kinds": self.kinds, "fitted_on": self.fitted_on}


# =============================================================================== checker
def check_contract(stack: ChannelStack, data, cfg, paths: dict | None = None) -> pd.DataFrame:
    """Static and numeric checks of a built stack. Returns a table (check, status, detail). Never fixes anything."""
    rows = []

    def chk(name, ok, detail="", warn=False):
        rows.append({"check": name, "status": "PASS" if ok else ("WARN" if warn else "FAIL"), "detail": str(detail)[:300]})

    m = stack.manifest
    chk("manifest present", m is not None)
    if m is None:
        return pd.DataFrame(rows)
    if paths:
        import rasterio

        from ..g13.contracts import check_grid

        try:
            with rasterio.open(paths["raw"]) as r, rasterio.open(paths["known"]) as k, rasterio.open(paths["template"]) as t:
                check_grid(r, k, t)
            chk("raster alignment (raw / known / template)", True)
        except Exception as e:  # noqa: BLE001
            chk("raster alignment (raw / known / template)", False, e)
    chk("grid shape matches data", list(data.shape) == list(m["shape"]), m["shape"])
    chk("transform matches data", np.allclose(list(data.transform)[:6], m["transform"]), m["transform"])
    chk("CRS EPSG:32611", "32611" in m["crs"], m["crs"])
    chans, _ = build_schema(data, cfg, None if not m.get("g13_store") else _store(m["g13_store"], data))
    want = [c.name for c in chans if c.source != "known"]
    got = [c["name"] for c in m["channels"]]
    chk("channel order reproducible from config", want == got, f"{len(got)} stored")
    chk("schema hash", m["schema_hash"] == schema_hash(chans), m["schema_hash"])
    chk("label-free stored channels", all(c["source"] != "known" for c in m["channels"]))
    for f in stack.factors:
        a = stack.level(f)
        exp = (len(got), -(-data.shape[0] // f), -(-data.shape[1] // f))
        chk(f"level {f} shape", tuple(a.shape) == exp, a.shape)
        mk = stack.masks(f)
        chk(f"level {f} mask shape", mk.shape[1:] == a.shape[1:], mk.shape)
    a1 = stack.level(1)
    rng = np.random.default_rng(0)
    vi = np.flatnonzero(data.valid.ravel())
    ids = rng.choice(vi, min(20000, vi.size), replace=False) if vi.size else vi
    for i, c in enumerate(m["channels"]):
        x = np.asarray(a1[i].reshape(-1)[ids], np.float32)
        fin = float(np.isfinite(x).mean()) if x.size else 0.0
        chk(f"finite inside valid: {c['name']}", fin > 0.5, f"{fin:.3f}", warn=fin > 0.05)
        out = np.asarray(a1[i].reshape(-1)[np.flatnonzero(~data.valid.ravel())[:20000]], np.float32)
        chk(f"NaN outside valid: {c['name']}", not np.isfinite(out).any(), int(np.isfinite(out).sum()))
    if cfg.data.augment:
        unsafe = [c["name"] for c in m["channels"] if not c["flip_safe"]]
        chk("no flip-unsafe channel with D4 augmentation", not unsafe, unsafe)
    return pd.DataFrame(rows)


def _store(path, data):
    from ..g13.features import FeatureStore

    return FeatureStore(path, data)
