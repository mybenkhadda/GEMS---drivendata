"""Label-free full-grid layers of M8: raw-band access, physics operators, worms / agreement / context layers.

Layer names (no prefix)::

    <family>:<operator>@<scale_m>      physics operator at one Gaussian scale, e.g. ``magnetic:tilt_edge@400``
    worms:<family>                     worm response (persistence x normalised strength) of section 03b
    worms:<family>:<stat>              persistence | orient_stability | ridge_stability | scale_max_m | strike | mask
    agree:mag_grav@<tol_m>             tolerant magnetic-gravity agreement of section 04
    context:<band>                     regional percentile of a context band (section 04)
    class:surface_subsurface           section-04 crest classes (uint8 codes)

Physics layers are computed on demand and cached as ``.npy`` under ``cache/m8`` (key = M8 version + definition +
band provenance status), then memory-mapped: the 19 raw bands and the 139 cached features are never all loaded.
Layers produced by a section (worms, agreement, context) are registered with ``put``; a pointer file records the
key of the producing stage, so a stale layer is never served after its stage changed.

Derivative provenance (``derivative_audit``): provided derivative bands (tmi_hg / tmi_vg, iso_grav_anom_hg / _vg,
*_slope) are only used inside physical formulas when their definition is confirmed by the user (``bands.verified``)
or empirically consistent with the same derivative computed from the parent field (|Spearman| >= consistency_rho);
hgrad and vgrad are combined (analytic signal / tilt from bands) only when their scale factors agree within
``unit_ratio_tolerance``. Otherwise the band is PROVISIONAL and every *_band operator using it is refused (logged).
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..artifacts.atomic import atomic_save, atomic_write_json
from . import potential_fields as PF
from .workspace import M8_VERSION

_NAME = re.compile(r"^(?P<fam>[a-z_]+):(?P<op>[a-z_]+)@(?P<scale>[0-9.]+)$")


def safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def parse(name: str) -> tuple[str, str, float] | None:
    m = _NAME.match(name)
    return (m["fam"], m["op"], float(m["scale"])) if m else None


class LayerStore:
    def __init__(self, ctx, cfg: dict, cache_dir: Path, compat: dict | None = None, max_bands: int = 3):
        from ..pipeline.m6 import fingerprint

        self.ctx, self.cfg = ctx, cfg
        self.dir = Path(cache_dir) / "layers"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.compat = compat or {}
        self._bands: dict = {}
        self._order: list[str] = []
        self._max = int(max_bands)
        self._fp = fingerprint
        g = ctx.data.grid
        self.px_km = (float(g.pixel_km_y), float(g.pixel_km_x))
        self.pixel_m = 1000.0 * float(g.pixel_km)
        self.log: list[dict] = []
        self._fresh: set[str] = set()

    # -- raw bands ---------------------------------------------------------------
    @property
    def families(self) -> dict:
        return self.cfg["bands"]["families"]

    def has_band(self, name: str | None) -> bool:
        return bool(name) and name in self.ctx.data.band_names

    def band(self, name: str) -> np.ndarray:
        """One raw band (float32, NaN nodata), small LRU cache."""
        if name not in self._bands:
            self._bands[name] = self.ctx.data.read_band(name)
            self._order.append(name)
            while len(self._order) > self._max:
                self._bands.pop(self._order.pop(0), None)
        return self._bands[name]

    def field(self, fam: str) -> np.ndarray:
        return self.band(self.families[fam]["field"])

    def valid(self, fam: str | None = None) -> np.ndarray:
        v = np.asarray(self.ctx.data.valid, bool)
        return v & np.isfinite(self.field(fam)) if fam else v

    def release(self) -> None:
        self._bands.clear()
        self._order.clear()

    # -- scales / permissions ------------------------------------------------------
    def eff_res_m(self, fam: str) -> float:
        return float(self.families[fam]["effective_resolution_m"])

    def scales(self, fam: str, scales=None) -> list[float]:
        """Physics scales a family supports: >= min_scale_factor x its effective resolution (others are skipped)."""
        f = float(self.cfg["physics"]["min_scale_factor"])
        s = [float(x) for x in (scales or self.cfg["physics"]["scales_m"])]
        return [x for x in s if x >= f * self.eff_res_m(fam)] or [max(s)]

    def allowed(self, fam: str, op: str) -> tuple[bool, str]:
        spec = self.families.get(fam)
        if spec is None:
            return False, f"unknown family {fam}"
        if op not in (self.cfg["physics"]["operators"].get(fam) or []):
            return False, f"{op} not configured for {fam}"
        if op in PF.NEEDS_VDR and not spec.get("potential_field"):
            return False, f"{op} needs a potential field (VDR via |k|); {fam} is not one"
        if op.endswith("_band"):
            hg, vg = spec.get("hgrad"), spec.get("vgrad")
            if not (self.has_band(hg) and self.has_band(vg)):
                return False, f"{op}: derivative bands {hg}/{vg} not available"
            if not self.compat.get("combinable", {}).get(fam, False):
                return False, (f"{op}: {hg}/{vg} not verified as compatible derivatives of "
                               f"{spec.get('parent_of_derivatives')} (PROVISIONAL)")
        if not self.has_band(spec["field"]):
            return False, f"field band {spec['field']} missing"
        return True, "ok"

    # -- naming / keys -------------------------------------------------------------
    @staticmethod
    def name(fam: str, op: str, scale_m: float) -> str:
        return f"{fam}:{op}@{int(round(float(scale_m)))}"

    def _parse(self, name: str):
        """Physics-layer parse restricted to configured families ('agree:mag_grav@800' is a PRODUCED layer)."""
        p = parse(name)
        return p if p is not None and p[0] in self.families else None

    def _key(self, name: str) -> str:
        p = self._parse(name)
        if p is None:
            ptr = self._pointer(name)
            return ptr.get("key", "missing") if ptr else "missing"
        fam, op, sc = p
        spec = self.families.get(fam, {})
        ph = self.cfg["physics"]
        return self._fp(M8_VERSION, "layer", name, spec, ph["amplitude_floor_quantile"], ph["lcn_window_m"],
                        self.compat.get("status", {}).get(spec.get("hgrad"), ""),
                        self.compat.get("status", {}).get(spec.get("vgrad"), ""))

    def _path(self, name: str) -> Path:
        return self.dir / f"{safe(name)}__{self._key(name)}.npy"

    def _pointer(self, name: str) -> dict | None:
        p = self.dir / f"{safe(name)}.pointer.json"
        try:
            return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
        except (OSError, json.JSONDecodeError):
            return None

    # -- access ------------------------------------------------------------------
    def has(self, name: str) -> bool:
        p = self._parse(name)
        if p is not None:
            return self.allowed(p[0], p[1])[0]
        ptr = self._pointer(name)
        return bool(ptr) and self._path(name).exists()

    def get(self, name: str) -> np.ndarray:
        """Full-grid float32 layer (memory-mapped). Physics layers are computed on first use."""
        p = self._path(name)
        if p.exists() and (not self.cfg.get("overwrite_existing") or name in self._fresh):
            return np.load(p, mmap_mode="r")
        parsed = self._parse(name)
        if parsed is None:
            if p.exists():
                return np.load(p, mmap_mode="r")
            raise KeyError(f"layer {name} has not been produced (run its section first)")
        fam, op, sc = parsed
        ok, why = self.allowed(fam, op)
        if not ok:
            raise KeyError(f"layer {name} refused: {why}")
        t0 = time.perf_counter()
        a, info = self.compute(fam, op, sc)
        atomic_save(p, lambda t: np.save(t, a))
        atomic_write_json(p.with_suffix(".json"), {"name": name, "family": fam, "operator": op, "scale_m": sc,
                                                   "band": self.families[fam]["field"], "info": info,
                                                   "seconds": round(time.perf_counter() - t0, 1),
                                                   "m8_version": M8_VERSION})
        self.log.append({"layer": name, "seconds": round(time.perf_counter() - t0, 1), **{k: v for k, v in info.items()
                                                                                            if not isinstance(v, dict)}})
        self._fresh.add(name)
        del a
        return np.load(p, mmap_mode="r")

    def compute(self, fam: str, op: str, scale_m: float) -> tuple[np.ndarray, dict]:
        spec = self.families[fam]
        sig = PF.sigma_px_of(scale_m, self.pixel_m)
        ph = self.cfg["physics"]
        kw = {"px_km": self.px_km, "amp_floor_q": float(ph["amplitude_floor_quantile"]),
              "lcn_sigma_px": float(ph["lcn_window_m"]) / 2.0 / self.pixel_m}
        if op.endswith("_band"):
            a, info = PF.operator(op, None, sig, hg=self.band(spec["hgrad"]), vg=self.band(spec["vgrad"]), **kw)
        else:
            a, info = PF.operator(op, self.field(fam), sig, **kw)
        a = np.where(self.ctx.data.valid, a, np.nan).astype(np.float32)
        info.update({"family": fam, "band": spec["field"], "scale_m": float(scale_m),
                     "units": (self.cfg["bands"].get("units") or {}).get(spec["field"], "unknown (not verified)")})
        return a, info

    def put(self, name: str, a: np.ndarray, key: str, meta: dict | None = None) -> Path:
        """Register a layer produced by a section (``key`` = the producing stage fingerprint)."""
        atomic_write_json(self.dir / f"{safe(name)}.pointer.json", {"name": name, "key": key,
                                                                    "written": time.strftime("%Y-%m-%dT%H:%M:%S")})
        p = self._path(name)
        atomic_save(p, lambda t: np.save(t, np.asarray(a)))
        if meta:
            atomic_write_json(p.with_suffix(".json"), {"name": name, "key": key, **meta})
        return p

    def crop(self, name: str, u) -> np.ndarray:
        return np.asarray(self.get(name)[u.sl], np.float32)

    def window(self, name: str, r0: int, r1: int, c0: int, c1: int) -> np.ndarray:
        return np.asarray(self.get(name)[r0:r1, c0:c1], np.float32)


# =============================================================================== derivative provenance audit
def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import spearmanr

    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 100:
        return np.nan
    return float(spearmanr(x[ok], y[ok]).correlation)


def derivative_audit(ctx, cfg: dict, seed: int = 2026) -> tuple[pd.DataFrame, dict]:
    """Table + compat dict {status: {band: VERIFIED_METADATA | CONSISTENT_EMPIRICAL | PROVISIONAL | ABSENT},
    combinable: {family: bool}} for every provided derivative / slope band.

    Empirical check (label-free): the band vs the same quantity computed from its declared parent field, over a
    fixed random sample of valid pixels, at the smallest scale (sigma 0.7 px) and at 200 m; the best |rho| counts.
    hgrad <-> THDR(parent); vgrad <-> |k| VDR(parent); slope <-> THDR(parent). Scale factor = median(|band|) /
    median(|computed|) (a unit-conversion estimate)."""
    bc = cfg["bands"]
    data = ctx.data
    valid = np.asarray(data.valid, bool)
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(valid.ravel())
    idx = np.sort(rng.choice(idx, min(int(bc["sample_px"]), idx.size), replace=False)) if idx.size else idx
    g = data.grid
    px_km = (float(g.pixel_km_y), float(g.pixel_km_x))
    rows, status, factors = [], {}, {}
    verified = bc.get("verified") or {}
    for fam, spec in bc["families"].items():
        parent = spec.get("parent_of_derivatives") or spec["field"]
        if parent not in data.band_names:
            continue
        par = data.read_band(parent)
        f, m = PF.fill(par)
        comp = {}
        for s in (0.7, 2.0):
            ge, gn = PF.hgrad(f, s, px_km)
            comp[("thdr", s)] = PF.restore(np.hypot(ge, gn), m).ravel()[idx]
            if spec.get("potential_field"):
                comp[("vdr", s)] = PF.restore(PF.vdr_fft(PF.smooth(f, s), px_km), m).ravel()[idx]
        del f
        for role, against in (("hgrad", "thdr"), ("vgrad", "vdr"), ("slope", "thdr")):
            b = spec.get(role)
            if not b:
                continue
            desc = data.band_desc.get(b, "") if hasattr(data, "band_desc") else ""
            if b not in data.band_names:
                status[b] = "ABSENT"
                rows.append({"family": fam, "band": b, "role": role, "parent": parent, "status": "ABSENT",
                             "description": desc})
                continue
            x = data.read_band(b).ravel()[idx]
            best, best_s, fac = np.nan, np.nan, np.nan
            for (kind, s), y in comp.items():
                if kind != against:
                    continue
                r = _spearman(x, y)
                if np.isfinite(r) and (not np.isfinite(best) or abs(r) > abs(best)):
                    best, best_s = r, s
                    ok = np.isfinite(x) & np.isfinite(y)
                    fac = float(np.median(np.abs(x[ok])) / max(np.median(np.abs(y[ok])), 1e-12)) if ok.any() else np.nan
            if b in verified:
                st = "VERIFIED_METADATA"
            elif np.isfinite(best) and abs(best) >= float(bc["consistency_rho"]):
                st = "CONSISTENT_EMPIRICAL"
            else:
                st = "PROVISIONAL"
            status[b] = st
            factors[b] = fac
            rows.append({"family": fam, "band": b, "role": role, "parent": parent, "compared_with": against,
                         "spearman": best, "best_sigma_px": best_s, "scale_factor_band_over_computed": fac,
                         "status": st, "verified_note": verified.get(b, ""), "description": desc,
                         "units": (bc.get("units") or {}).get(b, "unknown (not verified)")})
    comb = {}
    for fam, spec in bc["families"].items():
        hg, vg = spec.get("hgrad"), spec.get("vgrad")
        ok = (status.get(hg) in ("VERIFIED_METADATA", "CONSISTENT_EMPIRICAL") and
              status.get(vg) in ("VERIFIED_METADATA", "CONSISTENT_EMPIRICAL"))
        if ok and status.get(hg) != "VERIFIED_METADATA":
            a, b = factors.get(hg, np.nan), factors.get(vg, np.nan)
            ratio = a / b if np.isfinite(a) and np.isfinite(b) and b > 0 else np.inf
            ok = 1.0 / float(bc["unit_ratio_tolerance"]) <= ratio <= float(bc["unit_ratio_tolerance"])
        comb[fam] = bool(ok)
    t = pd.DataFrame(rows)
    if len(t):
        t["combinable_family"] = t.family.map(comb)
    return t, {"status": status, "combinable": comb, "factors": factors}


def resolution_table(store: LayerStore) -> pd.DataFrame:
    """Configured vs spectral effective resolution of every family's field (diagnostic; the configured value rules)."""
    rows = []
    for fam, spec in store.families.items():
        if not store.has_band(spec["field"]):
            continue
        e = PF.effective_resolution_m(store.field(fam), store.px_km, float(store.cfg["physics"]["resolution_snr"]))
        rows.append({"family": fam, "band": spec["field"], "configured_m": float(spec["effective_resolution_m"]),
                     **e, "scales_used_m": ";".join(f"{s:g}" for s in store.scales(fam))})
    return pd.DataFrame(rows)
