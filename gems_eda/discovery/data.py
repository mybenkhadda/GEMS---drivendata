"""Shared measurements of the exhaustive EDA (native 100 m grid). Reuses ``gems_eda`` (loader, fault network, fault
stations with local tangent / normal, profile extraction, enhancers, circular statistics) and ``gems_bench.reset``
(bilinear sampling, side-profile correlation, localisation).

Provenance classes of the 19 bands are inferred from the band names (verify against the band descriptions saved in
``audit/band_metadata.csv``): primary_survey, processed_survey, derivative, model_product, catalog_density.

Matched controls: for every fault station, ``k`` control locations in the SAME 3 x 3 region and the SAME tercile of two
label-free context covariates (2 km-smoothed terrain slope, 5 km magnetic gradient), between ``control_distance_km``
from any mapped trace, sampled with the fault station's own normal direction (orientation-matched). Controls are
"matched controls", NOT confirmed non-faults (unmapped != non-fault).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import ndimage

PROVENANCE = {"tmi": "primary_survey", "iso_grav_anom": "primary_survey", "det_elev": "primary_survey",
              "mag_anom": "processed_survey", "rtp": "processed_survey", "tc": "processed_survey",
              "tmi_hg": "derivative", "tmi_vg": "derivative", "iso_grav_anom_hg": "derivative",
              "iso_grav_anom_vg": "derivative", "iso_grav_anom_slope": "derivative", "det_elev_slope": "derivative",
              "depth_to_base_surf": "model_product", "cond_surf": "model_product", "geod_2ndinv": "model_product",
              "geod_shearrate": "model_product", "geod_dilaterate": "model_product",
              "deq_n100a15": "catalog_density", "ieq_n100a15": "catalog_density"}
PRIMARY = {"magnetic": "rtp", "gravity": "iso_grav_anom", "terrain": "det_elev", "basement": "depth_to_base_surf",
           "conductivity": "cond_surf", "strain": "geod_shearrate", "seismicity": "ieq_n100a15"}
LOCAL = ("magnetic", "gravity", "terrain", "basement", "conductivity")       # candidates for trace-scale evidence
PROFILE_SIGNALS = {"magnetic": ("rtp", "tmi_hg"), "gravity": ("iso_grav_anom", "iso_grav_anom_hg"),
                   "terrain": ("det_elev", "det_elev_slope"), "basement": ("depth_to_base_surf",),
                   "conductivity": ("cond_surf",), "strain": ("geod_shearrate",), "seismicity": ("ieq_n100a15",)}
RECORDED = {"epsg": 32611, "res_m": 100.0, "width": 3730, "height": 3292, "bands": 19, "valid_frac": 0.42,
            "fault_px": 60988}


@dataclass
class EDAData:
    stack: object
    labels: np.ndarray
    template_valid: np.ndarray
    net: object
    enh: object
    region_grid: tuple = (3, 3)
    cache: dict = field(default_factory=dict)

    @property
    def grid(self):
        return self.stack.grid

    @property
    def valid(self) -> np.ndarray:
        if "valid" not in self.cache:
            self.cache["valid"] = np.asarray(self.stack.valid, bool) & self.template_valid
        return self.cache["valid"]

    def has(self, band: str) -> bool:
        return band in self.stack.names

    def band(self, band: str) -> np.ndarray | None:
        return self.stack[band] if band in self.stack.names else None

    def modality(self, band: str) -> str:
        from gems_eda.config import band_modality

        return band_modality(band.split(":")[0]) if ":" not in band else band.split(":")[0]

    def region(self, r, c) -> np.ndarray:
        from gems_eda.morphology import region_ids

        return region_ids(r, c, self.grid, self.region_grid)


def load_data(root, region_grid=(3, 3)) -> EDAData:
    """Native-resolution stack (``max_dim=None``), mapped faults on the same grid, template footprint, fault network."""
    from rasterio.enums import Resampling

    from gems_eda.config import resolve_raw
    from gems_eda.enhance import Enhancer
    from gems_eda.io import load_feature_stack, load_on_grid
    from gems_eda.morphology import build_fault_network

    stack = load_feature_stack(resolve_raw(root, "features"), None)
    f, _ = load_on_grid(resolve_raw(root, "faults"), stack.grid, Resampling.max)
    s, _ = load_on_grid(resolve_raw(root, "submission"), stack.grid, Resampling.nearest)
    labels = np.isfinite(f) & (f > 0)
    net = build_fault_network(labels, stack.grid, region_grid)
    return EDAData(stack, labels, np.isfinite(s), net, Enhancer(stack.grid), tuple(region_grid))


# =============================================================================== inventory
def raster_inventory(root) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(file-level checks vs the recorded properties, per-band metadata + statistics) read with rasterio."""
    import rasterio

    from gems_eda.config import RAW_FILES, band_modality, resolve_raw

    files, bands = [], []
    for key in RAW_FILES:
        p = resolve_raw(root, key)
        if not p.exists():
            files.append({"file": key, "path": str(p), "exists": False})
            continue
        with rasterio.open(p) as src:
            epsg = src.crs.to_epsg() if src.crs else None
            row = {"file": key, "path": str(p), "exists": True, "epsg": epsg, "res_x": src.res[0], "res_y": src.res[1],
                   "width": src.width, "height": src.height, "count": src.count, "dtype": src.dtypes[0],
                   "nodata": src.nodata, "bounds": str(tuple(src.bounds)), "size_mb": p.stat().st_size / 2 ** 20}
            row.update(check_epsg=epsg == RECORDED["epsg"], check_res=abs(src.res[0] - RECORDED["res_m"]) < 1e-6,
                       check_shape=(src.width, src.height) == (RECORDED["width"], RECORDED["height"]))
            if key == "features":
                row["check_bands"] = src.count == RECORDED["bands"]
                for i in range(1, src.count + 1):
                    a = src.read(i, masked=True).astype("float64").filled(np.nan)
                    a[np.abs(a) > 1e30] = np.nan
                    v = a[np.isfinite(a)]
                    tags = src.tags(i)
                    name = tags.get("band_name") or (src.descriptions[i - 1] or f"band_{i}").split(" - ")[0].strip()
                    med = float(np.median(v)) if v.size else np.nan
                    mad = float(1.4826 * np.median(np.abs(v - med))) if v.size else np.nan
                    bands.append({"band_index": i, "band": name, "modality": band_modality(name),
                                  "provenance_class": PROVENANCE.get(name, "unknown (verify)"),
                                  "description": src.descriptions[i - 1] or tags.get("description", ""),
                                  "units": tags.get("units") or tags.get("unit") or "not tagged",
                                  "valid_frac": float(np.isfinite(a).mean()), "min": float(v.min()) if v.size else np.nan,
                                  "p01": float(np.quantile(v, 0.01)) if v.size else np.nan, "median": med,
                                  "p99": float(np.quantile(v, 0.99)) if v.size else np.nan,
                                  "max": float(v.max()) if v.size else np.nan, "robust_scale_mad": mad,
                                  "n_unique_sample": int(len(np.unique(v[:: max(len(v) // 200000, 1)])))})
                    del a, v
            if key == "faults":
                a = src.read(1, masked=True).filled(0)
                row["fault_px"] = int((a > 0).sum())
                row["check_fault_px"] = row["fault_px"] == RECORDED["fault_px"]
        files.append(row)
    return pd.DataFrame(files), pd.DataFrame(bands)


def zscore(a: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, float, float]:
    v = a[valid & np.isfinite(a)]
    med = float(np.median(v)) if v.size else 0.0
    mad = float(1.4826 * np.median(np.abs(v - med))) if v.size else 1.0
    mad = mad if mad > 0 else (float(np.std(v)) or 1.0)
    return ((a - med) / mad).astype(np.float32), med, mad


# =============================================================================== label-free domains
def domains(D: EDAData, n_domains: int, block_px: int, seed: int) -> tuple[np.ndarray, pd.DataFrame]:
    """Label-free structural domains: k-means on 5 km block descriptors (magnetic texture and gradient, gravity level
    and gradient, terrain relief and slope, basement depth, magnetic fabric orientation as doubled angle)."""
    from sklearn.cluster import KMeans

    g, e = D.grid, D.enh
    H, W = g.shape
    v = D.valid
    feats = {}
    rtp, grav, elev = D.band("rtp"), D.band("iso_grav_anom"), D.band("det_elev")
    if rtp is not None:
        feats["mag_texture"] = rtp
        feats["mag_grad_2km"] = e.thdr(rtp, 2.0)
        st, coh, _ = e.structure_tensor(rtp, 5.0)
        feats["fabric_cos"] = np.cos(np.radians(2 * st)) * coh
        feats["fabric_sin"] = np.sin(np.radians(2 * st)) * coh
    if grav is not None:
        feats["grav_level"], feats["grav_grad_2km"] = grav, e.thdr(grav, 2.0)
    if elev is not None:
        feats["terrain_relief"] = elev
    if D.has("det_elev_slope"):
        feats["terrain_slope"] = D.band("det_elev_slope")
    if D.has("depth_to_base_surf"):
        feats["basement_depth"] = D.band("depth_to_base_surf")
    nb_r, nb_c = int(np.ceil(H / block_px)), int(np.ceil(W / block_px))
    rows = []
    for i in range(nb_r):
        for j in range(nb_c):
            sl = (slice(i * block_px, min((i + 1) * block_px, H)), slice(j * block_px, min((j + 1) * block_px, W)))
            m = v[sl]
            if m.mean() < 0.5:
                continue
            r = {"bi": i, "bj": j}
            for k, a in feats.items():
                x = a[sl][m]
                x = x[np.isfinite(x)]
                if not x.size:
                    r[k] = np.nan
                elif k in ("mag_texture", "terrain_relief"):
                    r[k] = float(np.quantile(x, 0.9) - np.quantile(x, 0.1))
                else:
                    r[k] = float(np.mean(x))
            rows.append(r)
    B = pd.DataFrame(rows)
    X = B.drop(columns=["bi", "bj"])
    X = (X - X.median()) / (X.quantile(0.75) - X.quantile(0.25)).replace(0, 1)
    X = X.fillna(0).clip(-6, 6)
    km = KMeans(n_clusters=int(n_domains), n_init=10, random_state=int(seed)).fit(X.to_numpy())
    B["domain"] = km.labels_
    dom = np.full((nb_r, nb_c), -1, np.int16)
    dom[B.bi, B.bj] = B.domain
    full = np.repeat(np.repeat(dom, block_px, 0), block_px, 1)[:H, :W]
    x, y = g.rc_to_xy((B.bi + 0.5) * block_px, (B.bj + 0.5) * block_px)
    B["x_center"], B["y_center"] = x, y
    return np.where(v, full, -1).astype(np.int16), B


# =============================================================================== stations, controls, profiles
def fault_stations(D: EDAData, c: dict, rng) -> pd.DataFrame:
    from gems_eda.profiles import sample_fault_stations

    sc = c["sampling"]
    half_px = int(np.ceil(sc["profile_half_width_km"] / D.grid.pixel_km))
    st = sample_fault_stations(D.net, D.grid, D.valid, float(sc["along_strike_spacing_m"]) / 1000.0, 1.0, 0.5,
                               half_px + 2, D.region_grid, rng, int(c["max_fault_stations"]))
    seg = D.net.segments.set_index("seg_id")
    st["system"] = D.net.components[st.r, st.c].astype(int)
    st["seg_length_km"] = st.seg_id.map(seg.length_km)
    if sc.get("max_fault_systems"):
        keep = rng.choice(st.system.unique(), min(int(sc["max_fault_systems"]), st.system.nunique()), replace=False)
        st = st[st.system.isin(keep)].reset_index(drop=True)
    return st


def context_strata(D: EDAData) -> np.ndarray:
    """3 x 3 terciles of (2 km-smoothed terrain slope, 5 km magnetic gradient): label-free physical setting."""
    from gems_eda.enhance import nan_gaussian

    out = np.zeros(D.grid.shape, np.int16)
    for k, a in enumerate((nan_gaussian(D.band("det_elev_slope"), 10.0) if D.has("det_elev_slope") else None,
                           D.enh.thdr(D.band("rtp"), 5.0) if D.has("rtp") else None)):
        if a is None:
            continue
        q = np.nanquantile(a[D.valid], [1 / 3, 2 / 3])
        t = np.digitize(np.nan_to_num(a, nan=q[0]), q).astype(np.int16)
        out += t * (3 ** k)
    return out


def matched_controls(D: EDAData, st: pd.DataFrame, strata: np.ndarray, k: int, dmin: float, dmax: float, edge_px: int,
                     rng, kind: str = "control") -> pd.DataFrame:
    H, W = D.grid.shape
    ok = D.valid & (D.net.distance_km >= dmin) & (D.net.distance_km <= dmax)
    ok[:edge_px], ok[-edge_px:], ok[:, :edge_px], ok[:, -edge_px:] = False, False, False, False
    rr, cc = np.nonzero(ok)
    if not len(rr):
        raise ValueError(f"no control pixels between {dmin} and {dmax} km from mapped faults")
    key = D.region(rr, cc).astype(np.int64) * 100 + strata[rr, cc]
    reg = D.region(rr, cc)
    by_key = pd.Series(np.arange(len(rr))).groupby(key).apply(lambda s: s.to_numpy()).to_dict()
    by_reg = pd.Series(np.arange(len(rr))).groupby(reg).apply(lambda s: s.to_numpy()).to_dict()
    skey = D.region(st.r.to_numpy(), st.c.to_numpy()).astype(np.int64) * 100 + strata[st.r.to_numpy(), st.c.to_numpy()]
    rows = []
    for i, (s, kk) in enumerate(zip(st.itertuples(), skey)):
        pool, level = by_key.get(kk), "region+setting"
        if pool is None or len(pool) < 3 * k:
            pool, level = by_reg.get(int(s.region)), "region"
        if pool is None or not len(pool):
            pool, level = np.arange(len(rr)), "global"
        for j in rng.choice(pool, k, replace=len(pool) < k):
            rows.append({"r": int(rr[j]), "c": int(cc[j]), "dr": s.dr, "dc": s.dc, "strike": s.strike, "seg_id": -1,
                         "kind": kind, "region": int(reg[j]), "matched_to": i, "system": int(s.system),
                         "match_level": level, "dist_known_km": float(D.net.distance_km[rr[j], cc[j]]),
                         "seg_length_km": np.nan})
    return pd.DataFrame(rows)


def stations(D: EDAData, c: dict, rng, strata: np.ndarray, dom: np.ndarray) -> pd.DataFrame:
    sc = c["sampling"]
    half_px = int(np.ceil(sc["profile_half_width_km"] / D.grid.pixel_km)) + 2
    f = fault_stations(D, c, rng)
    f["matched_to"] = np.arange(len(f))
    f["match_level"], f["dist_known_km"] = "fault", 0.0
    lo, hi = sc["control_distance_km"]
    ctl = matched_controls(D, f, strata, int(sc["matched_controls_per_fault"]), lo, hi, half_px, rng)
    S = pd.concat([f, ctl], ignore_index=True)
    S.insert(0, "sample_id", np.arange(len(S)))
    S["domain"] = dom[S.r, S.c]
    S["strata"] = strata[S.r, S.c]
    x, y = D.grid.rc_to_xy(S.r, S.c)
    S["x_utm"], S["y_utm"] = x, y
    return S


def profile_signals(D: EDAData) -> tuple[dict, pd.DataFrame]:
    """Robust-z signals for profiles + their physical scaling (z * mad + median)."""
    sig, rows = {}, []
    for mod, names in PROFILE_SIGNALS.items():
        for n in names:
            a = D.band(n)
            if a is None:
                continue
            z, med, mad = zscore(a, D.valid)
            sig[n] = z
            rows.append({"signal": n, "modality": mod, "median": med, "mad": mad, "derived": False})
    for mod, n in (("magnetic", "rtp"), ("gravity", "iso_grav_anom")):
        if D.has(n):
            z, med, mad = zscore(D.enh.thdr(D.band(n), 1.0), D.valid)
            sig[f"{n}:thdr1km"] = z
            rows.append({"signal": f"{n}:thdr1km", "modality": mod, "median": med, "mad": mad, "derived": True})
    return sig, pd.DataFrame(rows)


def offsets(half_km: float, step_km: float = 0.1) -> np.ndarray:
    return np.round(np.arange(-half_km, half_km + step_km / 2, step_km), 4)


def station_metrics(cube: np.ndarray, S: pd.DataFrame, sigs: list[str], off: np.ndarray, core_km: float = 0.5,
                    loc_km: float = 5.0) -> pd.DataFrame:
    """Per (station, signal): |step| and |peak| (core vs flanks), gradient-peak offset (argmax |d/dx| within
    +-loc_km), flank asymmetry, local contrast (centre vs profile median, in profile IQR units)."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        core, lft, rgt = np.abs(off) <= core_km, off < -core_km, off > core_km
        c = np.nanmean(cube[..., core], -1)
        L, Rr = np.nanmean(cube[..., lft], -1), np.nanmean(cube[..., rgt], -1)
        gr = np.gradient(cube, off, axis=-1)
        win = np.abs(off) <= loc_km
        g = np.abs(np.where(np.isfinite(gr), gr, 0.0))[..., win]
        pk = off[win][np.argmax(g, -1)]
        gl = np.nanmean(np.abs(gr[..., lft & (off > -3)]), -1)
        grr = np.nanmean(np.abs(gr[..., rgt & (off < 3)]), -1)
        med = np.nanmedian(cube, -1)
        iqr = np.nanpercentile(cube, 75, -1) - np.nanpercentile(cube, 25, -1)
        cen = cube[..., int(np.argmin(np.abs(off)))]
    n, s = c.shape
    return pd.DataFrame({"sample_id": np.repeat(S.sample_id.to_numpy(), s), "kind": np.repeat(S.kind.to_numpy(), s),
                         "system": np.repeat(S.system.to_numpy(), s), "region": np.repeat(S.region.to_numpy(), s),
                         "domain": np.repeat(S.domain.to_numpy(), s), "signal": np.tile(sigs, n),
                         "abs_step": np.abs(Rr - L).ravel(), "abs_peak": np.abs(c - 0.5 * (L + Rr)).ravel(),
                         "grad_peak_offset_km": pk.ravel(), "abs_grad_peak_offset_km": np.abs(pk).ravel(),
                         "flank_asymmetry": (np.abs(grr - gl) / (grr + gl + 1e-9)).ravel(),
                         "local_contrast": (np.abs(cen - med) / (iqr + 1e-9)).ravel()})


class Sampler:
    """Bilinear (or nearest, for angles) sampling of one full-grid array at many points; the NaN fill and the
    validity weights are computed ONCE (per-call full-array work would dominate the cost)."""

    def __init__(self, a: np.ndarray, order: int = 1):
        a = np.asarray(a, np.float32)
        self.ok = np.isfinite(a).astype(np.float32)
        self.f = np.where(self.ok > 0, a, 0.0).astype(np.float32)
        self.order = order

    def __call__(self, rr: np.ndarray, cc: np.ndarray) -> np.ndarray:
        shp = np.shape(rr)
        co = [np.ravel(rr), np.ravel(cc)]
        v = ndimage.map_coordinates(self.f, co, order=self.order, mode="constant", cval=0.0, prefilter=False)
        w = ndimage.map_coordinates(self.ok, co, order=self.order, mode="constant", cval=0.0, prefilter=False)
        return np.where(w > 0.99, v, np.nan).reshape(shp)


def rowcorr(x: np.ndarray, y: np.ndarray, min_n: int = 8) -> np.ndarray:
    """Pearson correlation per row, NaNs ignored pairwise; NaN where fewer than ``min_n`` pairs or no variance."""
    m = np.isfinite(x) & np.isfinite(y)
    n = m.sum(1)
    x0, y0 = np.where(m, x, 0.0), np.where(m, y, 0.0)
    mx, my = x0.sum(1) / np.maximum(n, 1), y0.sum(1) / np.maximum(n, 1)
    dx, dy = np.where(m, x - mx[:, None], 0.0), np.where(m, y - my[:, None], 0.0)
    num = (dx * dy).sum(1)
    den = np.sqrt((dx * dx).sum(1) * (dy * dy).sum(1))
    return np.where((n >= min_n) & (den > 0), num / np.maximum(den, 1e-12), np.nan)


def structure_metrics(D: EDAData, S: pd.DataFrame, fields: dict, along_km: float = 2.0, side_km: float = 1.0,
                      max_lag: int = 8) -> pd.DataFrame:
    """Per station and primary field: along-strike persistence of the 1 km gradient at the trace (mean / std along
    the tangent), side-profile correlation and its lag (low = disruption / truncation; high at a non-zero lag =
    apparent lateral offset), strike change across the trace (structure tensor at +-1.5 km), local coherence."""
    out = []
    t = np.arange(-along_km, along_km + 1e-9, 0.1)
    r0, c0 = S.r.to_numpy(float)[:, None], S.c.to_numpy(float)[:, None]
    nr, nc = S.dr.to_numpy(float)[:, None], S.dc.to_numpy(float)[:, None]          # normal, px per km
    tr, tc = nc * (D.grid.pixel_km_x / D.grid.pixel_km_y), -nr * (D.grid.pixel_km_y / D.grid.pixel_km_x)
    rr, cc = r0 + t[None, :] * tr, c0 + t[None, :] * tc
    for name, f in fields.items():
        strike, coh, _ = D.enh.structure_tensor(f, 2.0)
        g1 = Sampler(D.enh.thdr(f, 1.0))
        fs, ss, cs = Sampler(f), Sampler(strike, order=0), Sampler(coh)
        a = g1(rr, cc)
        with np.errstate(invalid="ignore"):
            pers = np.nanmean(a, 1) / (np.nanstd(a, 1) + 1e-9)
        pers[np.isfinite(a).sum(1) <= 5] = np.nan
        s1, s2 = fs(rr + side_km * nr, cc + side_km * nc), fs(rr - side_km * nr, cc - side_km * nc)
        best = np.full(len(S), -np.inf)
        lag = np.full(len(S), np.nan)
        for k in range(-max_lag, max_lag + 1):
            x, y = (s1[:, k:], s2[:, :s2.shape[1] - k]) if k >= 0 else (s1[:, :k], s2[:, -k:])
            r = rowcorr(x, y)
            up = np.nan_to_num(r, nan=-np.inf) > best
            best = np.where(up, r, best)
            lag = np.where(up, k, lag)
        sa = ss(r0[:, 0] + 1.5 * nr[:, 0], c0[:, 0] + 1.5 * nc[:, 0])
        sb = ss(r0[:, 0] - 1.5 * nr[:, 0], c0[:, 0] - 1.5 * nc[:, 0])
        d = np.abs(sa - sb) % 180
        out.append(pd.DataFrame({"sample_id": S.sample_id.to_numpy(), "field": name, "along_persistence": pers,
                                 "side_corr": np.where(np.isfinite(best), best, np.nan),
                                 "side_lag_km": lag * 0.1, "strike_change_deg": np.minimum(d, 180 - d),
                                 "coherence": cs(r0[:, 0], c0[:, 0]), "local_strike_deg": ss(r0[:, 0], c0[:, 0])}))
    return pd.concat(out, ignore_index=True)


# =============================================================================== scales and lineaments
def along_normal(a, S: pd.DataFrame, off: np.ndarray) -> np.ndarray:
    """(n_stations, n_offsets) samples along each station's normal (``a``: array or ``Sampler``)."""
    rr = S.r.to_numpy()[:, None] + off[None, :] * S.dr.to_numpy()[:, None]
    cc = S.c.to_numpy()[:, None] + off[None, :] * S.dc.to_numpy()[:, None]
    return (a if isinstance(a, Sampler) else Sampler(a))(rr, cc)


def crest_mask(D: EDAData, g: np.ndarray, scale_km: float, top: float = 0.03) -> np.ndarray:
    """Label-free crest lines: Hessian line strength of the gradient map, top share inside the footprint."""
    ls, _ = D.enh.ridge(g, scale_km, "bright")
    v = ls[D.valid & np.isfinite(ls)]
    if not v.size:
        return np.zeros(D.grid.shape, bool)
    thr = np.quantile(v, 1 - top)
    return D.valid & (np.nan_to_num(ls) >= thr)


def scale_row(D: EDAData, S: pd.DataFrame, mod: str, scale: float, g: np.ndarray, crest: np.ndarray, c: dict) -> dict:
    """Detectability (max gradient within +-300 m: fault vs control), localisation (|offset| of the gradient maximum
    within +-5 km of the trace) and crest precision proxy (enrichment of crest pixels within 300 m of mapped faults,
    relative to the footprint share within 300 m)."""
    from .core import cluster_ci, cluster_effect

    near = along_normal(g, S, np.round(np.arange(-0.3, 0.31, 0.1), 2))
    wide_off = np.round(np.arange(-5, 5.01, 0.1), 2)
    wide = along_normal(g, S, wide_off)
    det = np.nanmax(near, 1)
    loc = np.abs(wide_off[np.nanargmax(np.where(np.isfinite(wide), wide, -np.inf), 1)])
    f, k = (S.kind == "fault").to_numpy(), (S.kind == "control").to_numpy()
    eff = cluster_effect(det[f], S.system[f], det[k], S.system[k], c["n_boot"], c["sampling"]["seed"])
    li = cluster_ci(loc[f], S.system[f], c["n_boot"], c["sampling"]["seed"])
    within = D.net.distance_km <= 0.3
    base = float(within[D.valid].mean())
    prec = float(within[crest].mean()) if crest.any() else np.nan
    return {"modality": mod, "scale_km": scale, "detect_delta": eff["effect"], "detect_lo": eff["ci_lo"],
            "detect_hi": eff["ci_hi"], "n_systems": eff["n_groups"], "loc_median_km": li["estimate"],
            "loc_lo": li["ci_lo"], "loc_hi": li["ci_hi"], "loc_within_300m": float(np.nanmean(loc[f] <= 0.3)),
            "crest_px": int(crest.sum()), "crest_share_within_300m": prec, "footprint_share_within_300m": base,
            "crest_enrichment": prec / base if base and np.isfinite(prec) else np.nan,
            "_det": det, "_loc": loc}


def lineaments(D: EDAData, crests: dict, min_km: float = 1.0) -> tuple[pd.DataFrame, np.ndarray]:
    """Label-free lineament inventory from the union of independent-family crests (1 km scale): components with
    their supporting families, orientation, length, mapped status (by distance to mapped traces, for analysis only)."""
    from skimage.morphology import skeletonize

    from gems_bench.reset.atlas import components

    union = np.zeros(D.grid.shape, bool)
    near = {}
    for fam, m in crests.items():
        union |= m
        near[fam] = ndimage.binary_dilation(m, iterations=3)
    sk = skeletonize(union)
    comps = components(sk, int(np.ceil(min_km / 0.112)))
    rows = []
    lab = np.zeros(D.grid.shape, np.int32)
    for i, (rr, cc) in enumerate(comps, start=1):
        lab[rr, cc] = i
        e, n = D.grid.rc_to_km(rr, cc)
        if len(rr) >= 3:
            w, v = np.linalg.eigh(np.cov(np.vstack([e, n])))
            az = float(np.degrees(np.arctan2(v[0, 1], v[1, 1])) % 180)
            lin = float(1 - w[0] / w[1]) if w[1] > 0 else np.nan
        else:
            az, lin = np.nan, np.nan
        d = D.net.distance_km[rr, cc]
        fams = [f for f in near if near[f][rr, cc].mean() >= 0.5]
        rows.append({"lineament_id": i, "n_px": len(rr), "length_km": len(rr) * 0.112, "azimuth": az,
                     "linearity": lin, "r_mid": int(np.median(rr)), "c_mid": int(np.median(cc)),
                     "region": int(D.region(np.array([int(np.median(rr))]), np.array([int(np.median(cc))]))[0]),
                     "families": ";".join(fams), "n_families": len(fams), "dist_known_min_km": float(d.min()),
                     "dist_known_median_km": float(np.median(d)), "near_known_share": float((d <= 0.3).mean()),
                     "status": "mapped" if (d <= 0.3).mean() >= 0.5 else "unmapped" if d.min() > 2.0
                     else "near_mapped"})
    return pd.DataFrame(rows), lab
