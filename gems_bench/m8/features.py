"""Section 10: one row per candidate (section-06 pool + section-07 MC-only structures) with grouped features.

Base: M7 ``cand_features.describe`` (geometry, M7 geophysics / terrain / DEM / neural / network descriptors and the
positive-unlabelled label: 1 near scored truth, 0 far from it, -1 unlabelled = never a negative). M8 adds::

    physics      phys_<layer>_{mean,p90,max}: M8 operator layers sampled along the path (THDR, tilt edge, TDX,
                 analytic signal where defined, LCN), with phys_as_defined_share (tilt / TDX undefined areas)
    worms        worm_<family>_{near_share, persistence, orient_stability, ridge_stability, scale_max_m, migration,
                 continuity_km}: statistics of the nearest worm pixel within 2 px
    agreement    agree_mg<tol>_mean: tolerant magnetic-gravity agreement; cls_<class>_share: surface-expressed /
                 potentially buried / terrain-only and magnetic-only / gravity-only / agreeing / discordant shares
    profiles     section-05 fault-normal profile summaries (contrasts, sign consistency, continuity, asymmetry, ...)
    monte carlo  section-07 stability (mc_freq, mc_pos_std_m, mc_orient_disp, mc_frag_mean, mc_len_cv, ...)
    network      comp_occupancy_mean / comp_dist_km: stochastic completion occupancy along the candidate (disc units)
    context      ctx_<band>: regional percentiles of strain / seismicity / conductivity / basement depth
    provenance   fam_<family> (0/1), n_families
    missingness  has_mc, has_dem, has_profile, has_worm_<family>: explicit, never hidden

Feature provenance is persisted as a table (feature -> group -> source layer / section).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

from .layers import safe

GROUP_PREFIX = {"physics": ("phys_",), "worms": ("worm_",), "agreement": ("agree_mg", "cls_"),
                "profiles": ("prof_", "xmod_contrast"), "monte_carlo": ("mc_",), "network_m8": ("comp_",),
                "context": ("ctx_",), "provenance": ("fam_", "n_families"), "missingness": ("has_",)}


def _along(a: np.ndarray, p: np.ndarray) -> np.ndarray:
    return a[p[:, 0], p[:, 1]]


def physics(m8, u, cs) -> tuple[pd.DataFrame, list]:
    L = m8.layers
    out, prov = {}, []
    for nm in m8.cfg["features"]["physics_layers"]:
        fam, rest = nm.split(":", 1)
        op = rest.split("@")[0]
        if not L.allowed(fam, op)[0]:
            continue
        a = ndimage.maximum_filter(np.nan_to_num(L.crop(nm, u), nan=-np.inf), size=3)
        a = np.where(np.isfinite(a), a, np.nan)
        col = f"phys_{safe(nm).replace('.', '_')}"
        vals = [_along(a, p) for p in cs.paths]
        out[f"{col}_mean"] = [float(np.nanmean(v)) if np.isfinite(v).any() else np.nan for v in vals]
        out[f"{col}_p90"] = [float(np.nanpercentile(v, 90)) if np.isfinite(v).any() else np.nan for v in vals]
        if op in ("tilt_edge", "tdx", "theta"):
            out[f"{col}_defined_share"] = [float(np.isfinite(v).mean()) for v in vals]
        prov.append({"feature_prefix": col, "group": "physics", "source": nm, "section": "s03"})
    return pd.DataFrame(out), prov


def worms(m8, u, cs) -> tuple[pd.DataFrame, list]:
    L = m8.layers
    out, prov = {}, []
    for fam in m8.cfg["features"]["worm_families"]:
        if not L.has(f"worms:{fam}:mask"):
            out[f"has_worm_{fam}"] = np.zeros(len(cs))
            continue
        m = L.crop(f"worms:{fam}:mask", u) > 0
        out[f"has_worm_{fam}"] = np.ones(len(cs))
        if not m.any():
            continue
        d, (ir, ic) = ndimage.distance_transform_edt(~m, return_indices=True)
        stats = {k: L.crop(f"worms:{fam}:{k}", u) for k in ("persistence", "orient_stability", "ridge_stability",
                                                             "scale_max_m", "migration", "continuity_km")}
        near_share, acc = [], {k: [] for k in stats}
        for p in cs.paths:
            near = d[p[:, 0], p[:, 1]] <= 2.0
            near_share.append(float(near.mean()))
            jr, jc = ir[p[near, 0], p[near, 1]], ic[p[near, 0], p[near, 1]]
            for k, a in stats.items():
                v = a[jr, jc]
                acc[k].append(float(np.nanmean(v)) if np.isfinite(v).any() else np.nan)
        out[f"worm_{fam}_near_share"] = near_share
        for k, v in acc.items():
            out[f"worm_{fam}_{k}"] = v
        prov.append({"feature_prefix": f"worm_{fam}_", "group": "worms", "source": f"worms:{fam}", "section": "s03b"})
    return pd.DataFrame(out), prov


CLASS_NAMES = {"class:surface_subsurface": {1: "surface_expressed", 2: "potentially_buried", 3: "terrain_only"},
               "class:mag_grav": {1: "magnetic_only", 2: "gravity_only", 3: "mag_grav_agree", 4: "mag_grav_discordant"}}


def agreement(m8, u, cs) -> tuple[pd.DataFrame, list]:
    L = m8.layers
    out, prov = {}, []
    for nm in m8.cfg["features"]["agreement_layers"]:
        if not L.has(nm):
            continue
        a = ndimage.maximum_filter(np.nan_to_num(L.crop(nm, u), nan=0.0), size=5)
        tol = nm.split("@")[-1]
        out[f"agree_mg{tol}_mean"] = [float(_along(a, p).mean()) for p in cs.paths]
        out[f"agree_mg{tol}_share"] = [float((_along(a, p) > 0).mean()) for p in cs.paths]
        prov.append({"feature_prefix": f"agree_mg{tol}", "group": "agreement", "source": nm, "section": "s04"})
    for nm, names in CLASS_NAMES.items():
        if not L.has(nm):
            continue
        c = np.asarray(L.crop(nm, u)).astype(np.int16)
        d, (ir, ic) = ndimage.distance_transform_edt(c == 0, return_indices=True)
        for code, cname in names.items():
            vals = []
            for p in cs.paths:
                near = d[p[:, 0], p[:, 1]] <= 2.0
                v = c[ir[p[near, 0], p[near, 1]], ic[p[near, 0], p[near, 1]]]
                vals.append(float((v == code).sum() / max(len(p), 1)))
            out[f"cls_{cname}_share"] = vals
        prov.append({"feature_prefix": "cls_", "group": "agreement", "source": nm, "section": "s04"})
    return pd.DataFrame(out), prov


def context(m8, u, cs) -> tuple[pd.DataFrame, list]:
    L = m8.layers
    out, prov = {}, []
    for b in m8.cfg["bands"]["context"]:
        nm = f"context:{b}"
        if not L.has(nm):
            continue
        a = L.crop(nm, u)
        out[f"ctx_{b}"] = [float(np.nanmean(_along(a, p))) if np.isfinite(_along(a, p)).any() else np.nan
                           for p in cs.paths]
        prov.append({"feature_prefix": f"ctx_{b}", "group": "context", "source": nm, "section": "s04"})
    return pd.DataFrame(out), prov


def completion(m8, u, cs) -> pd.DataFrame:
    occ = m8.completion_occupancy(u.key)
    comp = m8.completion_set(u.key)
    out = {}
    if occ is not None:
        a = ndimage.maximum_filter(occ, size=3)
        out["comp_occupancy_mean"] = [float(_along(a, p).mean()) for p in cs.paths]
    if comp is not None and len(comp):
        st = comp.table.get("origin", pd.Series("", index=comp.table.index)).astype(str).to_numpy()
        idx = [i for i in range(len(comp)) if st[i] == "m8_stochastic"]
        m = comp.subset(idx).raster(u.ev.shape) > 0 if idx else np.zeros(u.ev.shape, bool)
        d = ndimage.distance_transform_edt(~m) * 0.1 if m.any() else np.full(u.ev.shape, np.inf)
        out["comp_dist_km"] = [float(np.median(_along(d, p))) for p in cs.paths]
    return pd.DataFrame(out)


def provenance(cs, families: list[str]) -> pd.DataFrame:
    fams = cs.table.get("families", pd.Series("", index=cs.table.index)).astype(str)
    out = {f"fam_{f}": fams.map(lambda x, f=f: int(f in x.split(";"))).to_numpy() for f in families}
    out["n_families"] = cs.table.get("n_families", pd.Series(1, index=cs.table.index)).to_numpy()
    return pd.DataFrame(out)


def describe(m8, u, cs, *, with_labels: bool, scores: dict, profiles: pd.DataFrame | None, mc: pd.DataFrame | None,
             mos=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    from ..m7 import cand_features as CF

    if cs is None or not len(cs):
        return pd.DataFrame(), pd.DataFrame()
    base = CF.describe(m8.view, u, cs, scores, with_labels=with_labels, dem=mos)
    parts, prov = [base.reset_index(drop=True)], []
    for fn in (physics, worms, agreement, context):
        df, pv = fn(m8, u, cs)
        parts.append(df.reset_index(drop=True))
        prov += pv
    if m8.cfg["features"]["include_completion"] and u.task != "geo":
        parts.append(completion(m8, u, cs).reset_index(drop=True))
    fams = [s["name"] for s in m8.cfg["generation"]["families"]] + [s["name"] for s in m8.cfg["generation"]["model_families"]] \
        + ["tip1km", "tip2km", "tip4km", "mc_occupancy"]
    parts.append(provenance(cs, fams))
    T = pd.concat(parts, axis=1)
    T = T.loc[:, ~T.columns.duplicated()]
    if profiles is not None and len(profiles):
        T = T.merge(profiles, on="cand_id", how="left")
    T["has_profile"] = T[[c for c in T.columns if c.startswith("prof_")]].notna().any(axis=1).astype(int) \
        if any(c.startswith("prof_") for c in T.columns) else 0
    if mc is not None and len(mc):
        cols = [c for c in mc.columns if c.startswith("mc_") and c != "mc_region"]
        T = T.merge(mc[["cand_id"] + cols], on="cand_id", how="left")
        T["has_mc"] = T.mc_freq.notna().astype(int)
    else:
        T["has_mc"] = 0
    T["has_dem"] = T[[c for c in T.columns if c.startswith("dem_cov_")]].fillna(0).max(axis=1).gt(0).astype(int) \
        if any(c.startswith("dem_cov_") for c in T.columns) else 0
    T["unit"] = u.key
    return T, pd.DataFrame(prov)


def groups(T: pd.DataFrame) -> dict[str, list[str]]:
    """M7 groups (geometry, geophysics, topography, neural, network) + the M8 groups."""
    from ..m7 import cand_features as CF

    g = CF.groups(T)
    num = [c for c in T.columns if pd.api.types.is_numeric_dtype(T[c]) and c not in CF.LABEL_COLS | CF.ID_NUMERIC]
    for name, pre in GROUP_PREFIX.items():
        g[name] = [c for c in num if c.startswith(pre)]
    taken = set().union(*[set(v) for k, v in g.items() if k in GROUP_PREFIX])
    for k in ("geometry", "geophysics", "topography", "neural", "network"):
        g[k] = [c for c in g[k] if c not in taken]
    return g


def ablation_features(T: pd.DataFrame) -> dict[str, list[str]]:
    g = groups(T)
    phys = g["geometry"] + g["geophysics"] + g["physics"] + g["worms"] + g["agreement"] + g["profiles"] + g["provenance"]
    dem = g["topography"]
    mc = g["monte_carlo"]
    cnn = g["neural"]
    net = g["network"] + g["network_m8"]
    miss = g["missingness"]
    return {"physics_only": phys + miss,
            "physics_mc": phys + mc + miss,
            "physics_dem": phys + dem + miss,
            "physics_cnn": phys + cnn + miss,
            "all": phys + mc + dem + cnn + net + g["context"] + miss}
