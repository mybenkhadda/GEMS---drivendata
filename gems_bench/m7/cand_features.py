"""Section 10: one row per candidate with geological descriptors, grouped for the section-11 ablations.

Builds on ``gems3.descriptors.describe`` (geometry, provenance, |z| evidence, magnetic / gravity orientation
consistency, distances to the unit's VISIBLE known faults, member-score ranks, PU labels) and adds:

geometry   straightness, number of fused members (fragmentation), junction contacts (branching), endpoint
           connectivity to other candidates, along-line orientation persistence, generator persistence / coherence
geophysics side contrast of the magnetic / gravity fields at 300 / 600 m on either side (|mean(+n) - mean(-n)| /
           robust scale: a step, not just an edge), gradient coherence along the line
topography 100 m terrain side contrast and slope asymmetry, curvature / relief response; 1 m DEM evidence along the
           line when the M7 DEM mosaic covers it (scarp response, directional discontinuity, openness contrast)
neural     member-score ranks (mean / max / p90), agreement (min of means), disagreement (|difference|), rank
           within its ~20 km regional block
network    distances to known faults / endpoints, continuation angle to the nearest known tip tangent, share of the
           line running parallel to a known trace within ``parallel_radius_km``

Known-fault proximity is a FEATURE, never a rule: new faults can be independent structures or new parts of known
systems. Labels (``label``): 1 if mean k(dist to the unit's scored truth) >= ``positive_kd``; 0 if the median
distance >= ``negative_dist_km``; -1 (unlabelled, never a negative) otherwise. Label columns are excluded from
every feature group.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

from ..gems3 import descriptors as D
from ..gems3.oracle import candidate_truth_stats

GEOPHYS_MODS = ("magnetic", "gravity", "conductivity", "basement", "fused")
GEOMETRY_COLS = ("length_km", "chord_km", "sinuosity", "strike_deg", "curvature_deg_per_km", "n_px", "r_mid", "c_mid")


def _normals(strike_deg: np.ndarray) -> np.ndarray:
    a = np.radians(strike_deg)
    return np.c_[np.sin(a), np.cos(a)]                     # (row, col) unit normal of an azimuth-from-north strike


def side_contrast(field: np.ndarray, paths, strikes, off_px: float) -> np.ndarray:
    h, w = field.shape
    v = field[np.isfinite(field)]
    scale = float(np.subtract(*np.percentile(v, [75, 25]))) if v.size else 1.0
    scale = scale if scale > 0 else 1.0
    out = []
    for p, s in zip(paths, strikes):
        n = _normals(np.array([s]))[0] * off_px
        vals = []
        for sg in (1, -1):
            q = np.rint(p + sg * n).astype(int)
            q[:, 0], q[:, 1] = np.clip(q[:, 0], 0, h - 1), np.clip(q[:, 1], 0, w - 1)
            vals.append(np.nanmean(field[q[:, 0], q[:, 1]]))
        out.append(abs(vals[0] - vals[1]) / scale if np.all(np.isfinite(vals)) else np.nan)
    return np.array(out, float)


def gradient_coherence(field: np.ndarray, sigma_px: float = 2.0) -> np.ndarray:
    f = np.nan_to_num(field, nan=float(np.nanmedian(field)) if np.isfinite(field).any() else 0.0)
    gy, gx = np.gradient(ndimage.gaussian_filter(f, 1.0))
    jxx, jyy, jxy = (ndimage.gaussian_filter(a, sigma_px) for a in (gx * gx, gy * gy, gx * gy))
    lam = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2)
    return np.where(jxx + jyy > 1e-20, lam / (jxx + jyy + 1e-20), 0).astype(np.float32)


def orientation_persistence(p: np.ndarray, step: int = 3) -> float:
    if len(p) < 2 * step + 1:
        return np.nan
    d = p[step::step] - p[:-step:step]
    a = 2 * np.arctan2(d[:, 1], -d[:, 0])
    return float(np.hypot(np.cos(a).mean(), np.sin(a).mean()))      # 1 = constant strike


def endpoint_connectivity(cs, shape) -> np.ndarray:
    idx = np.zeros(shape, np.int32)
    for i, p in enumerate(cs.paths):
        idx[p[:, 0], p[:, 1]] = i + 1
    h, w = shape
    out = []
    for i, p in enumerate(cs.paths):
        n = 0
        for e in (p[0], p[-1]):
            r0, r1, c0, c1 = max(e[0] - 2, 0), min(e[0] + 3, h), max(e[1] - 2, 0), min(e[1] + 3, w)
            win = idx[r0:r1, c0:c1]
            n += int(((win > 0) & (win != i + 1)).any())
        out.append(n)
    return np.array(out)


def describe(m7, u, cs, scores: dict, *, with_labels: bool = True, dem=None) -> pd.DataFrame:
    """Feature table of one unit's candidate set (crop coordinates; ``r_full`` / ``c_full`` for the midpoints)."""
    from ..gems3.candidates import CandidateSet
    from .generator import structural_score

    ctx, cfg = m7.ctx, m7.cfg
    fc = cfg["features"]
    if not len(cs):
        return pd.DataFrame()
    # descriptors.describe appends the geometry columns itself: drop copies stored by section 03
    cs = CandidateSet(cs.table.drop(columns=[c for c in GEOMETRY_COLS if c in cs.table]).reset_index(drop=True),
                      cs.paths)
    rc = {"ranker": {"positive_kd": cfg["ranking"]["positive_kd"], "negative_dist_km": cfg["ranking"]["negative_dist_km"]}}
    layers = {s["name"]: _crop_layer(ctx, s, u) for s in cfg["candidates"]["sources"]}
    layers = {k: v for k, v in layers.items() if v is not None}
    elev = ctx.data.read_window("det_elev", u.sl[0].start, u.sl[0].stop, u.sl[1].start, u.sl[1].stop) \
        if "det_elev" in ctx.data.band_names else None
    st = candidate_truth_stats(cs, u) if with_labels else None
    df = D.describe(u, cs, scores=scores, layers=layers, elevation=elev, truth_stats=st, cfg=rc)
    if not with_labels:
        df["label"] = -1
    paths = cs.paths
    strikes = df.strike_deg.to_numpy()
    df["straightness"] = df.chord_km / df.length_km.clip(lower=1e-6)
    df["n_members"] = [len(str(x).split(";")) for x in cs.table.get("member_uids", pd.Series([""] * len(cs)))]
    df["endpoint_connectivity"] = endpoint_connectivity(cs, u.ev.shape)
    df["orient_persistence"] = [orientation_persistence(p) for p in paths]
    df["structural_score"] = structural_score(df)
    for k, band in (fc.get("side_fields") or {}).items():
        if band not in ctx.data.band_names:
            continue
        f = ctx.data.read_window(band, u.sl[0].start, u.sl[0].stop, u.sl[1].start, u.sl[1].stop)
        for off in fc["side_offsets_m"]:
            col = f"side_{k}_{int(off)}m" if k != "terrain" else f"topo_side_{int(off)}m"
            df[col] = side_contrast(f, paths, strikes, float(off) / 100.0)
        gc = gradient_coherence(f)
        df[f"gradcoh_{k}" if k != "terrain" else "topo_gradcoh"] = [float(gc[p[:, 0], p[:, 1]].mean()) for p in paths]
    if "det_elev_slope" in ctx.data.band_names:
        s = ctx.data.read_window("det_elev_slope", u.sl[0].start, u.sl[0].stop, u.sl[1].start, u.sl[1].stop)
        df["topo_slope_asymmetry"] = side_contrast(s, paths, strikes, 3.0)
        df["topo_slope_mean"] = [float(np.nanmean(s[p[:, 0], p[:, 1]])) for p in paths]
    for nm, col in (("terrain:curvature@1km", "topo_curvature"), ("terrain:relief@2km", "topo_relief")):
        if ctx.store.has(nm):
            a = np.asarray(ctx.store.get(nm), np.float32)[u.sl]
            df[col] = [float(np.nanmean(np.abs(a[p[:, 0], p[:, 1]]))) for p in paths]
    if dem is not None:
        for b in m7.cfg["dem"]["candidate_layers"] + ["openness_diff_mean", "scarp_length_m"]:
            if dem.has(b):
                a = dem.band(b)[u.sl]
                v = [a[p[:, 0], p[:, 1]] for p in paths]
                df[f"dem_{b}"] = [float(np.nanmean(x)) if np.isfinite(x).any() else np.nan for x in v]
                df[f"dem_cov_{b}"] = [float(np.isfinite(x).mean()) for x in v]
    mods = [m for m in ("b0", "b1") if f"{m}_mean" in df]
    if len(mods) == 2:
        df["neural_agree"] = df[["b0_mean", "b1_mean"]].min(axis=1)
        df["neural_disagree"] = (df.b0_mean - df.b1_mean).abs()
        blk = max(int(round(float(fc["regional_block_km"]) / 0.1)), 10)
        b_id = (df.r_full // blk).astype(int).astype(str) + "_" + (df.c_full // blk).astype(int).astype(str)
        for m in mods:
            df[f"{m}_regional_rank"] = df.groupby(b_id)[f"{m}_mean"].rank(pct=True)
    df = df.join(network_context(u, cs, df, fc))
    df["unit"] = u.key
    df["is_completion"] = (df.kind.astype(str) == "completion").astype(int) if "kind" in df else 0
    return df


def _crop_layer(ctx, src: dict, u):
    if src.get("feature"):
        return np.asarray(ctx.store.get(src["feature"]), np.float32)[u.sl] if ctx.store.has(src["feature"]) else None
    b = src.get("band")
    if b and b in ctx.data.band_names:
        from gems_eda.enhance import nan_gaussian

        a = ctx.data.read_window(b, u.sl[0].start, u.sl[0].stop, u.sl[1].start, u.sl[1].stop)
        g = nan_gaussian(a, 3.0)
        gr, gc = np.gradient(np.nan_to_num(g, nan=float(np.nanmedian(g)) if np.isfinite(g).any() else 0.0))
        v = np.hypot(gr, gc)
        sc = float(np.nanpercentile(v, 90)) or 1.0
        return (v / sc).astype(np.float32)
    return None


def network_context(u, cs, df: pd.DataFrame, fc: dict) -> pd.DataFrame:
    """Continuation angle to the nearest known tip (within 1 km of a candidate end) and parallel-trace share."""
    from ..postprocess.topology import _axial_diff, _endpoint_tangents, orientation_field

    out = pd.DataFrame(index=df.index)
    out["cont_angle_deg"], out["parallel_known_share"] = np.nan, np.nan
    if u.known is None or not u.known.any():
        return out
    from ..geology.skeleton import skeletonize

    sk = skeletonize(u.known)
    tips = _endpoint_tangents(sk, 5)
    if tips:
        from scipy.spatial import cKDTree

        tt = np.array([(r, c) for r, c, _, _ in tips], float)
        az = np.array([(90.0 - d) % 180.0 for _, _, d, _ in tips])
        tree = cKDTree(tt)
        ang = []
        for p, s in zip(cs.paths, df.strike_deg.to_numpy()):
            best = np.nan
            for e in (p[0], p[-1]):
                dd, j = tree.query(e.astype(float))
                if dd <= 10:
                    a = abs(((s - az[j]) + 90) % 180 - 90)
                    best = a if not np.isfinite(best) else min(best, a)
            ang.append(best)
        out["cont_angle_deg"] = ang
    a_k, _ = orientation_field(u.known, 2.0)
    azk = (90.0 - a_k) % 180.0
    d, (ir, ic) = ndimage.distance_transform_edt(~u.known, sampling=(0.1, 0.1), return_indices=True)
    R = float(fc["parallel_radius_km"])
    sh = []
    for p, s in zip(cs.paths, df.strike_deg.to_numpy()):
        dd = d[p[:, 0], p[:, 1]]
        near = (dd > 0.15) & (dd <= R)
        if not near.any():
            sh.append(0.0)
            continue
        kk = azk[ir[p[:, 0], p[:, 1]], ic[p[:, 0], p[:, 1]]]
        sh.append(float((near & (_axial_diff(kk, s) <= 15.0)).mean()))
    out["parallel_known_share"] = sh
    return out


# =============================================================================== feature groups (ablations)
LABEL_COLS = {"label", "kd_mean", "kd_max", "near300", "dist_median_km"}
ID_NUMERIC = {"cand_id", "zone", "r_mid", "c_mid", "r_full", "c_full", "strike_deg", "score_evidence", "score_lgbm",
              "n_px"}


def groups(df: pd.DataFrame) -> dict[str, list[str]]:
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in LABEL_COLS | ID_NUMERIC]
    geom = [c for c in num if c in ("length_km", "chord_km", "sinuosity", "strike_cos2", "strike_sin2",
                                    "curvature_deg_per_km", "straightness", "n_members", "junction_ends",
                                    "endpoint_connectivity", "orient_persistence", "persistence_mean", "coherence_mean",
                                    "strength_mean", "n_scales", "scale_m", "conf_lf", "n_sources", "linked_px",
                                    "is_feature", "is_model", "is_tip", "structural_score")]
    geoph = [c for c in num if c.startswith(("side_magnetic", "side_gravity", "gradcoh_magnetic", "gradcoh_gravity",
                                             "orient_consistency_", "n_agree_geophysical"))
             or (c.startswith("ev_") and not c.startswith("ev_terr"))
             or (c.startswith("agree_") and c[6:] in GEOPHYS_MODS)]
    topo = [c for c in num if c.startswith(("topo_", "dem_", "ev_terr", "normal_asymmetry_elev"))
            or c in ("agree_terrain", "agree_dem")]
    neural = [c for c in num if c.startswith(("b0_", "b1_", "neural_")) or c == "agree_neural"]
    network = [c for c in num if c.startswith(("dist_known_", "cont_angle", "parallel_known")) or
               c in ("agree_network", "is_completion", "prior_dependency", "evidence_mean", "L_km", "angle_offset",
                     "dem_alignment", "n_agree_modalities")]
    return {"geometry": geom, "geophysics": geoph, "topography": topo, "neural": neural, "network": network}


def ablation_features(df: pd.DataFrame) -> dict[str, list[str]]:
    g = groups(df)
    return {"geom_geophys": g["geometry"] + g["geophysics"],
            "geom_geophys_dem": g["geometry"] + g["geophysics"] + g["topography"],
            "geom_geophys_dem_neural": g["geometry"] + g["geophysics"] + g["topography"] + g["neural"] + g["network"],
            "plus_completion": g["geometry"] + g["geophysics"] + g["topography"] + g["neural"] + g["network"]}


def label_free_score(df: pd.DataFrame) -> np.ndarray:
    """Required control: per-unit rank mean of geometry + geophysics + topography evidence (no labels, no neural)."""
    g = groups(df)
    cols = [c for c in ("structural_score", "n_agree_geophysical", "persistence_mean", "coherence_mean",
                        "strength_mean", "length_km", "straightness", "orient_persistence")
            if c in df] + [c for c in g["geophysics"] + g["topography"] if c.startswith(("side_", "gradcoh_", "ev_",
                                                                                          "topo_side", "dem_scarp"))]
    cols = list(dict.fromkeys(cols))
    if not cols:
        return np.zeros(len(df))
    r = df.groupby("unit")[cols].rank(pct=True)
    return r.mean(axis=1).fillna(0).to_numpy()


def neural_score(df: pd.DataFrame) -> np.ndarray:
    cols = [c for c in ("b0_mean", "b1_mean", "b0_p90", "b1_p90") if c in df]
    if not cols:
        return np.zeros(len(df))
    return df.groupby("unit")[cols].rank(pct=True).mean(axis=1).fillna(0).to_numpy()
