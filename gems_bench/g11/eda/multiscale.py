"""Feature utility by scale and region: block-sampled ROC-AUC (near-fault vs local control) per feature, scale
and region; best scale per family; redundancy (Spearman clusters, mutual information) within and across
resolutions."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _auc(pos, neg):
    from sklearn.metrics import roc_auc_score

    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if pos.size < 20 or neg.size < 20:
        return np.nan
    a = roc_auc_score(np.r_[np.ones(pos.size), np.zeros(neg.size)], np.r_[pos, neg])
    return float(max(a, 1 - a))                                      # direction-free separability


def utility_by_scale(store, meta: pd.DataFrame, pos_rc, neg_rc, regions) -> pd.DataFrame:
    (rp, cp), (rn, cn) = pos_rc, neg_rc
    rows = []
    for r in meta.itertuples():
        a = store.get(r.name)
        vp, vn = np.asarray(a[rp, cp], np.float64), np.asarray(a[rn, cn], np.float64)
        rows.append({"feature": r.name, "family": r.family, "scale_m": r.scale_m, "region": "all", "auc": _auc(vp, vn)})
        gp, gn = regions[rp, cp], regions[rn, cn]
        for k in np.unique(gp[gp >= 0]):
            rows.append({"feature": r.name, "family": r.family, "scale_m": r.scale_m, "region": int(k),
                         "auc": _auc(vp[gp == k], vn[gn == k])})
    return pd.DataFrame(rows)


def native_scale_contrast(reader, segments, grid, cfg: dict, n_sites: int = 120, half_m: float = 160.0,
                          seed: int = 2026) -> pd.DataFrame:
    """Terrain features at NATIVE DEM resolution, on mapped-fault stations vs local controls (1.5 km normal offset).

    Each site: a window of +-``half_m`` read at ``cross_fault.resolution_m``; the median of each feature in the central
    11 x 11 m is kept. Scales are limited to those the window and resolution support (``supported_scales``);
    unsupported ones (e.g. 250-500 m) are reported only at 100 m via ``dem100_*_ctx*``. Direction-free AUC per
    feature / scale (fault vs control).
    """
    import re

    from ..features.terrain import compute_terrain_features, supported_scales

    if reader is None or segments is None or not len(segments):
        return pd.DataFrame()
    res = float(cfg["feature_engineering"]["cross_fault"]["resolution_m"])
    n_px = int(2 * half_m / res)
    scales, skipped = supported_scales(cfg["feature_engineering"]["terrain"]["scales_m"], res, (n_px, n_px))
    rng = np.random.default_rng(seed)
    segs = segments.sample(min(len(segments), n_sites), random_state=int(rng.integers(1 << 31)))
    vals = {"fault": [], "control": []}
    for g in segs.geometry:
        t = rng.uniform(0.2, 0.8)
        p, q = g.interpolate(max(t - 0.01, 0), normalized=True), g.interpolate(min(t + 0.01, 1), normalized=True)
        tx, ty = q.x - p.x, q.y - p.y
        nn = np.hypot(tx, ty) + 1e-12
        c = g.interpolate(t, normalized=True)
        for kind, (x, y) in (("fault", (c.x, c.y)), ("control", (c.x + 1500 * ty / nn, c.y - 1500 * tx / nn))):
            try:
                w = reader.read_dem_window((x - half_m, y - half_m, x + half_m, y + half_m), grid.crs, resolution=res)
            except Exception:                                    # noqa: BLE001 - missing DEM at a site
                continue
            if w.valid.mean() < 0.6:
                continue
            tf = compute_terrain_features(w.elevation, w.valid, res, scales, cfg,
                                          families=("first_order", "second_order", "structural"))
            h, wd = w.elevation.shape
            k = max(int(5.5 / res), 1)
            core = (slice(h // 2 - k, h // 2 + k + 1), slice(wd // 2 - k, wd // 2 + k + 1))
            vals[kind].append({n: float(np.nanmedian(a[core])) for n, a in tf.arrays.items()})
    F, C = pd.DataFrame(vals["fault"]), pd.DataFrame(vals["control"])
    rows = []
    for n in sorted(set(F.columns) & set(C.columns)):
        m = re.search(r"_(\d+)m$", n)
        rows.append({"feature": n, "family": "dem_native", "scale_m": float(m.group(1)) if m else None, "region": "all",
                     "auc": _auc(F[n].to_numpy(), C[n].to_numpy()), "n_fault": int(F[n].notna().sum()),
                     "n_control": int(C[n].notna().sum()), "resolution_m": res})
    out = pd.DataFrame(rows)
    out.attrs["skipped_scales"] = skipped
    return out


def best_scales(util: pd.DataFrame) -> pd.DataFrame:
    u = util[(util.region == "all") & util.scale_m.notna()].copy()
    u["base"] = u.feature.str.replace(r"_\d+m$", "", regex=True)
    return u.sort_values("auc", ascending=False).groupby("base").head(1)[["base", "family", "scale_m", "auc"]]


def redundancy(store, names, rows, cols, rho: float = 0.9, mi_n: int = 5000, seed: int = 2026) -> pd.DataFrame:
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    from scipy.stats import spearmanr

    X = np.stack([np.asarray(store.get(n)[rows, cols], np.float64) for n in names], 1)
    ok = np.isfinite(X).all(1)
    X = X[ok]
    if X.shape[0] < 50 or X.shape[1] < 2:
        return pd.DataFrame({"feature": names, "cluster": np.arange(len(names))})
    R = spearmanr(X).correlation
    R = np.atleast_2d(R)
    D = np.clip(1 - np.abs(np.nan_to_num(R)), 0, 2)
    np.fill_diagonal(D, 0)
    cl = fcluster(linkage(squareform(D, checks=False), "average"), t=1 - rho, criterion="distance")
    rng = np.random.default_rng(seed)
    sub = X[rng.choice(len(X), min(mi_n, len(X)), replace=False)]
    from sklearn.feature_selection import mutual_info_regression

    mi_first = mutual_info_regression(sub, sub[:, 0], random_state=seed) if sub.shape[1] > 1 else np.zeros(1)
    out = pd.DataFrame({"feature": names, "cluster": cl, "max_abs_rho_other": [
        float(np.max(np.abs(np.delete(R[i], i)))) for i in range(len(names))]})
    out["mi_with_first_feature"] = mi_first
    out["representative"] = out.groupby("cluster").max_abs_rho_other.transform(lambda s: s == s.min())
    return out
