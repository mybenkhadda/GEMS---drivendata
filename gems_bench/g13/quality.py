"""Feature quality, physical plausibility, redundancy and compact selection (sections 14, 21 A10/A11, 27)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.stats import ks_2samp

REJECT_RULES = {"min_coverage": 0.5, "min_unique": 6, "max_boundary_dependence": 0.6, "duplicate_rho": 0.98}


def quality_table(store, data, sp, rng, n: int = 40_000, names: list[str] | None = None) -> pd.DataFrame:
    names = names or store.names()
    valid_ids = np.flatnonzero(data.valid.ravel())
    ids = np.sort(rng.choice(valid_ids, min(n, valid_ids.size), replace=False))
    bd_name = "validity:all:boundary_dist_km"
    bd = store.column(bd_name, ids) if store.has(bd_name) else None
    if bd is not None and len(store.registry):
        r = store.registry.loc[store.registry.name == bd_name].iloc[0]
        bd = bd * float(r.scale_iqr) + float(r.scale_median)
    fold = sp.fold_map.reshape(-1)[ids]
    rows = []
    reg = store.registry.set_index("name") if len(store.registry) else pd.DataFrame()
    for nm in names:
        x = store.column(nm, ids)
        fin = np.isfinite(x)
        v = x[fin]
        row = {
            "feature": nm,
            "family": store.family_of(nm),
            "finite_frac_sample": float(fin.mean()),
            "variance": float(np.var(v)) if v.size else np.nan,
            "dynamic_range_p01_p99": float(np.subtract(*np.percentile(v, [99, 1]))) if v.size else np.nan,
            "n_unique": int(len(np.unique(v[:20000]))) if v.size else 0,
        }
        if nm in reg.index:
            rr = reg.loc[nm]
            row.update(
                coverage_frac=float(rr.coverage_frac),
                seconds=float(rr.seconds),
                storage_mb=float(rr.bytes) / 2**20,
                scale_m=rr.get("scale_m"),
                modality=rr.get("modality"),
                common_source=bool(rr.get("common_source")),
            )
        else:
            row.update(coverage_frac=data.coverage.get(nm[4:], np.nan), seconds=0.0, storage_mb=0.0, modality="raw")
        # nodata-boundary dependence: |Spearman(feature, boundary distance)| within 3 km of a boundary
        if bd is not None:
            m = fin & np.isfinite(bd) & (bd < 3.0)
            row["boundary_dependence"] = (
                float(abs(pd.Series(x[m]).corr(pd.Series(bd[m]), method="spearman"))) if m.sum() > 200 else np.nan
            )
        # train / validation distribution shift (label-free): mean KS over CV folds
        ks = []
        for k in sp.cv_folds:
            a, b = x[fin & (fold != k) & (fold >= 0)], x[fin & (fold == k)]
            if len(a) > 200 and len(b) > 200:
                ks.append(ks_2samp(a[:5000], b[:5000]).statistic)
        row["shift_ks_mean"] = float(np.mean(ks)) if ks else np.nan
        rows.append(row)
    q = pd.DataFrame(rows)
    return q


def smoothing_sensitivity(store, names: list[str], data, rng, n: int = 20_000, sigma: float = 2.0) -> pd.Series:
    """1 - Spearman(feature, feature smoothed by sigma px): high = dominated by pixel-scale noise."""
    valid_ids = np.flatnonzero(data.valid.ravel())
    ids = np.sort(rng.choice(valid_ids, min(n, valid_ids.size), replace=False))
    out = {}
    for nm in names:
        a = store.get(nm)
        v = np.isfinite(a)
        f = np.where(v, a, 0)
        sm = ndimage.gaussian_filter(f, sigma) / np.maximum(ndimage.gaussian_filter(v.astype(np.float32), sigma), 1e-6)
        x, y = a.ravel()[ids], sm.ravel()[ids]
        m = np.isfinite(x) & np.isfinite(y)
        out[nm] = float(1 - pd.Series(x[m]).corr(pd.Series(y[m]), method="spearman")) if m.sum() > 100 else np.nan
    return pd.Series(out, name="smoothing_sensitivity")


def redundancy_clusters(store, names: list[str], data, rng, n: int = 30_000, rho: float = 0.98) -> pd.DataFrame:
    """Groups of features with |Spearman| >= rho (complete-linkage); one representative per group."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    valid_ids = np.flatnonzero(data.valid.ravel())
    ids = np.sort(rng.choice(valid_ids, min(n, valid_ids.size), replace=False))
    X = pd.DataFrame(store.matrix(ids, names), columns=names)
    C = np.array(X.rank().corr().abs().fillna(0).to_numpy(), dtype=float, copy=True)
    np.fill_diagonal(C, 1.0)
    D = np.clip(1 - C, 0, None)
    if len(names) < 2:
        return pd.DataFrame({"feature": names, "cluster": [0] * len(names)})
    Z = linkage(squareform(D, checks=False), method="complete")
    lab = fcluster(Z, t=1 - rho, criterion="distance")
    return pd.DataFrame({"feature": names, "cluster": lab})


def apply_rejection(
    q: pd.DataFrame, clusters: pd.DataFrame | None, utility: pd.Series | None = None, rules: dict = REJECT_RULES
) -> pd.DataFrame:
    """Mark features as kept / rejected with an explicit reason (raw bands are never rejected)."""
    q = q.copy()
    q["reject_reason"] = ""
    is_raw = q.family == "raw"
    q.loc[~is_raw & (q.coverage_frac < rules["min_coverage"]), "reject_reason"] += "low_coverage;"
    q.loc[~is_raw & (q.n_unique < rules["min_unique"]), "reject_reason"] += "degenerate;"
    if "boundary_dependence" in q:
        q.loc[~is_raw & (q.boundary_dependence > rules["max_boundary_dependence"]), "reject_reason"] += "nodata_boundary;"
    if clusters is not None and len(clusters):
        q = q.merge(clusters, on="feature", how="left")
        u = utility.reindex(q.feature).fillna(0).to_numpy() if utility is not None else -q.seconds.fillna(0).to_numpy()
        q["_u"] = u + is_raw.to_numpy() * 1e6  # raw bands always represent their cluster
        rep = q.sort_values("_u", ascending=False).groupby("cluster").feature.first()
        dup = ~q.feature.isin(rep.values) & ~is_raw
        q.loc[dup, "reject_reason"] += "duplicate;"
        q = q.drop(columns="_u")
    q["kept"] = q.reject_reason == ""
    return q


def compact_selection(importance: pd.DataFrame, quality: pd.DataFrame, max_features: int = 40) -> list[str]:
    """A10: highest-importance kept features (raw bands always included)."""
    kept = set(quality.loc[quality.kept, "feature"])
    raw = quality.loc[quality.family == "raw", "feature"].tolist()
    imp = importance[importance.feature.isin(kept) & ~importance.feature.isin(raw)].sort_values(
        "importance", ascending=False
    )
    return raw + imp.feature.head(max(max_features - len(raw), 0)).tolist()


def complementary_families(ablation: pd.DataFrame, metric: str, base: str = "A0_raw") -> list[str]:
    """A11: families whose single-family experiment beat raw-only on ``metric`` (greedy, ordered by gain)."""
    if ablation.empty or metric not in ablation:
        return []
    b = ablation.loc[ablation.experiment == base, metric]
    if b.empty:
        return []
    gains = ablation.set_index("experiment")[metric] - float(b.iloc[0])
    single = [
        e for e in gains.index if e.startswith("A") and e[1:].split("_")[0].isdigit() and 1 <= int(e[1:].split("_")[0]) <= 8
    ]
    return [e for e in gains.loc[single].sort_values(ascending=False).index if gains[e] > 0]
