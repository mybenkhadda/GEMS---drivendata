"""Feature quality, uniqueness and redundancy - label-free.

Per scalar feature (axial components excluded from correlations): valid fraction over the footprint, robust
location / spread (median, IQR, MAD), lag-1 spatial autocorrelation (noise proxy), Spearman correlation with the
distance to the footprint edge and the seam / non-seam median ratio for DEM features (artifact proxies), and the
adjacent-scale Spearman of the same operator (scale consistency). Spearman matrix on a block sample; redundancy
clusters by average-linkage on 1 - |rho| cut at 1 - ``redundant_rho``; the cluster representative is the member
with the highest valid fraction, then the highest lag-1 autocorrelation.

Classes (first rule that applies): spatially_limited (valid < ``spatially_limited``), potentially_artifact
(seam ratio > ``artifact_ratio`` or |rho(edge distance)| > 0.5), noisy (lag-1 < ``noisy_lag1``), redundant
(non-representative cluster member), informative_distinct. ``retain`` = informative_distinct or representative
(and not artifact / noisy). No label is used anywhere in this module.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .magnetic import block_sample


def _scale_key(fid: str):
    """('mag_thd_uc', 1000) for mag_thd_uc1000, ('terrain_slope', 300) for terrain_slope_300m, else (None, None)."""
    m = re.match(r"^(.*?[a-z_])(\d+)m?$", fid)
    return (m.group(1).rstrip("_"), int(m.group(2))) if m else (None, None)


def quality(feature_ids: list[str], loader, footprint: np.ndarray, edge_dist: np.ndarray, seam: np.ndarray | None,
            dem_cov: np.ndarray | None, cfg, rng) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """``loader(feature_id) -> array``; features are read one at a time (bounded memory)."""
    a = cfg["analytics"]
    rr, cc = block_sample(footprint, int(a["sample_block_px"]), rng, int(a["max_samples"]))
    rows, samples = [], {}
    cols = np.clip(cc + 1, 0, footprint.shape[1] - 1)
    for fid in feature_ids:
        x = loader(fid)
        v = x[rr, cc]
        fin = np.isfinite(v)
        vf = float(np.isfinite(x[footprint]).mean())
        rec = {"feature_id": fid, "valid_fraction": vf, "n_sample": int(fin.sum())}
        if fin.sum() > 20:
            q = np.percentile(v[fin], [25, 50, 75])
            rec.update(median=q[1], iqr=q[2] - q[0], mad=float(np.median(np.abs(v[fin] - q[1]))))
            nb = x[rr, cols]
            ok = fin & np.isfinite(nb)
            rec["lag1_autocorr"] = float(np.corrcoef(v[ok], nb[ok])[0, 1]) if ok.sum() > 20 and np.std(v[ok]) > 0 else np.nan
            rec["rho_edge_distance"] = float(spearmanr(v[fin], edge_dist[rr, cc][fin]).statistic)
            if seam is not None and dem_cov is not None and fid.startswith("dem_"):
                s_in = np.abs(x[seam & np.isfinite(x)])
                s_out = np.abs(x[~seam & (dem_cov > 0) & np.isfinite(x)])
                med_out = float(np.median(s_out)) if s_out.size else 0.0
                rec["seam_ratio"] = float(np.median(s_in) / med_out) if s_in.size > 20 and med_out > 0 else np.nan
            samples[fid] = v
        rows.append(rec)
    q = pd.DataFrame(rows)
    names = [f for f in samples if np.nanstd(samples[f]) > 0]
    M = np.column_stack([samples[f] for f in names]) if names else np.zeros((0, 0))
    rho = pd.DataFrame(np.nan, index=names, columns=names)
    if len(names) > 1:
        df = pd.DataFrame(M, columns=names)
        rho = df.corr(method="spearman", min_periods=200)
    # scale consistency (adjacent scales of the same operator)
    sc = []
    keyed = {}
    for f in names:
        k, s = _scale_key(f)
        if k:
            keyed.setdefault(k, []).append((s, f))
    for k, lst in keyed.items():
        lst.sort()
        for (s1, f1), (s2, f2) in zip(lst[:-1], lst[1:]):
            sc.append({"operator": k, "scale_a": s1, "scale_b": s2, "feature_a": f1, "feature_b": f2,
                       "spearman": float(rho.loc[f1, f2])})
    clusters = _clusters(rho, float(a["redundant_rho"]))
    q = q.merge(clusters, on="feature_id", how="left")
    if "seam_ratio" not in q:
        q["seam_ratio"] = np.nan
    for c in ("lag1_autocorr", "rho_edge_distance"):
        if c not in q:
            q[c] = np.nan
    # features without a usable sample form their own cluster
    q["cluster"] = np.where(q["cluster"].isna(), -1 - np.arange(len(q)), q["cluster"]).astype(int)
    q["class"] = [_classify(r, a) for r in q.itertuples()]
    return finalize_classes(q), rho, pd.DataFrame(sc)


def _clusters(rho: pd.DataFrame, thr: float) -> pd.DataFrame:
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    names = list(rho.index)
    if len(names) < 2:
        return pd.DataFrame({"feature_id": names, "cluster": range(len(names)), "representative": True})
    D = 1 - rho.abs().fillna(0).to_numpy()
    np.fill_diagonal(D, 0)
    Z = linkage(squareform(np.clip(D, 0, None), checks=False), "average")
    lab = fcluster(Z, t=1 - thr, criterion="distance")
    return pd.DataFrame({"feature_id": names, "cluster": lab})


def _classify(r, a) -> str:
    if r.valid_fraction < float(a["spatially_limited"]):
        return "spatially_limited"
    seam = getattr(r, "seam_ratio", np.nan)
    if (np.isfinite(seam) and seam > float(a["artifact_ratio"])) or abs(getattr(r, "rho_edge_distance", 0) or 0) > 0.5:
        return "potentially_artifact"
    if np.isfinite(getattr(r, "lag1_autocorr", np.nan)) and r.lag1_autocorr < float(a["noisy_lag1"]):
        return "noisy"
    return "candidate"


def finalize_classes(q: pd.DataFrame) -> pd.DataFrame:
    """Pick a representative per redundancy cluster among 'candidate' members; others become 'redundant'."""
    q = q.copy()
    q["representative"] = False
    cand = q[q["class"] == "candidate"].sort_values(["valid_fraction", "lag1_autocorr"], ascending=False)
    rep = cand.drop_duplicates("cluster").index
    q.loc[rep, "representative"] = True
    q.loc[(q["class"] == "candidate") & ~q.index.isin(rep), "class"] = "redundant"
    single = q.groupby("cluster").feature_id.transform("count") == 1
    q.loc[(q["class"] == "candidate") & single, "class"] = "informative_distinct"
    q.loc[q["class"] == "candidate", "class"] = "informative_distinct"
    q["retain"] = (q["class"] == "informative_distinct") & q.representative
    q["retention_reason"] = np.select(
        [q.retain, q["class"] == "redundant", q["class"] == "spatially_limited", q["class"] == "noisy",
         q["class"] == "potentially_artifact"],
        ["cluster representative, spatially coherent, wide coverage", "near-duplicate of the cluster representative",
         "covers too little of the footprint", "low spatial autocorrelation (pixel noise)",
         "pattern follows DEM seams or the footprint edge"], "not retained")
    return q
