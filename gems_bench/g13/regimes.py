"""Unsupervised geological regimes and fault archetypes (sections 15-16).

Regimes cluster *geophysical signatures* (never coordinates). Transformations and clusterings are fitted on a sample
of CV training pixels only; the number of clusters is chosen by a rule fixed in advance: among k whose mean
seed-to-seed adjusted Rand index is >= 0.75, the k with the best silhouette (ties -> smaller k); if none is stable,
the most stable k. UMAP is not used as evidence of cluster validity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

REGIME_FEATURES = (
    "raw:rtp",
    "raw:iso_grav_anom",
    "raw:det_elev",
    "raw:cond_surf",
    "raw:depth_to_base_surf",
    "raw:tmi_hg",
    "raw:iso_grav_anom_hg",
    "raw:det_elev_slope",
    "raw:geod_shearrate",
    "magnetic:rtp:uc2000_thd",
    "gravity:iso_grav_anom:uc2000_thd",
    "terrain:det_elev:lstd_s20",
    "magnetic:rtp:coh_s1600",
    "gravity:iso_grav_anom:coh_s1600",
    "magnetic:rtp:entropy_w9",
)


def _matrix(store, ids, names):
    X = store.matrix(ids, names)
    med = np.nanmedian(X, 0)
    X = np.where(np.isfinite(X), X, med)
    return X


def fit_regimes(store, data, sp, rng, k_range=range(3, 11), n_fit: int = 60_000, seeds=(0, 1, 2, 3, 4)) -> dict:
    from sklearn.cluster import HDBSCAN, MiniBatchKMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
    from sklearn.preprocessing import RobustScaler

    names = [n for n in REGIME_FEATURES if store.has(n)]
    train = np.zeros(data.shape, bool)
    for k in sp.cv_folds:
        train |= sp.train(k)
    pool = np.flatnonzero((train & data.valid).ravel())
    ids = np.sort(rng.choice(pool, min(n_fit, pool.size), replace=False))
    X = _matrix(store, ids, names)
    scaler = RobustScaler().fit(X)
    Z = np.clip(scaler.transform(X), -10, 10)
    pca = PCA(n_components=min(8, Z.shape[1]), random_state=0).fit(Z)
    P = pca.transform(Z)
    hold = rng.choice(len(P), min(8000, len(P)), replace=False)
    rows = []
    for k in k_range:
        labs = [MiniBatchKMeans(k, random_state=s, n_init=3, batch_size=4096).fit(P).predict(P[hold]) for s in seeds]
        ari = [adjusted_rand_score(labs[0], l) for l in labs[1:]]
        sil = silhouette_score(P[hold], labs[0])
        g = GaussianMixture(k, covariance_type="full", random_state=0).fit(P[::4])
        rows.append(
            {
                "k": k,
                "kmeans_ari_mean": float(np.mean(ari)),
                "kmeans_silhouette": float(sil),
                "gmm_bic": float(g.bic(P[::4])),
            }
        )
    sel = pd.DataFrame(rows)
    stable = sel[sel.kmeans_ari_mean >= 0.75]
    k_best = (
        int(stable.sort_values(["kmeans_silhouette", "k"], ascending=[False, True]).k.iloc[0])
        if len(stable)
        else int(sel.sort_values("kmeans_ari_mean", ascending=False).k.iloc[0])
    )
    km = MiniBatchKMeans(k_best, random_state=0, n_init=5, batch_size=4096).fit(P)
    bgm = BayesianGaussianMixture(n_components=10, weight_concentration_prior=0.1, random_state=0, max_iter=300).fit(P[::4])
    hdb = HDBSCAN(min_cluster_size=200).fit(P[hold[:5000]])
    comparison = {
        "bgmm_effective_components": int((bgm.weights_ > 0.02).sum()),
        "hdbscan_clusters": int(len(set(hdb.labels_)) - (1 if -1 in hdb.labels_ else 0)),
        "hdbscan_noise_frac": float((hdb.labels_ == -1).mean()),
    }
    return {
        "names": names,
        "scaler": scaler,
        "pca": pca,
        "kmeans": km,
        "k": k_best,
        "selection": sel,
        "comparison": comparison,
        "explained_variance": pca.explained_variance_ratio_.tolist(),
    }


def assign_regimes(model: dict, store, data, chunk_rows: int = 256) -> np.ndarray:
    h, w = data.shape
    out = np.full((h, w), -1, np.int16)
    for r0 in range(0, h, chunk_rows):
        r1 = min(h, r0 + chunk_rows)
        m = data.valid[r0:r1]
        if not m.any():
            continue
        ids = (np.flatnonzero(m.ravel()) + r0 * w).astype(np.int64)
        X = _matrix(store, ids, model["names"])
        P = model["pca"].transform(np.clip(model["scaler"].transform(X), -10, 10))
        out.ravel()[ids] = model["kmeans"].predict(P).astype(np.int16)
    return out


def regime_atlas(regimes: np.ndarray, data, sp, store) -> pd.DataFrame:
    cv = sp.cv_region()
    rows = []
    same = 0.5 * ((regimes[1:, :] == regimes[:-1, :])[:, :-1] + (regimes[:, 1:] == regimes[:, :-1])[:-1, :])
    for z in sorted(set(np.unique(regimes)) - {-1}):
        m = regimes == z
        r = {
            "regime": int(z),
            "area_km2": float(m.sum() * 0.01),
            "area_frac": float(m.sum() / max(data.valid.sum(), 1)),
            "fault_density_cv": float((data.labels & m & cv).sum() / max((m & cv).sum(), 1)),
            "spatial_coherence": float(same[m[:-1, :-1]].mean()) if m[:-1, :-1].any() else np.nan,
        }
        for b in ("rtp", "iso_grav_anom", "det_elev", "cond_surf", "depth_to_base_surf"):
            if data.has(b):
                x = data.band(b)
                r[f"median_{b}"] = float(np.nanmedian(x[m]))
        rows.append(r)
    return pd.DataFrame(rows)


ARCHETYPE_ORDER = (
    "joint_magnetic_gravity",
    "strong_magnetic",
    "strong_gravity",
    "terrain_visible",
    "buried_no_terrain",
    "weak_persistent",
    "weak",
)


def evidence_ranks(store, data) -> dict[str, np.ndarray]:
    """Per-modality edge evidence as a global percentile, maximum within 300 m (label-free)."""
    from .features import _rank01

    cand = {
        "magnetic": ("magnetic:rtp:thd", "raw:tmi_hg"),
        "gravity": ("gravity:iso_grav_anom:thd", "raw:iso_grav_anom_hg"),
        "terrain": ("terrain:det_elev:slope_break_s400", "raw:det_elev_slope"),
        "persistence": ("structural:rtp:orient_persistence",),
    }
    out = {}
    for k, names in cand.items():
        for n in names:
            if store.has(n):
                r = _rank01(store.get(n), data.valid)
                out[k] = ndimage.maximum_filter(np.nan_to_num(r), size=7)
                break
    return out


def archetypes(ev: dict[str, np.ndarray], mask: np.ndarray) -> np.ndarray:
    """Empirical signal category of every pixel in ``mask`` (rules fixed in advance; signal classes, not geology)."""
    z = np.zeros(mask.shape, np.float32)
    mg, gr, te, pe = (ev.get(k, z) for k in ("magnetic", "gravity", "terrain", "persistence"))
    out = np.full(mask.shape, -1, np.int8)
    rules = [
        (mg >= 0.9) & (gr >= 0.9),
        mg >= 0.9,
        gr >= 0.9,
        (te >= 0.9),
        ((mg >= 0.8) | (gr >= 0.8)) & (te < 0.5),
        pe >= 0.7,
        np.ones(mask.shape, bool),
    ]
    for i, r in enumerate(rules):
        sel = mask & (out < 0) & r
        out[sel] = i
    return out


def archetype_table(arch: np.ndarray, mask: np.ndarray, support: dict | None = None) -> pd.DataFrame:
    rows = []
    for i, name in enumerate(ARCHETYPE_ORDER):
        m = mask & (arch == i)
        r = {"archetype": name, "n_px": int(m.sum()), "share": float(m.sum() / max(mask.sum(), 1))}
        if support:
            for k, (rc, sup) in support.items():
                sel = m[rc[:, 0], rc[:, 1]]
                r[f"recall_w_{k}"] = float(sup[sel].mean()) if sel.any() else np.nan
        rows.append(r)
    return pd.DataFrame(rows)
