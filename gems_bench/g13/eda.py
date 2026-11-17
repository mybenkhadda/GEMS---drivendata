"""Deep exploratory geological analysis (sections 05-08).

All label-conditioned statistics are computed on the CV region only (the TEST partition stays untouched), and
unmapped pixels are treated as *unlabeled*, never as proven fault-free: every contrast is reported against several
control populations so that a signal that only separates faults from flat basin floor is visible as such.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

from . import signal as S
from .models import distance_px

CONTROL_NAMES = (
    "random_background",
    "matched_1_3km",
    "high_gradient",
    "ridge_drainage",
    "magnetic_lineament",
    "gravity_lineament",
    "dem_boundary",
    "model_disagreement",
)


# =============================================================================== populations
def populations(data, store, sp, rng, n: int = 20_000, disagreement: np.ndarray | None = None) -> dict:
    """Flat pixel indices of the fault corridor and the control populations (CV region only)."""
    cv = sp.cv_region()
    d = distance_px(data.labels) * 0.1  # km
    far = cv & (d >= 2.0)
    pick = lambda m: (
        rng.choice(np.flatnonzero(m.ravel()), min(n, int(m.sum())), replace=False) if m.any() else np.empty(0, np.int64)
    )
    out = {
        "fault_corridor": pick(cv & (d <= 0.1)),
        "random_background": pick(far),
        "matched_1_3km": pick(cv & (d >= 1.0) & (d <= 3.0)),
    }

    def top(name, q=0.9):
        if not store.has(name):
            return None
        a = store.get(name)
        v = far & np.isfinite(a)
        return v & (a >= np.quantile(a[v], q)) if v.any() else None

    for key, names in (
        ("high_gradient", ("terrain:det_elev:slope_s400", "magnetic:rtp:grad_s400")),
        ("ridge_drainage", ("terrain:det_elev:ridge_s400", "terrain:det_elev:valley_s400")),
        ("magnetic_lineament", ("magnetic:rtp:tdx",)),
        ("gravity_lineament", ("gravity:iso_grav_anom:tdx",)),
    ):
        m = None
        for nm in names:
            t = top(nm, 0.95 if "lineament" in key else 0.9)
            if t is not None:
                m = t if m is None else (m | t)
        if m is not None:
            out[key] = pick(m)
    if store.has("validity:all:boundary_dist_km"):
        b = store.physical("validity:all:boundary_dist_km")
        out["dem_boundary"] = pick(far & np.isfinite(b) & (b < 1.5))
    if disagreement is not None:
        v = far & np.isfinite(disagreement)
        if v.any():
            out["model_disagreement"] = pick(v & (disagreement >= np.quantile(disagreement[v], 0.95)))
    return {k: v for k, v in out.items() if len(v) >= 100}


# =============================================================================== contrasts
def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """P(a > b) - P(a < b) via ranks (O(n log n))."""
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if not len(a) or not len(b):
        return np.nan
    from scipy.stats import rankdata

    r = rankdata(np.r_[a, b])
    ua = r[: len(a)].sum() - len(a) * (len(a) + 1) / 2
    return float(2 * ua / (len(a) * len(b)) - 1)


def overlap_coefficient(a: np.ndarray, b: np.ndarray, bins: int = 64) -> float:
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if not len(a) or not len(b):
        return np.nan
    lo, hi = np.percentile(np.r_[a, b], [0.5, 99.5])
    if hi <= lo:
        return 1.0
    ha, e = np.histogram(np.clip(a, lo, hi), bins, (lo, hi), density=True)
    hb, _ = np.histogram(np.clip(b, lo, hi), bins, (lo, hi), density=True)
    return float(np.minimum(ha, hb).sum() * (e[1] - e[0]))


def mutual_information(a: np.ndarray, b: np.ndarray, rng, n_perm: int = 3) -> tuple[float, float]:
    """MI (nats) between the feature and population membership, and the mean MI of label permutations (bias)."""
    from sklearn.feature_selection import mutual_info_classif

    x = np.r_[a, b]
    y = np.r_[np.ones(len(a)), np.zeros(len(b))]
    ok = np.isfinite(x)
    x, y = x[ok][:, None], y[ok]
    if len(np.unique(y)) < 2 or len(x) < 50:
        return np.nan, np.nan
    mi = float(mutual_info_classif(x, y, random_state=0)[0])
    perm = [float(mutual_info_classif(x, rng.permutation(y), random_state=0)[0]) for _ in range(n_perm)]
    return mi, float(np.mean(perm))


def weighted_cliffs(a: np.ndarray, wa: np.ndarray, c_sorted: np.ndarray, wc_sorted: np.ndarray) -> float:
    """Cliff's delta with integer resampling weights (bootstrap without materialising repeated samples)."""
    cum = np.r_[0.0, np.cumsum(wc_sorted)]
    lo = np.searchsorted(c_sorted, a, side="left")
    hi = np.searchsorted(c_sorted, a, side="right")
    below, above = cum[lo], cum[-1] - cum[hi]
    tot = wa.sum() * cum[-1]
    return float((wa * (below - above)).sum() / tot) if tot > 0 else np.nan


def block_bootstrap_cliffs(fa, fault_ids, ca, ids, width, block_px, rng, n_boot: int = 60, cap: int = 4000) -> list:
    """Spatial block bootstrap of Cliff's delta: 20 km blocks resampled with replacement, shared by both
    populations (vectorised; each draw is one weighted Mann-Whitney evaluation)."""
    ok_a, ok_c = np.isfinite(fa), np.isfinite(ca)
    fa, fault_ids, ca, ids = fa[ok_a], fault_ids[ok_a], ca[ok_c], ids[ok_c]
    if len(fa) > cap:
        s = rng.choice(len(fa), cap, replace=False)
        fa, fault_ids = fa[s], fault_ids[s]
    if len(ca) > cap:
        s = rng.choice(len(ca), cap, replace=False)
        ca, ids = ca[s], ids[s]
    ba = (fault_ids // width // block_px) * 100_000 + (fault_ids % width) // block_px
    bc = (ids // width // block_px) * 100_000 + (ids % width) // block_px
    ub, inv = np.unique(np.r_[ba, bc], return_inverse=True)
    ia, ic = inv[: len(ba)], inv[len(ba) :]
    order = np.argsort(ca, kind="stable")
    cs, ics = ca[order], ic[order]
    out = []
    for _ in range(n_boot):
        cnt = np.bincount(rng.integers(0, len(ub), len(ub)), minlength=len(ub)).astype(np.float64)
        wa, wc = cnt[ia], cnt[ics]
        if wa.sum() > 0 and wc.sum() > 0:
            out.append(weighted_cliffs(fa, wa, cs, wc))
    return out


def contrast_table(
    store, names: list[str], pops: dict, sp, rng, n_boot: int = 100, block_px: int = 200, with_mi: bool = True
) -> pd.DataFrame:
    """Fault corridor vs every control population, per feature: robust standardized median difference, Cliff's
    delta with a spatial-block bootstrap interval, overlap coefficient, MI vs permutation baseline, and the share
    of CV folds where the contrast has the same sign (spatial stability)."""
    fault = pops["fault_corridor"]
    w = sp.valid.shape[1]
    fold_of = sp.fold_map.reshape(-1)
    rows = []
    for name in names:
        fa = store.column(name, np.sort(fault))
        fault_ids = np.sort(fault)
        for ctrl, ids in pops.items():
            if ctrl == "fault_corridor":
                continue
            ids = np.sort(ids)
            ca = store.column(name, ids)
            ok_a, ok_c = np.isfinite(fa), np.isfinite(ca)
            if ok_a.sum() < 50 or ok_c.sum() < 50:
                continue
            med_a, med_c = np.median(fa[ok_a]), np.median(ca[ok_c])
            mad = 1.4826 * np.median(np.abs(ca[ok_c] - med_c)) or np.std(ca[ok_c]) or 1.0
            cd = cliffs_delta(fa, ca)
            boots = block_bootstrap_cliffs(fa, fault_ids, ca, ids, w, block_px, rng, n_boot)
            fa_f, ca_f = fold_of[fault_ids], fold_of[ids]
            signs = []
            for k in sp.cv_folds:
                x, y = fa[(fa_f == k) & ok_a], ca[(ca_f == k) & ok_c]
                if len(x) > 30 and len(y) > 30:
                    signs.append(np.sign(np.median(x) - np.median(y)))
            mi, mi0 = (
                mutual_information(fa[ok_a][:3000], ca[ok_c][:3000], rng)
                if with_mi and ctrl in ("random_background", "matched_1_3km")
                else (np.nan, np.nan)
            )
            rows.append(
                {
                    "feature": name,
                    "family": store.family_of(name),
                    "control": ctrl,
                    "n_fault": int(ok_a.sum()),
                    "n_control": int(ok_c.sum()),
                    "robust_std_diff": float((med_a - med_c) / mad),
                    "cliffs_delta": cd,
                    "cd_ci_lo": float(np.percentile(boots, 2.5)) if boots else np.nan,
                    "cd_ci_hi": float(np.percentile(boots, 97.5)) if boots else np.nan,
                    "overlap": overlap_coefficient(fa, ca),
                    "mi": mi,
                    "mi_perm_baseline": mi0,
                    "mi_excess": mi - mi0 if np.isfinite(mi) else np.nan,
                    "fold_sign_agreement": float(np.mean(np.array(signs) == np.sign(med_a - med_c))) if signs else np.nan,
                    "n_folds": len(signs),
                }
            )
    return pd.DataFrame(rows)


def discriminative_summary(ct: pd.DataFrame) -> pd.DataFrame:
    """Per feature: is the contrast robust across controls (genuinely discriminative) or only vs random background
    (confounded)? Classification rules are fixed in advance."""
    if ct.empty:
        return ct
    g = ct.groupby(["feature", "family"])
    out = g.agg(
        cd_random=("cliffs_delta", lambda s: s[ct.loc[s.index, "control"] == "random_background"].mean()),
        cd_min_abs=("cliffs_delta", lambda s: np.nanmin(np.abs(s))),
        cd_matched=("cliffs_delta", lambda s: s[ct.loc[s.index, "control"] == "matched_1_3km"].mean()),
        cd_high_grad=("cliffs_delta", lambda s: s[ct.loc[s.index, "control"] == "high_gradient"].mean()),
        stable=("fold_sign_agreement", "min"),
        ci_excludes_zero=("cd_ci_lo", lambda s: bool(((s > 0) | (ct.loc[s.index, "cd_ci_hi"] < 0)).all())),
    ).reset_index()

    def label(r):
        if not np.isfinite(r.cd_random) or abs(r.cd_random) < 0.1:
            return "weak"
        same = np.sign(r.cd_matched) == np.sign(r.cd_random) and abs(r.cd_matched) >= 0.1
        hard = (
            np.isfinite(r.cd_high_grad) and np.sign(r.cd_high_grad) == np.sign(r.cd_random) and abs(r.cd_high_grad) >= 0.05
        )
        if same and hard and r.ci_excludes_zero and (not np.isfinite(r.stable) or r.stable >= 1.0):
            return "discriminative"
        if same:
            return "partly_confounded"
        return "confounded"

    out["verdict"] = out.apply(label, axis=1)
    return out.sort_values("cd_random", key=np.abs, ascending=False)


# =============================================================================== multiscale (section 05)
def scale_space_auc(data, sp, bands: list[str], scales_m, rng, n: int = 30_000) -> pd.DataFrame:
    """AUC of the scale-normalised gradient (near <= 300 m vs far >= 2 km) per band and scale on the CV region."""
    from sklearn.metrics import roc_auc_score

    d = distance_px(data.labels) * 0.1
    cv = sp.cv_region()
    near = np.flatnonzero((cv & (d <= 0.3)).ravel())
    far = np.flatnonzero((cv & (d >= 2.0)).ravel())
    near = rng.choice(near, min(n, near.size), replace=False)
    far = rng.choice(far, min(n, far.size), replace=False)
    rows = []
    for b in bands:
        if not data.has(b):
            continue
        x = data.band(b)
        v = data.valid & np.isfinite(x)
        f = S.fill_nodata(x, v)
        for sm in scales_m:
            s = max(sm / 100.0 / 2, 0.7)
            g = np.hypot(*S.gaussian_gradient(f, s)) * s
            g[~v] = np.nan
            a, c = g.ravel()[near], g.ravel()[far]
            ok = np.r_[np.isfinite(a), np.isfinite(c)]
            y = np.r_[np.ones(len(a)), np.zeros(len(c))]
            sc = np.r_[a, c]
            auc = roc_auc_score(y[ok], sc[ok]) if len(np.unique(y[ok])) == 2 else np.nan
            rows.append({"band": b, "scale_m": sm, "gradient_auc": float(auc)})
    return pd.DataFrame(rows)


def variogram(x: np.ndarray, valid: np.ndarray, rng, lags_px=(1, 2, 4, 8, 16, 32, 64, 128), n: int = 40_000) -> pd.DataFrame:
    """Directional semivariogram gamma(h) = 0.5 E[(z(x) - z(x+h))^2] at 0/45/90/135 deg, and correlation length."""
    h, w = x.shape
    rows = []
    dirs = {"E-W": (0, 1), "NE-SW": (-1, 1), "N-S": (-1, 0), "NW-SE": (-1, -1)}
    r = rng.integers(0, h, n)
    c = rng.integers(0, w, n)
    for dname, (dr, dc) in dirs.items():
        for L in lags_px:
            r2, c2 = r + dr * L, c + dc * L
            ok = (r2 >= 0) & (r2 < h) & (c2 >= 0) & (c2 < w)
            a = x[r[ok], c[ok]]
            b = x[r2[ok], c2[ok]]
            m = valid[r[ok], c[ok]] & valid[r2[ok], c2[ok]] & np.isfinite(a) & np.isfinite(b)
            rows.append(
                {
                    "direction": dname,
                    "lag_km": L * 0.1,
                    "gamma": float(0.5 * np.mean((a[m] - b[m]) ** 2)) if m.any() else np.nan,
                    "pairs": int(m.sum()),
                }
            )
    df = pd.DataFrame(rows)
    sill = np.nanvar(x[valid])
    df["gamma_over_var"] = df.gamma / sill if sill > 0 else np.nan
    return df


def correlation_length(vg: pd.DataFrame) -> pd.DataFrame:
    out = []
    for d, g in vg.groupby("direction"):
        g = g.sort_values("lag_km")
        above = g[g.gamma_over_var >= 0.95 * g.gamma_over_var.max()]
        out.append(
            {
                "direction": d,
                "corr_length_km": float(above.lag_km.iloc[0]) if len(above) else np.nan,
                "nugget_ratio": float(g.gamma_over_var.iloc[0]),
            }
        )
    return pd.DataFrame(out)


def radial_power_spectrum(x: np.ndarray, valid: np.ndarray, pixel_m: float = 100.0, nbins: int = 40) -> pd.DataFrame:
    sf = S.SpectralField(x, valid, pixel_m)
    P = np.abs(sf.F) ** 2
    k = sf.K
    kb = np.geomspace(max(k[k > 0].min(), 1e-6), k.max(), nbins + 1)
    idx = np.digitize(k.ravel(), kb) - 1
    ok = (idx >= 0) & (idx < nbins)
    pw = np.bincount(idx[ok], P.ravel()[ok], nbins) / np.maximum(np.bincount(idx[ok], minlength=nbins), 1)
    kc = np.sqrt(kb[:-1] * kb[1:])
    return pd.DataFrame({"wavelength_km": 2 * np.pi / kc / 1000.0, "log_power": np.log10(pw + 1e-30)})


# =============================================================================== cross-fault profiles (section 07)
def label_orientation(labels: np.ndarray, sigma: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """Strike (axial) and coherence of the mapped traces from the structure tensor of the label mask."""
    coh, th, _ = S.structure_tensor(labels.astype(np.float32), 1.0, sigma)
    return th, coh


def cross_fault_profiles(
    data, sp, bands: list[str], rng, n_stations: int = 1500, half_km: float = 5.0, step_km: float = 0.1
) -> dict:
    """Perpendicular transects at sampled skeleton stations of mapped faults (CV region only).

    Returns profiles (n, bands, L) normalised per station by the robust spread of the whole profile, station
    metadata (strike, coherence, fold) and per-station response descriptors (contrast, width, asymmetry)."""
    from skimage.morphology import skeletonize

    sk = skeletonize(data.labels & sp.cv_region())
    th, coh = label_orientation(data.labels)
    rr, cc = np.nonzero(sk & (coh > 0.3))
    if not len(rr):
        return {"profiles": np.empty((0, len(bands), 1)), "meta": pd.DataFrame()}
    sel = rng.choice(len(rr), min(n_stations, len(rr)), replace=False)
    rr, cc = rr[sel], cc[sel]
    t = th[rr, cc]
    nx, ny = -np.sin(t), np.cos(t)  # unit normal (east, north)
    offs = np.arange(-half_km, half_km + 1e-9, step_km) / 0.1
    R = rr[:, None] - ny[:, None] * offs[None, :]  # north positive -> row decreases
    C = cc[:, None] + nx[:, None] * offs[None, :]
    prof = np.full((len(rr), len(bands), len(offs)), np.nan, np.float32)
    for j, b in enumerate(bands):
        if not data.has(b):
            continue
        x = data.band(b)
        v = data.valid & np.isfinite(x)
        f = S.fill_nodata(x, v)
        vals = ndimage.map_coordinates(f, [R.ravel(), C.ravel()], order=1, mode="nearest").reshape(R.shape)
        okv = ndimage.map_coordinates(v.astype(np.float32), [R.ravel(), C.ravel()], order=0, mode="constant").reshape(
            R.shape
        )
        vals[okv < 0.5] = np.nan
        med = np.nanmedian(vals, 1, keepdims=True)
        spread = np.nanpercentile(vals, 84, 1, keepdims=True) - np.nanpercentile(vals, 16, 1, keepdims=True)
        prof[:, j] = (vals - med) / np.where(spread > 0, spread, 1.0)
    mid = len(offs) // 2
    win = int(round(1.0 / step_km))
    desc = []
    for j, b in enumerate(bands):
        p = prof[:, j]
        left = np.nanmean(p[:, mid - 3 * win : mid - win], 1)
        right = np.nanmean(p[:, mid + win : mid + 3 * win], 1)
        g = np.abs(np.gradient(p, axis=1))
        peak = np.nanmax(g[:, mid - win : mid + win + 1], 1)
        half = peak[:, None] / 2
        above = g[:, mid - 2 * win : mid + 2 * win + 1] >= half
        width = above.sum(1) * step_km
        centre = p[:, mid - 1 : mid + 2].mean(1)
        sym = np.abs(right - left) / (np.abs(centre - 0.5 * (left + right)) + np.abs(right - left) + 1e-6)
        desc.append(
            pd.DataFrame(
                {f"{b}_step": right - left, f"{b}_peak_grad": peak, f"{b}_width_km": width, f"{b}_step_vs_line": sym}
            )
        )
    meta = pd.DataFrame(
        {"row": rr, "col": cc, "strike_deg": np.degrees(t), "coherence": coh[rr, cc], "fold": sp.fold_map[rr, cc]}
    )
    meta = pd.concat([meta] + desc, axis=1)
    return {"profiles": prof, "offsets_km": offs * 0.1, "meta": meta, "bands": bands}


def profile_families(prof: np.ndarray, rng, k_range=range(2, 7)) -> tuple[np.ndarray, pd.DataFrame]:
    """Cluster stations into physical-response families (k by silhouette); |profile| orientation-invariant."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    X = np.nan_to_num(np.abs(prof)).reshape(len(prof), -1)
    if len(X) < 20:
        return np.zeros(len(X), int), pd.DataFrame()
    scores = []
    for k in k_range:
        lab = KMeans(k, n_init=5, random_state=0).fit_predict(X)
        sub = rng.choice(len(X), min(len(X), 2000), replace=False)
        scores.append({"k": k, "silhouette": float(silhouette_score(X[sub], lab[sub]))})
    sc = pd.DataFrame(scores)
    k = int(sc.loc[sc.silhouette.idxmax(), "k"])
    lab = KMeans(k, n_init=10, random_state=0).fit_predict(X)
    return lab, sc


# =============================================================================== directional (section 08)
def orientation_rose(data, sp, bands: list[str], scale_m: int = 800, bins: int = 18) -> pd.DataFrame:
    """Coherence-weighted strike histograms: near mapped faults vs background, plus the mapped-trace strikes."""
    d = distance_px(data.labels) * 0.1
    cv = sp.cv_region()
    near, far = cv & (d <= 0.5), cv & (d >= 3.0)
    edges = np.linspace(0, 180, bins + 1)
    rows = []
    th, coh = label_orientation(data.labels)
    m = data.labels & cv & (coh > 0.3)
    h, _ = np.histogram(np.degrees(th[m]), edges)
    for i in range(bins):
        rows.append({"source": "mapped_faults", "zone": "trace", "bin_lo": edges[i], "share": h[i] / max(h.sum(), 1)})
    for b in bands:
        if not data.has(b):
            continue
        x = data.band(b)
        v = data.valid & np.isfinite(x)
        c, t, _ = S.structure_tensor(S.fill_nodata(x, v), scale_m / 200.0, scale_m / 200.0)
        for zname, z in (("near_faults", near & v), ("background", far & v)):
            hh, _ = np.histogram(np.degrees(t[z]), edges, weights=c[z])
            for i in range(bins):
                rows.append({"source": b, "zone": zname, "bin_lo": edges[i], "share": hh[i] / max(hh.sum(), 1e-12)})
    return pd.DataFrame(rows)


def modality_correlation(store, names: list[str], ids: np.ndarray) -> pd.DataFrame:
    """Spearman correlation between features on a pixel sample (redundancy / modality relationships)."""
    X = store.matrix(np.sort(ids), names)
    df = pd.DataFrame(X, columns=names)
    return df.rank().corr()
