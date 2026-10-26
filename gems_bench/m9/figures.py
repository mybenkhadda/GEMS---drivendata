"""M9 figures. Analytical panels are 16:9 (``m7.figures`` helpers, fixed validated palette); maps keep true geometry
(equal aspect, real extents in metres). Every function returns (figure or None, plotting data as a DataFrame) and the
caller saves both (PNG + SVG + CSV)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..gems3.figures import INK, SERIES, _style
from ..m7.figures import H9, W16, bar_panels16, line_panels16, map_panels


def _window(shape, center, half: int = 250) -> tuple[int, int, int, int]:
    r, c = int(center[0]), int(center[1])
    return max(r - half, 0), min(r + half, shape[0]), max(c - half, 0), min(c + half, shape[1])


def unit_center(u) -> tuple[int, int]:
    rr, cc = np.nonzero(u.eval_truth() if u.eval_truth().any() else u.ev)
    return (int(np.median(rr)) + u.sl[0].start, int(np.median(cc)) + u.sl[1].start) if len(rr) else (0, 0)


# 1-2 ------------------------------------------------------------------------------------------- features
def feature_family_comparison(pooled: pd.DataFrame):
    if not len(pooled):
        return None, pooled
    d = pooled[["task", "feature_set", "dti_pooled", "worst_zone_dti", "auc_pr_mean", "n_features"]].copy()
    return bar_panels16(d, "feature_set", "dti_pooled", "task", ref_col="worst_zone_dti",
                        title="Feature sets: pixel LightGBM screen, held-out official-style DTI (tick = worst zone)"), d


def raw_vs_structural(pooled: pd.DataFrame):
    if not len(pooled):
        return None, pooled
    d = pooled[pooled.feature_set.isin(["raw", "base", "base_spectral", "base_no_new"])].copy()
    return bar_panels16(d, "feature_set", "auc_pr_mean", "task", xlabel="held-out AUC-PR (pixels; not the metric)",
                        title="Raw fields vs raw + geological structural inputs"), d


# 3-5 ------------------------------------------------------------------------------------------- physics / spectral
def agreement_map(m9, name: str, center):
    from .features import available, read

    if not available(m9, name):
        return None, pd.DataFrame()
    a = np.asarray(read(m9, name), np.float32)
    w = _window(a.shape, center)
    lab = np.asarray(m9.ctx.data.labels, bool)
    v = a[np.isfinite(a)]
    hist, edges = np.histogram(v, bins=40)
    fig = map_panels(m9.ctx.data.grid, w, [{"title": name, "raster": a[w[0]:w[1], w[2]:w[3]], "cmap": "viridis",
                                            "known": lab[w[0]:w[1], w[2]:w[3]], "legend": True}],
                     title="Magnetic-gravity tolerant edge agreement (mapped faults in ink)", ncols=1)
    return fig, pd.DataFrame({"bin_lo": edges[:-1], "bin_hi": edges[1:], "count": hist})


def fourier_spectra(prod: dict, fam: str, n: int = 6, seed: int = 2026):
    """Rose diagrams of the short-band sector power of ``n`` windows + the anisotropy grid (true orientation:
    sector = WAVEVECTOR azimuth; the trace strike is perpendicular)."""
    import matplotlib.pyplot as plt

    sec, an = prod["sectors"], prod["anisotropy"]
    ok = np.argwhere(np.isfinite(an))
    if not len(ok):
        return None, pd.DataFrame()
    rng = np.random.default_rng(seed)
    pick = ok[rng.choice(len(ok), min(n, len(ok)), replace=False)]
    ns = sec.shape[-1]
    ang = np.radians(np.arange(ns) * 180.0 / ns + 90.0 / ns)
    fig = plt.figure(figsize=(W16, H9))
    rows = []
    for i, (r, c) in enumerate(pick):
        ax = fig.add_subplot(2, 4, i + 1, projection="polar")
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        p = sec[r, c]
        th = np.r_[ang, ang + np.pi]
        ax.bar(th, np.r_[p, p], width=np.pi / ns, color=SERIES[0], edgecolor="white")
        ax.set_title(f"window ({int(prod['rows'][r])}, {int(prod['cols'][c])}) A={an[r, c]:.2f}", fontsize=8, color=INK)
        ax.set_yticklabels([])
        rows += [{"window_row": int(prod["rows"][r]), "window_col": int(prod["cols"][c]), "sector": j,
                  "k_azimuth_deg": float(np.degrees(ang[j])), "power_share": float(p[j]), "anisotropy": float(an[r, c])}
                 for j in range(ns)]
    ax = fig.add_subplot(2, 4, 8)
    ax.imshow(an, cmap="magma", vmin=0, vmax=1, interpolation="nearest")
    ax.set_aspect("equal")
    ax.set_title("anisotropy (window grid)", fontsize=8, color=INK)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.suptitle(f"{fam}: directional spectra (wavevector azimuth; trace strike = +90 deg)", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    return fig, pd.DataFrame(rows)


def wavelet_maps(m9, fam: str, center):
    from .features import available, read

    names = [f"m9:spec:{fam}_wav_energy@{lam:g}km" for lam in m9.cfg["spectral"]["wavelet"]["wavelengths_km"]] + \
        [f"m9:spec:{fam}_wav_coherence_mean", f"m9:spec:{fam}_wav_persistence"]
    names = [n for n in names if available(m9, n)]
    if not names:
        return None, pd.DataFrame()
    w = _window(m9.ctx.data.shape, center)
    lay, stats = [], []
    for n in names:
        a = np.asarray(read(m9, n), np.float32)[w[0]:w[1], w[2]:w[3]]
        lay.append({"title": n.split(":")[-1], "raster": a, "cmap": "viridis"})
        stats.append({"layer": n, "p05": float(np.nanpercentile(a, 5)), "p50": float(np.nanmedian(a)),
                      "p95": float(np.nanpercentile(a, 95))})
    return map_panels(m9.ctx.data.grid, w, lay, title=f"{fam}: log-Gabor energy by scale, coherence, persistence"), \
        pd.DataFrame(stats)


# 6 --------------------------------------------------------------------------------------------- training
def loss_components(hist: pd.DataFrame):
    if not len(hist):
        return None, hist
    cols = [c for c in hist.columns if c.startswith("loss_l")]
    d = hist.melt(id_vars=["epoch", "run"], value_vars=cols, var_name="part", value_name="value").dropna()
    return line_panels16(d, "epoch", "value", "part", "run", xlabel="epoch", ylabel="loss",
                         title="Training loss decomposition per unit (L1 dense, L2 Tversky, L3 skeleton, L4 orientation, "
                               "L5 topology, L6 reconstruction)"), d


# 7-10 ------------------------------------------------------------------------------------------ validation
def fold_performance(per_zone: pd.DataFrame, methods: list[str]):
    d = per_zone[per_zone.method.isin(methods)].copy() if len(per_zone) else per_zone
    if not len(d):
        return None, d
    d["cat"] = d.method + " | zone " + d.zone.astype(str)
    return bar_panels16(d[["task", "cat", "dti"]], "cat", "dti", "task",
                        title="Held-out DTI per geographic fold / discovery split (nested choices)"), d


def system_recall(pooled: pd.DataFrame):
    d = pooled.dropna(subset=["system_recall"]) if len(pooled) and "system_recall" in pooled else pd.DataFrame()
    if not len(d):
        return None, d
    return bar_panels16(d[["task", "method", "system_recall"]], "method", "system_recall", "task",
                        xlabel="system recall (support >= 0.5 on >= half the pixels)",
                        title="Whole-system discovery recall"), d


def short_fault(strata: pd.DataFrame):
    if not len(strata):
        return None, strata
    return line_panels16(strata, "length_bin_km", "system_recall", "method", "task", xlabel="system length (km)",
                         ylabel="system recall", title="Recall by fault-system length (short faults left)"), strata


def known_distance(prox: pd.DataFrame):
    if not len(prox):
        return None, prox
    return line_panels16(prox, "distance_band", "recall_w", "method", "task", xlabel="distance to known faults",
                         ylabel="weighted recall", title="Known-fault-distance sensitivity (independent vs continuation)"), prox


# 11-13 ----------------------------------------------------------------------------------------- dense / structured
def dense_vs_structured(m9, u, dense: np.ndarray | None, struct: np.ndarray | None):
    if dense is None:
        return None, pd.DataFrame()
    c = unit_center(u)
    w = _window(u.ev.shape, (c[0] - u.sl[0].start, c[1] - u.sl[1].start), 200)          # crop coordinates
    s = (slice(w[0], w[1]), slice(w[2], w[3]))
    gw = (w[0] + u.sl[0].start, w[1] + u.sl[0].start, w[2] + u.sl[1].start, w[3] + u.sl[1].start)
    lay = [{"title": "dense evidence (held-out)", "raster": dense[s], "cmap": "magma", "vmin": 0, "vmax": 1,
            "truth": u.eval_truth()[s], "known": None if u.known is None else u.known[s], "legend": True}]
    if struct is not None:
        lay.append({"title": "structured channel (ranked traces)", "raster": struct[s], "cmap": "magma", "vmin": 0,
                    "vmax": 1, "truth": u.eval_truth()[s]})
    d = pd.DataFrame({"unit": [u.key], "window": [str(gw)]})
    return map_panels(m9.ctx.data.grid, gw, lay, title=f"{u.key}: dense vs structured predictions", ncols=2), d


def fault_normal_profiles(T: pd.DataFrame):
    cols = [c for c in T.columns if c.startswith("fn_")] if len(T) else []
    if not cols:
        return None, pd.DataFrame()
    d = T[T.label >= 0].melt(id_vars=["label"], value_vars=cols, var_name="feature", value_name="contrast")
    d["modality"] = d.feature.str.split("_").str[1]
    d["offset_m"] = d.feature.str.extract(r"_(\d+)m$")[0].astype(float)
    g = d.groupby(["modality", "offset_m", "label"]).contrast.median().reset_index()
    g["class"] = np.where(g.label == 1, "near truth", "far from truth")
    return line_panels16(g, "offset_m", "contrast", "class", "modality", xlabel="offset from the trace (m)",
                         ylabel="median fault-normal contrast", title="Candidate fault-normal profiles by label class"), g


def tracing_failures(m9, u, missed: list[int], E: np.ndarray, cs):
    if not missed:
        return None, pd.DataFrame()
    sm = np.asarray(m9.ctx.splits.system_map)[u.sl]
    m = np.isin(sm, missed) & u.eval_truth()
    rr, cc = np.nonzero(m)
    if not len(rr):
        return None, pd.DataFrame()
    c = (int(np.median(rr)), int(np.median(cc)))
    w = _window(u.ev.shape, c, 150)
    s = (slice(w[0], w[1]), slice(w[2], w[3]))
    gw = (w[0] + u.sl[0].start, w[1] + u.sl[0].start, w[2] + u.sl[1].start, w[3] + u.sl[1].start)
    cand = (cs.raster(u.ev.shape) > 0)[s] if cs is not None and len(cs) else None
    fig = map_panels(m9.ctx.data.grid, gw, [{"title": "missed systems (red) / traces (blue) on dense evidence", "raster": E[s],
                                             "cmap": "Greys", "truth": m[s], "cand": cand, "legend": True}], ncols=1)
    return fig, pd.DataFrame({"unit": u.key, "system_id": missed})


# 14-16 ----------------------------------------------------------------------------------------- ranking / MC / failures
def gbm_comparison(summary: pd.DataFrame):
    if not len(summary):
        return None, summary
    return bar_panels16(summary[["task", "method", "dti_crossfit", "fold_min"]], "method", "dti_crossfit", "task",
                        ref_col="fold_min", title="XGBoost / LightGBM / CatBoost: rasterised cross-fitted DTI"), summary


def mc_uncertainty(m9, u, occ: np.ndarray, stab: pd.DataFrame):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(W16, H9))
    ext = m9.ctx.data.grid.window_extent(u.sl[0].start, u.sl[0].stop, u.sl[1].start, u.sl[1].stop)
    axes[0].imshow(np.where(u.ev, occ, np.nan), extent=ext, cmap="magma", vmin=0, vmax=1, origin="upper")
    axes[0].set_aspect("equal")
    axes[0].set_title("path occupancy (share of realisations; NOT a probability)", fontsize=9, color=INK, loc="left")
    v = stab.mc_a_detection.dropna() if len(stab) and "mc_a_detection" in stab else pd.Series(dtype=float)
    axes[1].hist(v, bins=20, color=SERIES[0], edgecolor="white")
    _style(axes[1], "MC-A detection frequency", "candidates", "candidate stability")
    fig.tight_layout()
    h, e = np.histogram(v, bins=20, range=(0, 1))
    return fig, pd.DataFrame({"bin_lo": e[:-1], "bin_hi": e[1:], "count": h})


def failure_modes(domains: pd.DataFrame, regions: pd.DataFrame):
    parts = []
    if len(domains):
        parts.append(domains.assign(panel="label-free domain", cat=domains.method + " | d" + domains.domain.astype(str)))
    if len(regions):
        parts.append(regions.assign(panel="3 x 3 region", cat=regions.method + " | " + regions.region.astype(str)))
    if not parts:
        return None, pd.DataFrame()
    d = pd.concat(parts, ignore_index=True)[["panel", "cat", "dti"]]
    return bar_panels16(d, "cat", "dti", "panel", title="Geographic failure modes: DTI by geological domain and region"), d
