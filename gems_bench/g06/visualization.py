"""Figures for geological analysis and ML benchmarking (matplotlib, Agg backend).

Every figure function writes ``<name>.png`` + ``<name>.svg`` AND its numerical source data (``<name>.csv`` or
``<name>.parquet``) through ``save_figure``. Conventions:

* Colour follows the entity, never its rank: models get fixed categorical slots in configuration order
  (``model_colors``; reference palette of the dataviz method), the W2 reference is drawn in secondary ink.
  More than 8 series fold into small multiples (never a generated 9th hue).
* Magnitude rasters use one sequential hue (blue, light -> dark); signed rasters a blue <-> red diverging map with a
  neutral grey midpoint centred on 0. Robust colour limits (2-98 % quantiles inside the footprint).
* One y-axis per chart; recessive grid; text in ink colours, never in series colours; a legend for >= 2 series.
* Maps keep the native grid extent (pixel rows / columns, north up) and a common downsampling factor.
* Validation truth is drawn POST HOC only (outline overlays); no figure output is ever a model input.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIV = ["#0d366b", "#256abf", "#86b6ef", "#f0efec", "#f2a3a2", "#e34948", "#9c2121"]
REFERENCE = "W2"


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.edgecolor": AXIS,
                         "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": False,
                         "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "axes.titlecolor": INK,
                         "axes.titlesize": 10, "legend.frameon": False, "axes.spines.top": False,
                         "axes.spines.right": False, "lines.linewidth": 2.0, "svg.fonttype": "none"})
    return plt


def _cmaps():
    from matplotlib.colors import LinearSegmentedColormap

    return (LinearSegmentedColormap.from_list("g06_seq", SEQ), LinearSegmentedColormap.from_list("g06_div", DIV))


def model_colors(models) -> dict:
    """Fixed colour per model in the given order; the W2 reference in secondary ink."""
    out, i = {}, 0
    for m in models:
        if m == REFERENCE:
            out[m] = INK2
            continue
        out[m] = SERIES[i] if i < len(SERIES) else MUTED
        i += 1
    return out


def save_figure(fig, data, out_dir, name: str, dpi: int = 140) -> dict:
    """PNG + SVG + source data (DataFrame -> CSV when small, Parquet otherwise; dict of DataFrames -> one file each)."""
    plt = _plt()
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / f"{name}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(d / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)
    files = [f"{name}.png", f"{name}.svg"]
    items = data.items() if isinstance(data, dict) else [("", data)]
    for k, df in items:
        if df is None:
            continue
        stem = f"{name}_{k}" if k else name
        if len(df) > 200000:
            df.to_parquet(d / f"{stem}.parquet", index=False)
            files.append(f"{stem}.parquet")
        else:
            df.to_csv(d / f"{stem}.csv", index=False)
            files.append(f"{stem}.csv")
    return {"name": name, "dir": str(d), "files": files}


def _down(a: np.ndarray, f: int) -> np.ndarray:
    return np.asarray(a)[::f, ::f]


def _limits(a: np.ndarray) -> tuple[float, float, bool]:
    v = a[np.isfinite(a)]
    if not v.size:
        return 0.0, 1.0, False
    lo, hi = np.quantile(v, [0.02, 0.98])
    signed = lo < 0 < hi
    if signed:
        m = max(abs(lo), abs(hi))
        return -m, m, True
    return float(lo), float(hi if hi > lo else lo + 1e-9), False


def _long(arrays: dict, f: int) -> pd.DataFrame:
    parts = []
    for k, a in arrays.items():
        s = _down(a, f)
        r, c = np.indices(s.shape)
        parts.append(pd.DataFrame({"feature": k, "row": (r * f).ravel(), "col": (c * f).ravel(),
                                   "value": s.ravel().astype(np.float32)}))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# =============================================================================== physical / feature atlases
def plot_feature_atlas(arrays: dict, out_dir, name: str, title: str, units: dict | None = None, factor: int = 8,
                       ncols: int = 4, overlay: np.ndarray | None = None) -> dict:
    """Small-multiple maps of full-grid layers (same extent, robust limits, unit in each panel title)."""
    plt = _plt()
    seq, div = _cmaps()
    keys = list(arrays)
    nrows = max(1, int(np.ceil(len(keys) / ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.4 * nrows), squeeze=False)
    stats = []
    for ax, k in zip(axes.ravel(), keys):
        a = _down(arrays[k], factor).astype(np.float32)
        lo, hi, signed = _limits(a)
        im = ax.imshow(a, cmap=div if signed else seq, vmin=lo, vmax=hi, interpolation="nearest")
        if overlay is not None:
            ov = _down(overlay, factor)
            ax.contour(ov.astype(float), levels=[0.5], colors=[INK], linewidths=0.4)
        ax.set_title(f"{k}" + (f" [{units.get(k)}]" if units and units.get(k) else ""), fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=6)
        v = arrays[k][np.isfinite(arrays[k])]
        stats.append({"feature": k, "finite_share": float(np.isfinite(arrays[k]).mean()),
                      "p02": float(np.quantile(v, 0.02)) if v.size else np.nan,
                      "p50": float(np.median(v)) if v.size else np.nan,
                      "p98": float(np.quantile(v, 0.98)) if v.size else np.nan, "downsample": factor})
    for ax in axes.ravel()[len(keys):]:
        ax.axis("off")
    fig.suptitle(title, color=INK)
    return save_figure(fig, {"stats": pd.DataFrame(stats), "grid": _long(arrays, max(factor * 2, 16))}, out_dir, name)


def plot_missingness(arrays: dict, footprint: np.ndarray, out_dir, name: str = "missingness", factor: int = 8
                     ) -> dict:
    """Count of missing bands per pixel inside the footprint + per-band missing share."""
    plt = _plt()
    seq, _ = _cmaps()
    miss = sum((~np.isfinite(a)).astype(np.int16) for a in arrays.values())
    miss = np.where(footprint, miss, np.nan)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.2, 1]})
    im = a1.imshow(_down(miss, factor), cmap=seq, interpolation="nearest")
    a1.set_title("Missing bands per pixel (inside the footprint)")
    a1.set_xticks([])
    a1.set_yticks([])
    fig.colorbar(im, ax=a1, fraction=0.046, pad=0.02, label="bands missing")
    T = pd.DataFrame({"band": list(arrays), "missing_share": [float((~np.isfinite(a))[footprint].mean())
                                                              for a in arrays.values()]})
    T = T.sort_values("missing_share")
    a2.barh(T.band, T.missing_share, color=SERIES[0], height=0.7)
    a2.set_xlabel("missing share inside the footprint")
    a2.grid(axis="x", color=GRID, linewidth=0.6)
    return save_figure(fig, T, out_dir, name)


def plot_feature_correlations(X: pd.DataFrame, out_dir, name: str = "feature_correlations", max_features: int = 60
                              ) -> dict:
    """Spearman correlation matrix of a pixel sample (features ordered by hierarchical clustering)."""
    plt = _plt()
    _, div = _cmaps()
    cols = [c for c in X.columns if c != "row_id" and X[c].notna().mean() > 0.5][:max_features]
    C = X[cols].rank().corr().fillna(0.0)
    try:
        from scipy.cluster.hierarchy import leaves_list, linkage

        order = leaves_list(linkage(1 - np.abs(C.to_numpy()), "average"))
        C = C.iloc[order, order]
    except Exception:
        pass
    n = len(C)
    fig, ax = plt.subplots(figsize=(0.16 * n + 3, 0.16 * n + 2))
    im = ax.imshow(C.to_numpy(), cmap=div, vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels(C.columns, rotation=90, fontsize=5)
    ax.set_yticks(range(n))
    ax.set_yticklabels(C.index, fontsize=5)
    fig.colorbar(im, ax=ax, fraction=0.03, label="Spearman rho")
    ax.set_title("Feature correlations (pixel sample)")
    return save_figure(fig, C.reset_index().rename(columns={"index": "feature"}), out_dir, name)


def plot_fault_profiles(dist_km: np.ndarray, arrays: dict, footprint: np.ndarray, out_dir,
                        name: str = "fault_profiles", max_km: float = 5.0, step_km: float = 0.2) -> dict:
    """Median and IQR of each feature against distance to the nearest mapped fault (post-hoc description)."""
    plt = _plt()
    bins = np.arange(0, max_km + step_km, step_km)
    m = footprint & np.isfinite(dist_km) & (dist_km <= max_km)
    b = np.digitize(dist_km[m], bins) - 1
    rows = []
    for k, a in arrays.items():
        v = np.asarray(a)[m]
        df = pd.DataFrame({"b": b, "v": v}).dropna()
        g = df.groupby("b").v.quantile([0.25, 0.5, 0.75]).unstack()
        for i, r in g.iterrows():
            rows.append({"feature": k, "dist_km": float(bins[int(i)] + step_km / 2), "q25": r[0.25], "q50": r[0.5],
                         "q75": r[0.75]})
    T = pd.DataFrame(rows)
    keys = list(arrays)
    nc = min(4, len(keys))
    nr = int(np.ceil(len(keys) / max(nc, 1)))
    fig, axes = plt.subplots(nr, nc, figsize=(3.3 * nc, 2.6 * nr), squeeze=False)
    for ax, k in zip(axes.ravel(), keys):
        t = T[T.feature == k]
        ax.fill_between(t.dist_km, t.q25, t.q75, color=SEQ[1], linewidth=0)
        ax.plot(t.dist_km, t.q50, color=SERIES[0])
        ax.set_title(k, fontsize=8)
        ax.set_xlabel("distance to mapped fault (km)")
        ax.grid(axis="y", color=GRID, linewidth=0.6)
    for ax in axes.ravel()[len(keys):]:
        ax.axis("off")
    fig.suptitle("Fault-centred profiles: median (line) and interquartile range (band)", color=INK)
    fig.tight_layout()
    return save_figure(fig, T, out_dir, name)


# =============================================================================== training diagnostics
def plot_class_distributions(X: pd.DataFrame, cls: pd.Series, features: list[str], out_dir,
                             name: str = "class_distributions") -> dict:
    """Per feature, the distribution in each supervision class (POSITIVE / UNLABELED / LIKELY_BACKGROUND)."""
    from .supervision import CLASS_NAMES

    plt = _plt()
    classes = [1, 0, 2]
    colors = {1: SERIES[1], 0: SERIES[0], 2: SERIES[2]}
    nc = min(4, len(features))
    nr = int(np.ceil(len(features) / max(nc, 1)))
    fig, axes = plt.subplots(nr, nc, figsize=(3.3 * nc, 2.6 * nr), squeeze=False)
    rows = []
    for ax, f in zip(axes.ravel(), features):
        v = X[f].to_numpy(float)
        ok = np.isfinite(v)
        if not ok.any():
            ax.axis("off")
            continue
        lo, hi = np.quantile(v[ok], [0.01, 0.99])
        edges = np.linspace(lo, hi if hi > lo else lo + 1, 40)
        for c in classes:
            s = v[ok & (cls.to_numpy() == c)]
            if not s.size:
                continue
            h, _ = np.histogram(np.clip(s, lo, hi), edges, density=True)
            ax.step(edges[:-1], h, where="post", color=colors[c], linewidth=1.5, label=CLASS_NAMES[c])
            rows.append({"feature": f, "class": CLASS_NAMES[c], "n": int(s.size), "q10": float(np.quantile(s, .1)),
                         "q50": float(np.median(s)), "q90": float(np.quantile(s, .9))})
        ax.set_title(f, fontsize=8)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
    for ax in axes.ravel()[len(features):]:
        ax.axis("off")
    axes.ravel()[0].legend(fontsize=7)
    fig.suptitle("Feature distributions per supervision class (density)", color=INK)
    fig.tight_layout()
    return save_figure(fig, pd.DataFrame(rows), out_dir, name)


def plot_sample_locations(meta: pd.DataFrame, labels: pd.DataFrame, shape, out_dir, name: str = "sample_locations"
                          ) -> dict:
    """Map of the sampled pixels of one fit by class (and the strata balance table)."""
    from .supervision import CLASS_NAMES

    plt = _plt()
    J = labels.merge(meta[["row_id", "row", "col"]], on="row_id")
    fig, ax = plt.subplots(figsize=(7, 7 * shape[0] / shape[1]))
    for c, col in ((0, SERIES[0]), (2, SERIES[2]), (1, SERIES[1])):
        s = J[J.cls == c]
        if len(s) > 60000:
            s = s.sample(60000, random_state=0)
        ax.scatter(s.col, s.row, s=1.5, color=col, label=CLASS_NAMES[c], linewidths=0)
    ax.set_xlim(0, shape[1])
    ax.set_ylim(shape[0], 0)
    ax.set_aspect("equal")
    ax.legend(markerscale=6, fontsize=7, loc="lower right")
    ax.set_title(f"Training sample locations ({labels.fit_id.iloc[0] if len(labels) else ''})")
    T = J.groupby(["stratum", "cls"]).agg(n=("row_id", "size"), weight=("weight", "sum")).reset_index()
    T["class"] = T.cls.map(CLASS_NAMES)
    return save_figure(fig, T, out_dir, name)


# =============================================================================== model diagnostics
def plot_feature_importance(imp: pd.DataFrame, out_dir, name: str, top: int = 25, title: str = "") -> dict:
    plt = _plt()
    T = imp.sort_values("importance", ascending=False).head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 0.24 * len(T) + 1.2))
    ax.barh(T.feature, T.share if "share" in T else T.importance, color=SERIES[0], height=0.7)
    ax.set_xlabel("share of total gain")
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_title(title or "Feature importance")
    return save_figure(fig, imp, out_dir, name)


def plot_model_comparison(summary: pd.DataFrame, out_dir, name: str, rep: str, frac: float, models=None) -> dict:
    """Pooled DTI per model at the fixed policy, one panel per scenario (worst zone as a marker)."""
    plt = _plt()
    S = summary[(summary.rep == rep) & (summary.frac == frac)]
    models = list(models or sorted(S.model.unique(), key=lambda m: (m != REFERENCE, m)))
    cols = model_colors(models)
    scen = [s for s in ("V1", "V3", "V2") if s in set(S.scenario)]
    fig, axes = plt.subplots(1, max(len(scen), 1), figsize=(4.2 * max(len(scen), 1), 0.32 * len(models) + 1.6),
                             squeeze=False)
    for ax, sc in zip(axes.ravel(), scen):
        t = S[S.scenario == sc].set_index("model").reindex(models)
        y = np.arange(len(models))
        ax.barh(y, t.dti, color=[cols[m] for m in models], height=0.62)
        ax.scatter(t.worst_zone_dti, y, marker="|", s=120, color=INK, label="worst zone", zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels(models, fontsize=7)
        ax.set_xlabel("pooled DTI (summed components)")
        ax.set_title(f"{sc} - {rep} @ {frac:.3g}")
        ax.grid(axis="x", color=GRID, linewidth=0.6)
    axes.ravel()[0].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    return save_figure(fig, S, out_dir, name)


def plot_dti_frontier(rows: pd.DataFrame, out_dir, name: str, scenario: str, rep: str, models=None) -> dict:
    """Pooled DTI against the final positive area (share of the eligible domain) per model."""
    plt = _plt()
    from .evaluation import aggregate_components

    S = aggregate_components(rows[(rows.scenario == scenario) & (rows.rep == rep)])
    if not len(S):
        return {}
    models = list(models or sorted(S.model.unique(), key=lambda m: (m != REFERENCE, m)))
    cols = model_colors(models)
    fig, ax = plt.subplots(figsize=(6.2, 4))
    for m in models:
        t = S[S.model == m].sort_values("frac")
        ax.plot(100 * t.area_frac_eligible, t.dti, color=cols[m], marker="o", markersize=4, label=m)
    ax.set_xlabel("final positive area (% of the eligible domain)")
    ax.set_ylabel("pooled DTI")
    ax.grid(color=GRID, linewidth=0.6)
    ax.legend(fontsize=7)
    ax.set_title(f"DTI vs area - {scenario}, {rep}")
    return save_figure(fig, S, out_dir, name)


def plot_geographic_robustness(per_zone: pd.DataFrame, out_dir, name: str, rep: str, frac: float, models=None
                               ) -> dict:
    """Per-zone DTI of every model (one panel per scenario, zones on the x axis)."""
    plt = _plt()
    Z = per_zone[(per_zone.rep == rep) & (per_zone.frac == frac)]
    models = list(models or sorted(Z.model.unique(), key=lambda m: (m != REFERENCE, m)))
    cols = model_colors(models)
    scen = [s for s in ("V1", "V3", "V2") if s in set(Z.scenario)]
    fig, axes = plt.subplots(1, max(len(scen), 1), figsize=(4.2 * max(len(scen), 1), 3.6), squeeze=False)
    for ax, sc in zip(axes.ravel(), scen):
        t = Z[Z.scenario == sc]
        for i, m in enumerate(models):
            u = t[t.model == m].sort_values("zone")
            ax.plot(u.zone + (i - len(models) / 2) * 0.04, u.dti, color=cols[m], marker="o", markersize=5,
                    linewidth=1.0, label=m)
        ax.set_xlabel("zone")
        ax.set_ylabel("zone DTI")
        ax.set_title(sc)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
    axes.ravel()[-1].legend(fontsize=7, loc="best")
    fig.tight_layout()
    return save_figure(fig, Z, out_dir, name)


def plot_prediction_overlay(score: np.ndarray, selected: np.ndarray, truth: np.ndarray, known: np.ndarray, out_dir,
                            name: str, title: str, factor: int = 2) -> dict:
    """Score map with the selected area, POST-HOC scored truth (outline) and the permitted known faults."""
    plt = _plt()
    seq, _ = _cmaps()
    fig, ax = plt.subplots(figsize=(7, 7 * score.shape[0] / max(score.shape[1], 1)))
    ax.imshow(_down(score, factor), cmap=seq, interpolation="nearest")
    for m, col, lw in ((selected, SERIES[1], 0.6), (truth, INK, 0.8), (known, MUTED, 0.4)):
        if m is not None and np.asarray(m).any():
            ax.contour(_down(np.asarray(m, float), factor), levels=[0.5], colors=[col], linewidths=lw)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title + "\n(orange: selected area, black: held-out truth - post hoc, grey: permitted known faults)",
                 fontsize=8)
    T = pd.DataFrame({"selected_px": [int(np.asarray(selected).sum())], "truth_px": [int(np.asarray(truth).sum())],
                      "known_px": [int(np.asarray(known).sum())], "score_mean": [float(np.nanmean(score))]})
    return save_figure(fig, T, out_dir, name)


def plot_failure_cases(cases: list[dict], out_dir, name: str, title: str, ncols: int = 4) -> dict:
    """Small multiples of system windows: ``cases`` = [{"label", "score", "truth", "selected"}] (crops)."""
    plt = _plt()
    seq, _ = _cmaps()
    if not cases:
        return {}
    nr = int(np.ceil(len(cases) / ncols))
    fig, axes = plt.subplots(nr, ncols, figsize=(3 * ncols, 3 * nr), squeeze=False)
    rows = []
    for ax, c in zip(axes.ravel(), cases):
        ax.imshow(c["score"], cmap=seq, interpolation="nearest")
        for m, col in ((c.get("selected"), SERIES[1]), (c.get("truth"), INK)):
            if m is not None and np.asarray(m).any():
                ax.contour(np.asarray(m, float), levels=[0.5], colors=[col], linewidths=0.7)
        ax.set_title(c["label"], fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
        rows.append({k: v for k, v in c.items() if not isinstance(v, np.ndarray)})
    for ax in axes.ravel()[len(cases):]:
        ax.axis("off")
    fig.suptitle(title, color=INK)
    fig.tight_layout()
    return save_figure(fig, pd.DataFrame(rows), out_dir, name)


def plot_pr_and_calibration(pr: pd.DataFrame, cal: pd.DataFrame, out_dir, name: str) -> dict:
    """PR-AUC per model / unit (diagnostic) and the reliability curve of the probability models."""
    plt = _plt()
    models = sorted(pr.model.unique()) if len(pr) else []
    cols = model_colors(models)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    for i, m in enumerate(models):
        t = pr[pr.model == m]
        a1.scatter([i] * len(t), t.pr_auc, color=cols[m], s=30)
    a1.set_xticks(range(len(models)))
    a1.set_xticklabels(models, rotation=45, ha="right", fontsize=7)
    a1.set_ylabel("PR-AUC per unit (diagnostic)")
    a1.grid(axis="y", color=GRID, linewidth=0.6)
    for m in sorted(cal.model.unique()) if len(cal) else []:
        t = cal[cal.model == m].groupby("bin").agg(p=("mean_pred", "mean"), y=("share_near_truth", "mean"))
        a2.plot(t.p, t.y, marker="o", markersize=4, color=cols.get(m, MUTED), label=m)
    a2.plot([0, 1], [0, 1], color=AXIS, linewidth=1, linestyle="--")
    a2.set_xlabel("mean predicted probability")
    a2.set_ylabel("share of pixels within 300 m of truth")
    a2.grid(color=GRID, linewidth=0.6)
    if len(cal):
        a2.legend(fontsize=7)
    fig.tight_layout()
    return save_figure(fig, {"pr_auc": pr, "calibration": cal}, out_dir, name)
