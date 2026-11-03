"""G07 research atlas. Reuses the G06 figure layer (``g06.visualization``: palette, ``save_figure`` = PNG + SVG +
CSV / Parquet source data, model comparison, DTI frontier, geographic robustness, PR / calibration, importance) and
adds DEM, feasibility, candidate-recall, exact-budget, change-map and case-atlas figures.

Maps carry real coordinates (``extent`` in the raster CRS, metres); panels of one figure share their extent. Held-out
truth appears only as a post-hoc outline in validation figures and never enters a model input.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..g06 import visualization as _V6

AXIS, GRID, INK, INK2, MUTED, SERIES = _V6.AXIS, _V6.GRID, _V6.INK, _V6.INK2, _V6.MUTED, _V6.SERIES
_cmaps, _limits, _plt, model_colors, save_figure = _V6._cmaps, _V6._limits, _V6._plt, _V6.model_colors, _V6.save_figure
plot_dti_frontier, plot_feature_importance = _V6.plot_dti_frontier, _V6.plot_feature_importance
plot_geographic_robustness, plot_model_comparison = _V6.plot_geographic_robustness, _V6.plot_model_comparison
plot_pr_and_calibration = _V6.plot_pr_and_calibration


def _extent(transform, shape):
    """(left, right, bottom, top) of an array with an affine ``transform`` (a, b, c, d, e, f)."""
    a, _, c, _, e, f = tuple(transform)[:6]
    h, w = shape
    return (c, c + w * a, f + h * e, f)


def _show(ax, arr, transform, cmap=None, signed=None, title="", unit=""):
    lo, hi, sg = _limits(np.asarray(arr, float))
    seq, div = _cmaps()
    sg = sg if signed is None else signed
    im = ax.imshow(arr, cmap=cmap or (div if sg else seq), vmin=lo, vmax=hi, extent=_extent(transform, arr.shape),
                   interpolation="nearest")
    ax.set_title(title + (f" [{unit}]" if unit else ""), fontsize=8)
    ax.tick_params(labelsize=6)
    return im


def plot_dem_native_panels(z: np.ndarray, feats: dict, transform, out_dir, name: str, title: str,
                           units: dict | None = None, overlays: dict | None = None) -> dict:
    """Native-resolution DEM, hillshades and geomorphometry of one window (same extent for every panel)."""
    plt = _plt()
    keys = ["elevation"] + list(feats)
    arrays = {"elevation": z, **feats}
    nc = 4
    nr = int(np.ceil(len(keys) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(3.4 * nc, 3.2 * nr), squeeze=False)
    rows = []
    for ax, k in zip(axes.ravel(), keys):
        a = np.asarray(arrays[k], float)
        cmap = "gray" if k.startswith("dem_hs") else None
        im = _show(ax, a, transform, cmap, title=k, unit=(units or {}).get(k, "m" if k == "elevation" else ""))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=6)
        for nm, (m, col) in (overlays or {}).items():
            if np.asarray(m).any():
                ax.contour(np.asarray(m, float), levels=[0.5], colors=[col], linewidths=0.5,
                           extent=_extent(transform, np.shape(m)), origin="upper")
        v = a[np.isfinite(a)]
        rows.append({"panel": k, "p02": float(np.quantile(v, .02)) if v.size else np.nan,
                     "p50": float(np.median(v)) if v.size else np.nan,
                     "p98": float(np.quantile(v, .98)) if v.size else np.nan, "finite_share": float(np.isfinite(a).mean())})
    for ax in axes.ravel()[len(keys):]:
        ax.axis("off")
    fig.suptitle(title, color=INK)
    fig.tight_layout()
    return save_figure(fig, pd.DataFrame(rows), out_dir, name)


def plot_resolution_comparison(hs_native: np.ndarray, t_native, elev_100m: np.ndarray, t_100m, out_dir, name: str
                               ) -> dict:
    """The same area: native-resolution hillshade vs the 100 m GeoDAWN elevation (shared extent)."""
    plt = _plt()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.6))
    _show(a1, hs_native, t_native, "gray", title="native-resolution hillshade")
    _show(a2, elev_100m, t_100m, title="GeoDAWN det_elev (100 m)", unit="m")
    a2.set_xlim(a1.get_xlim())
    a2.set_ylim(a1.get_ylim())
    fig.tight_layout()
    return save_figure(fig, pd.DataFrame({"native_shape": [str(hs_native.shape)], "grid_shape": [str(elev_100m.shape)]}),
                       out_dir, name)


def plot_dem_coverage(local: np.ndarray, catalog: np.ndarray, eligible: np.ndarray, transform, out_dir,
                      name: str = "dem_coverage", factor: int = 4) -> dict:
    """Eligible region coloured by DEM availability: local valid DEM / catalog only / none."""
    plt = _plt()
    from matplotlib.colors import ListedColormap

    cat = np.full(np.shape(eligible), np.nan)
    e = np.asarray(eligible, bool)
    cat[e] = 0
    cat[e & np.asarray(catalog, bool)] = 1
    cat[e & np.asarray(local, bool)] = 2
    cmap = ListedColormap([GRID, SERIES[3], SERIES[0]])
    fig, ax = plt.subplots(figsize=(7, 7.5))
    t = list(transform)
    t[0], t[4] = t[0] * factor, t[4] * factor
    ax.imshow(cat[::factor, ::factor], cmap=cmap, vmin=0, vmax=2, extent=_extent(t, cat[::factor, ::factor].shape),
              interpolation="nearest")
    from matplotlib.patches import Patch

    ax.legend(handles=[Patch(color=SERIES[0], label="local DEM (headers verified)"),
                       Patch(color=SERIES[3], label="catalog only (not downloaded)"),
                       Patch(color=GRID, label="no DEM")], fontsize=7, loc="lower right")
    ax.set_title("DEM coverage of the eligible fault-discovery region")
    T = pd.DataFrame({"class": ["none", "catalog_only", "local"],
                      "px": [int((cat == v).sum()) for v in (0, 1, 2)]})
    T["share_of_eligible"] = T.px / max(int(e.sum()), 1)
    return save_figure(fig, T, out_dir, name)


def plot_signal_auc(T: pd.DataFrame, out_dir, name: str = "dem_fault_signal", top: int = 25) -> dict:
    """Per DEM feature: AUC of fault-near vs background cells (held-out truth, DIAGNOSTIC) with a 0.5 reference."""
    plt = _plt()
    S = T.sort_values("auc_abs", ascending=False).head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 0.24 * len(S) + 1.3))
    ax.barh(S.feature, S.auc_abs, color=SERIES[0], height=0.7)
    ax.axvline(0.5, color=AXIS, linewidth=1)
    ax.set_xlabel("|AUC - 0.5| + 0.5 (fault-near vs background cells, DEM-covered V1 units)")
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_title("DEM fault signal (diagnostic - held-out truth)")
    return save_figure(fig, T, out_dir, name)


def plot_candidate_recall(R: pd.DataFrame, out_dir, name: str = "candidate_recall") -> dict:
    """Systems covered per pool / union / inventory and their area share (one panel per scenario)."""
    plt = _plt()
    if not len(R):
        return {}
    G = R.groupby(["scenario", "source"]).agg(n_systems=("n_systems", "sum"), n_covered=("n_covered", "sum"),
                                              area_frac=("area_frac", "mean")).reset_index()
    G["recall"] = G.n_covered / G.n_systems.clip(lower=1)
    sc = sorted(G.scenario.unique())
    fig, axes = plt.subplots(1, len(sc), figsize=(4.6 * len(sc), 0.3 * G.source.nunique() + 1.8), squeeze=False)
    for ax, s in zip(axes.ravel(), sc):
        t = G[G.scenario == s].sort_values("recall")
        ax.barh(t.source, t.recall, color=SERIES[0], height=0.65)
        for y, (r, a) in enumerate(zip(t.recall, t.area_frac)):
            ax.text(r, y, f" {a:.1%} area", va="center", fontsize=6, color=INK2)
        ax.set_xlim(0, 1)
        ax.set_xlabel("share of truth systems covered (geometric, 300 m)")
        ax.set_title(s)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
    fig.tight_layout()
    return save_figure(fig, G, out_dir, name)


def plot_exact_budget(rows: pd.DataFrame, out_dir, name: str = "exact_budget") -> dict:
    """Requested vs actual selected pixels and tied pixels at the cutoff, per model (exact-k audit)."""
    plt = _plt()
    if not len(rows) or "sel_k" not in rows:
        return {}
    G = rows.groupby("model").agg(k=("sel_k", "sum"), actual=("sel_actual_px", "sum"),
                                  legacy=("sel_legacy_px", "sum"), tied=("sel_n_tied_at_cutoff", "max")).reset_index()
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(G) + 1.5))
    y = np.arange(len(G))
    ax.barh(y - 0.2, G.k, height=0.38, color=SERIES[0], label="requested k")
    ax.barh(y + 0.2, G.legacy, height=0.38, color=SERIES[1], label="legacy >= cutoff rule")
    ax.set_yticks(y)
    ax.set_yticklabels(G.model, fontsize=7)
    ax.set_xlabel("pixels (summed over units and cells)")
    ax.legend(fontsize=7)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_title("Exact-k selection vs the legacy tie-inflated rule")
    return save_figure(fig, G, out_dir, name)


def plot_tp_fp_fn_change(q_ref: np.ndarray, q_new: np.ndarray, truth_near: np.ndarray, transform, out_dir, name: str,
                         title: str) -> dict:
    """Where a model changes the W2 selection: gained / lost truth-near pixels, added / removed far pixels (post hoc)."""
    plt = _plt()
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    a, b, t = np.asarray(q_ref) > 0, np.asarray(q_new) > 0, np.asarray(truth_near, bool)
    cat = np.full(a.shape, np.nan)
    cat[b & ~a & t] = 0
    cat[a & ~b & t] = 1
    cat[b & ~a & ~t] = 2
    cat[a & ~b & ~t] = 3
    cols = [SERIES[2], SERIES[7], SERIES[1], SERIES[0]]
    labels = ["gained near truth", "lost near truth", "added far from truth", "removed far from truth"]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(cat, cmap=ListedColormap(cols), vmin=0, vmax=3, extent=_extent(transform, cat.shape),
              interpolation="nearest")
    ax.legend(handles=[Patch(color=c, label=lb) for c, lb in zip(cols, labels)], fontsize=7, loc="lower right")
    ax.set_title(title + "\n(post-hoc diagnostic: truth proximity)", fontsize=8)
    T = pd.DataFrame({"class": labels, "px": [int((cat == v).sum()) for v in range(4)]})
    return save_figure(fig, T, out_dir, name)


def plot_case_atlas(case: dict, out_dir, name: str) -> dict:
    """One fault case: geophysics, DEM (native hillshade + scarp evidence), W2 / surface / subsurface / candidates /
    final selection on the 100 m grid, truth outline (post hoc), and an error-attribution note."""
    plt = _plt()
    panels = [(k, v) for k, v in case["panels"].items() if v is not None]
    nc = 4
    nr = int(np.ceil(len(panels) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(3.3 * nc, 3.3 * nr + 0.6), squeeze=False)
    for ax, (k, (arr, T, cmap)) in zip(axes.ravel(), panels):
        _show(ax, np.asarray(arr, float), T, cmap, title=k)
        tr = case.get("truth")
        if tr is not None:
            ax.contour(np.asarray(tr[0], float), levels=[0.5], colors=[INK], linewidths=0.6,
                       extent=_extent(tr[1], np.shape(tr[0])), origin="upper")
        ax.set_xlim(*case["xlim"])
        ax.set_ylim(*case["ylim"])
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    fig.suptitle(f"{case['title']}\n{case.get('attribution', '')}  (black: held-out truth - post hoc)", fontsize=9,
                 color=INK)
    fig.tight_layout()
    return save_figure(fig, pd.DataFrame([{k: v for k, v in case.get("facts", {}).items()}]), out_dir, name)
