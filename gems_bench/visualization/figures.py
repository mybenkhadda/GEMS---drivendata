"""Phase-5 figures. Every function takes saved tables / rasters (never a model), so any figure can be
regenerated from ``metrics/`` and ``predictions/`` without retraining.

Style rules (shared with Phase 3/4): colour follows the model (``phase4_eval.colors_for``: fixed order,
never cycled, never by rank); one quantity per axis (no dual axes); thin marks; recessive grid; a legend
whenever >= 2 series; values and labels in neutral ink; geographic and discovery scores in separate panels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..phase4_eval import colors_for

INK, MUTED, GRID = "#2b2b2b", "#6b6b6b", "#e6e6e6"


def _ax_style(ax, xlabel="", ylabel="", title=""):
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    if title:
        ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#bdbdbd")
    ax.tick_params(colors=MUTED, labelsize=8)


def experiment_dashboard(board: pd.DataFrame, ref: str | None = None):
    """Dot plot per score family (geographic area-selected, discovery excl, discovery retained_fp, <2 km recall)
    with geographic fold range as a thin bar. Panels share the model order, never the scale."""
    import matplotlib.pyplot as plt

    metrics = [("geo_dti_area", "geographic DTI (area-selected)"), ("disc_dti_excl", "pseudo-discovery DTI (excl)"),
               ("disc_dti_retained_fp", "pseudo-discovery DTI (retained_fp)"), ("short_recall_w", "<2 km weighted recall")]
    metrics = [m for m in metrics if m[0] in board and board[m[0]].notna().any()]
    b = board.sort_values("geo_dti_area")
    col = colors_for(list(board.model))
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.4 * len(metrics), 0.38 * len(b) + 1.6), sharey=True)
    axes = np.atleast_1d(axes)
    y = np.arange(len(b))
    for ax, (m, title) in zip(axes, metrics):
        if m == "geo_dti_area" and "geo_fold_min" in b:
            ax.hlines(y, b.geo_fold_min, b.geo_fold_max, color="#c9c9c9", lw=2)
        ax.scatter(b[m], y, s=36, c=[col[x] for x in b.model], zorder=3, edgecolors="white", linewidths=1.5)
        if ref is not None and ref in set(b.model):
            ax.axvline(float(b.loc[b.model == ref, m].iloc[0]), color=MUTED, lw=0.8, ls="--")
        _ax_style(ax, title=title)
    axes[0].set_yticks(y, b.model)
    fig.tight_layout()
    return fig


def dti_area_curves(curves: pd.DataFrame, models: list[str], metric: str = "dti", title: str = ""):
    """Pooled metric vs predicted-area fraction (log x), one line per model (geographic or discovery curves)."""
    import matplotlib.pyplot as plt

    from ..p3.curves import dti_from

    col = colors_for(models)
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for m in models:
        g = curves[curves.model == m]
        if not len(g):
            continue
        s = g.groupby("target_frac").apply(dti_from) if metric == "dti" else g.groupby("target_frac")[metric].mean()
        ax.plot(s.index, s.values, lw=2, color=col[m], label=m)
    ax.set_xscale("log")
    _ax_style(ax, "predicted area fraction", metric, title)
    if len(models) > 1:
        ax.legend(frameon=False, fontsize=8)
    return fig


def short_fault_recall(geom: pd.DataFrame, strata=("len_lt1", "len_1_2", "len_2_5", "len_ge5", "endpoint", "intersection")):
    """Grouped horizontal bars: weighted recall per trace stratum and model (matched budget)."""
    import matplotlib.pyplot as plt

    strata = [s for s in strata if f"recall_w_{s}" in geom]
    models = list(geom.model)
    col = colors_for(models)
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(strata) * max(len(models), 2) / 2 + 1.5))
    h = 0.8 / max(len(models), 1)
    for i, m in enumerate(models):
        r = geom[geom.model == m].iloc[0]
        ax.barh(np.arange(len(strata)) + i * h, [r[f"recall_w_{s}"] for s in strata], height=h - 0.03,
                color=col[m], label=m)
    ax.set_yticks(np.arange(len(strata)) + 0.4 - h / 2, strata)
    _ax_style(ax, "weighted recall (matched budget)", "", "Recall by trace stratum")
    ax.legend(frameon=False, fontsize=8)
    return fig


def pu_diagnostics(history: pd.DataFrame, units: list[str] | None = None):
    """Mean predicted probability on unlabeled training pixels per epoch (collapse monitor), one line per unit."""
    import matplotlib.pyplot as plt

    h = history[history.unit.isin(units)] if units else history
    us = list(dict.fromkeys(h.unit))
    col = colors_for(us)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for u in us:
        g = h[h.unit == u]
        ax.plot(g.epoch, g.loss_unlabeled_mean_p, lw=2, color=col[u], label=u)
    _ax_style(ax, "epoch", "mean p on unlabeled pixels", "PU collapse monitor")
    if len(us) > 1:
        ax.legend(frameon=False, fontsize=8)
    return fig


def raster_windows(layers: dict[str, np.ndarray], windows: list[tuple[int, int, int]], labels: np.ndarray | None = None,
                   cmap: str = "viridis", title: str = ""):
    """Grid: one row per window (r, c, half), one column per layer; mapped faults as a thin black overlay."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(windows), len(layers), figsize=(2.6 * len(layers), 2.6 * len(windows)), squeeze=False)
    for i, (r, c, hw) in enumerate(windows):
        for j, (name, a) in enumerate(layers.items()):
            ax = axes[i, j]
            sub = a[max(r - hw, 0):r + hw, max(c - hw, 0):c + hw]
            ax.imshow(sub, cmap=cmap, interpolation="nearest")
            if labels is not None:
                lab = labels[max(r - hw, 0):r + hw, max(c - hw, 0):c + hw]
                ax.contour(lab.astype(float), levels=[0.5], colors="k", linewidths=0.5)
            ax.set_xticks([]), ax.set_yticks([])
            if i == 0:
                ax.set_title(name, fontsize=9, color=INK)
    if title:
        fig.suptitle(title, x=0.01, ha="left", color=INK)
    fig.tight_layout()
    return fig


def coverage_map(coverage: np.ndarray, valid: np.ndarray, tiles: pd.DataFrame | None = None, transform=None):
    """DEM valid-coverage fraction on the native grid (sequential single hue), data footprint in light gray."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 7))
    ax.imshow(np.where(valid, 0.0, np.nan), cmap="Greys", vmin=0, vmax=4, interpolation="nearest")
    im = ax.imshow(np.where(coverage > 0, coverage, np.nan), cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    fig.colorbar(im, ax=ax, shrink=0.6, label="1 m DEM coverage")
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_title("DEM source coverage (native 100 m grid)", loc="left", color=INK)
    return fig


def champion_selection(board: pd.DataFrame, champion: str, challengers: list[str]):
    """Paired panels: geographic vs primary pseudo-discovery DTI (two separate axes, same model colours)."""
    import matplotlib.pyplot as plt

    models = [champion] + [c for c in challengers if c != champion]
    b = board.set_index("model").reindex(models)
    col = colors_for(models)
    fig, axes = plt.subplots(1, 2, figsize=(9, 0.4 * len(models) + 1.6), sharey=True)
    for ax, m, t in zip(axes, ("geo_dti_area", "disc_dti_excl"), ("geographic DTI", "pseudo-discovery DTI (excl)")):
        ax.barh(np.arange(len(models)), b[m], color=[col[x] for x in models], height=0.6)
        _ax_style(ax, title=t)
    axes[0].set_yticks(np.arange(len(models)), models)
    axes[0].invert_yaxis()
    fig.tight_layout()
    return fig
