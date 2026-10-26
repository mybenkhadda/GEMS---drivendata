"""M7 figures. Analytical panels are 16:9; geographic maps keep true geometry (equal aspect, real extents in metres,
never stretched to fit a frame). Colours follow the entity in the fixed validated order of ``gems3.figures``.
Labels (mapped / withheld faults) are always drawn in ink / red line colours; predictions as filled rasters, so
the two can never be confused."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..gems3.figures import GRID, INK, INK2, SERIES, _style, colors_for

W16, H9 = 16.0, 9.0
TRUTH = "#e34948"          # withheld / scored truth
KNOWN = "#0b0b0b"          # known (retained / mapped) faults
CAND = "#2a78d6"           # candidate lines


def bar_panels16(df: pd.DataFrame, cat: str, y: str, panel: str, *, ref_col: str | None = None, title: str = "",
                 xlabel: str = "cross-fitted DTI"):
    """Horizontal bars per category, one panel per ``panel`` value, 16:9; ``ref_col`` drawn as a tick per bar."""
    import matplotlib.pyplot as plt

    panels = list(dict.fromkeys(df[panel]))
    if not panels:
        return None
    fig, axes = plt.subplots(1, len(panels), figsize=(W16, H9), squeeze=False)
    for ax, pv in zip(axes[0], panels):
        d = df[df[panel] == pv].sort_values(y)
        yy = np.arange(len(d))
        ax.barh(yy, d[y], color=SERIES[0], height=0.65, edgecolor="white", linewidth=2)
        if ref_col and ref_col in d:
            ax.scatter(d[ref_col], yy, marker="|", s=220, color=INK, zorder=3, label=ref_col)
        ax.set_yticks(yy, d[cat].astype(str), fontsize=9)
        _style(ax, xlabel, "", str(pv))
        ax.grid(axis="y", visible=False)
    if title:
        fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    return fig


def line_panels16(df: pd.DataFrame, x: str, y: str, series: str, panel: str, *, xlabel="", ylabel="", logx=False,
                  title="", hline: dict | None = None):
    import matplotlib.pyplot as plt

    panels = list(dict.fromkeys(df[panel]))
    if not panels:
        return None
    col = colors_for(df[series])
    fig, axes = plt.subplots(1, len(panels), figsize=(W16, H9), squeeze=False)
    for ax, pv in zip(axes[0], panels):
        d = df[df[panel] == pv]
        for s, g in d.groupby(series, sort=False):
            g = g.sort_values(x)
            ax.plot(g[x], g[y], lw=2, marker="o", ms=5, color=col[s], label=str(s))
        if hline and pv in hline and np.isfinite(hline[pv]):
            ax.axhline(hline[pv], color=INK2, lw=1, ls="--")
        if logx:
            ax.set_xscale("log")
        _style(ax, xlabel, ylabel, str(pv))
    h, lab = axes[0][0].get_legend_handles_labels()
    if len(lab) >= 2:
        fig.legend(h, lab, loc="lower center", ncol=min(len(lab), 6), frameon=False, fontsize=9,
                   bbox_to_anchor=(0.5, -0.02))
    if title:
        fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    return fig


def stacked_shares16(df: pd.DataFrame, cat: str, parts: list[str], *, title="", xlabel="share of systems"):
    """One horizontal stacked bar per category (e.g. both / B0 only / B1 only / neither per stratum)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(W16, H9))
    d = df.reset_index(drop=True)
    left = np.zeros(len(d))
    col = colors_for(parts)
    for p in parts:
        v = d[p].to_numpy(float) if p in d else np.zeros(len(d))
        ax.barh(np.arange(len(d)), v, left=left, color=col[p], height=0.7, edgecolor="white", linewidth=1.5, label=p)
        left += v
    ax.set_yticks(np.arange(len(d)), d[cat].astype(str), fontsize=9)
    _style(ax, xlabel, "", title)
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, fontsize=9, ncol=len(parts), loc="lower center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    return fig


def ledger_figure(led: pd.DataFrame):
    """Positive area (share of the eligible domain) after each masking step, per task."""
    import matplotlib.pyplot as plt

    d = led.groupby(["task", "step"], sort=False).frac_of_eligible.mean().reset_index()
    tasks = list(dict.fromkeys(d.task))
    fig, axes = plt.subplots(1, len(tasks), figsize=(W16, H9), squeeze=False)
    for ax, t in zip(axes[0], tasks):
        g = d[d.task == t]
        ax.barh(np.arange(len(g)), g.frac_of_eligible, color=SERIES[0], height=0.6)
        ax.set_yticks(np.arange(len(g)), g.step, fontsize=8)
        ax.invert_yaxis()
        _style(ax, "positive area / eligible pixels (mean over methods and zones)", "", t)
    fig.suptitle("Area ledger: selected area before and after every masking step", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    return fig


# =============================================================================== maps
def _extent(grid, r0, r1, c0, c1):
    return grid.window_extent(r0, r1, c0, c1)


def _lines(ax, mask, extent, color, lw=1.0, label=None):
    """Draw a thin binary mask as an outline-free overlay (labels), never as a filled prediction."""
    if mask is None or not np.asarray(mask).any():
        return
    from matplotlib.colors import ListedColormap

    ax.imshow(np.where(mask, 1.0, np.nan), extent=extent, cmap=ListedColormap([color]), interpolation="nearest",
              origin="upper", alpha=0.95)
    if label:
        ax.plot([], [], color=color, lw=2, label=label)


def map_panels(grid, window: tuple[int, int, int, int], layers: list[dict], *, title: str = "", ncols: int = 3):
    """Aligned map panels on one window (same extent, equal aspect, metres). ``layers`` items:
    {"title", "raster" (optional), "cmap", "vmin", "vmax", "truth", "known", "cand", "pred_mask"}."""
    import matplotlib.pyplot as plt

    r0, r1, c0, c1 = window
    ext = _extent(grid, r0, r1, c0, c1)
    n = len(layers)
    nrows = int(np.ceil(n / ncols))
    h, w = r1 - r0, c1 - c0
    cell = 4.6
    fig, axes = plt.subplots(nrows, ncols, figsize=(cell * ncols, cell * nrows * h / max(w, 1) + 0.6), squeeze=False)
    for ax, L in zip(axes.ravel(), layers):
        a = L.get("raster")
        if a is not None:
            v = np.asarray(a, np.float32)
            lo = L.get("vmin", np.nanpercentile(v, 2) if np.isfinite(v).any() else 0)
            hi = L.get("vmax", np.nanpercentile(v, 98) if np.isfinite(v).any() else 1)
            ax.imshow(v, extent=ext, cmap=L.get("cmap", "Greys"), vmin=lo, vmax=hi, interpolation="nearest",
                      origin="upper")
        if L.get("pred_mask") is not None:
            from matplotlib.colors import ListedColormap

            pm = np.asarray(L["pred_mask"]) > 0
            ax.imshow(np.where(pm, 1.0, np.nan), extent=ext, cmap=ListedColormap([SERIES[1]]), alpha=0.75,
                      interpolation="nearest", origin="upper")
            ax.plot([], [], color=SERIES[1], lw=6, alpha=0.75, label="prediction")
        _lines(ax, L.get("known"), ext, KNOWN, label="known faults" if L.get("legend") else None)
        _lines(ax, L.get("cand"), ext, CAND, label="candidates" if L.get("legend") else None)
        _lines(ax, L.get("truth"), ext, TRUTH, label="scored truth (label)" if L.get("legend") else None)
        ax.set_aspect("equal")
        ax.set_title(L.get("title", ""), fontsize=9, loc="left", color=INK)
        ax.tick_params(labelsize=6, colors=INK2)
        ax.ticklabel_format(style="plain", useOffset=False)
        if L.get("legend"):
            ax.legend(fontsize=7, frameon=False, loc="lower left")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    return fig


def hillshade(elev: np.ndarray, az_deg: float = 315.0, alt_deg: float = 40.0, pixel_m: float = 100.0) -> np.ndarray:
    """Plain hillshade in [0, 1] (NaN-safe)."""
    z = np.nan_to_num(np.asarray(elev, np.float64), nan=float(np.nanmedian(elev)) if np.isfinite(elev).any() else 0.0)
    gy, gx = np.gradient(z, pixel_m)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az, alt = np.radians(az_deg), np.radians(alt_deg)
    hs = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    out = np.clip(hs, 0, 1).astype(np.float32)
    out[~np.isfinite(elev)] = np.nan
    return out


def overview_map(grid, valid: np.ndarray, points: pd.DataFrame, *, color_col: str, title: str = "", step: int = 6):
    """Whole-region map (true geometry) with points coloured by a category (e.g. system class)."""
    import matplotlib.pyplot as plt

    H, W = valid.shape
    ext = grid.extent
    fig, ax = plt.subplots(figsize=(10, 10 * H / W))
    ax.imshow(np.where(valid[::step, ::step], 0.92, np.nan), extent=ext, cmap="Greys", vmin=0, vmax=1, origin="upper")
    col = colors_for(points[color_col])
    for k, g in points.groupby(color_col, sort=False):
        x, y = grid.rc_to_xy(g.r.to_numpy(), g.c.to_numpy())
        ax.scatter(x, y, s=14 + 4 * np.sqrt(g.get("size", pd.Series(1, index=g.index)).to_numpy()), color=col[k],
                   label=f"{k} ({len(g)})", alpha=0.85, edgecolor="white", linewidth=0.4)
    ax.set_aspect("equal")
    ax.ticklabel_format(style="plain", useOffset=False)
    ax.tick_params(labelsize=7, colors=INK2)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.grid(True, color=GRID, lw=0.4)
    fig.tight_layout()
    return fig
