"""Shared plotting helpers (consistent palette, UTM-km extents, scale bars, downsampled full-region maps)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .core import PALETTE

DIVERGING = {"mag_anom", "rtp", "tmi", "tmi_vg", "tc", "iso_grav_anom", "iso_grav_anom_vg", "det_elev",
             "geod_dilaterate"}


def plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as p

    p.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7,
                       "figure.dpi": 110})
    return p


def grid(n: int, ncols: int = 4, w: float = 4.0, h: float = 3.4):
    p = plt()
    nr = int(np.ceil(n / ncols))
    fig, axs = p.subplots(nr, ncols, figsize=(w * ncols, h * nr), squeeze=False)
    ax = axs.ravel()
    for a in ax[n:]:
        a.set_axis_off()
    return fig, ax[:n]


def win(D, r: int, c: int, half_px: int):
    H, W = D.grid.shape
    return (slice(max(r - half_px, 0), min(r + half_px, H)), slice(max(c - half_px, 0), min(c + half_px, W)))


def extent_km(D, sl=None):
    if sl is None:
        sl = (slice(0, D.grid.height), slice(0, D.grid.width))
    l, r, b, t = D.grid.window_extent(sl[0].start, sl[0].stop, sl[1].start, sl[1].stop)
    return [l / 1e3, r / 1e3, b / 1e3, t / 1e3]


def stretch(a, pct=(2, 98), symmetric=False):
    v = np.asarray(a)[np.isfinite(a)]
    if not v.size:
        return 0.0, 1.0
    if symmetric:
        m = float(np.percentile(np.abs(v), pct[1]))
        return -m, m
    lo, hi = np.percentile(v, pct)
    return float(lo), float(hi if hi > lo else lo + 1e-9)


def show(ax, D, a, title="", cmap=None, sl=None, step=1, band=None, units="", pct=(2, 98), faults=False,
         footprint=True, vmin=None, vmax=None, colorbar=True):
    """Map panel: UTM km extent, equal aspect, recorded stretch. Returns the stretch used."""
    sub = a if sl is None else a[sl]
    sub = sub[::step, ::step]
    sym = band in DIVERGING
    lo, hi = (vmin, vmax) if vmin is not None else stretch(sub, pct, sym)
    cm = cmap or ("RdBu_r" if sym else "viridis")
    ext = extent_km(D, sl)
    im = ax.imshow(sub, extent=ext, cmap=cm, vmin=lo, vmax=hi, interpolation="nearest", rasterized=True)
    if footprint:
        v = (D.valid if sl is None else D.valid[sl])[::step, ::step]
        ax.contour(np.linspace(ext[0], ext[1], v.shape[1]), np.linspace(ext[3], ext[2], v.shape[0]), v.astype(float),
                   levels=[0.5], colors="0.3", linewidths=0.4)
    if faults:
        overlay_faults(ax, D, sl, step)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=6)
    ax.set_xlabel("E (km, UTM 11N)", fontsize=6)
    if colorbar:
        cb = ax.figure.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.ax.tick_params(labelsize=6)
        if units:
            cb.set_label(units, fontsize=6)
    scalebar(ax, ext)
    return {"vmin": lo, "vmax": hi, "cmap": cm, "step": step}


def overlay_faults(ax, D, sl=None, step=1, color=None, ms=0.3):
    sk = D.net.skeleton if sl is None else D.net.skeleton[sl]
    r, c = np.nonzero(sk[::step, ::step])
    r0, c0 = (0, 0) if sl is None else (sl[0].start, sl[1].start)
    x, y = D.grid.rc_to_xy(r * step + r0, c * step + c0)
    ax.plot(np.asarray(x) / 1e3, np.asarray(y) / 1e3, ",", color=color or PALETTE["faults"], ms=ms)


def scalebar(ax, ext, km=None):
    span = ext[1] - ext[0]
    km = km or (10 if span > 60 else 5 if span > 15 else 1)
    x0, y0 = ext[0] + 0.05 * span, ext[2] + 0.05 * (ext[3] - ext[2])
    ax.plot([x0, x0 + km], [y0, y0], "k-", lw=2.5)
    ax.text(x0 + km / 2, y0 + 0.02 * (ext[3] - ext[2]), f"{km} km", ha="center", fontsize=6)


def ecdf(ax, v, label, color=None, weights=None):
    v = np.asarray(v, float)
    m = np.isfinite(v)
    v = v[m]
    if not v.size:
        return
    o = np.argsort(v)
    w = np.ones_like(v) if weights is None else np.asarray(weights, float)[m]
    y = np.cumsum(w[o]) / w.sum()
    ax.step(v[o], y, where="post", label=f"{label} (n={len(v)})", color=color)


def effect_bars(ax, df: pd.DataFrame, label_col="label", title="", min_effect=0.1, color_col=None):
    d = df.reset_index(drop=True)
    y = np.arange(len(d))
    cols = [PALETTE.get(x, PALETTE["other"]) for x in d[color_col]] if color_col else "k"
    ax.errorbar(d.effect, y, xerr=[d.effect - d.ci_lo, d.ci_hi - d.effect], fmt="none", ecolor="0.5", lw=1)
    ax.scatter(d.effect, y, c=cols, s=18, zorder=3)
    ax.axvspan(-min_effect, min_effect, color="0.9", zorder=0)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(d[label_col], fontsize=6)
    ax.set_xlabel("Cliff's delta (fault vs matched control), 95% cluster CI")
    ax.set_title(title)


def map_data(name: str, arrays: dict, D, extra: dict | None = None) -> pd.DataFrame:
    """Compact CSV for a map figure: plotted layers with their statistics (the arrays themselves come from the
    named source bands / rasters, which are the reproducible inputs)."""
    rows = []
    for k, a in arrays.items():
        v = np.asarray(a)[np.isfinite(a)]
        rows.append({"figure": name, "layer": k, "n_finite": int(v.size), "p02": float(np.percentile(v, 2)) if v.size
                     else np.nan, "median": float(np.median(v)) if v.size else np.nan,
                     "p98": float(np.percentile(v, 98)) if v.size else np.nan, "crs": "EPSG:32611",
                     "source": "gems-geodawn-numerical-features.tif / derived (see figure interpretation)",
                     **(extra or {})})
    return pd.DataFrame(rows)
