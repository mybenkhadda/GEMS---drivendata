"""Static PNG + SVG figures (matplotlib, Agg) with their source data as CSV, for every saved analysis.
Interactive views in the app use Plotly; these are the persistent, reproducible copies."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save(fig, store, run_dir: Path, name: str, data: pd.DataFrame | None) -> list[Path]:
    plt = _plt()
    out = []
    for fmt in ("png", "svg"):
        p = run_dir / f"{name}.{fmt}"
        store._atomic(p, lambda t, f=fmt: fig.savefig(t, dpi=150, bbox_inches="tight", format=f))
        out.append(p)
    plt.close(fig)
    if data is not None:
        out += store.table(data, run_dir / "data" / name, parquet=False)
    return out


def extent_km(window, transform):
    a = transform
    x0 = a[2] + window.col_off * a[0]
    y0 = a[5] + window.row_off * a[4]
    return [x0 / 1e3, (x0 + window.width * a[0]) / 1e3, (y0 + window.height * a[4]) / 1e3, y0 / 1e3]


def map_png(store, run_dir, name, arr, window, transform, faults=None, cmap="viridis", pct=(2, 98),
            title=None, categorical=False):
    plt = _plt()
    a = np.asarray(arr, np.float32)
    fig, ax = plt.subplots(figsize=(7, 6))
    v = a[np.isfinite(a)]
    lo, hi = (np.percentile(v, pct) if v.size and not categorical else (np.nanmin(a), np.nanmax(a)))
    im = ax.imshow(a, extent=extent_km(window, transform), cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
    if faults is not None and np.any(faults):
        rr, cc = np.nonzero(faults)
        ex = extent_km(window, transform)
        ax.plot(ex[0] + (cc + 0.5) * (ex[1] - ex[0]) / a.shape[1], ex[3] - (rr + 0.5) * (ex[3] - ex[2]) / a.shape[0],
                ",", color="crimson")
    fig.colorbar(im, ax=ax, fraction=0.04)
    ax.set(title=title or name, xlabel="UTM E (km)", ylabel="UTM N (km)")
    ax.set_aspect("equal")
    stats = pd.DataFrame([{"layer": name, "n_valid": int(v.size), "p_lo": float(lo) if v.size else np.nan,
                           "p_hi": float(hi) if v.size else np.nan, "row_off": int(window.row_off),
                           "col_off": int(window.col_off), "height": int(window.height), "width": int(window.width),
                           "note": "pixel values: save the GeoTIFF export; this CSV holds the display statistics"}])
    return _save(fig, store, run_dir, name, stats)


def bar_png(store, run_dir, name, df: pd.DataFrame, x: str, y: str, title=None):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(df) + 1.5))
    d = df[[x, y]].dropna()
    ax.barh(d[x].astype(str), d[y], color="#2b7bb9")
    ax.set(xlabel=y, title=title or name)
    ax.invert_yaxis()
    return _save(fig, store, run_dir, name, df)


def heatmap_png(store, run_dir, name, grid: np.ndarray, title=None, cmap="RdBu_r", vmin=None, vmax=None,
                data: pd.DataFrame | None = None):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax)
    for (i, j), v in np.ndenumerate(grid):
        ax.text(j, i, "n/a" if not np.isfinite(v) else f"{v:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.04)
    ax.set(title=title or name, xlabel="tile column", ylabel="tile row")
    return _save(fig, store, run_dir, name, data)


def lines_png(store, run_dir, name, df: pd.DataFrame, x: str, ys: list, band: tuple | None = None, title=None,
              xlabel=None, markers: list | None = None):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 4))
    for y in ys:
        ax.plot(df[x], df[y], label=y)
    if band:
        ax.fill_between(df[x], df[band[0]], df[band[1]], alpha=0.25, color="0.5", label=f"{band[0]}-{band[1]}")
    for m in markers or []:
        ax.axvline(m, color="crimson", lw=0.8, ls="--")
    ax.set(xlabel=xlabel or x, title=title or name)
    ax.legend(fontsize=7)
    return _save(fig, store, run_dir, name, df)
