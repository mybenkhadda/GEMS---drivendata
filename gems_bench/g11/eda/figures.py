"""EDA figures (PNG, deterministic names). Maps use the master-grid extent in km, equal aspect, a scale bar and
grey nodata. Every function returns the saved path or None (with the reason printed) when its input is missing."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OKABE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]


def _plt():
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})
    return plt


def _save(fig, path) -> Path:
    plt = _plt()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.stem + ".partial.png")
    fig.savefig(tmp, dpi=170)
    tmp.replace(p)
    plt.close(fig)
    return p


def _extent(grid):
    x0, y0, x1, y1 = grid.bounds
    return (x0 / 1e3, x1 / 1e3, y0 / 1e3, y1 / 1e3)


def _bar(ax, km=20):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    km = min(km, (x1 - x0) / 4)
    ax.plot([x0 + 0.05 * (x1 - x0), x0 + 0.05 * (x1 - x0) + km], [y0 + 0.05 * (y1 - y0)] * 2, "k", lw=3)
    ax.text(x0 + 0.05 * (x1 - x0) + km / 2, y0 + 0.07 * (y1 - y0), f"{km:g} km", ha="center", fontsize=7)


def coverage_map(grid, valid, dem_cov, fault, split_code, path):
    plt = _plt()
    from matplotlib.colors import ListedColormap

    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    base = np.where(valid, 0.0, np.nan)
    for a in ax:
        cm = ListedColormap(["#f0f0f0"]).with_extremes(bad="#d0d0d0")
        a.imshow(base, extent=_extent(grid), cmap=cm)
    ax[0].imshow(np.where(dem_cov, 1.0, np.nan), extent=_extent(grid), cmap=ListedColormap([OKABE[5]]), alpha=0.8)
    ax[0].imshow(np.where(fault, 1.0, np.nan), extent=_extent(grid), cmap=ListedColormap(["black"]))
    ax[0].set_title("1 m DEM coverage (blue) and mapped faults (black)")
    if split_code is not None:
        cols = ListedColormap([OKABE[0], OKABE[2], OKABE[1], OKABE[3], "#bbbbbb"])
        ax[1].imshow(np.where(split_code > 0, split_code.astype(float), np.nan), extent=_extent(grid), cmap=cols,
                     vmin=0.5, vmax=5.5)
        ax[1].set_title("Splits: train / validation / stack / test / embargo")
    for a in ax:
        a.set_xlabel("Easting (km)")
        a.set_ylabel("Northing (km)")
        a.set_aspect("equal")
        _bar(a)
    return _save(fig, path)


def band_with_faults(grid, band: np.ndarray, fault: np.ndarray, path, title: str, unit: str = ""):
    """Geophysical band (2-98 % stretch) with mapped faults overlaid - visual alignment check."""
    plt = _plt()
    from matplotlib.colors import ListedColormap

    fig, ax = plt.subplots(figsize=(8, 7))
    v = band[np.isfinite(band)]
    lo, hi = (np.percentile(v, [2, 98]) if v.size else (0, 1))
    im = ax.imshow(band, extent=_extent(grid), cmap="viridis", vmin=lo, vmax=hi)
    ax.imshow(np.where(fault, 1.0, np.nan), extent=_extent(grid), cmap=ListedColormap(["#ff2a2a"]), interpolation="nearest")
    fig.colorbar(im, ax=ax, shrink=0.7, label=unit or "value")
    ax.set(title=title + " (red = mapped faults)", xlabel="Easting (km)", ylabel="Northing (km)", aspect="equal")
    _bar(ax)
    return _save(fig, path)


def dem_footprints(grid, count: np.ndarray, usable: np.ndarray, fault: np.ndarray, path):
    """DEM tile overlap count, usable (verified-unit) coverage and mapped faults with / without DEM."""
    plt = _plt()
    from matplotlib.colors import ListedColormap

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    im = ax[0].imshow(np.where(count > 0, count, np.nan), extent=_extent(grid), cmap="magma", interpolation="nearest")
    fig.colorbar(im, ax=ax[0], shrink=0.7, label="DEM tiles covering the cell")
    ax[0].set_title("DEM footprints (all tiles)")
    ax[1].imshow(np.where(usable, 1.0, np.nan), extent=_extent(grid), cmap=ListedColormap([OKABE[5]]), alpha=0.6)
    ax[1].imshow(np.where(fault & usable, 1.0, np.nan), extent=_extent(grid), cmap=ListedColormap([OKABE[2]]))
    ax[1].imshow(np.where(fault & ~usable, 1.0, np.nan), extent=_extent(grid), cmap=ListedColormap([OKABE[3]]))
    ax[1].set_title("Usable DEM (blue); faults with DEM (green) / without (red)")
    for a in ax:
        a.set(xlabel="Easting (km)", ylabel="Northing (km)", aspect="equal")
        _bar(a)
    return _save(fig, path)


def native_window(hillshade: np.ndarray, slope: np.ndarray, extent, title: str, path):
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    ax[0].imshow(hillshade, extent=extent, cmap="gray")
    ax[0].set_title("hillshade (native window)")
    im = ax[1].imshow(slope, extent=extent, cmap="viridis")
    fig.colorbar(im, ax=ax[1], shrink=0.7, label="slope (deg)")
    ax[1].set_title("slope")
    for a in ax:
        a.set(xlabel="Easting (km)", ylabel="Northing (km)", aspect="equal")
    fig.suptitle(title, fontsize=8)
    return _save(fig, path)


def missing_data(grid, valid: np.ndarray, band_valid_count: np.ndarray, n_bands: int, path):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(np.where(band_valid_count > 0, band_valid_count, np.nan), extent=_extent(grid), cmap="cividis",
                   vmin=0, vmax=n_bands)
    ax.contour(valid.astype(float), levels=[0.5], colors="red", linewidths=0.6, extent=_extent(grid), origin="upper")
    fig.colorbar(im, ax=ax, shrink=0.7, label=f"finite GeoDAWN bands (of {n_bands})")
    ax.set(title="Geophysical data availability (red = valid footprint)", xlabel="Easting (km)", ylabel="Northing (km)",
           aspect="equal")
    _bar(ax)
    return _save(fig, path)


def split_map(grid, split_code: np.ndarray, fault: np.ndarray, path):
    plt = _plt()
    from matplotlib.colors import ListedColormap

    fig, ax = plt.subplots(figsize=(8, 7))
    cols = ListedColormap([OKABE[0], OKABE[2], OKABE[1], OKABE[3], "#bbbbbb"])
    ax.imshow(np.where(split_code > 0, split_code.astype(float), np.nan), extent=_extent(grid), cmap=cols, vmin=0.5,
              vmax=5.5, interpolation="nearest")
    ax.imshow(np.where(fault, 1.0, np.nan), extent=_extent(grid), cmap=ListedColormap(["black"]), interpolation="nearest")
    import matplotlib.patches as mp

    ax.legend(handles=[mp.Patch(color=c, label=n) for c, n in zip(cols.colors, ["train", "validation", "stack (calibration)",
                                                                              "test", "embargo"])], loc="lower right",
              fontsize=7)
    ax.set(title="Geographic splits (black = mapped faults)", xlabel="Easting (km)", ylabel="Northing (km)", aspect="equal")
    _bar(ax)
    return _save(fig, path)


def contrast_forest(contrast: pd.DataFrame, path, top: int = 30):
    if contrast is None or not len(contrast):
        print("[eda figure] contrast table empty - figure skipped")
        return None
    plt = _plt()
    c = contrast.assign(a=contrast.near_vs_local_control.abs()).nlargest(top, "a").iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.25 * len(c) + 1.5))
    y = np.arange(len(c))
    ax.errorbar(c.near_vs_local_control, y, xerr=[c.near_vs_local_control - c.ci_low, c.ci_high - c.near_vs_local_control],
                fmt="o", color=OKABE[0], ms=3, capsize=2)
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(c.feature, fontsize=7)
    ax.set_xlabel("near-fault vs local control (robust standardised difference), 90% block-bootstrap CI")
    ax.set_title("Fault contrast (top features)")
    return _save(fig, path)


def utility_heatmap(util: pd.DataFrame, path):
    if util is None or not len(util):
        print("[eda figure] utility table empty - figure skipped")
        return None
    plt = _plt()
    u = util[(util.region == "all") & util.scale_m.notna()].copy()
    u["base"] = u.feature.str.replace(r"_\d+m$", "", regex=True)
    piv = u.pivot_table(index="base", columns="scale_m", values="auc")
    piv = piv.loc[piv.max(axis=1).sort_values(ascending=False).index[:40]]
    fig, ax = plt.subplots(figsize=(1 + 0.6 * piv.shape[1], 0.22 * len(piv) + 1.5))
    im = ax.imshow(piv.values, aspect="auto", cmap="viridis", vmin=0.5, vmax=max(0.6, np.nanmax(piv.values)))
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels([f"{c:g} m" for c in piv.columns], rotation=45)
    ax.set_yticks(range(len(piv)))
    ax.set_yticklabels(piv.index, fontsize=7)
    fig.colorbar(im, ax=ax, label="direction-free AUC (near fault vs local control)")
    ax.set_title("Feature utility by physical scale")
    return _save(fig, path)


def atlas_site(site: dict, path):
    """Multiscale atlas panel: geophysics, model score, mapped faults, DEM hillshade, slope, curvature, profiles."""
    plt = _plt()
    panels = [k for k in ("geophysics", "score", "hillshade", "slope", "curvature") if site.get(k) is not None]
    n = len(panels) + (1 if site.get("profiles") is not None else 0)
    fig, ax = plt.subplots(1, max(n, 1), figsize=(4.2 * max(n, 1), 4.2), squeeze=False)
    for a, k in zip(ax[0], panels):
        arr, ext = site[k]
        a.imshow(arr, extent=ext, cmap="gray" if k == "hillshade" else "viridis")
        if site.get("faults_xy") is not None:
            for xs, ys in site["faults_xy"]:
                a.plot(np.asarray(xs) / 1e3, np.asarray(ys) / 1e3, color=OKABE[3], lw=1)
        a.set_title(k)
        a.set_aspect("equal")
        a.set_xlabel("km")
    if site.get("profiles") is not None:
        a = ax[0][len(panels)]
        offs, Z = site["profiles"]
        for z in Z[:: max(len(Z) // 10, 1)]:
            a.plot(offs, z - np.nanmean(z), lw=0.6, color=OKABE[0], alpha=0.6)
        a.set_xlabel("offset from trace (m)")
        a.set_ylabel("relative elevation (m)")
        a.set_title("cross-fault profiles")
    fig.suptitle(f"{site.get('title', '')} | tiles: {site.get('tiles', 'none')}", fontsize=8)
    return _save(fig, path)
