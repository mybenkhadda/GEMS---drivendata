"""Scientific figure style for the atlas.

Palettes: earth tones for elevation (``TERRAIN``), diverging ``RdBu_r`` centred at 0 for signed anomalies
(robust symmetric limits), perceptually uniform ``cividis`` for gradients / magnitudes, ``viridis`` for
coherence / fractions, ``magma`` for evidence. Every map: km axes in EPSG:32611, colour bar with unit, CRS note,
scale bar, consistent width. Maps are saved as PNG (rasters in SVG are huge); charts as PNG + SVG.
A missing source produces an explicit data-limitation panel, never an empty or invented map.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

INK, MUTED, GRID = "#1f2328", "#59636e", "#d0d7de"
CATEG = ("#2a6fdb", "#e0782f", "#2f9e72", "#b8438f", "#7a5af5", "#8a6d1f")
MAP_W = 7.2


def style():
    import matplotlib as mpl

    mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                         "axes.edgecolor": GRID, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.titleweight": "semibold", "figure.dpi": 110, "savefig.facecolor": "white",
                         "axes.spines.top": False, "axes.spines.right": False})


def terrain_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("atlas_terrain", ["#2f4b3a", "#5f7f4f", "#a3a86b", "#cdbb84", "#b48a5a",
                                                               "#8c6a4f", "#e8e2d6"])


def downsample(a: np.ndarray, max_px: int) -> tuple[np.ndarray, int]:
    f = max(int(np.ceil(max(a.shape) / max_px)), 1)
    if f == 1:
        return a, 1
    H, W = (a.shape[0] // f) * f, (a.shape[1] // f) * f
    with np.errstate(all="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(a[:H, :W].reshape(H // f, f, W // f, f), axis=(1, 3)), f


def _limits(a, kind):
    v = a[np.isfinite(a)]
    if not v.size:
        return None, None
    if kind == "signed":
        m = float(np.percentile(np.abs(v), 98)) or 1.0
        return -m, m
    if kind in ("fraction", "rank", "coherence"):
        return 0.0, 1.0
    if kind == "axial":
        return -1.0, 1.0
    return float(np.percentile(v, 2)), float(np.percentile(v, 98))


CMAPS = {"terrain": None, "signed": "RdBu_r", "gradient": "cividis", "coherence": "viridis", "fraction": "viridis",
         "rank": "magma", "axial": "twilight_shifted", "count": "magma", "categorical": "tab20"}


def map_panel(ax, a: np.ndarray, grid, kind: str, title: str, unit: str, max_px: int = 1400, vmin=None, vmax=None,
              cbar=True, extent=None):
    import matplotlib.pyplot as plt

    d, _ = downsample(np.asarray(a, np.float32), max_px)
    lo, hi = _limits(d, kind)
    vmin = lo if vmin is None else vmin
    vmax = hi if vmax is None else vmax
    cmap = terrain_cmap() if kind == "terrain" else plt.get_cmap(CMAPS.get(kind, "cividis"))
    ext = extent or grid.extent_km()
    im = ax.imshow(np.ma.masked_invalid(d), cmap=cmap, vmin=vmin, vmax=vmax, extent=ext, interpolation="nearest")
    ax.set_title(title, color=INK, loc="left")
    ax.set_xlabel("Easting (km)")
    ax.set_ylabel("Northing (km)")
    ax.tick_params(labelsize=7)
    if cbar:
        cb = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cb.set_label(unit, color=MUTED, fontsize=8)
        cb.ax.tick_params(labelsize=7, colors=MUTED)
    scale_bar(ax, ext)
    ax.text(0.01, 0.01, "EPSG:32611 (UTM 11N, WGS 84)", transform=ax.transAxes, fontsize=6, color=MUTED)
    return im


def scale_bar(ax, ext, km=None):
    x0, x1, y0, y1 = ext
    span = x1 - x0
    km = km or (50 if span > 200 else 20 if span > 60 else 5 if span > 15 else 1)
    xs, ys = x0 + 0.05 * span, y0 + 0.06 * (y1 - y0)
    ax.plot([xs, xs + km], [ys, ys], color=INK, linewidth=2.5, solid_capstyle="butt")
    ax.text(xs + km / 2, ys + 0.015 * (y1 - y0), f"{km} km", ha="center", fontsize=7, color=INK)


def limitation_panel(ax, title: str, text: str):
    ax.set_axis_off()
    ax.set_title(title, color=INK, loc="left")
    ax.text(0.5, 0.5, "DATA LIMITATION\n\n" + text, ha="center", va="center", fontsize=9, color=MUTED, wrap=True,
            transform=ax.transAxes, bbox=dict(boxstyle="round", fc="#f6f8fa", ec=GRID))


def save(fig, stem: Path, svg: bool = False, dpi: int = 200) -> list[Path]:
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    out = [stem.with_suffix(".png")]
    fig.savefig(out[0], dpi=dpi, bbox_inches="tight")
    if svg:
        out.append(stem.with_suffix(".svg"))
        fig.savefig(out[1], bbox_inches="tight")
    return out


def rose(ax, az_deg, w=None, bins: int = 36, color=CATEG[0], title=""):
    """Axial rose: 0-180 histogram mirrored to 180-360; north up, clockwise."""
    from .circular_statistics import rose as hist

    edges, h = hist(az_deg, w, bins)
    th = np.radians(np.r_[edges[:-1], edges[:-1] + 180])
    hh = np.r_[h, h]
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.bar(th, hh, width=np.radians(180 / bins), bottom=0, color=color, edgecolor="white", linewidth=0.5, alpha=0.9)
    ax.set_yticklabels([])
    ax.set_title(title, color=INK, fontsize=9)


def lines_on(ax, gdf, color="#e0782f", lw=0.8, km=True, label=None):
    for i, geom in enumerate(gdf.geometry):
        if geom is None:
            continue
        x, y = geom.xy
        ax.plot(np.asarray(x) / 1e3 if km else x, np.asarray(y) / 1e3 if km else y, color=color, linewidth=lw,
                label=label if i == 0 else None)


def interactive_map(layers: dict, grid, lineaments=None, faults_rc=None, max_px: int = 600):
    """Plotly heat-map with a layer dropdown (downsampled previews). Returns the figure (``fig.show()``)."""
    import plotly.graph_objects as go

    x0, x1, y0, y1 = grid.extent_km()
    fig = go.Figure()
    names = list(layers)
    for i, (name, (a, kind)) in enumerate(layers.items()):
        d, f = downsample(np.asarray(a, np.float32), max_px)
        lo, hi = _limits(d, kind)
        xs = np.linspace(x0, x1, d.shape[1])
        ys = np.linspace(y1, y0, d.shape[0])
        fig.add_trace(go.Heatmap(z=d, x=xs, y=ys, zmin=lo, zmax=hi, visible=(i == 0), name=name, colorbar={"title": name},
                                 colorscale="RdBu_r" if kind == "signed" else "Viridis" if kind in ("coherence", "fraction")
                                 else "Cividis" if kind == "gradient" else "Earth" if kind == "terrain" else "Magma"))
    extra = 0
    if lineaments is not None and len(lineaments):
        xs, ys = [], []
        for g in lineaments.geometry:
            if g is not None:
                x, y = g.xy
                xs += list(np.asarray(x) / 1e3) + [None]
                ys += list(np.asarray(y) / 1e3) + [None]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line={"color": "#e0782f", "width": 1},
                                 name="candidate lineaments (unverified)"))
        extra += 1
    if faults_rc is not None:
        rr, cc = faults_rc
        a, _, c, _, e, f = grid.transform[:6]
        fig.add_trace(go.Scattergl(x=(c + a * (cc + 0.5)) / 1e3, y=(f + e * (rr + 0.5)) / 1e3, mode="markers",
                                   marker={"size": 1.5, "color": "black"}, name="mapped faults"))
        extra += 1
    buttons = [{"label": n, "method": "update",
                "args": [{"visible": [j == i for j in range(len(names))] + [True] * extra}]} for i, n in enumerate(names)]
    fig.update_layout(updatemenus=[{"buttons": buttons, "x": 0, "y": 1.12, "xanchor": "left"}], height=650,
                      xaxis_title="Easting (km)", yaxis_title="Northing (km)", yaxis_scaleanchor="x",
                      margin={"l": 50, "r": 20, "t": 60, "b": 40}, template="simple_white")
    return fig
