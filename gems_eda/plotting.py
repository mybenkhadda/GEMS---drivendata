"""Map plotting on a shared geographic frame (equal ground aspect, consistent windows)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import LightSource
from matplotlib.patches import Rectangle
from scipy import ndimage

from .io import Grid

DPI = 220

DIVERGING = {"mag_anom", "rtp", "tmi", "tmi_vg", "tc", "iso_grav_anom", "iso_grav_anom_vg",
             "det_elev", "geod_dilaterate", "magnetic:tilt", "gravity:tilt"}
CMAPS = {"magnetic": "RdBu_r", "gravity": "PuOr_r", "terrain": "gist_earth", "basement": "viridis_r",
         "conductivity": "magma", "strain": "inferno", "seismicity": "cividis"}


def set_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": DPI, "font.size": 8, "axes.titlesize": 9,
        "axes.labelsize": 8, "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 7,
        "axes.titleweight": "bold", "figure.constrained_layout.use": True,
    })


def cmap_for(name: str) -> str:
    from .config import band_modality

    m = band_modality(name)
    if name in DIVERGING and m in ("magnetic", "gravity"):
        return "RdBu_r" if m == "magnetic" else "PuOr_r"
    if name.endswith("_hg") or name.endswith("_slope") or ":thdr" in name or name == "tmi_hg":
        return "magma"
    return CMAPS.get(m, "viridis")


@dataclass(frozen=True)
class Window:
    name: str
    label: str
    r0: int
    r1: int
    c0: int
    c1: int

    @property
    def slices(self) -> tuple[slice, slice]:
        return slice(self.r0, self.r1), slice(self.c0, self.c1)

    def extent(self, grid: Grid):
        return grid.window_extent(self.r0, self.r1, self.c0, self.c1)

    def contains(self, r, c) -> np.ndarray:
        return (np.asarray(r) >= self.r0) & (np.asarray(r) < self.r1) & (np.asarray(c) >= self.c0) & (np.asarray(c) < self.c1)

    @classmethod
    def around(cls, name, label, r, c, half_px, grid: Grid) -> "Window":
        half_px = int(max(half_px, 8))
        r0 = int(np.clip(r - half_px, 0, max(grid.height - 2 * half_px, 0)))
        c0 = int(np.clip(c - half_px, 0, max(grid.width - 2 * half_px, 0)))
        return cls(name, label, r0, min(r0 + 2 * half_px, grid.height), c0, min(c0 + 2 * half_px, grid.width))


def pick_windows(score_maps: list[tuple[str, str, np.ndarray]], grid: Grid, valid: np.ndarray,
                 size_km: float) -> list[Window]:
    """One square window per score map at the max of its window-averaged score, without overlap."""
    size_px = int(min(grid.km_to_px(size_km), 0.45 * min(grid.shape)))
    half = size_px // 2
    chosen: list[Window] = []
    interior = np.zeros(grid.shape, bool)
    interior[half:grid.height - half, half:grid.width - half] = True
    cover = ndimage.uniform_filter(valid.astype(float), size_px) > 0.9
    for name, label, score in score_maps:
        sm = ndimage.uniform_filter(np.nan_to_num(score).astype(float), size_px)
        sm[~(interior & cover)] = -np.inf
        for w in chosen:
            cr, cc = (w.r0 + w.r1) // 2, (w.c0 + w.c1) // 2
            rr, ccs = np.ogrid[:grid.height, :grid.width]
            sm[np.hypot(rr - cr, ccs - cc) < 0.9 * size_px] = -np.inf
        if not np.isfinite(sm).any():
            continue
        r, c = np.unravel_index(np.nanargmax(sm), sm.shape)
        chosen.append(Window.around(name, label, r, c, half, grid))
    return chosen


def windows_table(windows: list[Window], grid: Grid) -> pd.DataFrame:
    rows = []
    for w in windows:
        l, r, b, t = w.extent(grid)
        rows.append({"name": w.name, "label": w.label, "r0": w.r0, "r1": w.r1, "c0": w.c0, "c1": w.c1,
                     "xmin": l, "xmax": r, "ymin": b, "ymax": t})
    return pd.DataFrame(rows)


def windows_from_table(df: pd.DataFrame) -> list[Window]:
    return [Window(r.name, r.label, int(r.r0), int(r.r1), int(r.c0), int(r.c1)) for r in df.itertuples()]


# ---------------------------------------------------------------- primitives
def stretch(a: np.ndarray, pct=(2, 98), symmetric: bool = False) -> tuple[float, float]:
    v = a[np.isfinite(a)]
    if v.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(v, pct)
    if symmetric:
        m = max(abs(lo), abs(hi))
        return -m, m
    return float(lo), float(hi)


def show_raster(ax, a: np.ndarray, grid: Grid, window: Window | None = None, cmap="viridis",
                pct=(2, 98), symmetric=False, title=None, colorbar=False, vmin=None, vmax=None, alpha=1.0):
    if window is not None:
        sub, ext = a[window.slices], window.extent(grid)
    else:
        sub, ext = a, grid.extent
    if vmin is None or vmax is None:
        vmin, vmax = stretch(sub, pct, symmetric)
    im = ax.imshow(sub, extent=ext, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest", alpha=alpha)
    frame(ax, grid, ext)
    if title:
        ax.set_title(title)
    if colorbar:
        plt.colorbar(im, ax=ax, shrink=0.75, pad=0.01)
    return im


def show_rgb(ax, rgb: np.ndarray, grid: Grid, window: Window | None = None, title=None):
    sub, ext = (rgb[window.slices], window.extent(grid)) if window is not None else (rgb, grid.extent)
    ax.imshow(sub, extent=ext, interpolation="nearest")
    frame(ax, grid, ext)
    if title:
        ax.set_title(title)


def frame(ax, grid: Grid, ext) -> None:
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_aspect(grid.aspect)
    ax.tick_params(length=2)


def paths_xy(paths: list[np.ndarray], grid: Grid) -> list[np.ndarray]:
    out = []
    for p in paths:
        x, y = grid.rc_to_xy(p[:, 0], p[:, 1])
        out.append(np.column_stack([x, y]))
    return out


def add_lines(ax, lines_xy: list[np.ndarray], color="k", lw=0.5, alpha=0.9, label=None, zorder=3):
    lc = LineCollection(lines_xy, colors=color, linewidths=lw, alpha=alpha, label=label, zorder=zorder)
    ax.add_collection(lc)
    return lc


def add_window_boxes(ax, windows: list[Window], grid: Grid, color="w") -> None:
    for w in windows:
        l, r, b, t = w.extent(grid)
        ax.add_patch(Rectangle((l, b), r - l, t - b, fill=False, ec=color, lw=1.0, zorder=5))
        ax.text(l, t, f" {w.name}", color=color, fontsize=7, va="top", ha="left", zorder=6, fontweight="bold")


def rgb_composite(channels: list[np.ndarray], pct=(2, 98)) -> np.ndarray:
    out = []
    for ch in channels:
        lo, hi = stretch(ch, pct)
        out.append(np.clip((ch - lo) / (hi - lo + 1e-12), 0, 1))
    rgb = np.dstack(out)
    rgb[~np.isfinite(rgb).all(-1)] = 1.0
    return rgb


def hillshade_colored(elev: np.ndarray, color_field: np.ndarray, grid: Grid, cmap="RdBu_r", vert_exag=None):
    """Shaded relief of detrended elevation, coloured by another field."""
    ls = LightSource(azdeg=315, altdeg=40)
    lo, hi = stretch(color_field, (2, 98), symmetric=True)
    norm = np.clip((np.nan_to_num(color_field) - lo) / (hi - lo + 1e-12), 0, 1)
    rgb = plt.get_cmap(cmap)(norm)[..., :3]
    e = np.nan_to_num(elev, nan=np.nanmedian(elev))
    ve = vert_exag or 1.0 / max(np.nanstd(elev), 1e-6) * grid.pixel_km * 3
    shaded = ls.shade_rgb(rgb, e, vert_exag=ve, blend_mode="soft", dx=grid.pixel_km_x, dy=grid.pixel_km_y)
    shaded[~np.isfinite(elev) | ~np.isfinite(color_field)] = 1.0
    return shaded


def axial_rose(ax, centers_deg: np.ndarray, density: np.ndarray, color="k", label=None, fill=True, lw=1.0):
    """Bidirectional rose on a polar axis (north up, clockwise)."""
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    th = np.radians(np.concatenate([centers_deg, centers_deg + 180, centers_deg[:1]]))
    r = np.concatenate([density, density, density[:1]])
    ax.plot(th, r, color=color, lw=lw, label=label)
    if fill:
        ax.fill(th, r, color=color, alpha=0.25)
    ax.set_yticklabels([])
    ax.set_xticks(np.radians([0, 45, 90, 135, 180, 225, 270, 315]))
    ax.set_xticklabels(["N", "", "E", "", "S", "", "W", ""])


def savefig(fig, path: Path, manifest=None, name: str | None = None, description: str = "", show=True) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    if manifest is not None:
        manifest.add(name or path.stem, path, "figure", description)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return path
