"""Figures regenerated from saved tables / rasters only (PNG + SVG in ``figures/``)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

INK, MUTED, GRID = "#1f2328", "#59636e", "#d0d7de"
SERIES = ("#2a6fdb", "#e0782f", "#2f9e72", "#b8438f", "#7a5af5", "#8a6d1f")


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)


def _save(fig, out: Path, name: str) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / f"{name}.png", out / f"{name}.svg"]
    fig.savefig(paths[0], dpi=150, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    return paths


def loss_curves(history: pd.DataFrame, out: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    units = sorted(history.unit.unique())
    fig, axes = plt.subplots(2, len(units), figsize=(3.0 * len(units), 5.2), sharex=False, squeeze=False)
    for j, u in enumerate(units):
        h = history[history.unit == u].copy()
        h["step"] = np.arange(len(h))
        for i, (col, lab) in enumerate((("loss", "training loss"), ("inner_val_dti", "inner-val DTI"))):
            ax = axes[i, j]
            for k, (stage, g) in enumerate(h.groupby("stage", sort=False)):
                ax.plot(g.step, g[col], color=SERIES[k % len(SERIES)], linewidth=2, label=f"stage {stage}")
            _style(ax)
            ax.set_title(u if i == 0 else "", color=INK, fontsize=10)
            if j == 0:
                ax.set_ylabel(lab, color=INK, fontsize=9)
            if i == 1:
                ax.set_xlabel("epoch (stages concatenated)", color=MUTED, fontsize=8)
    axes[0, -1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    p = _save(fig, out, "loss_curves")
    plt.close(fig)
    return p


def raster_map(a: np.ndarray, out: Path, name: str, title: str, label: str, cmap: str, vmin=0.0, vmax=1.0):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 7.6))
    im = ax.imshow(np.ma.masked_invalid(a), cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_axis_off()
    ax.set_title(title, color=INK, fontsize=11)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label(label, color=MUTED, fontsize=9)
    cb.ax.tick_params(colors=MUTED, labelsize=8)
    p = _save(fig, out, name)
    plt.close(fig)
    return p


def gate_weights(df: pd.DataFrame, out: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    g = df.assign(wsum=df.w_dem * df.px).groupby(["level", "coverage_lo", "coverage_hi"]).agg(
        wsum=("wsum", "sum"), px=("px", "sum")).reset_index()
    g["w_dem"] = g.wsum / g.px
    g["mid"] = (g.coverage_lo + g.coverage_hi) / 2
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for k, (lvl, gg) in enumerate(g.groupby("level")):
        ax.plot(gg.mid, gg.w_dem, marker="o", markersize=5, linewidth=2, color=SERIES[k], label=f"level {lvl}")
    _style(ax)
    ax.set_xlabel("measured DEM coverage (bin centre)", color=INK, fontsize=9)
    ax.set_ylabel("mean DEM gate weight", color=INK, fontsize=9)
    ax.set_ylim(0, max(0.05, float(g.w_dem.max()) * 1.15))
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    p = _save(fig, out, "gate_weights")
    plt.close(fig)
    return p
