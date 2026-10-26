"""Static report figures (PNG + SVG via the workspace). Fixed categorical order of the validated default palette
(the same hexes as p3.context.MODEL_COLORS); colour follows the entity, never its rank. Thin 2 px lines,
>= 8 px markers, recessive grid, one y-axis, legend for >= 2 series, text in ink colours."""
from __future__ import annotations

import numpy as np
import pandas as pd

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def colors_for(names) -> dict:
    """Stable slot per entity (first-seen order); a 9th entity folds into ink grey."""
    out = {}
    for i, n in enumerate(dict.fromkeys(names)):
        out[n] = SERIES[i] if i < len(SERIES) else INK2
    return out


def _style(ax, xlabel="", ylabel="", title=""):
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    if title:
        ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)


def line_panels(df: pd.DataFrame, x: str, y: str, series: str, panel: str, *, xlabel="", ylabel="DTI",
                logx=False, title=""):
    """One panel per ``panel`` value, one line per ``series`` value."""
    import matplotlib.pyplot as plt

    panels = list(dict.fromkeys(df[panel]))
    if not panels:
        return None
    col = colors_for(df[series])
    fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 3.4), squeeze=False)
    for ax, pv in zip(axes[0], panels):
        d = df[df[panel] == pv]
        for s, g in d.groupby(series, sort=False):
            g = g.sort_values(x)
            ax.plot(g[x], g[y], lw=2, marker="o", ms=4, color=col[s], label=str(s))
        if logx:
            ax.set_xscale("log")
        _style(ax, xlabel, ylabel, f"{title}{pv}")
    handles, labels = axes[0][0].get_legend_handles_labels()
    if len(labels) >= 2:
        fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 5), frameon=False, fontsize=8,
                   bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    return fig


def bar_panels(df: pd.DataFrame, cat: str, y: str, panel: str, *, ylabel="cross-fitted DTI", ref: dict | None = None,
               title=""):
    """Horizontal bars per category, one panel per ``panel``; ``ref`` = {panel: value} drawn as a dashed line."""
    import matplotlib.pyplot as plt

    panels = list(dict.fromkeys(df[panel]))
    if not panels:
        return None
    n = max(df.groupby(panel)[cat].nunique().max(), 1)
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 0.32 * n + 1.4), squeeze=False)
    for ax, pv in zip(axes[0], panels):
        d = df[df[panel] == pv].sort_values(y)
        ax.barh(np.arange(len(d)), d[y], color=SERIES[0], height=0.7, edgecolor="white", linewidth=2)
        ax.set_yticks(np.arange(len(d)), d[cat].astype(str), fontsize=8)
        if ref and pv in ref and np.isfinite(ref[pv]):
            ax.axvline(ref[pv], color=INK2, lw=1, ls="--")
        _style(ax, ylabel, "", f"{title}{pv}")
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig
