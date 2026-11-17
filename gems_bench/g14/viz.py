"""Plotting helpers for G14 reports (Phase 13). They draw only from tables / arrays produced by real stages; no
function here invents data. UNTESTED: written without execution."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def training_curves(history: list[dict], title: str = ""):
    plt = _plt()
    h = pd.DataFrame(history)
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6), layout="constrained")
    for c in [c for c in h if c.startswith("train_") and c != "train_total"]:
        ax[0].plot(h.epoch, h[c], label=c[6:])
    ax[0].set_title("training loss terms")
    ax[0].legend(fontsize=6)
    ax[1].plot(h.epoch, h.val_loss, label="inner val loss")
    ax[1].legend(fontsize=7)
    ax[2].plot(h.epoch, h.val_dti_surrogate, label="inner DTI surrogate (not the official metric)")
    ax[2].legend(fontsize=7)
    fig.suptitle(title)
    return fig


def comparison_bars(df: pd.DataFrame, metric: str = "disc_dti"):
    plt = _plt()
    d = df.dropna(subset=[metric]).sort_values(metric)
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(d) + 1.5), layout="constrained")
    ax.barh(d.model, d[metric])
    ax.set_xlabel(f"{metric} (nested offline proxy, not a leaderboard score)")
    return fig


def ablation_panel(df: pd.DataFrame, groups: dict[str, list[str]], metric: str = "disc_dti"):
    """One panel per ablation group with the control highlighted."""
    plt = _plt()
    fig, axs = plt.subplots(1, len(groups), figsize=(4 * len(groups), 3.5), layout="constrained", squeeze=False)
    for ax, (g, names) in zip(axs[0], groups.items()):
        d = df[df.model.isin(names)].set_index("model").reindex(names)
        ax.bar(range(len(d)), d[metric])
        ax.set_xticks(range(len(d)), [n.replace("A_", "") for n in d.index], rotation=60, ha="right", fontsize=7)
        ax.set_title(g)
    return fig


def map_panel(prob: np.ndarray, truth: np.ndarray, window: tuple[slice, slice], gate: np.ndarray | None = None):
    plt = _plt()
    n = 3 if gate is not None else 2
    fig, ax = plt.subplots(1, n, figsize=(4 * n, 4), layout="constrained")
    ax[0].imshow(np.nan_to_num(prob[window]), cmap="magma")
    ax[0].set_title("probability")
    ax[1].imshow(truth[window], cmap="gray_r")
    ax[1].set_title("held-out truth")
    if gate is not None:
        ax[2].imshow(gate, cmap="viridis")
        ax[2].set_title("fusion weight (diagnostic, not causal)")
    for a in ax:
        a.set_axis_off()
    return fig


def modality_weight_summary(weights: np.ndarray, names: list[str]) -> pd.DataFrame:
    """Mean gate weight per modality over a batch (B, M, H, W) -> table."""
    w = np.asarray(weights)
    return pd.DataFrame({"modality": names, "mean_weight": w.mean(axis=(0, 2, 3)), "p90_weight":
                         np.percentile(w, 90, axis=(0, 2, 3))})
