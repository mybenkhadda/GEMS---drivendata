"""The 24 research figures (Phase T). Each function saves ``<out>.png`` AND its source table ``<out>.csv`` and
returns the PNG path. Inputs are the tables written by the pipeline stages; nothing here computes a metric.

Conventions: one message per figure; shared axes / colour scales across compared panels; DTI axes start at 0;
offline values are labelled "offline (proxy)" and never "leaderboard".

UNTESTED: written without execution.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OFFLINE = "offline region DTI (proxy, not leaderboard)"


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save(fig, table: pd.DataFrame | None, out: str | Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=130)
    _plt().close(fig)
    if table is not None:
        table.to_csv(out.with_suffix(".csv"), index=False)
    return out.with_suffix(".png")


def f01_training_curves(history: pd.DataFrame, out):
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for unit, h in history.groupby("unit"):
        ax[0].plot(h.epoch, h.train_total, label=f"{unit} train")
        if "inner_loss" in h:
            ax[0].plot(h.epoch, h.inner_loss, "--", label=f"{unit} inner")
        for c, ls in (("enriched_surrogate", ":"), ("inner_full_region_dti", "-")):
            if c in h:
                ax[1].plot(h.epoch, h[c], ls, label=f"{unit} {c}")
    ax[0].set(xlabel="epoch", ylabel="loss", title="Training / inner loss")
    ax[1].set(xlabel="epoch", ylabel="score", ylim=(0, 1), title="Inner metrics: patch diagnostic vs region DTI")
    for a in ax:
        a.legend(fontsize=7)
    return _save(fig, history, out)


def f02_surrogate_vs_region(df: pd.DataFrame, out):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(df.enriched_surrogate, df.inner_full_region_dti)
    for _, r in df.iterrows():
        ax.annotate(str(r.get("label", "")), (r.enriched_surrogate, r.inner_full_region_dti), fontsize=7)
    ax.plot([0, 1], [0, 1], "k:", lw=0.8)
    ax.set(xlabel="enriched patch surrogate (G14 'inner')", ylabel="inner region DTI", xlim=(0, 1), ylim=(0, 1),
           title="Patch surrogate vs region DTI")
    return _save(fig, df, out)


def f03_rank_correlation(rc: pd.DataFrame, out):
    plt = _plt()
    piv = rc.pivot(index="inner", columns="external", values="spearman")
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(piv.values, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(piv.columns)), piv.columns, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(piv.index)), piv.index, fontsize=8)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Spearman")
    ax.set_title("Do inner metrics rank models like held-out DTI?")
    return _save(fig, rc, out)


def f04_probability_histograms(hist: pd.DataFrame, out):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 4))
    for g, h in hist.groupby("group"):
        x = np.log10(h.hi.clip(1e-7))
        ax.step(x, h.share, where="post", label=g)
    ax.set(xlabel="log10 probability (bin upper edge)", ylabel="share of pixels", yscale="log",
           title="Raw probability by position relative to known faults")
    ax.legend()
    return _save(fig, hist, out)


def f05_area_vs_dti(sweep: pd.DataFrame, out, reps=("top", "crest_w1", "sparse_soft")):
    plt = _plt()
    t = sweep[sweep.rep.isin(reps) & (sweep.target_frac > 0)]
    g = t.groupby(["task", "rep", "target_frac"])[["tp_w", "fp_w", "fn_w"]].sum().reset_index()
    g["dti"] = g.tp_w / (g.tp_w + 0.2 * g.fp_w + 0.8 * g.fn_w + 1e-9)
    tasks = list(g.task.unique())
    fig, axes = plt.subplots(1, max(len(tasks), 1), figsize=(6 * max(len(tasks), 1), 4), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        for rep, h in g[g.task == task].groupby("rep"):
            ax.plot(h.target_frac, h.dti, marker="o", label=rep)
        ax.set(xscale="log", xlabel="requested area fraction", ylabel=f"pooled {OFFLINE}", title=task, ylim=(0, None))
        ax.legend()
    return _save(fig, g, out)


def f06_tp_fp_fn(sweep: pd.DataFrame, out, rep="top"):
    plt = _plt()
    g = sweep[(sweep.rep == rep) & (sweep.target_frac > 0)].groupby(["task", "target_frac"])[
        ["tp_w", "fp_w", "fn_w"]].sum().reset_index()
    g["alpha_fp"], g["beta_fn"] = 0.2 * g.fp_w, 0.8 * g.fn_w
    fig, ax = plt.subplots(figsize=(7, 4))
    for task, h in g.groupby("task"):
        ax.plot(h.target_frac, h.tp_w, label=f"{task} TP")
        ax.plot(h.target_frac, h.alpha_fp, "--", label=f"{task} 0.2 FP")
        ax.plot(h.target_frac, h.beta_fn, ":", label=f"{task} 0.8 FN")
    ax.set(xscale="log", yscale="log", xlabel="area fraction", ylabel="weighted components",
           title=f"Denominator terms ({rep})")
    ax.legend(fontsize=7)
    return _save(fig, g, out)


def f07_representations(best: pd.DataFrame, out):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 4))
    lab = best.rep + "|" + best.order
    for i, (task, h) in enumerate(best.groupby("task")):
        ax.barh([f"{task}:{x}" for x in lab[h.index]], h.dti_crossfit, label=task)
    ax.set(xlabel=f"cross-fitted {OFFLINE}", title="Representation x masking order", xlim=(0, None))
    return _save(fig, best, out)


def f08_context_dependence(curves: pd.DataFrame, out):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4))
    for (iv, reg), h in curves.groupby(["intervention", "region"]):
        ax.plot(h.dist_lo_km + 0.01, h.mean_p, marker=".", label=f"{iv} / {reg}")
    ax.set(xscale="log", xlabel="distance to context faults (km)", ylabel="mean probability",
           title="Known-context dependence under interventions")
    ax.legend(fontsize=7)
    return _save(fig, curves, out)


def f09_perf_vs_known_distance(df: pd.DataFrame, out):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4))
    for m, h in df.groupby("model"):
        ax.plot(h.dist_bin_km, h.recall_w, marker="o", label=m)
    ax.set(xlabel="distance of held-out truth to nearest permitted known fault (km)", ylabel="weighted recall",
           ylim=(0, 1), title="Discovery recall vs distance to known faults")
    ax.legend()
    return _save(fig, df, out)


def _bar(df, x, y, out, title, xlabel=OFFLINE, err=None):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * len(df))))
    ax.barh(df[x].astype(str), df[y], xerr=None if err is None else df[err])
    ax.set(xlabel=xlabel, title=title, xlim=(0, None))
    return _save(fig, df, out)


def f10_feature_ablation(df, out):
    return _bar(df.sort_values("disc_dti"), "spec", "disc_dti", out, "Feature-family ablations (discovery)")


def f11_regime_performance(df, out):
    return _bar(df, "regime", "dti", out, "Performance by geological regime")


def f12_length_recall(df, out):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4))
    for m, h in df.groupby("model"):
        ax.plot(h.length_bin, h.recall_w, marker="o", label=m)
    ax.set(xlabel="fault-system length bin", ylabel="weighted recall", ylim=(0, 1), title="Length-stratified recall")
    ax.legend()
    return _save(fig, df, out)


def _density(mask: np.ndarray, block: int = 50) -> np.ndarray:
    H, W = mask.shape
    hh, ww = H // block, W // block
    return mask[:hh * block, :ww * block].reshape(hh, block, ww, block).mean((1, 3))


def f13_fp_density(fp_mask: np.ndarray, out, extent=None):
    plt = _plt()
    d = _density(fp_mask.astype(np.float32))
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(d, cmap="magma", extent=extent)
    fig.colorbar(im, ax=ax, label="FP pixel share per 5 km block")
    ax.set_title("Spatial false-positive density")
    return _save(fig, pd.DataFrame({"block_row": np.repeat(np.arange(d.shape[0]), d.shape[1]),
                                    "block_col": np.tile(np.arange(d.shape[1]), d.shape[0]), "fp_share": d.ravel()}),
                 out)


def f14_fn_density(fn_mask: np.ndarray, out, extent=None):
    plt = _plt()
    d = _density(fn_mask.astype(np.float32))
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(d, cmap="viridis", extent=extent)
    fig.colorbar(im, ax=ax, label="missed-truth share per 5 km block")
    ax.set_title("Spatial false-negative density")
    return _save(fig, pd.DataFrame({"fn_share": d.ravel()}), out)


def f15_calibration_curves(df: pd.DataFrame, out):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5, 5))
    for k, h in df.groupby("calibration"):
        ax.plot(h.mean_p, h.observed, marker="o", label=k)
    ax.plot([0, 1], [0, 1], "k:", lw=0.8)
    ax.set(xlabel="mean predicted p (bin)", ylabel="observed truth share (within 300 m)", title="Reliability")
    ax.legend(fontsize=7)
    return _save(fig, df, out)


def f16_champion_vs_g14(sample: pd.DataFrame, out):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.hexbin(sample.champion_rank, sample.g14_rank, gridsize=50, bins="log", cmap="Greys")
    if "truth" in sample:
        t = sample[sample.truth]
        ax.scatter(t.champion_rank, t.g14_rank, s=2, c="crimson", label="held-out truth")
        ax.legend()
    ax.set(xlabel="champion rank", ylabel="G14 rank", title="Champion vs G14 (per-zone ranks)")
    return _save(fig, sample, out)


def f17_unique_recovery(pc: pd.DataFrame, out):
    cols = ["both_tp", "champ_tp_g14_miss", "champ_miss_g14_tp", "both_miss"]
    plt = _plt()
    g = pc.groupby("task")[cols].sum()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    left = np.zeros(len(g))
    for c in cols:
        ax.barh(g.index, g[c], left=left, label=c)
        left += g[c].to_numpy()
    ax.set(xlabel="held-out truth pixels", title="Who recovers what")
    ax.legend(fontsize=7)
    return _save(fig, pc, out)


def f18_ensemble_weight(bt: pd.DataFrame, out):
    plt = _plt()
    g = bt.groupby(["task", "kind", "w"])[["tp_w", "fp_w", "fn_w"]].sum().reset_index()
    g["dti"] = g.tp_w / (g.tp_w + 0.2 * g.fp_w + 0.8 * g.fn_w + 1e-9)
    fig, ax = plt.subplots(figsize=(7, 4))
    for (task, kind), h in g.groupby(["task", "kind"]):
        ax.plot(h.w, h.dti, marker="o", label=f"{task}:{kind}")
    ax.set(xlabel="G14 weight", ylabel=f"pooled {OFFLINE} (in-sample pooled, see cross-fitted table)", ylim=(0, None))
    ax.legend(fontsize=7)
    return _save(fig, g, out)


def f19_fragmentation(df, out):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df.fragmentation_per_100km.dropna(), bins=30)
    ax.set(xlabel="components per 100 km of skeleton", ylabel="units", title="Fragmentation")
    return _save(fig, df, out)


def f20_orientation(angles: dict[str, np.ndarray], out):
    plt = _plt()
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(projection="polar")
    rows = []
    for k, a in angles.items():
        h, e = np.histogram(np.mod(a, np.pi), bins=36, range=(0, np.pi))
        h = h / max(h.sum(), 1)
        ax.plot(np.r_[e[:-1], e[:-1] + np.pi], np.r_[h, h], label=k)
        rows += [{"set": k, "bin_lo_rad": lo, "share": s} for lo, s in zip(e[:-1], h)]
    ax.legend(fontsize=7)
    ax.set_title("Strike distribution (axial)")
    return _save(fig, pd.DataFrame(rows), out)


def f22_alignment(expected: np.ndarray, actual: np.ndarray, out, window=(slice(None), slice(None))):
    plt = _plt()
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    e, a = np.nan_to_num(expected[window]), np.nan_to_num(actual[window])
    for x, t, im in ((ax[0], "expected", e), (ax[1], "exported", a), (ax[2], "difference", a - e)):
        h = x.imshow(im, cmap="RdBu_r" if t == "difference" else "viridis")
        x.set_title(t)
        fig.colorbar(h, ax=x, fraction=0.046)
    return _save(fig, None, out)


def f23_fold_distribution(df, out):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4))
    models = list(df.model.unique())
    ax.boxplot([df[df.model == m].dti for m in models], labels=models, vert=False)
    ax.set(xlabel=f"per-zone {OFFLINE}", title="Discovery fold score distribution")
    return _save(fig, df, out)


def f24_architecture(df, out):
    return _bar(df.sort_values("disc_dti"), "spec", "disc_dti", out,
                "Local vs regional vs global (identical features, supervision, folds, masks)")


FIGURES = {1: f01_training_curves, 2: f02_surrogate_vs_region, 3: f03_rank_correlation, 4: f04_probability_histograms,
           5: f05_area_vs_dti, 6: f06_tp_fp_fn, 7: f07_representations, 8: f08_context_dependence,
           9: f09_perf_vs_known_distance, 10: f10_feature_ablation, 11: f11_regime_performance, 12: f12_length_recall,
           13: f13_fp_density, 14: f14_fn_density, 15: f15_calibration_curves, 16: f16_champion_vs_g14,
           17: f17_unique_recovery, 18: f18_ensemble_weight, 19: f19_fragmentation, 20: f20_orientation,
           22: f22_alignment, 23: f23_fold_distribution, 24: f24_architecture}
# 21 (geological failure montages) lives in ``atlas``.
