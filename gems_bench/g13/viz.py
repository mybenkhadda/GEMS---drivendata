"""Scientific figures (section 38). Every function saves through ``FigureIndex`` with a one-line takeaway that is
computed from the plotted numbers, so the takeaway cannot drift from the figure."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .report import extent_km, map_axes


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def robust_lim(a, valid):
    v = a[valid & np.isfinite(a)]
    return (np.percentile(v, 2), np.percentile(v, 98)) if v.size else (0, 1)


# =============================================================================== inventory
def coverage_maps(fx, data, inv: pd.DataFrame):
    plt = _plt()
    fig, ax = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    map_axes(
        ax[0],
        data,
        np.where(data.template_valid, data.valid.astype(float), np.nan),
        "Study area: valid footprint",
        "Greys",
        0,
        1,
    )
    step = max(1, max(data.shape) // 1200)
    lab = np.ma.masked_where(~data.labels[::step, ::step], data.labels[::step, ::step])
    ax[0].imshow(lab, extent=extent_km(data), cmap="autumn", interpolation="nearest")
    miss = np.zeros(data.shape, np.int16)
    for i in range(len(data.names)):
        miss += ~np.isfinite(np.asarray(data.raw[i]))
    map_axes(
        ax[1], data, np.where(data.template_valid, miss, np.nan), "Missing bands per pixel", "magma", 0, len(data.names)
    )
    inv = inv.sort_values("finite_frac_in_template")
    ax[2].barh(inv.name, inv.finite_frac_in_template, color="#4a7")
    ax[2].set_xlabel("finite fraction inside template")
    ax[2].tick_params(labelsize=7)
    worst = inv.iloc[0]
    return fx.save(
        fig,
        "inventory_coverage",
        "03",
        f"Lowest band coverage: {worst['name']} ({worst.finite_frac_in_template:.1%}); "
        f"{int((miss[data.template_valid] > 0).sum())} template pixels miss >= 1 band",
    )


def coverage_matrix(fx, cov: pd.DataFrame):
    plt = _plt()
    piv = cov.groupby(["band", "block_r"]).finite_frac.mean().unstack()
    fig, ax = plt.subplots(figsize=(8, 7), layout="constrained")
    im = ax.imshow(piv.to_numpy(), aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_yticks(range(len(piv)), piv.index, fontsize=7)
    ax.set_xlabel("north -> south regional row block")
    fig.colorbar(im, ax=ax, label="finite fraction")
    low = (piv < 0.9).sum().sum()
    return fx.save(fig, "inventory_coverage_matrix", "03", f"{low} band x region cells have < 90 % coverage")


def raw_atlas(fx, data):
    plt = _plt()
    n = len(data.names)
    cols = 5
    rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.0 * rows), layout="constrained")
    for ax, (i, nm) in zip(axs.flat, enumerate(data.names)):
        a = np.asarray(data.raw[i])
        lo, hi = robust_lim(a, data.valid)
        map_axes(ax, data, a, nm, "RdBu_r" if a[np.isfinite(a)].min() < 0 else "viridis", lo, hi)
    for ax in list(axs.flat)[n:]:
        ax.set_axis_off()
    return fx.save(fig, "raw_multiband_atlas", "05", f"{n} raw GeoDAWN bands at 2-98 % robust colour limits")


def valid_by_modality(fx, inv: pd.DataFrame):
    plt = _plt()
    g = inv.groupby("modality").finite_frac_in_template.agg(["mean", "min"]).sort_values("mean")
    fig, ax = plt.subplots(figsize=(6, 3.5), layout="constrained")
    ax.barh(g.index, g["mean"], label="mean band coverage")
    ax.barh(g.index, g["min"], alpha=0.6, label="worst band")
    ax.legend(fontsize=7)
    ax.set_xlabel("finite fraction inside template")
    return fx.save(fig, "valid_by_modality", "03", f"Worst-covered modality: {g.index[0]} (mean {g['mean'].iloc[0]:.1%})")


# =============================================================================== splits
def partitions(fx, data, sp):
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.5), layout="constrained")
    fm = np.where(data.valid, sp.fold_map.astype(float), np.nan)
    fm[sp.test()] = -2
    map_axes(ax[0], data, fm, f"Geographic folds (TEST = fold {sp.test_fold}, shown dark)", "tab10")
    wh = np.full(data.shape, np.nan)
    for d in range(sp.n_splits):
        wh[sp.withheld(d)] = d
    base = np.where(data.valid, 0.0, np.nan)
    map_axes(ax[1], data, base, "Discovery splits: withheld fault systems", "Greys", 0, 3)
    step = max(1, max(data.shape) // 1200)
    ax[1].imshow(np.ma.masked_invalid(wh[::step, ::step]), extent=extent_km(data), cmap="Set1", interpolation="nearest")
    return fx.save(
        fig,
        "validation_design",
        "18",
        f"{len(sp.cv_folds)} CV folds + 1 untouched TEST fold; "
        f"{sp.n_splits} discovery splits withhold {int(sum(sp.withheld(d).sum() for d in range(sp.n_splits)))} fault px",
    )


# =============================================================================== features / EDA
def feature_atlas(fx, data, store, names: list[str]):
    plt = _plt()
    names = [n for n in names if store.has(n)][:8]
    if not names:
        return None
    fig, axs = plt.subplots(2, 4, figsize=(16, 7.5), layout="constrained")
    for ax, n in zip(axs.flat, names):
        a = store.get(n)
        lo, hi = robust_lim(a, data.valid)
        map_axes(ax, data, a, n, "magma", lo, hi)
    for ax in list(axs.flat)[len(names) :]:
        ax.set_axis_off()
    return fx.save(fig, "feature_atlas", "12-17", f"{len(names)} representative physics features")


def scale_space(fx, ss: pd.DataFrame):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    for b, g in ss.groupby("band"):
        ax.plot(g.scale_m / 1000, g.gradient_auc, marker="o", label=b)
    ax.set_xscale("log")
    ax.axhline(0.5, color="k", lw=0.5)
    ax.set_xlabel("smoothing scale (km)")
    ax.set_ylabel("gradient AUC (<= 300 m vs >= 2 km)")
    ax.legend(fontsize=7)
    best = ss.loc[ss.gradient_auc.idxmax()]
    return fx.save(
        fig,
        "scale_space_auc",
        "05",
        f"Strongest fault signal: {best.band} at {best.scale_m / 1000:g} km (AUC {best.gradient_auc:.3f})",
    )


def variograms(fx, vg: dict):
    plt = _plt()
    fig, axs = plt.subplots(1, len(vg), figsize=(4 * len(vg), 3.5), layout="constrained", squeeze=False)
    for ax, (b, df) in zip(axs[0], vg.items()):
        for d, g in df.groupby("direction"):
            ax.plot(g.lag_km, g.gamma_over_var, marker=".", label=d)
        ax.set_xscale("log")
        ax.set_title(b, fontsize=9)
        ax.set_xlabel("lag (km)")
    axs[0][0].set_ylabel("gamma / variance")
    axs[0][0].legend(fontsize=7)
    return fx.save(fig, "directional_variograms", "05", "Directional variograms: anisotropy shows as diverging curves")


def spectra(fx, sp_tabs: dict):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    for b, df in sp_tabs.items():
        ax.plot(df.wavelength_km, df.log_power - df.log_power.max(), label=b)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("wavelength (km)")
    ax.set_ylabel("log10 power (normalised)")
    ax.legend(fontsize=7)
    return fx.save(fig, "radial_power_spectra", "05", "Radially averaged power spectra of the primary fields")


def contrasts(fx, ct: pd.DataFrame, summary: pd.DataFrame, top: int = 15):
    plt = _plt()
    if ct.empty:
        return None
    feats = summary.head(top).feature.tolist()
    piv = ct[ct.feature.isin(feats)].pivot_table(index="feature", columns="control", values="cliffs_delta").reindex(feats)
    fig, ax = plt.subplots(figsize=(10, 0.35 * len(feats) + 2), layout="constrained")
    im = ax.imshow(piv.to_numpy(), cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto")
    ax.set_yticks(range(len(feats)), [f[:48] for f in feats], fontsize=7)
    ax.set_xticks(range(piv.shape[1]), piv.columns, rotation=40, ha="right", fontsize=7)
    fig.colorbar(im, ax=ax, label="Cliff's delta (fault corridor vs control)")
    n_disc = int((summary.verdict == "discriminative").sum())
    return fx.save(
        fig,
        "fault_conditioned_contrasts",
        "06",
        f"{n_disc} features stay discriminative against every "
        f"control; {int((summary.verdict == 'confounded').sum())} only separate faults from random background",
    )


def profiles(fx, prof: dict, families: np.ndarray):
    plt = _plt()
    P, x, bands = prof["profiles"], prof["offsets_km"], prof["bands"]
    if not len(P):
        return None
    fams = sorted(set(families))
    fig, axs = plt.subplots(
        len(fams), len(bands), figsize=(3 * len(bands), 2.4 * len(fams)), layout="constrained", squeeze=False
    )
    for i, f in enumerate(fams):
        m = families == f
        for j, b in enumerate(bands):
            ax = axs[i][j]
            med = np.nanmedian(P[m, j], 0)
            lo, hi = np.nanpercentile(P[m, j], [25, 75], 0)
            ax.fill_between(x, lo, hi, alpha=0.3)
            ax.plot(x, med)
            ax.axvline(0, color="k", lw=0.5)
            if i == 0:
                ax.set_title(b, fontsize=8)
            if j == 0:
                ax.set_ylabel(f"family {f}\n(n={int(m.sum())})", fontsize=8)
    for ax in axs[-1]:
        ax.set_xlabel("distance across fault (km)")
    return fx.save(
        fig,
        "cross_fault_profiles",
        "07",
        f"{len(P)} stations in {len(fams)} response families (median and IQR envelopes, strike-normal transects)",
    )


def roses(fx, rose: pd.DataFrame):
    plt = _plt()
    srcs = rose.source.unique()
    fig, axs = plt.subplots(
        1, len(srcs), subplot_kw={"projection": "polar"}, figsize=(3.2 * len(srcs), 3.4), layout="constrained", squeeze=False
    )
    for ax, s in zip(axs[0], srcs):
        for z, g in rose[rose.source == s].groupby("zone"):
            th = np.radians(g.bin_lo.to_numpy() + 5)
            r = g.share.to_numpy()
            ax.bar(np.r_[th, th + np.pi], np.r_[r, r], width=np.radians(10), alpha=0.5, label=z)
        ax.set_theta_zero_location("E")
        ax.set_title(s, fontsize=8)
        ax.set_yticklabels([])
    axs[0][0].legend(fontsize=6, loc="lower left")
    t = rose[rose.source == "mapped_faults"]
    peak = t.loc[t.share.idxmax(), "bin_lo"] if len(t) else np.nan
    return fx.save(
        fig,
        "orientation_roses",
        "08",
        f"Mapped-fault strike peak near {peak:.0f} deg "
        "(counter-clockwise from east); coherence-weighted field strikes near faults vs background",
    )


def regime_map(fx, data, regimes: np.ndarray, sel: pd.DataFrame, k: int):
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    map_axes(ax[0], data, np.where(regimes >= 0, regimes, np.nan), f"Geological regimes (k = {k})", "tab10")
    ax[1].plot(sel.k, sel.kmeans_ari_mean, marker="o", label="seed stability (ARI)")
    ax[1].plot(sel.k, sel.kmeans_silhouette, marker="s", label="silhouette")
    ax2 = ax[1].twinx()
    ax2.plot(sel.k, sel.gmm_bic, color="gray", ls="--", label="GMM BIC")
    ax[1].axvline(k, color="k", lw=0.6)
    ax[1].legend(fontsize=7)
    ax[1].set_xlabel("k")
    return fx.save(fig, "geological_regimes", "15", f"k = {k} chosen by the fixed stability-then-silhouette rule")


def ablation_bars(fx, board: pd.DataFrame, metric: str):
    plt = _plt()
    b = board.dropna(subset=[metric]).sort_values(metric)
    if b.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(b) + 1.5), layout="constrained")
    err = None
    if f"{metric}_ci_lo" in b:
        err = np.vstack([b[metric] - b[f"{metric}_ci_lo"], b[f"{metric}_ci_hi"] - b[metric]])
        err = np.where(np.isfinite(err), np.maximum(err, 0), 0)
    ax.barh(b.model, b[metric], xerr=err, color="#58a")
    ax.set_xlabel(f"{metric} (offline proxy, cross-fitted area budget)")
    ax.tick_params(labelsize=7)
    best = b.iloc[-1]
    return fx.save(fig, f"scoreboard_{metric}", "21-22", f"Best {metric}: {best.model} ({best[metric]:.4f})")


def dti_sensitivity(fx, syn: pd.DataFrame):
    plt = _plt()
    exps = syn.experiment.unique()
    fig, axs = plt.subplots(2, int(np.ceil(len(exps) / 2)), figsize=(14, 6), layout="constrained")
    for ax, e in zip(axs.flat, exps):
        g = syn[syn.experiment == e]
        ax.plot(g.x, g.dti, marker="o")
        ax.set_title(e, fontsize=9)
        ax.set_ylim(0, 1.02)
    for ax in list(axs.flat)[len(exps) :]:
        ax.set_axis_off()
    w = syn[syn.experiment == "width_px"]
    return fx.save(
        fig,
        "dti_synthetic_sensitivity",
        "29",
        f"Synthetic fault: DTI {w.dti.iloc[0]:.2f} at 1 px width, {w.dti.iloc[-1]:.2f} at {int(w.x.iloc[-1])} px half-width",
    )


def mass_curve(fx, mc: pd.DataFrame):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    for t, g in mc.groupby("task"):
        ax.plot(g.target_frac * 100, g.dti, marker=".", label=t)
    ax.set_xscale("log")
    ax.set_xlabel("predicted area (% of evaluation zone)")
    ax.set_ylabel("pooled DTI (offline proxy)")
    ax.legend(fontsize=7)
    best = mc.loc[mc.dti.idxmax()]
    return fx.save(fig, "dti_vs_area", "29", f"Best area {best.target_frac:.2%} ({best.task}); DTI falls off on both sides")


def candidate_atlas(fx, data, seg_raster: np.ndarray, cands: pd.DataFrame):
    plt = _plt()
    if cands.empty:
        return None
    cls = {c: i for i, c in enumerate(sorted(cands["class"].unique()))}
    lut = np.full(int(seg_raster.max()) + 2, np.nan)
    for sid, c in zip(cands.segment_id, cands["class"]):
        lut[int(sid)] = cls[c]
    arr = np.where(seg_raster >= 0, lut[np.maximum(seg_raster, 0)], np.nan)
    arr = __import__("scipy").ndimage.maximum_filter(np.nan_to_num(arr, nan=-1), 3)
    arr = np.where(arr >= 0, arr, np.nan)
    fig, ax = plt.subplots(figsize=(9, 8), layout="constrained")
    map_axes(
        ax,
        data,
        np.where(data.valid, 0.0, np.nan),
        "Candidate lineaments by relation to mapped faults (predictions, not confirmed faults)",
        "Greys",
        0,
        4,
    )
    step = max(1, max(data.shape) // 1200)
    lab = np.ma.masked_where(~data.labels[::step, ::step], np.ones_like(data.labels[::step, ::step], float))
    ax.imshow(lab, extent=extent_km(data), cmap="gray_r", alpha=0.5, interpolation="nearest")
    im = ax.imshow(
        np.ma.masked_invalid(arr[::step, ::step]),
        extent=extent_km(data),
        cmap="tab10",
        vmin=0,
        vmax=9,
        interpolation="nearest",
    )
    cb = fig.colorbar(im, ax=ax, shrink=0.6, ticks=list(cls.values()))
    cb.ax.set_yticklabels(list(cls.keys()), fontsize=7)
    km = cands.groupby("class").length_km.sum()
    return fx.save(
        fig, "discovery_atlas", "25", "Candidate km by class: " + ", ".join(f"{k} {v:.0f}" for k, v in km.items())
    )


def error_panels(fx, data, windows: list[dict], score: np.ndarray, size: int = 80):
    plt = _plt()
    if not windows:
        return None
    w = windows[:6]
    fig, axs = plt.subplots(1, len(w), figsize=(3 * len(w), 3.2), layout="constrained", squeeze=False)
    for ax, win in zip(axs[0], w):
        r0, c0 = max(win["row"] - size // 2, 0), max(win["col"] - size // 2, 0)
        sl = (slice(r0, r0 + size), slice(c0, c0 + size))
        ax.imshow(np.nan_to_num(score[sl]), cmap="magma", vmin=0, vmax=np.nanpercentile(score, 99.5))
        rr, cc = np.nonzero(data.labels[sl])
        ax.plot(cc, rr, ",", color="cyan")
        ax.set_title(f"{win['kind']} ({win['task']}:{win['zone']})", fontsize=8)
        ax.set_axis_off()
    return fx.save(
        fig,
        "failure_mode_panels",
        "17",
        "Largest missed-truth and false-positive windows (cyan = mapped faults, colour = score)",
    )
