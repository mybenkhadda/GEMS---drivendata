"""Scientific figures (PNG + SVG, deterministic names) for every notebook ``plot_group``.

All maps share the analysis grid: EPSG:32611 extent in km, equal aspect, 20 km scale bar, colourbar with
units, NaN / nodata in light grey. Map previews are 4x4 block maxima (thin structures survive; stated in
every caption). A figure whose evidence is unavailable is skipped with a reason - never drawn empty.
Every caption names the metric as an OFFLINE DTI PROXY where DTI appears.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OKABE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
NODATA = "#e3e3e3"
PROXY = "offline DTI proxy (not the leaderboard DTI)"


def plt():
    import matplotlib.pyplot as p

    p.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight", "font.size": 9, "axes.titlesize": 10,
                       "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    return p


class Writer:
    def __init__(self, run, group: str):
        self.run, self.group = run, group
        self.dir = run.paths.fig_dir(group)
        self.saved: list[Path] = []
        self.skipped: list[str] = []
        self.inline = bool(run.cfg["figures"]["inline"])

    def save(self, fig, name: str, caption: str, key: bool = True):
        P = plt()
        if caption:
            fig.text(0.0, -0.02, caption, ha="left", va="top", fontsize=7.5, wrap=True, color="#333333",
                     transform=fig.transFigure)
        dpi = int(self.run.cfg["figures"]["dpi"])
        for ext, on in (("png", self.run.cfg["save_png"]), ("svg", self.run.cfg["save_svg"])):
            if on:
                p = self.dir / f"{name}.{ext}"
                self.run.paths.check_writable(p)
                tmp = p.with_name(p.stem + ".partial." + ext)
                fig.savefig(tmp, dpi=dpi, format=ext)
                tmp.replace(p)
                self.saved.append(p)
        if self.inline and key:
            P.show()
        P.close(fig)

    def skip(self, name: str, reason: str):
        self.skipped.append(f"{name}: {reason}")
        print(f"[figure skipped] {name}: {reason}")


# ---------------------------------------------------------------------------- map helpers
def preview(a: np.ndarray, f: int, how: str = "max") -> np.ndarray:
    a = np.asarray(a, np.float32)
    if f <= 1:
        return a
    h, w = a.shape
    H, W = -(-h // f) * f, -(-w // f) * f
    b = np.full((H, W), np.nan, np.float32)
    b[:h, :w] = a
    b = b.reshape(H // f, f, W // f, f)
    with np.errstate(all="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmax(b, axis=(1, 3)) if how == "max" else np.nanmean(b, axis=(1, 3))


def extent_km(transform, shape):
    h, w = shape
    x0, y0 = transform.c, transform.f
    return (x0 / 1e3, (x0 + transform.a * w) / 1e3, (y0 + transform.e * h) / 1e3, y0 / 1e3)


def scale_bar(ax, km=20):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    xs, ys = x0 + 0.05 * (x1 - x0), y0 + 0.05 * (y1 - y0)
    ax.plot([xs, xs + km], [ys, ys], color="black", lw=3, solid_capstyle="butt")
    ax.text(xs + km / 2, ys + 0.015 * (y1 - y0), f"{km:g} km", ha="center", va="bottom", fontsize=7)


def map_panel(ax, run, arr, title, label, cmap="magma", vmin=None, vmax=None, norm=None, how="max", f=None,
              bar=True, window=None):
    import matplotlib as mpl

    tr, shape = run.bank.transform, arr.shape
    f = f or int(run.cfg["figures"]["preview_factor"])
    if window is not None:                                   # (r0, r1, c0, c1) full-resolution zoom
        r0, r1, c0, c1 = window
        sub = arr[r0:r1, c0:c1]
        ext = (( tr.c + tr.a * c0) / 1e3, (tr.c + tr.a * c1) / 1e3, (tr.f + tr.e * r1) / 1e3, (tr.f + tr.e * r0) / 1e3)
        img = sub
    else:
        ext = extent_km(tr, shape)
        img = preview(arr, f, how)
    cm = mpl.colormaps[cmap].copy() if isinstance(cmap, str) else cmap
    cm.set_bad(NODATA)
    im = ax.imshow(np.ma.masked_invalid(img), extent=ext, origin="upper", cmap=cm, vmin=None if norm else vmin,
                   vmax=None if norm else vmax, norm=norm, interpolation="nearest")
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("Easting (km, EPSG:32611)")
    ax.set_ylabel("Northing (km)")
    if label is not None:
        cb = ax.figure.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
        cb.set_label(label)
    if bar:
        scale_bar(ax, 20 if window is None else 5)
    return im


def overlay(ax, run, mask, color, alpha=1.0, f=None, window=None, label=None):
    """Boolean overlay (block-max preview keeps thin traces)."""
    from matplotlib.colors import ListedColormap

    tr = run.bank.transform
    f = f or int(run.cfg["figures"]["preview_factor"])
    if window is not None:
        r0, r1, c0, c1 = window
        img = mask[r0:r1, c0:c1].astype(np.float32)
        ext = ((tr.c + tr.a * c0) / 1e3, (tr.c + tr.a * c1) / 1e3, (tr.f + tr.e * r1) / 1e3, (tr.f + tr.e * r0) / 1e3)
    else:
        img = preview(mask.astype(np.float32), f, "max")
        ext = extent_km(tr, mask.shape)
    img = np.where(img > 0, 1.0, np.nan)
    ax.imshow(img, extent=ext, origin="upper", cmap=ListedColormap([color]), alpha=alpha, interpolation="nearest",
              vmin=0, vmax=1)
    if label:
        ax.plot([], [], color=color, lw=2, label=label)


def window_around(run, row, col, half_km=None):
    half = int((half_km or run.cfg["figures"]["zoom_half_km"]) * 10)
    h, w = run.bank.shape
    r0, c0 = max(int(row) - half, 0), max(int(col) - half, 0)
    return r0, min(r0 + 2 * half, h), c0, min(c0 + 2 * half, w)


def blocks_to_full(grid: np.ndarray, block_px: int, shape) -> np.ndarray:
    """Expand a per-block grid to the analysis grid so maps keep the true geographic extent."""
    full = np.repeat(np.repeat(np.asarray(grid, np.float32), block_px, 0), block_px, 1)
    return full[: shape[0], : shape[1]]


def logn(vmax, vmin=1e-4):
    from matplotlib.colors import LogNorm

    return LogNorm(vmin=vmin, vmax=max(vmax, vmin * 10))


# ---------------------------------------------------------------------------- A: confidence distribution
def prediction_distribution(run, W: Writer, R):
    P = plt()
    p = run.v1_full_prob()
    s = R.stats
    fig, ax = P.subplots(1, 2, figsize=(13, 6.2))
    map_panel(ax[0], run, p, "V1 full-model confidence (fixed 0-1 scale)", "p (dimensionless)", vmin=0, vmax=1)
    map_panel(ax[1], run, p, "Same raster, log scale to q99.9", "p (log)", norm=logn(s["q999"]))
    overlay(ax[1], run, run.bank.labels, "#00c8ff", 0.8, label="mapped faults")
    ax[1].legend(loc="upper right", fontsize=7)
    W.save(fig, "01_confidence_map", "4x4 block-maximum preview of the original V1 full-model raster (submission "
           "source). Grey = outside the eligible footprint. Mapped faults overlaid only for orientation.")
    h, lz, ex = R.histogram, R.lorenz, R.exceedance
    fig, ax = P.subplots(1, 3, figsize=(14, 4.2))
    mid = np.sqrt(np.maximum(h.bin_lo, 1e-7) * h.bin_hi)
    ax[0].step(mid, np.maximum(h["count"], 0.5), where="mid", color=OKABE[0])
    ax[0].set(xscale="log", yscale="log", xlabel="confidence p", ylabel="pixels per log bin",
              title="Confidence histogram (eligible, outside mapped faults)")
    ax[0].axvline(s["median"], color=OKABE[1], ls="--", lw=1, label=f"median {s['median']:.2g}")
    ax[0].axvline(s["q99"], color=OKABE[3], ls="--", lw=1, label=f"q99 {s['q99']:.2g}")
    ax[0].legend(fontsize=7)
    ax[1].plot(ex.threshold, ex.eligible_fraction_above, color=OKABE[0])
    ax[1].set(xscale="log", yscale="log", xlabel="threshold t", ylabel="eligible fraction with p >= t",
              title="Exceedance curve")
    ax[2].plot(lz.top_pixel_fraction, lz.mass_fraction, color=OKABE[2], label="model")
    ax[2].plot([1e-5, 1], [1e-5, 1], color="grey", lw=0.8, ls=":", label="uniform")
    ax[2].set(xscale="log", xlabel="top fraction of pixels (by p)", ylabel="share of total mass",
              title=f"Concentration (Gini {s['gini']:.2f})")
    ax[2].legend(fontsize=7)
    W.save(fig, "02_confidence_distribution", f"Total mass outside mapped faults {s['total_mass_outside_mapped']:.0f} "
           f"over {s['eligible_pixels_outside_mapped']:,} pixels; top 1% of pixels carry "
           f"{100 * s['share_mass_top_1pct_pixels']:.1f}% of the mass.")


def known_fault_proximity(run, W: Writer, R):
    P = plt()
    t, cum, ends = R.table, R.cumulative, R.endpoints
    fig, ax = P.subplots(2, 2, figsize=(12, 8))
    x = np.arange(len(t))
    ax[0, 0].bar(x - 0.2, t.area_share, 0.4, color=OKABE[5], label="area share")
    ax[0, 0].bar(x + 0.2, t.mass_share, 0.4, color=OKABE[3], label="confidence-mass share")
    ax[0, 0].set(xticks=x, xticklabels=t.band, ylabel="share", title="Mass vs area by distance to mapped faults")
    ax[0, 0].legend(fontsize=7)
    ax[0, 1].bar(x, t.enrichment, color=OKABE[0])
    ax[0, 1].axhline(1, color="grey", ls=":", lw=1)
    for i, v in enumerate(t.enrichment):
        ax[0, 1].text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax[0, 1].set(xticks=x, xticklabels=t.band, ylabel="mass share / area share", title="Area-normalised enrichment")
    ax[1, 0].plot(cum.distance_m / 1e3, cum.cumulative_mass_share, color=OKABE[3], label="confidence mass")
    ax[1, 0].plot(cum.distance_m / 1e3, cum.cumulative_area_share, color=OKABE[5], label="area")
    ax[1, 0].set(xlabel="distance to nearest mapped fault d (km)", ylabel="cumulative share within d",
                 title="Cumulative mass vs distance")
    ax[1, 0].legend(fontsize=7)
    ax[1, 1].bar(range(len(t)), t.mean_confidence, color=OKABE[2], label="by band")
    ax[1, 1].set(xticks=range(len(t)), xticklabels=t.band, ylabel="mean confidence", title="Mean confidence by band")
    for i, r in enumerate(ends.itertuples()):
        ax[1, 1].axhline(r.mean_confidence, color=OKABE[1 + 3 * i], ls="--", lw=1,
                         label=("near trace endpoints" if i == 0 else "mid-trace, same distance") + f" {r.mean_confidence:.3g}")
    ax[1, 1].legend(fontsize=7)
    W.save(fig, "03_mapped_fault_proximity", "Distances to mapped faults are diagnostic only (never model inputs or "
           "postprocessing). Eligible pixels outside mapped faults; enrichment > 1 = more mass than area.")
    high = run.v1_full_prob() >= R.stats["high_conf_threshold"]
    high &= run.bank.valid & ~run.bank.labels
    fig, ax = P.subplots(1, 2, figsize=(13, 6.2))
    cls = np.where(high, np.digitize(run.dist_mapped(), [300, 1000, 3000]).astype(np.float32), np.nan)
    from matplotlib.colors import BoundaryNorm, ListedColormap

    cm = ListedColormap([OKABE[3], OKABE[1], OKABE[5], OKABE[0]])
    map_panel(ax[0], run, cls, f"Top {100 * (1 - run.cfg['diagnostics']['high_conf_quantile']):.1f}% pixels by "
              "distance class", "distance class", cmap=cm, norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], 4))
    overlay(ax[0], run, run.bank.labels, "black", 0.35)
    far = R.far_components
    if len(far):
        ax[0].scatter(far.centroid_e / 1e3, far.centroid_n / 1e3, s=18, facecolors="none", edgecolors="black",
                      lw=0.8, label=f"{len(far)} high-confidence components > 3 km from any mapped trace")
        ax[0].legend(fontsize=7, loc="upper right")
    g = run._mem.get("conc_grid")
    if g is not None:
        from matplotlib.colors import LogNorm

        B = int(run.cfg["diagnostics"]["region_block_px"])
        img = blocks_to_full(g, B, run.bank.shape)
        img[~run.bank.valid] = np.nan
        map_panel(ax[1], run, img, "Concentration near mapped faults per 10 km block",
                  "(mass share <= 1 km) / (area share <= 1 km)", cmap="RdBu_r", norm=LogNorm(0.25, 4), how="mean")
    else:
        ax[1].axis("off")
    W.save(fig, "04_high_confidence_and_concentration", "Classes: <=300 m, 300 m-1 km, 1-3 km, >3 km. Right: blocks "
           "with < 25% valid area or no pixel within 1 km of a mapped fault are blank.")


# ---------------------------------------------------------------------------- B: validation integrity
def split_integrity(run, W: Writer, R):
    P = plt()
    U = run.units
    geo = [U[n] for n in run.cfg["units"]["geographic"]]
    fig, ax = P.subplots(1, 1, figsize=(7, 6.5))
    z = np.full(run.bank.shape, np.nan, np.float32)
    for i, g in enumerate(geo):
        z[g.zone] = i
    from matplotlib.colors import BoundaryNorm, ListedColormap

    map_panel(ax, run, z, "Geographic CV: held-out zones", "zone (geo0, geo1, geo2)", cmap=ListedColormap(OKABE[:3]),
              norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3), how="mean")
    buf = np.zeros(run.bank.shape, bool)
    for g in geo:
        buf |= run.bank.valid & ~g.train & ~g.zone
    overlay(ax, run, buf, "white", 0.85, label="1.5 km training buffers")
    overlay(ax, run, run.bank.labels, "black", 0.5, label="mapped faults")
    ax.legend(fontsize=7, loc="upper right")
    W.save(fig, "05_geographic_folds", "Each geographic checkpoint was trained outside its zone and a 1.5 km buffer.")
    disc = [U[n] for n in run.cfg["units"]["discovery"]]
    fig, ax = P.subplots(1, len(disc), figsize=(5.2 * len(disc), 5.5))
    for a, d in zip(np.atleast_1d(ax), disc):
        bg = np.where(run.bank.valid, np.where(d.train, 0.15, 0.45), np.nan)
        map_panel(a, run, bg, f"{d.name}: held components and buffers", None, cmap="Greys", vmin=0, vmax=1, how="mean")
        overlay(a, run, run.bank.valid & ~d.train, OKABE[1], 0.5, label="excluded from training (800 m)")
        overlay(a, run, d.training_visible, "black", 0.6, label="training-visible mapped")
        overlay(a, run, d.truth, OKABE[3], 1.0, label="held (pseudo-discovery truth)")
        a.legend(fontsize=6.5, loc="upper right")
    W.save(fig, "06_discovery_holdouts", "Held groups are 8-connected raster components balanced by pixel count; "
           "they are not verified to be geologically independent fault systems.")
    fr = R.fragmentation
    if len(fr):
        fig, ax = P.subplots(1, 2, figsize=(12, 4.5))
        bins = np.r_[0, 500, 800, 1000, 1500, 2000, 3000, 5000, 10000, max(10001, fr.min_dist_to_train_visible_m.max() + 1)]
        for i, (u, g) in enumerate(fr.groupby("unit")):
            ax[0].hist(g.min_dist_to_train_visible_m, bins=bins, histtype="step", lw=1.5, color=OKABE[i], label=u)
        ax[0].set(xscale="symlog", xlabel="min distance from a held component to a training-visible trace (m)",
                  ylabel="held components", title="Adjacency of held and training-visible traces")
        ax[0].axvline(1000, color="grey", ls=":")
        ax[0].legend(fontsize=7)
        ax[1].scatter(fr.length_km, fr.min_dist_to_train_visible_m / 1e3, s=10,
                      c=[OKABE[int(u[-1]) % 8] for u in fr.unit])
        ax[1].set(xscale="log", xlabel="held component length (km)", ylabel="distance to visible trace (km)",
                  title="Held fragment length vs adjacency")
        W.save(fig, "07_split_leakage_audit", "Held components within ~1 km of a visible trace may be continuations of "
               "a structure the checkpoint saw in training (raster-connectivity split, not a geological one).")
    fig, ax = P.subplots(1, len(disc), figsize=(5.2 * len(disc), 5.5))
    for a, d in zip(np.atleast_1d(ax), disc):
        m = np.where(run.bank.valid, 1.0, np.nan)
        map_panel(a, run, m, f"{d.name}: narrow vs full-area scoring", None, cmap="Greys", vmin=0, vmax=3, how="mean")
        overlay(a, run, d.zone, OKABE[5], 0.6, label="V1 narrow zone (3 km around held)")
        overlay(a, run, d.training_visible, "black", 0.7, label="excluded (training-visible)")
        overlay(a, run, d.truth, OKABE[3], 1.0, label="held truth")
        a.legend(fontsize=6.5, loc="upper right")
    W.save(fig, "08_narrow_vs_full_masks", "Narrow V1 protocol counts false positives only inside the light-blue zone; "
           "the full-area protocol counts them over the whole grey eligible footprint.")


def original_dti(run, W: Writer, R):
    P = plt()
    c = R.components
    c = c[c.source == "recomputed"] if "source" in c else c
    if not len(c):
        return W.skip("09_original_components", "no V1 fold prediction could be rescored")
    fig, ax = P.subplots(1, 2, figsize=(13, 4.5))
    x = np.arange(len(c))
    a = run.cfg["metric"]["alpha"]
    b = run.cfg["metric"]["beta"]
    ax[0].bar(x, c.tp_w, color=OKABE[2], label="TP_w")
    ax[0].bar(x, a * c.fp_w, bottom=c.tp_w, color=OKABE[3], label=f"{a} FP_w")
    ax[0].bar(x, b * c.fn_w, bottom=c.tp_w + a * c.fp_w, color=OKABE[5], label=f"{b} FN_w")
    ax[0].set(xticks=x, xticklabels=c.unit, ylabel="weighted pixels", title="DTI denominator terms (original V1 protocol)")
    ax[0].legend(fontsize=7)
    ax[1].scatter(x, c.dti, color=OKABE[0], zorder=3, label="recomputed")
    if "dti_v1_table" in c:
        ax[1].scatter(x, c.dti_v1_table, marker="x", color=OKABE[1], zorder=4, label="V1 validation_components.csv")
    ax[1].set(xticks=x, xticklabels=c.unit, ylabel=PROXY, title="Original fold DTI: recomputed vs V1 table")
    ax[1].legend(fontsize=7)
    W.save(fig, "09_original_components", "Original protocols: GEOGRAPHIC (zone, no exclusion) and WITHHELD_SYSTEM "
           "(3 km neighbourhood, retained labels excluded pixel-exactly).")


def full_area_coverage(run, W: Writer, R):
    P = plt()
    units = [u for u in run.cfg["units"]["geographic"] + run.cfg["units"]["discovery"] if run.inference.cached(u)[0]]
    if not units:
        return W.skip("10_full_area_predictions", "no full-area prediction is complete")
    n = len(units)
    cols = 3
    rows = -(-n // cols)
    fig, ax = P.subplots(rows, cols, figsize=(5.2 * cols, 5 * rows), squeeze=False)
    vmax = max(float(np.nanquantile(run.full_area(u)[run.bank.valid], 0.999)) for u in units)
    for a, u in zip(ax.ravel(), units):
        map_panel(a, run, run.full_area(u), f"{u}: full-area re-inference", "p (log)", norm=logn(vmax))
        if run.units[u].protocol == "WITHHELD_SYSTEM":
            overlay(a, run, run.units[u].truth, "#00c8ff", 0.9)
        else:
            overlay(a, run, run.units[u].zone & ~np.roll(run.units[u].zone, 3, 1), "#00c8ff", 0.9)
    for a in ax.ravel()[n:]:
        a.axis("off")
    W.save(fig, "10_full_area_predictions", "Existing checkpoints, no retraining; one shared log colour scale. Cyan: "
           "held traces (discovery) or zone edge (geographic).")
    t = R.artifact_table
    t = t[t.reproduction_max_abs_diff.notna()] if "reproduction_max_abs_diff" in t else t
    if len(t):
        fig, ax = P.subplots(figsize=(7, 3.8))
        ax.bar(t.unit, t.reproduction_max_abs_diff.astype(float), color=OKABE[0])
        ax.axhline(run.cfg["inference"]["reproduction_tol"], color=OKABE[3], ls="--", lw=1, label="tolerance")
        ax.set(yscale="log", ylabel="max |V2 - V1| on V1-predicted pixels", title="Re-inference reproduces V1 inside V1 ROIs?")
        ax.legend(fontsize=7)
        W.save(fig, "11_reproduction_check", "Inside each V1 ROI the covering windows and blend weights are identical, "
               "so differences measure numerical nondeterminism (AMP / cuDNN) or a provenance problem.")


def full_area_dti(run, W: Writer, R):
    P = plt()
    c = R.components
    a, b = run.cfg["metric"]["alpha"], run.cfg["metric"]["beta"]
    d = c[c.protocol.isin(["WITHHELD_SYSTEM_NARROW_V1", "DISCOVERY_NARROW_REINFERRED", "DISCOVERY_FULL_AREA"])
          & (c.exclusion.isin(["v1", run.cfg["scoring"]["primary_exclusion"]]))]
    if not len(d):
        return W.skip("12_dti_breakdown", "no discovery components")
    fig, ax = P.subplots(1, 2, figsize=(14, 4.8))
    lab = d.unit + "\n" + d.protocol.str.replace("_", " ").str.lower()
    x = np.arange(len(d))
    tot = d.tp_w + a * d.fp_w + b * d.fn_w
    ax[0].bar(x, d.tp_w / tot, color=OKABE[2], label="TP_w")
    ax[0].bar(x, a * d.fp_w / tot, bottom=d.tp_w / tot, color=OKABE[3], label=f"{a} FP_w")
    ax[0].bar(x, b * d.fn_w / tot, bottom=(d.tp_w + a * d.fp_w) / tot, color=OKABE[5], label=f"{b} FN_w")
    ax[0].set(xticks=x, ylabel="share of DTI denominator", title="Narrow vs full-area: where the penalty comes from")
    ax[0].set_xticklabels(lab, fontsize=6, rotation=90)
    ax[0].legend(fontsize=7)
    pooled = R.pooled
    pr = pooled.protocol.unique()
    for i, p in enumerate(pr):
        sub = c[(c.protocol == p) & (c.exclusion == pooled[pooled.protocol == p].exclusion.iloc[0])]
        ax[1].scatter(np.full(len(sub), i) + np.linspace(-0.15, 0.15, len(sub)), sub.dti, color=OKABE[i % 8], s=25)
        pv = pooled[pooled.protocol == p].dti.iloc[0]
        ax[1].hlines(pv, i - 0.3, i + 0.3, color="black", lw=2)
    ax[1].set(xticks=range(len(pr)), ylabel=PROXY, title="Fold DTI (dots) and pooled DTI (bar = summed components)")
    ax[1].set_xticklabels([p.replace("_", "\n").lower() for p in pr], fontsize=7)
    W.save(fig, "12_dti_breakdown_and_folds", "Pooled values sum TP/FP/FN over units; they are not means of fold DTI. "
           "Full-area scoring counts unmapped real faults as false positives - a pseudo-discovery proxy, not an "
           "unbiased leaderboard estimate.")


# ---------------------------------------------------------------------------- C: spatial errors
def error_atlas(run, W: Writer, R):
    P = plt()
    disc = [u for u in run.cfg["units"]["discovery"] if run.error_maps(u) is not None]
    if not disc:
        return W.skip("13_tp_fp_fn_maps", "no spatial error maps")
    tp = np.zeros(run.bank.shape, np.float32)
    fp, fn, G = tp.copy(), tp.copy(), np.zeros(run.bank.shape, bool)
    for u in disc:
        m = run.error_maps(u)
        tp += np.nan_to_num(m["tp_support"])
        fp += np.nan_to_num(m["fp_mass"])
        fn += np.nan_to_num(m["fn_support"])
        G |= run.units[u].truth
    fig, ax = P.subplots(1, 3, figsize=(17, 5.8))
    map_panel(ax[0], run, np.where(G, tp, np.nan), "Weighted TP support M(g) at held truth", "M(g) (0-1)", cmap="viridis",
              vmin=0, vmax=1)
    map_panel(ax[1], run, np.where(run.bank.valid, fp, np.nan), "False-positive mass p(1-k)", "FP mass per pixel (log)",
              norm=logn(float(np.nanquantile(fp[fp > 0], 0.999)) if (fp > 0).any() else 1))
    map_panel(ax[2], run, np.where(G, fn, np.nan), "Missed support 1 - M(g) at held truth", "1 - M(g)", cmap="Reds",
              vmin=0, vmax=1)
    W.save(fig, "13_tp_fp_fn_maps", f"Full-area discovery protocol, summed over {', '.join(disc)} (each pixel scored "
           "by its own fold checkpoint); 300 m linear kernel; training-visible mapped pixels excluded exactly.")
    B = int(run.cfg["diagnostics"]["fp_density_block_px"])
    h, w = run.bank.shape
    H, Wd = -(-h // B), -(-w // B)
    pad = np.zeros((H * B, Wd * B), np.float32)
    pad[:h, :w] = fp
    dens = pad.reshape(H, B, Wd, B).sum((1, 3)) / (B * B * 0.01)            # per km^2
    fig, ax = P.subplots(1, 2, figsize=(13, 5.5))
    img = blocks_to_full(np.where(dens > 0, dens, np.nan), B, run.bank.shape)
    img[~run.bank.valid] = np.nan
    vmax_d = float(np.nanmax(dens)) if (dens > 0).any() else 1.0
    map_panel(ax[0], run, img, "FP-mass density (5 km blocks)", "FP mass per km^2 (log)",
              norm=logn(vmax_d, max(vmax_d * 1e-4, 1e-6)), how="mean")
    rp = R.fp_by_visible_distance if "fp_by_visible_distance" in R.tables else None
    if rp is not None and len(rp):
        for i, (u, g) in enumerate(rp.groupby("unit")):
            ax[1].plot(g.d_mid_m / 1e3, g.mean_p, color=OKABE[i], label=u)
        ax[1].set(xlabel="distance to nearest training-visible mapped trace (km)", ylabel="mean confidence of scored pixels",
                  xscale="log", title="Known-fault proximity vs confidence")
        ax[1].legend(fontsize=7)
    W.save(fig, "14_fp_density_and_proximity", "Right panel is diagnostic: distances are never used for prediction.")
    zoom_panels(run, W, R)
    prof = R.profiles if "profiles" in R.tables else None
    if prof is not None and len(prof):
        fig, ax = P.subplots(figsize=(7.5, 4.2))
        for i, ((u, ref), g) in enumerate(prof.groupby(["unit", "reference"])):
            ax.plot(g.d_mid_m, g.mean_p, color=OKABE[int(u[-1]) % 8], ls="-" if ref == "held truth" else "--",
                    label=f"{u}: from {ref}")
        ax.axvline(300, color="grey", ls=":", lw=1)
        ax.set(xlabel="perpendicular distance from trace (m)", ylabel="mean confidence", title="Fault-normal confidence profiles")
        ax.legend(fontsize=6.5, ncol=2)
        W.save(fig, "16_fault_normal_profiles", "Mean full-area confidence vs distance from held (solid) and "
               "training-visible (dashed) traces; a peak at 0 m for held traces is required for weighted TP.")


def zoom_panels(run, W: Writer, R):
    P = plt()
    reg = R.regions if "regions" in R.tables else pd.DataFrame()
    if not len(reg):
        return W.skip("15_zoom_panels", "no regions")
    picks = []
    for kind, col, asc in (("correct", "recall_w", False), ("false_positive", "fp_mass", False), ("missed", "n_px", False)):
        k = reg[reg.kind == kind].sort_values(col, ascending=asc).head(int(run.cfg["figures"]["n_zoom"]))
        picks += [(kind, r) for r in k.itertuples()]
    if not picks:
        return W.skip("15_zoom_panels", "no correct / FP / missed regions")
    n = len(picks)
    cols = 3
    rows = -(-n // cols)
    fig, ax = P.subplots(rows, cols, figsize=(5 * cols, 4.8 * rows), squeeze=False)
    for a, (kind, r) in zip(ax.ravel(), picks):
        win = window_around(run, r.row, r.col)
        p = run.full_area(r.unit)
        sub = p[win[0]:win[1], win[2]:win[3]]
        map_panel(a, run, p, f"{kind.replace('_', ' ')} - {r.region_id}", "p", vmin=0,
                  vmax=max(float(np.nanmax(sub)) if np.isfinite(sub).any() else 1e-3, 1e-3), window=win)
        overlay(a, run, run.units[r.unit].truth, "#00c8ff", 1, window=win)
        overlay(a, run, run.units[r.unit].training_visible, "white", 0.9, window=win)
    for a in ax.ravel()[n:]:
        a.axis("off")
    W.save(fig, "15_zoom_panels", "Full resolution (100 m). Cyan = held truth, white = training-visible mapped traces. "
           "A false positive here may be a real but unmapped structure: held labels are incomplete.")


# ---------------------------------------------------------------------------- D: experts
def expert_complementarity(run, W: Writer, R):
    P = plt()
    u = run.cfg["units"]["full"] if run.inference.cached(run.cfg["units"]["full"])[0] else None
    if u is None:
        units = [x for x in run.cfg["units"]["discovery"] if run.inference.cached(x)[0]]
        u = units[0] if units else None
    if u is None:
        return W.skip("17_heads", "no cached heads")
    heads = {h: run.full_area(u, h) for h in ("global", "expert0", "expert1", "prob")}
    vmax = max(float(np.nanquantile(heads[h][run.bank.valid], 0.999)) for h in heads)
    fig, ax = P.subplots(1, 4, figsize=(21, 5.5))
    for a, h in zip(ax, heads):
        map_panel(a, run, heads[h], f"{u}: {h}", "p (log)", norm=logn(vmax))
    W.save(fig, "17_global_vs_specialists", "Same log scale for all heads. Specialists are mixed as "
           "prob = (1 - s) global + s (r0 e0 + r1 e1).")
    r0, s = run.full_area(u, "route0"), run.full_area(u, "specialist_weight")
    fig, ax = P.subplots(1, 3, figsize=(17, 5.5))
    map_panel(ax[0], run, r0, "Router weight of expert 0", "r0 (0-1)", cmap="RdBu_r", vmin=0, vmax=1, how="mean")
    map_panel(ax[1], run, s, "Specialist mixture strength s", "s (0-1)", cmap="viridis", vmin=0,
              vmax=float(np.nanquantile(s, 0.99)), how="mean")
    d = heads["prob"] - heads["global"]
    lim = float(np.nanquantile(np.abs(d), 0.999)) or 1e-3
    from matplotlib.colors import TwoSlopeNorm

    map_panel(ax[2], run, d, "Where mixing changes the global head", "prob - global", cmap="RdBu_r",
              norm=TwoSlopeNorm(0, -lim, lim), how="mean")
    W.save(fig, "18_router_strength_mixing", "Mean-pooled previews. A textured router does not by itself show "
           "geological specialisation.")
    smp = run.sample
    rr, cc = smp
    fig, ax = P.subplots(1, 3, figsize=(15, 4.6))
    for a, (x, y) in zip(ax, (("global", "expert0"), ("expert0", "expert1"), ("global", "prob"))):
        xv, yv = heads[x][rr, cc], heads[y][rr, cc]
        ok = np.isfinite(xv) & np.isfinite(yv)
        a.hexbin(np.log10(np.clip(xv[ok], 1e-6, 1)), np.log10(np.clip(yv[ok], 1e-6, 1)), gridsize=60, bins="log",
                 cmap="Greys")
        a.plot([-6, 0], [-6, 0], color=OKABE[3], lw=0.8, ls=":")
        a.set(xlabel=f"log10 {x}", ylabel=f"log10 {y}", title=f"{y} vs {x} (1 px per km^2)")
    W.save(fig, "19_joint_head_distributions", "Points on a line parallel to the diagonal = same ranking, different "
           "scale; scatter away from any monotone curve = different structures.")
    dd = (heads["expert0"] - heads["expert1"])
    fig, ax = P.subplots(1, 2, figsize=(13, 5))
    lim = float(np.nanquantile(np.abs(dd[run.bank.valid]), 0.995)) or 1e-3
    map_panel(ax[0], run, dd, "Expert disagreement e0 - e1", "e0 - e1", cmap="RdBu_r", norm=TwoSlopeNorm(0, -lim, lim),
              how="mean")
    reg = run.region_map()
    zs = [k for k in range(int(reg.max()) + 1)]
    vals = [dd[rr, cc][(reg[rr, cc] == k) & np.isfinite(dd[rr, cc])] for k in zs]
    keep = [i for i, v in enumerate(vals) if v.size]
    if keep:
        ax[1].boxplot([vals[i] for i in keep], showfliers=False)
        ax[1].set_xticks(range(1, len(keep) + 1))
        ax[1].set_xticklabels([f"zone {zs[i]}" for i in keep])
    ax[1].set(ylabel="e0 - e1", title="Expert difference by geographic zone")
    W.save(fig, "20_expert_disagreement", "Zones are the V1 geographic CV zones (west to east); sampled 1 px / km^2.")
    perf = R.performance if "performance" in R.tables else pd.DataFrame()
    env = R.environment if "environment" in R.tables else pd.DataFrame()
    if len(perf):
        fig, ax = P.subplots(1, 2, figsize=(14, 4.6))
        units = sorted(perf.unit.unique())
        heads_ = ["global", "expert0", "expert1", "prob"]
        for j, h in enumerate(heads_):
            v = [perf[(perf.unit == x) & (perf.head == h)].dti.iloc[0] for x in units]
            ax[0].bar(np.arange(len(units)) + (j - 1.5) * 0.2, v, 0.2, color=OKABE[j], label=h)
        ax[0].set(xticks=range(len(units)), xticklabels=units, ylabel=PROXY, title="Each head on withheld labels (full area)")
        ax[0].legend(fontsize=7)
        if len(env):
            e = env.groupby(["head", "environment"]).apply(
                lambda g: np.nansum(g.recall_w * g.n_truth) / max(g.n_truth.sum(), 1)).unstack()
            for j, h in enumerate(heads_):
                if h in e.index:
                    ax[1].bar(np.arange(e.shape[1]) + (j - 1.5) * 0.2, e.loc[h].values, 0.2, color=OKABE[j], label=h)
            ax[1].set(xticks=range(e.shape[1]), ylabel="weighted recall of held truth",
                      title="By label-free structural environment")
            ax[1].set_xticklabels(e.columns, fontsize=7)
        W.save(fig, "21_expert_performance", "Environments: V1 router prior 0.5 + 1.5 (magnetic - gravity edge "
               "persistence), label-free.")
    pri = run.regime_prior()
    fig, ax = P.subplots(figsize=(6, 4.8))
    xv, yv = pri[rr, cc], r0[rr, cc]
    ok = np.isfinite(xv) & np.isfinite(yv)
    ax.hexbin(xv[ok], yv[ok], gridsize=50, bins="log", cmap="Greys")
    ax.set(xlabel="label-free prior (magnetic - gravity edge persistence)", ylabel="router weight r0",
           title="Expert assignment vs edge persistence")
    W.save(fig, "22_router_vs_persistence_prior", "If r0 tracks only its training prior, the router added no "
           "new partition of the geology.")


# ---------------------------------------------------------------------------- E: geology
def geological_disagreement(run, W: Writer, R):
    P = plt()
    rc = R.response if "response" in R.tables else pd.DataFrame()
    if len(rc):
        ev = list(rc.evidence.unique())[:4]
        fig, ax = P.subplots(1, len(ev), figsize=(4.4 * len(ev), 4), squeeze=False)
        for a, e in zip(ax[0], ev):
            g = rc[rc.evidence == e]
            a.fill_between(g.x_mid, g.ci_lo, g.ci_hi, color=OKABE[5], alpha=0.35, lw=0)
            a.plot(g.x_mid, g.mean_p, "o-", color=OKABE[0], ms=3)
            a.set(xlabel=e.replace("_", " "), ylabel="mean V1 full-model p", title=e.replace("_", " "))
        W.save(fig, "23_confidence_vs_evidence", "Deciles of each label-free Atlas layer; 90% bootstrap band over "
               "1 km block samples. Rising curves show what the model responds to, not where faults are.")
    cp = R.class_profiles if "class_profiles" in R.tables else pd.DataFrame()
    if len(cp):
        ev = [e for e in ("magnetic_edge", "gravity_edge", "terrain_slope_break", "xm_evidence_mean",
                          "xm_multiscale_persistence", "xm_continuity") if e in set(cp.evidence)]
        classes = ["supported truth", "missed truth", "high-confidence FP", "background"]
        fig, ax = P.subplots(figsize=(12, 4.5))
        for j, cl in enumerate(classes):
            g = cp[cp["class"] == cl].groupby("evidence")[["q25", "median", "q75"]].median().reindex(ev)
            x = np.arange(len(ev)) + (j - 1.5) * 0.18
            ax.errorbar(x, g["median"], yerr=[g["median"] - g.q25, g.q75 - g["median"]], fmt="o", color=OKABE[j], label=cl,
                        capsize=2)
        ax.set(xticks=range(len(ev)), ylabel="evidence value (rank / fraction)", title="Label-free evidence by error class")
        ax.set_xticklabels([e.replace("_", "\n") for e in ev], fontsize=7)
        ax.legend(fontsize=7)
        W.save(fig, "24_evidence_by_error_class", "Median and inter-quartile range, pooled over discovery folds.")
    ctx = run.gravity_context()
    if ctx is not None:
        fig, ax = P.subplots(figsize=(7.5, 6.5))
        map_panel(ax, run, ctx, "Long-wavelength gravity context (sigma 15 km)", "isostatic anomaly, smoothed (mGal)",
                  cmap="cividis", how="mean")
        p = run.v1_full_prob()
        q = float(run.cfg["diagnostics"]["high_conf_quantile"])
        elig = run.bank.valid & ~run.bank.labels & np.isfinite(p)
        hi = elig & (p >= float(np.quantile(p[elig], q)))
        overlay(ax, run, hi, OKABE[3], 0.9, label=f"top {100 * (1 - q):.1f}% confidence")
        ax.legend(fontsize=7)
        W.save(fig, "25_gravity_context_vs_predictions", "Context channel of the V1 bank (label-free, 1 km coarsened "
               "then smoothed). Units follow the Atlas isostatic anomaly band.")
    else:
        W.skip("25_gravity_context_vs_predictions", "V1 context bank not loaded")
    reg = R.regions if "regions" in R.tables else pd.DataFrame()
    ev = run.evidence
    for kind in ("false_positive", "missed"):
        k = reg[reg.kind == kind].sort_values("fp_mass" if kind == "false_positive" else "n_px", ascending=False).head(
            int(run.cfg["figures"]["n_zoom"])) if len(reg) else pd.DataFrame()
        if not len(k) or ev is None:
            W.skip(f"26_evidence_zoom_{kind}", "no regions or no Atlas evidence")
            continue
        layers = [("p", None), ("magnetic_edge", "magnetic edge rank"), ("gravity_edge", "gravity edge rank"),
                  ("xm_evidence_mean", "cross-modal evidence")]
        fig, ax = P.subplots(len(k), 4, figsize=(18, 4.4 * len(k)), squeeze=False)
        for i, r in enumerate(k.itertuples()):
            win = window_around(run, r.row, r.col)
            for j, (name, lab) in enumerate(layers):
                a = ax[i, j]
                if name == "p":
                    arr = run.full_area(r.unit)
                    sub = arr[win[0]:win[1], win[2]:win[3]]
                    map_panel(a, run, arr, f"{r.region_id}: p", "p", vmin=0,
                              vmax=max(float(np.nanmax(sub)) if np.isfinite(sub).any() else 1e-3, 1e-3), window=win)
                else:
                    arr = ev.get(name)
                    if arr is None:
                        a.text(0.5, 0.5, f"{name} unavailable in the Atlas", ha="center", transform=a.transAxes)
                        a.axis("off")
                        continue
                    map_panel(a, run, arr, lab, "rank (0-1)", cmap="viridis", vmin=0, vmax=1, window=win)
                overlay(a, run, run.units[r.unit].truth, "#ff2a2a", 1, window=win)
                overlay(a, run, run.units[r.unit].training_visible, "white", 0.9, window=win)
            ax[i, 0].text(0.01, -0.22, r.flags, transform=ax[i, 0].transAxes, fontsize=6.5, va="top", wrap=True)
        W.save(fig, f"26_evidence_zoom_{kind}", "Red = held truth, white = training-visible mapped traces. Flags are "
               "descriptive ('consistent with'), not confirmations of faults.")


def structural_strata(run, W: Writer, R):
    P = plt()
    t = R.components if "components" in R.tables else pd.DataFrame()
    if not len(t):
        return W.skip("27_recall_by_length", "no held components")
    fig, ax = P.subplots(1, 2, figsize=(12, 4.3))
    lc = R.length_classes
    pl = lc[lc.unit == "pooled"]
    ax[0].bar(pl.length_class.astype(str), pl.recall_w, color=OKABE[0])
    for i, r in enumerate(pl.itertuples()):
        ax[0].text(i, r.recall_w, f"n={r.n_components}", ha="center", va="bottom", fontsize=7)
    ax[0].set(ylabel="weighted recall (pooled)", title="Recall by held-fault length class")
    ax[1].scatter(t.linearity, t.recall_w, s=np.clip(t.n_px / 3, 4, 80), c=[OKABE[int(u[-1]) % 8] for u in t.unit], alpha=0.7)
    ax[1].set(xlabel="linearity l1/(l1+l2) of the held component", ylabel="weighted recall",
              title="Recall vs trace linearity (size = pixels)")
    W.save(fig, "27_recall_by_length_and_linearity", "Full-area discovery protocol; components are 8-connected raster "
           "fragments of the mapped network.")
    sa = R.strike_agreement if "strike_agreement" in R.tables else pd.DataFrame()
    if len(sa):
        fig, ax = P.subplots(1, 2, figsize=(12, 4.2))
        g = sa.groupby(["diff_lo_deg"]).n_px.sum()
        ax[0].bar(g.index + 5, g.values / g.values.sum(), width=9, color=OKABE[2])
        ax[0].set(xlabel="|predicted - mapped| local strike (deg)", ylabel="share of supported held pixels",
                  title="Strike agreement at supported truth")
        sk = R.strikes if "strikes" in R.tables else pd.DataFrame(columns=["kind", "strike_deg"])
        ms = sk[sk.kind == "mapped"].strike_deg.to_numpy()
        ps = sk[sk.kind == "predicted"].strike_deg.to_numpy()
        if len(ms) and len(ps):
            b = np.arange(0, 181, 10)
            ax[1].hist(ms, bins=b, density=True, histtype="step", lw=1.5, color=OKABE[0], label="mapped (held)")
            ax[1].hist(ps, bins=b, density=True, histtype="step", lw=1.5, color=OKABE[3], label="predicted (supported)")
            ax[1].set(xlabel="strike azimuth (deg, clockwise from N, axial)", ylabel="density", title="Strike distributions")
            ax[1].legend(fontsize=7)
        W.save(fig, "28_strike_agreement", "Structure-tensor strike of the smoothed held trace vs of the smoothed "
               "prediction (coherence > 0.2).")
    rg = R.regional if "regional" in R.tables else pd.DataFrame()
    if len(rg):
        B = int(run.cfg["diagnostics"]["region_block_px"])
        img = np.full(run.bank.shape, np.nan, np.float32)
        agg = rg.groupby(["row0", "col0"]).apply(lambda g: g.tp_w.sum() / g.n_truth.sum() if g.n_truth.sum() >= run.cfg[
            "diagnostics"]["min_truth_per_block"] else np.nan)
        for (r0, c0), v in agg.items():
            img[int(r0):int(r0) + B, int(c0):int(c0) + B] = v
        img[~run.bank.valid] = np.nan
        fig, ax = P.subplots(figsize=(7.5, 6.5))
        map_panel(ax, run, img, "Weighted recall of held truth per 10 km block", "recall (0-1)", cmap="viridis", vmin=0,
                  vmax=1, how="mean")
        W.save(fig, "29_regional_recall", "Blocks with fewer than %d held pixels are blank (coverage-aware)."
               % run.cfg["diagnostics"]["min_truth_per_block"])


def candidate_lineaments(run, W: Writer, R):
    P = plt()
    t = R.table if "table" in R.tables else pd.DataFrame()
    lin = run._mem.get("lineaments")
    if not len(t) or lin is None:
        return W.skip("30_lineaments", "Atlas candidate lineaments unavailable")
    import matplotlib as mpl

    fig, ax = P.subplots(figsize=(9, 7.5))
    p = run.v1_full_prob()
    q999 = float(np.nanquantile(p[run.bank.valid], 0.999))
    map_panel(ax, run, p, "Atlas lineament proposals over V1 confidence", "p (log)", norm=logn(q999), cmap="Greys")
    cm = mpl.colormaps["plasma"]
    m = dict(zip(t.candidate_id, t.model_pct_rank))
    s = dict(zip(t.candidate_id, t.label_free_support))
    for cid, g in zip(lin.candidate_id, lin.geometry):
        if g is None or g.is_empty:
            continue
        x, y = g.xy
        ax.plot(np.asarray(x) / 1e3, np.asarray(y) / 1e3, color=cm(float(np.nan_to_num(m.get(cid, 0)))),
                lw=0.6 + 2.5 * float(np.nan_to_num(s.get(cid, 0))))
    sm = mpl.cm.ScalarMappable(cmap=cm, norm=mpl.colors.Normalize(0, 1))
    fig.colorbar(sm, ax=ax, shrink=0.6, label="model percentile rank along the trace")
    W.save(fig, "30_lineaments_vs_confidence", "Line width = label-free support. All proposals are UNVERIFIED "
           "structures and are never used as labels.")
    fig, ax = P.subplots(figsize=(6.5, 4.8))
    sc = ax.scatter(t.label_free_support, t.model_pct_rank, c=t.evidence_count, cmap="viridis", s=14 + 4 * t.length_km.clip(0, 20))
    fig.colorbar(sc, ax=ax, label="independent evidence count (0-3)")
    ax.set(xlabel="label-free support (V1 review formula)", ylabel="model percentile rank",
           title="Geological support vs model confidence")
    W.save(fig, "31_support_vs_model", f"Spearman {R.stats.get('spearman_support_vs_model', np.nan):.2f}; size = length.")
    top = R.top_candidates.head(int(run.cfg["figures"]["n_zoom"]))
    pix = run._mem.get("lineament_pixels", {})
    ev = run.evidence.get("xm_evidence_mean") if run.evidence is not None else None
    if len(top) and pix:
        fig, ax = P.subplots(len(top), 2, figsize=(10, 4.4 * len(top)), squeeze=False)
        for i, r in enumerate(top.itertuples()):
            rr, cc = pix.get(r.candidate_id, (np.array([]), np.array([])))
            if not len(rr):
                continue
            win = window_around(run, np.median(rr), np.median(cc))
            sub = p[win[0]:win[1], win[2]:win[3]]
            map_panel(ax[i, 0], run, p, f"{r.candidate_id}: V1 p", "p", vmin=0,
                      vmax=max(float(np.nanmax(sub)) if np.isfinite(sub).any() else 1e-3, 1e-3), window=win)
            if ev is not None:
                map_panel(ax[i, 1], run, ev, "cross-modal evidence", "rank", cmap="viridis", vmin=0, vmax=1, window=win)
            line = np.zeros(run.bank.shape, bool)
            line[rr, cc] = True
            for a in ax[i]:
                overlay(a, run, line, "#00c8ff", 1, window=win)
                overlay(a, run, run.bank.labels, "white", 0.8, window=win)
        W.save(fig, "32_candidate_zooms", "Cyan = candidate trace, white = mapped faults. Limitations per candidate are "
               "in candidate_model_agreement.csv.")


# ---------------------------------------------------------------------------- F: recovery
def transform_tradeoffs(run, W: Writer, R):
    P = plt()
    rc = R.response_curves
    fig, ax = P.subplots(figsize=(5.5, 4.6))
    for i, (key, g) in enumerate(rc.groupby(["policy", "support"], dropna=False)):
        pol, s = key
        ax.plot(g.p, g.q, color=OKABE[i % 8], label=pol + ("" if pd.isna(s) else f" (S={s:g})"))
    ax.set(xlabel="model probability p", ylabel="transformed score q", title="Predeclared transform responses")
    ax.legend(fontsize=7)
    W.save(fig, "33_transform_response", "At most four policies, fixed in config before any held-out score.")
    c = R.components_by_fold
    d = c[c.protocol == run.cfg["transforms"]["selection_protocol"]]
    ide = d[d.policy == "identity"].set_index("unit")
    fig, ax = P.subplots(1, 2, figsize=(13, 4.6))
    for i, (pol, g) in enumerate(d[d.policy != "identity"].groupby("policy")):
        g = g.set_index("unit")
        ax[0].scatter(g.fp_w - ide.loc[g.index].fp_w, g.tp_w - ide.loc[g.index].tp_w, color=OKABE[i + 1], label=pol, s=30)
    ax[0].axhline(0, color="grey", lw=0.7)
    ax[0].axvline(0, color="grey", lw=0.7)
    ax[0].set(xlabel="added FP_w vs identity", ylabel="TP_w gain vs identity", title="TP gain vs added FP mass (per fold)")
    ax[0].legend(fontsize=7)
    units = sorted(d.unit.unique())
    pols = list(dict.fromkeys(d.policy))
    for j, pol in enumerate(pols):
        v = [d[(d.unit == u) & (d.policy == pol)].dti.iloc[0] for u in units]
        ax[1].bar(np.arange(len(units)) + (j - (len(pols) - 1) / 2) * 0.8 / len(pols), v, 0.8 / len(pols),
                  color=OKABE[j], label=pol)
    ax[1].set(xticks=range(len(units)), xticklabels=units, ylabel=PROXY, title="Held-out DTI of each policy")
    ax[1].legend(fontsize=7)
    W.save(fig, "34_policy_tradeoffs", "Per-fold scores are shown for transparency; selection uses other folds only.")
    p = run.v1_full_prob()
    S = run.support_raster()
    pols = run.cfg["transforms"]["policies"]
    fig, ax = P.subplots(1, len(pols), figsize=(5 * len(pols), 5), squeeze=False)
    from .transforms import apply

    vmax = float(np.nanquantile(p[run.bank.valid], 0.999))
    for a, pol in zip(ax[0], pols):
        if pol["kind"] == "support" and S is None:
            a.text(0.5, 0.5, "support raster unavailable", ha="center", transform=a.transAxes)
            a.axis("off")
            continue
        map_panel(a, run, apply(pol, p, S), pol["name"], "score (log)", norm=logn(vmax))
    W.save(fig, "35_identity_vs_candidates", "Same colour scale; applied to the V1 full-model raster. A visually "
           "cleaner map is not a reason to promote a policy.")


def crossfit_uncertainty(run, W: Writer, R):
    P = plt()
    t = R.decision_table
    per = t[t.fold != "POOLED"]
    fig, ax = P.subplots(1, 3, figsize=(17, 4.6))
    x = np.arange(len(per))
    ax[0].bar(x - 0.2, per.dti_heldout_identity, 0.4, color=OKABE[7], label="identity")
    ax[0].bar(x + 0.2, per.dti_heldout_chosen, 0.4, color=OKABE[0], label="cross-fitted choice")
    for i, r in enumerate(per.itertuples()):
        ax[0].text(i + 0.2, r.dti_heldout_chosen, r.chosen_policy, ha="center", va="bottom", fontsize=7, rotation=90)
    ax[0].set(xticks=x, xticklabels=per.fold, ylabel=PROXY, title="Untouched held-out fold scores")
    ax[0].legend(fontsize=7)
    bs = R.bootstrap.delta_dti.dropna() if "bootstrap" in R.tables else pd.Series(dtype=float)
    if len(bs):
        ax[1].hist(bs, bins=40, color=OKABE[5])
        ax[1].axvline(0, color="black", lw=1)
        ax[1].axvline(R.stats["ci_low"], color=OKABE[3], ls="--")
        ax[1].axvline(R.stats["ci_high"], color=OKABE[3], ls="--")
        ax[1].set(xlabel="pooled DTI(cross-fitted) - DTI(identity)", ylabel="bootstrap replicates",
                  title="10 km block bootstrap (paired)")
    rg = R.regional if "regional" in R.tables else pd.DataFrame()
    if len(rg):
        ax[2].bar(rg.region.astype(str), rg.delta_dti, color=[OKABE[3] if v < 0 else OKABE[2] for v in rg.delta_dti])
        ax[2].axhline(0, color="black", lw=0.8)
        ax[2].set(xlabel="geographic zone (-1 = outside)", ylabel="delta DTI (all-fold choice)", title="Regional effect")
    W.save(fig, "36_crossfit_uncertainty", f"Pooled cross-fitted {R.stats['crossfit_pooled_dti']:.5f} vs identity "
           f"{R.stats['identity_pooled_dti']:.5f} ({PROXY}).")
    g = R.gates
    fig, ax = P.subplots(figsize=(10, 0.5 + 0.42 * len(g)))
    ax.axis("off")
    for i, r in enumerate(g.itertuples()):
        y = 1 - (i + 1) / (len(g) + 1)
        ax.text(0.0, y, "PASS" if r.passed else "FAIL", color=OKABE[2] if r.passed else OKABE[3], fontweight="bold",
                transform=ax.transAxes)
        ax.text(0.08, y, f"{r.gate}  -  {r.detail}", transform=ax.transAxes, fontsize=8)
    ax.set_title(R.stats["decision"], fontweight="bold", loc="left")
    W.save(fig, "37_recommendation_gates", "Gates were fixed in config before scoring. Export additionally requires "
           "the user opt-in and the guarded readback checks.")


GROUPS = {f.__name__: f for f in (prediction_distribution, known_fault_proximity, split_integrity, original_dti,
                                   full_area_coverage, full_area_dti, error_atlas, expert_complementarity,
                                   geological_disagreement, structural_strata, candidate_lineaments,
                                   transform_tradeoffs, crossfit_uncertainty)}
