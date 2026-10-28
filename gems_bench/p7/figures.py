"""P7 diagnostic figures. Every function returns ``(fig, data)``; ``P7GeoFaultFormer.figures`` saves PNG + SVG and
the plotting data as ``<name>_data.csv`` (``G3Workspace.save_fig``). A figure whose inputs do not exist returns
``(None, None)`` and is recorded as ABSENT: nothing is drawn from invented numbers.

Attention: ``attention_context`` shows GENUINE softmax weights of a window-attention block (captured in the
forward pass) next to an input-gradient saliency map, which is an attribution proxy and is labelled as such.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FIGURES = ("training_curves", "fold_dti", "probability_maps", "attention_context", "centerline_comparison",
           "orientation_field", "gru_refinement", "tracer_trajectories", "heldout_examples", "error_maps",
           "recall_curves", "pareto", "disagreement", "coverage_by_budget", "regime_errors")


def _plt():
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    return plt


def _crop(a: np.ndarray, ev: np.ndarray, size: int = 400) -> tuple[slice, slice]:
    """A ``size`` window centred on the evaluation mask's densest area (label-free choice)."""
    r, c = np.nonzero(ev)
    if not len(r):
        return slice(0, a.shape[0]), slice(0, a.shape[1])
    rc, cc = int(np.median(r)), int(np.median(c))
    h = size // 2
    return slice(max(rc - h, 0), rc + h), slice(max(cc - h, 0), cc + h)


def training_curves(hist: pd.DataFrame):
    if hist is None or not len(hist):
        return None, None
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for u, g in hist.groupby("unit"):
        ax[0].plot(g.epoch, g.loss_total, label=u)
        v = g[g.inner_status == "VALID"]
        if len(v):
            ax[1].plot(v.epoch, v.inner_dti, marker="o", label=u)
    ax[0].set(title="training loss (total)", xlabel="epoch")
    ax[1].set(title="inner checkpoint DTI (VALID epochs only; not an outer score)", xlabel="epoch")
    for a in ax:
        a.legend(fontsize=7)
    return fig, hist[[c for c in ("experiment_id", "unit", "epoch", "loss_total", "inner_dti", "inner_status")
                      if c in hist]]


def fold_dti(per_zone: pd.DataFrame):
    if per_zone is None or not len(per_zone):
        return None, None
    plt = _plt()
    tasks = sorted(per_zone.task.unique())
    fig, ax = plt.subplots(1, len(tasks), figsize=(6 * len(tasks), 4), squeeze=False)
    for a, t in zip(ax[0], tasks):
        d = per_zone[per_zone.task == t].pivot_table(index="zone", columns="policy", values="dti")
        d.plot.bar(ax=a)
        a.set(title=f"nested outer DTI per zone ({t})", ylabel="DTI")
        a.legend(fontsize=7)
    return fig, per_zone[["policy", "task", "zone", "dti", "rep", "frac", "blend"]]


def probability_maps(maps: dict, ev: np.ndarray, title: str):
    if not maps:
        return None, None
    plt = _plt()
    sl = _crop(next(iter(maps.values())), ev)
    fig, ax = plt.subplots(1, len(maps), figsize=(4.5 * len(maps), 4.5), squeeze=False)
    rows = []
    for a, (k, m) in zip(ax[0], maps.items()):
        im = a.imshow(np.where(ev[sl], m[sl], np.nan), cmap="magma", vmin=0, vmax=1)
        a.set(title=k)
        a.axis("off")
        fig.colorbar(im, ax=a, fraction=0.046)
        v = m[ev]
        rows.append({"map": k, "mean": float(np.nanmean(v)), "p99": float(np.nanpercentile(v, 99))})
    fig.suptitle(title)
    return fig, pd.DataFrame(rows)


def attention_context(x_patch: np.ndarray, attn: np.ndarray | None, saliency: np.ndarray | None, window: int,
                      query_rc: tuple[int, int]):
    """attn: (heads, N, N) weights of the query's window; saliency: (P, P) |d output / d input| (proxy)."""
    if attn is None and saliency is None:
        return None, None
    plt = _plt()
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    ax[0].imshow(x_patch, cmap="gray")
    ax[0].set(title="input channel 0 (scaled)")
    rows = []
    if attn is not None:
        q = query_rc[0] * window + query_rc[1]
        w = attn.mean(0)[q].reshape(window, window)
        ax[1].imshow(w, cmap="viridis")
        ax[1].set(title="GENUINE attention weights (mean over heads)")
        rows += [{"kind": "attention", "r": int(i), "c": int(j), "value": float(w[i, j])}
                 for i in range(window) for j in range(window)]
    if saliency is not None:
        ax[2].imshow(saliency, cmap="inferno")
        ax[2].set(title="input-gradient saliency (ATTRIBUTION PROXY)")
        rows.append({"kind": "saliency_summary", "r": -1, "c": -1, "value": float(np.nanmean(saliency))})
    for a in ax:
        a.axis("off")
    return fig, pd.DataFrame(rows)


def centerline_comparison(seg: np.ndarray, center: np.ndarray | None, truth: np.ndarray, ev: np.ndarray):
    if center is None:
        return None, None
    plt = _plt()
    sl = _crop(seg, ev)
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.5))
    for a, (k, m) in zip(ax, (("fault probability", seg), ("centerline probability", center),
                              ("held-out truth (evaluation only)", truth.astype(np.float32)))):
        a.imshow(np.where(ev[sl], m[sl], np.nan), cmap="magma", vmin=0, vmax=1)
        a.set(title=k)
        a.axis("off")
    t = truth & ev
    data = pd.DataFrame([{"map": "seg", "mean_on_truth": float(seg[t].mean()) if t.any() else np.nan,
                          "mean_off_truth": float(seg[ev & ~t].mean())},
                         {"map": "center", "mean_on_truth": float(center[t].mean()) if t.any() else np.nan,
                          "mean_off_truth": float(center[ev & ~t].mean())}])
    return fig, data


def orientation_field(ocos: np.ndarray | None, osin: np.ndarray | None, seg: np.ndarray, ev: np.ndarray,
                      step: int = 8):
    if ocos is None or osin is None:
        return None, None
    from ..geology.orientation import axial_angle

    plt = _plt()
    sl = _crop(seg, ev, 240)
    th = np.radians(axial_angle(ocos[sl], osin[sl]))
    conf = np.hypot(ocos[sl], osin[sl])
    rr, cc = np.mgrid[0:th.shape[0]:step, 0:th.shape[1]:step]
    t, cf = th[::step, ::step], conf[::step, ::step]
    dr, dc = -np.cos(t) * cf, np.sin(t) * cf                 # along-strike vector (rows grow southwards)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(np.where(ev[sl], seg[sl], np.nan), cmap="Greys", vmin=0, vmax=1)
    ax.quiver(cc, rr, dc, -dr, cf, cmap="viridis", headlength=0, headaxislength=0, pivot="middle", scale=25)
    ax.set(title="predicted strike (axial; length = |(cos 2t, sin 2t)|)")
    ax.axis("off")
    return fig, pd.DataFrame({"r": rr.ravel(), "c": cc.ravel(), "strike_deg": np.degrees(t).ravel(),
                              "confidence": cf.ravel()})


def gru_refinement(states: list[np.ndarray] | None):
    """|h_t| and |h_t - h_{t-1}| maps of the ConvGRU state (channel-mean of one patch); hidden-state diagnostics."""
    if not states:
        return None, None
    plt = _plt()
    n = len(states)
    fig, ax = plt.subplots(2, n, figsize=(3.5 * n, 7), squeeze=False)
    rows, prev = [], None
    for t, h in enumerate(states):
        mag = np.abs(h).mean(0)
        ax[0][t].imshow(mag, cmap="viridis")
        ax[0][t].set(title=f"|h_{t + 1}|")
        d = np.abs(h - prev).mean(0) if prev is not None else np.zeros_like(mag)
        ax[1][t].imshow(d, cmap="magma")
        ax[1][t].set(title=f"|h_{t + 1} - h_{t}|")
        rows.append({"step": t + 1, "mean_abs_h": float(mag.mean()), "mean_abs_update": float(d.mean())})
        prev = h
    for a in ax.ravel():
        a.axis("off")
    return fig, pd.DataFrame(rows)


def tracer_trajectories(evidence: np.ndarray, base: np.ndarray, paths: list, reasons: list[str], ev: np.ndarray):
    if not paths:
        return None, None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(np.where(ev, evidence, np.nan), cmap="Greys", vmin=0, vmax=1)
    ax.contour(base.astype(float), levels=[0.5], colors="tab:blue", linewidths=0.6)
    rows = []
    for i, (p, why) in enumerate(zip(paths, reasons)):
        ax.plot(p[:, 1], p[:, 0], lw=1.0, color="tab:red" if why in ("gap", "boundary") else "tab:orange")
        rows += [{"path": i, "k": k, "r": float(r), "c": float(c), "stop": why} for k, (r, c) in enumerate(p)]
    ax.set(title="tracer rollouts (red: gap / boundary stop) over evidence; blue: base map")
    ax.axis("off")
    return fig, pd.DataFrame(rows)


def heldout_examples(q: np.ndarray, truth: np.ndarray, known: np.ndarray | None, ev: np.ndarray, title: str):
    plt = _plt()
    sl = _crop(q, ev)
    rgb = np.zeros(q[sl].shape + (3,), np.float32)
    rgb[..., 0] = q[sl] > 0
    rgb[..., 1] = truth[sl]
    if known is not None:
        rgb[..., 2] = known[sl] * 0.6
    rgb[~ev[sl]] = 0.15
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(rgb)
    ax.set(title=f"{title}: red prediction, green held-out truth, yellow overlap, blue known (masked)")
    ax.axis("off")
    return fig, pd.DataFrame([{"pred_px": int((q > 0)[ev].sum()), "truth_px": int(truth[ev].sum()),
                               "overlap_px": int(((q > 0) & truth & ev).sum())}])


def error_maps(q: np.ndarray, truth: np.ndarray, ev: np.ndarray, radius_px: int = 3, block: int = 50):
    """FP pixels (> 300 m from truth) and FN truth pixels (> 300 m from prediction), aggregated in blocks."""
    from scipy import ndimage

    plt = _plt()
    pred = (q > 0) & ev
    t = truth & ev
    dt = ndimage.distance_transform_edt(~t) if t.any() else np.full(t.shape, np.inf)
    dp = ndimage.distance_transform_edt(~pred) if pred.any() else np.full(t.shape, np.inf)
    fp, fn = pred & (dt > radius_px), t & (dp > radius_px)
    H, W = q.shape
    hb, wb = H // block + 1, W // block + 1
    agg = {}
    for k, m in (("fp", fp), ("fn", fn)):
        a = np.zeros((hb, wb))
        r, c = np.nonzero(m)
        np.add.at(a, (r // block, c // block), 1)
        agg[k] = a
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for a_, k in zip(ax, ("fp", "fn")):
        im = a_.imshow(agg[k], cmap="Reds" if k == "fp" else "Blues")
        a_.set(title=f"{k.upper()} pixels per {block} x {block} block")
        fig.colorbar(im, ax=a_, fraction=0.046)
    rows = [{"kind": k, "br": int(i), "bc": int(j), "count": float(agg[k][i, j])}
            for k in agg for i, j in zip(*np.nonzero(agg[k]))]
    return fig, pd.DataFrame(rows)


def recall_curves(rt: pd.DataFrame):
    if rt is None or not len(rt):
        return None, None
    plt = _plt()
    reps = sorted(rt.rep.unique())
    fig, ax = plt.subplots(1, len(reps), figsize=(6 * len(reps), 4), squeeze=False)
    for a, rep in zip(ax[0], reps):
        for c, g in rt[rt.rep == rep].groupby("candidate"):
            a.plot(g.frac * 100, g.recall, marker="o", label=c)
        a.set(title=f"system recall vs area budget ({rep})", xlabel="area % of eligible", ylabel="system recall")
        a.legend(fontsize=7)
    return fig, rt


def pareto(summary: pd.DataFrame):
    s = summary[summary.status == "COMPLETE"] if summary is not None and "status" in summary else None
    if s is None or not len(s):
        return None, None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(s.geo_dti, s.disc_dti)
    for r in s.itertuples():
        ax.annotate(r.policy, (r.geo_dti, r.disc_dti), fontsize=7)
    ax.set(xlabel="geographic DTI (nested)", ylabel="discovery DTI (nested)", title="discovery vs geographic")
    return fig, s[["policy", "disc_dti", "geo_dti", "disc_worst"]]


def disagreement(a: np.ndarray, b: np.ndarray, ev: np.ndarray, names: tuple[str, str]):
    plt = _plt()
    sl = _crop(a, ev)
    d = np.where(ev, a - b, np.nan)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    im = ax.imshow(d[sl], cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set(title=f"rank({names[0]}) - rank({names[1]})")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046)
    v = d[ev]
    return fig, pd.DataFrame([{"a": names[0], "b": names[1], "mean_abs": float(np.nanmean(np.abs(v))),
                               "share_abs_gt_0_5": float(np.nanmean(np.abs(v) > 0.5))}])


def coverage_by_budget(rt: pd.DataFrame):
    if rt is None or not len(rt):
        return None, None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 4))
    for (c, rep), g in rt.groupby(["candidate", "rep"]):
        ax.plot(g.frac * 100, g.coverage_w, marker=".", label=f"{c} {rep}")
    ax.set(xlabel="area % of eligible", ylabel="length-weighted coverage (TP_w / truth)", title="coverage by budget")
    ax.legend(fontsize=6)
    return fig, rt[["candidate", "rep", "frac", "coverage_w", "area_frac_eligible"]]


def regime_errors(rows: pd.DataFrame):
    if rows is None or not len(rows):
        return None, None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4))
    rows.pivot_table(index="domain", columns="candidate", values="dti").plot.bar(ax=ax)
    ax.set(title="DTI by label-free geological domain (CV-E; held-out cells)", ylabel="DTI")
    return fig, rows
