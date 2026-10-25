"""M6.5: do two models (B0, B1) make different geological errors? Paired maps at identical area budgets.

``pair_statistics``      score correlation (Pearson / Spearman on a pixel sample), mask overlap, IoU and the
                         disagreement area per budget, pooled and per fold (masks: per-fold top-a, the scored object)
``fp_distance_profile``  predicted area by distance to the nearest mapped fault, per model and agreement class
``discovery_classes``    withheld systems recovered by both / one / neither at a matched discovery budget
``case_windows``         reproducible windows (kind, source map, row / col, size, budget) for the panels:
                         a-only, b-only, missed by both, short faults, complex junctions, false-positive-prone
                         unmapped structures, recovered withheld systems
``fig_case_study``       six matched panels with identical extent and scale (geophysics, mapped faults, a, b,
                         ensemble, agreement) for one window, regenerated from the window table alone

Predictions away from mapped faults are labelled "unmapped prediction": they are not confirmed faults.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

DIST_BINS_KM = (0.0, 0.3, 1.0, 3.0, np.inf)
CASE_KINDS = ("a_only", "b_only", "missed_by_both", "short_fault", "complex_junction", "fp_prone",
              "withheld_discovery")


def budget_masks(ctx, p: np.ndarray, frac: float) -> np.ndarray:
    from ..phase4_eval import fold_masks

    return fold_masks(ctx, p, frac)


def pair_statistics(ctx, maps: dict[str, np.ndarray], a: str, b: str, budgets=(0.01, 0.02, 0.05, 0.10),
                    sample: int = 300_000, seed: int = 2026) -> pd.DataFrame:
    from scipy.stats import pearsonr, spearmanr

    v, fm = ctx.data.valid, ctx.folds.fold_map
    km2 = ctx.data.grid.pixel_km_x * ctx.data.grid.pixel_km_y
    far = ctx.dist_to_labels_km() > 1.0
    rng = np.random.default_rng(seed)
    rows = []
    zones = [("all", v & (fm >= 0))] + [(str(k), v & (fm == k)) for k in ctx.fold_ids]
    corr = {}
    for z, m in zones:
        idx = np.flatnonzero(m.ravel())
        idx = rng.choice(idx, min(sample, idx.size), replace=False) if idx.size else idx
        pa, pb = maps[a].ravel()[idx], maps[b].ravel()[idx]
        ok = np.isfinite(pa) & np.isfinite(pb)
        corr[z] = (float(pearsonr(pa[ok], pb[ok])[0]) if ok.sum() > 2 else np.nan,
                   float(spearmanr(pa[ok], pb[ok]).correlation) if ok.sum() > 2 else np.nan)
    for f in budgets:
        A, B = budget_masks(ctx, maps[a], f), budget_masks(ctx, maps[b], f)
        for z, m in zones:
            Am, Bm = A & m, B & m
            inter, union = float((Am & Bm).sum()), float((Am | Bm).sum())
            rows.append({"model_a": a, "model_b": b, "budget": f, "zone": z, "pearson": corr[z][0], "spearman": corr[z][1],
                         "area_a_km2": Am.sum() * km2, "area_b_km2": Bm.sum() * km2, "iou": inter / max(union, 1.0),
                         "overlap_a_in_b": inter / max(float(Am.sum()), 1.0),
                         "a_only_km2": float((Am & ~Bm).sum() * km2), "b_only_km2": float((Bm & ~Am).sum() * km2),
                         "both_km2": inter * km2, "a_only_far_km2": float((Am & ~Bm & far).sum() * km2),
                         "b_only_far_km2": float((Bm & ~Am & far).sum() * km2),
                         "both_far_km2": float((Am & Bm & far).sum() * km2)})
    return pd.DataFrame(rows)


def fp_distance_profile(ctx, masks: dict[str, np.ndarray], a: str, b: str, bins=DIST_BINS_KM) -> pd.DataFrame:
    """Predicted area by distance to mapped faults (false-positive exposure grows with distance)."""
    d = ctx.dist_to_labels_km()
    km2 = ctx.data.grid.pixel_km_x * ctx.data.grid.pixel_km_y
    A, B = masks[a], masks[b]
    groups = {**{m: masks[m] for m in masks}, f"{a}_only": A & ~B, f"{b}_only": B & ~A, "both": A & B}
    rows = []
    for name, m in groups.items():
        tot = float(m.sum()) or 1.0
        for lo, hi in zip(bins[:-1], bins[1:]):
            sel = m & (d >= lo) & (d < hi)
            rows.append({"group": name, "dist_km_lo": lo, "dist_km_hi": hi, "area_km2": float(sel.sum() * km2),
                         "share": float(sel.sum() / tot)})
    return pd.DataFrame(rows)


def discovery_classes(ctx, objs: dict, a: str, b: str, frac: float, policy: str = "excl", k_thr: float = 0.5) -> pd.DataFrame:
    from ..p3 import discovery3

    per = discovery3.systems_at_budget(ctx, {k: v for k, v in objs.items() if k[1] in (a, b)}, frac, policy=policy)
    w = per.pivot_table(index=["split", "system_id"], columns="model", values="weighted_recall").reset_index()
    if a not in w or b not in w:
        return pd.DataFrame()
    attrs = per[["split", "system_id", "trace_km", "region_name", "n_px", "length_bin"]].drop_duplicates(["split", "system_id"])
    w = w.merge(attrs, on=["split", "system_id"], how="left")
    ra, rb = w[a] >= k_thr, w[b] >= k_thr
    w["class"] = np.select([ra & rb, ra & ~rb, ~ra & rb], ["both", f"{a}_only", f"{b}_only"], "neither")
    return w.rename(columns={a: f"recall_{a}", b: f"recall_{b}"}).assign(frac=frac, policy=policy)


# =============================================================================== case-study windows
def _window(r, c, half, shape) -> tuple[int, int, int]:
    H, W = shape
    size = 2 * half
    return int(np.clip(r - half, 0, max(H - size, 0))), int(np.clip(c - half, 0, max(W - size, 0))), size


def case_windows(ctx, seg_cls: pd.DataFrame, a: str, b: str, frac: float, *, masks: dict | None = None,
                 sys_cls: pd.DataFrame | None = None, n_per_kind: int = 3, half_px: int = 60,
                 ens: str | None = None) -> pd.DataFrame:
    rows = []
    shape = ctx.data.shape
    k_e = f"k_{ens}" if ens and f"k_{ens}" in seg_cls else None

    def add(kind, s, source="oof", note=""):
        for r in s.head(n_per_kind).itertuples():
            r0, c0, size = _window(r.r_mid, r.c_mid, half_px, shape)
            rows.append({"kind": kind, "source": source, "id": int(getattr(r, "seg_id", getattr(r, "system_id", -1))),
                         "length_km": float(getattr(r, "length_km", getattr(r, "trace_km", np.nan))), "r0": r0,
                         "c0": c0, "size": size, "frac": frac, "model_a": a, "model_b": b, "ensemble": ens or "",
                         "note": note})

    s = seg_cls.sort_values("length_km", ascending=False)
    add("a_only", s[s["class"] == f"{a}_only"])
    add("b_only", s[s["class"] == f"{b}_only"])
    add("missed_by_both", s[s["class"] == "neither"])
    short = seg_cls[seg_cls.length_km < 2.0]
    if k_e:
        short = short.sort_values(k_e, ascending=False)
    add("short_fault", short, note="mapped trace < 2 km")
    if "share_intersection" in seg_cls:
        add("complex_junction", seg_cls.sort_values(["share_intersection", "length_km"], ascending=False),
            note="junction-rich mapped trace")
    if masks is not None:
        far = ctx.dist_to_labels_km() > 1.0
        lab, n = ndimage.label(masks[a] & masks[b] & far, structure=np.ones((3, 3), bool))
        if n:
            cnt = np.bincount(lab.ravel())[1:]
            com = ndimage.center_of_mass(np.ones_like(lab), lab, np.argsort(cnt)[::-1][:n_per_kind] + 1)
            fp = pd.DataFrame({"r_mid": [x[0] for x in com], "c_mid": [x[1] for x in com], "seg_id": -1,
                               "length_km": np.nan})
            add("fp_prone", fp, note="unmapped prediction selected by both models > 1 km from mapped faults "
                                     "(not a confirmed fault)")
    if sys_cls is not None and len(sys_cls):
        sm = ctx.splits.system_map
        for kind, sel in (("withheld_discovery", sys_cls[sys_cls["class"] == "both"]),
                          ("withheld_discovery_a_only", sys_cls[sys_cls["class"] == f"{a}_only"]),
                          ("withheld_discovery_b_only", sys_cls[sys_cls["class"] == f"{b}_only"])):
            for r in sel.sort_values("trace_km", ascending=False).head(n_per_kind).itertuples():
                rr, cc = np.nonzero(sm == r.system_id)
                if not len(rr):
                    continue
                r0, c0, size = _window(np.median(rr), np.median(cc), half_px, shape)
                rows.append({"kind": kind, "source": f"disc{int(r.split)}", "id": int(r.system_id),
                             "length_km": float(r.trace_km), "r0": r0, "c0": c0, "size": size, "frac": frac,
                             "model_a": a, "model_b": b, "ensemble": ens or "",
                             "note": f"withheld system of split {int(r.split)}"})
    return pd.DataFrame(rows)


def read_band_window(data, name: str, r0: int, c0: int, size: int) -> np.ndarray:
    """Windowed read of one raw band (never the full raster)."""
    import rasterio
    from rasterio.windows import Window

    i = data.band_names.index(name) + 1
    with rasterio.open(data.paths.raw("features")) as src:
        a = src.read(i, window=Window(c0, r0, size, size), masked=True)
    return np.ma.filled(a.astype(np.float32), np.nan)


def _source_maps(ctx, w: dict, maps: dict[str, np.ndarray], disc_maps: dict | None):
    """(score maps for a / b / ensemble, scored zone, truth / retained layers) for a window's source."""
    if w["source"] == "oof":
        return {m: maps.get(m) for m in (w["model_a"], w["model_b"], w["ensemble"]) if m}, None
    d = int(w["source"][4:])
    out = {}
    for m in (w["model_a"], w["model_b"], w["ensemble"]):
        if not m:
            continue
        p = (disc_maps or {}).get((m, d))
        out[m] = p if p is not None else ctx.discovery_pred(m, d)
    return out, d


def fig_case_study(ctx, w: dict, maps: dict[str, np.ndarray], *, disc_maps: dict | None = None,
                   context_band: str | None = "det_elev", geophysics_band: str | None = None):
    """Six panels on one window: geophysics, mapped faults, a, b, ensemble, agreement at the window's budget."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    from ..geo_eval import _hillshade
    from ..p3.budget import binary_mask
    from ..phase4_eval import colors_for

    r0, c0, size = int(w["r0"]), int(w["c0"]), int(w["size"])
    sl = (slice(r0, r0 + size), slice(c0, c0 + size))
    a, b, e = w["model_a"], w["model_b"], w.get("ensemble") or ""
    score, d = _source_maps(ctx, w, maps, disc_maps)
    if d is None:
        masks = {m: budget_masks(ctx, p, float(w["frac"])) for m, p in score.items() if p is not None}
        withheld = None
    else:
        ez = ctx.splits.eval_zone(d) & ctx.data.valid
        masks = {m: binary_mask(np.where(ez, p, np.nan), ez, None, float(w["frac"])) for m, p in score.items()
                 if p is not None}
        withheld = ctx.splits.withheld(d)
    col = colors_for([a, b] + ([e] if e else []))
    fig, axs = plt.subplots(2, 3, figsize=(11.4, 7.8), constrained_layout=True)
    ax = axs[0, 0]
    names = ctx.data.band_names
    if context_band and context_band in names:
        ax.imshow(_hillshade(np.nan_to_num(read_band_window(ctx.data, context_band, r0, c0, size))), cmap="gray",
                  interpolation="nearest")
    if geophysics_band and geophysics_band in names:
        g = read_band_window(ctx.data, geophysics_band, r0, c0, size)
        lo, hi = np.nanpercentile(g, [2, 98]) if np.isfinite(g).any() else (0, 1)
        ax.imshow(g, cmap="viridis", vmin=lo, vmax=hi, alpha=0.55 if context_band in names else 1.0,
                  interpolation="nearest")
    ax.set_title(f"context: {geophysics_band or ''} {'over ' + context_band if context_band in names else ''}",
                 fontsize=8, loc="left")
    lab = ctx.data.labels[sl]
    ax = axs[0, 1]
    ax.imshow(np.zeros(lab.shape), cmap="Greys", vmin=0, vmax=1)
    if withheld is None:
        ax.imshow(np.ma.masked_where(~lab, lab), cmap=ListedColormap(["#000000"]), interpolation="nearest")
        ax.set_title("mapped faults", fontsize=8, loc="left")
    else:
        wh = withheld[sl]
        ax.imshow(np.ma.masked_where(~(lab & ~wh), lab), cmap=ListedColormap(["#000000"]), interpolation="nearest")
        ax.imshow(np.ma.masked_where(~wh, wh), cmap=ListedColormap(["#e34948"]), interpolation="nearest")
        ax.set_title(f"mapped faults (red = withheld, split {d})", fontsize=8, loc="left")
    for ax, m in zip((axs[0, 2], axs[1, 0], axs[1, 1]), (a, b, e)):
        if not m or score.get(m) is None:
            ax.axis("off")
            continue
        ax.imshow(score[m][sl], cmap="magma", vmin=np.nanpercentile(score[m][sl], 1) if np.isfinite(score[m][sl]).any() else 0,
                  vmax=np.nanpercentile(score[m][sl], 99.5) if np.isfinite(score[m][sl]).any() else 1, interpolation="nearest")
        ax.contour(masks[m][sl].astype(float), levels=[0.5], colors=[col[m]], linewidths=0.9)
        ax.contour(lab.astype(float), levels=[0.5], colors="#ffffff", linewidths=0.4)
        ax.set_title(f"{m} score (outline: top-{float(w['frac']):.0%} mask)", fontsize=8, loc="left")
    ax = axs[1, 2]
    code = np.zeros(lab.shape, np.int8)
    if a in masks and b in masks:
        A, B = masks[a][sl], masks[b][sl]
        code[A & ~B], code[B & ~A], code[A & B] = 1, 2, 3
    code[lab] = 4
    ax.imshow(code, cmap=ListedColormap(["#f4f3ef", col[a], col[b], "#3a3a38", "#000000"]), vmin=0, vmax=4,
              interpolation="nearest")
    ax.legend(handles=[Patch(color=col[a], label=f"only {a}"), Patch(color=col[b], label=f"only {b}"),
                       Patch(color="#3a3a38", label="both (unmapped prediction)"), Patch(color="#000000", label="mapped")],
              fontsize=6, frameon=False, loc="lower left")
    ax.set_title("agreement at the matched budget", fontsize=8, loc="left")
    km = size * ctx.data.grid.pixel_km
    for ax in axs.ravel():
        ax.set_xticks([]), ax.set_yticks([])
    fig.suptitle(f"{w['kind']} | {w['source']} | id {w['id']} | rows {r0}-{r0 + size}, cols {c0}-{c0 + size} "
                 f"({km:.0f} km window) | {w.get('note', '')}", fontsize=8.5, x=0.01, ha="left")
    return fig


def fig_short_fault_analysis(rec: pd.DataFrame, cls_summary: pd.DataFrame, models: list[str]):
    """Left: weighted recall by length bin (matched budget). Right: recovery classes by length bin (km)."""
    import matplotlib.pyplot as plt

    from .comparative import _colors, _style

    col = _colors(models)
    lb = rec[rec.stratum == "length_bin"]
    order = ["<1", "1-2", "2-5", "5-10", ">10"]
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.9), constrained_layout=True)
    for m in models:
        g = lb[lb.model == m].set_index("value").reindex(order)
        axs[0].plot(range(len(order)), g.weighted_recall, marker="o", color=col[m], label=m, lw=1.5)
    axs[0].set_xticks(range(len(order)), [f"{o} km" for o in order], fontsize=8)
    _style(axs[0], "mapped segment length", "weighted recall", "Recall by fault length (matched budget)")
    axs[0].legend(fontsize=7, frameon=False)
    cs = cls_summary[cls_summary.stratum == "length_bin"] if len(cls_summary) else cls_summary
    if len(cs):
        piv = cs.pivot_table(index="value", columns="class", values="trace_km", aggfunc="sum").reindex(order).fillna(0)
        bottom = np.zeros(len(piv))
        palette = {"both": "#3a3a38", "neither": "#c9c8c2"}
        for c in piv.columns:
            axs[1].bar(range(len(piv)), piv[c], bottom=bottom, label=c,
                       color=palette.get(c, col.get(c.replace("_only", ""), "#8a8985")))
            bottom += piv[c].to_numpy()
        axs[1].set_xticks(range(len(piv)), [f"{o} km" for o in piv.index], fontsize=8)
        axs[1].legend(fontsize=7, frameon=False)
    _style(axs[1], "mapped segment length", "trace km", "Recovered by both / one / neither")
    data = lb[lb.model.isin(models)][["model", "value", "weighted_recall", "n_px", "trace_km"]]
    return fig, data
