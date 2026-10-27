"""P6 full GPU figures: inputs, references, predictions, confidence, differences, held-out errors, zone DTI, cost,
continuity. Every figure is saved as PNG + SVG with its plotting data (CSV); maps are down-sampled for display only.

Labels appear only in the evaluation panels (held-out discovery units, the reference fault map); no figure output
feeds a model, a threshold or a choice.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import final_submission as FS

PX_KM = 0.1


def _down(a: np.ndarray, step: int, how: str = "max") -> np.ndarray:
    from scipy import ndimage

    a = np.asarray(a)
    if step <= 1:
        return a
    if how == "max":
        return ndimage.maximum_filter(np.nan_to_num(a.astype(np.float32), nan=0.0), size=step)[::step, ::step]
    return a[::step, ::step]


def _step(shape, max_px: int) -> int:
    return max(1, int(np.ceil(max(shape) / int(max_px))))


def _read(path):
    import rasterio

    with rasterio.open(path) as s:
        return s.read(1).astype(np.float32)


def inputs(fg) -> None:
    """Geophysical input layers (robust 2-98 % stretch) with the mapped-fault reference overlaid."""
    import matplotlib.pyplot as plt

    data = fg.p6.ctx.data
    st = _step(data.shape, fg.cfg["figures"]["max_px"])
    bands = {k: b for k, b in fg.cfg["figures"]["bands"].items() if b in data.band_names}
    kn = _down(data.labels, st) > 0
    fig, ax = plt.subplots(1, len(bands) + 1, figsize=(5.2 * (len(bands) + 1), 5.6))
    rows = []
    for i, (k, b) in enumerate(bands.items()):
        a = data.read_band(b)[::st, ::st]
        lo, hi = np.nanpercentile(a, [2, 98]) if np.isfinite(a).any() else (0, 1)
        ax[i].imshow(a, cmap="gray" if k == "terrain" else "viridis", vmin=lo, vmax=hi)
        ax[i].contour(kn, levels=[0.5], colors="r", linewidths=0.3)
        ax[i].set(title=f"{k} ({b})", xticks=[], yticks=[])
        rows.append({"layer": k, "band": b, "p2": float(lo), "p98": float(hi), "step_px": st})
    ax[-1].imshow(kn, cmap="Greys")
    ax[-1].set(title=f"mapped faults (reference, {int(np.asarray(data.labels).sum())} px)", xticks=[], yticks=[])
    fg.save_fig(fig, "input_layers", pd.DataFrame(rows), "figures")


def predictions(fg) -> None:
    """Winner / baseline binary maps, the winner's zone-stitched score (confidence = within-zone rank), and the
    difference map (winner only / baseline only / both)."""
    import matplotlib.pyplot as plt

    data = fg.p6.ctx.data
    st = _step(data.shape, fg.cfg["figures"]["max_px"])
    idx = fg.load_table("submission_index", "submissions")
    if not len(idx):
        return
    maps = {}
    for r in idx.to_dict("records"):
        p = fg.root / "submissions" / "rasters" / f"{r['submission'].rsplit('.', 1)[0]}_maps.tif"
        if p.exists():
            import rasterio

            with rasterio.open(p) as s:
                maps[r["role"]] = {"score": s.read(1), "q": s.read(2)}
    if "winner" not in maps:
        return
    w = maps["winner"]
    b = maps.get("baseline")
    n = 3 + (b is not None)
    fig, ax = plt.subplots(1, n, figsize=(5.4 * n, 5.6))
    ax[0].imshow(np.where(_down(data.valid, st, "nn"), _down(w["score"], st, "nn"), np.nan), cmap="magma")
    ax[0].set(title="winner score (within-zone rank, 0-1)", xticks=[], yticks=[])
    ax[1].imshow(_down(w["q"] > 0, st), cmap="Reds")
    ax[1].set(title=f"winner binary ({int((w['q'] > 0).sum())} px)", xticks=[], yticks=[])
    kn = _down(data.labels, st) > 0
    ax[2].imshow(kn, cmap="Greys", alpha=0.6)
    ax[2].imshow(np.ma.masked_where(~(_down(w["q"] > 0, st) > 0), np.ones(kn.shape)), cmap="autumn", alpha=0.9)
    ax[2].set(title="winner over mapped faults", xticks=[], yticks=[])
    rows = [{"map": "winner", "positive_px": int((w["q"] > 0).sum())}]
    if b is not None:
        a1, a0 = w["q"] > 0, b["q"] > 0
        D = np.zeros(a1.shape, np.int8)
        D[a1 & ~a0], D[a0 & ~a1], D[a1 & a0] = 1, 2, 3
        ax[3].imshow(_down(D, st), cmap="tab10", vmin=0, vmax=9, interpolation="nearest")
        ax[3].set(title="difference: 1 winner only, 2 baseline only, 3 both", xticks=[], yticks=[])
        rows += [{"map": "winner_only", "positive_px": int((D == 1).sum())},
                 {"map": "baseline_only", "positive_px": int((D == 2).sum())},
                 {"map": "both", "positive_px": int((D == 3).sum())}]
    fg.save_fig(fig, "prediction_maps", pd.DataFrame(rows), "figures")


def heldout_errors(fg) -> None:
    """Held-out discovery units: the winner's per-zone cell (chosen on the other zones) vs the withheld systems.
    hit = truth within 300 m of a prediction, miss = truth farther, FP = prediction farther than 300 m from truth
    (an approximation of the kernel for display; the numbers in the tables are the official components)."""
    import matplotlib.pyplot as plt
    from scipy import ndimage

    ms = fg.read_json("benchmark", "model_selection") or {}
    Z = fg.load_table("zone_metrics", "benchmark")
    M = fg.load_table("candidate_metrics", "benchmark")
    if not ms.get("winner") or not len(Z) or not len(M):
        return
    row = M[M.name == ms["winner"]].iloc[0].to_dict()
    kind = row["deploy"]
    if kind not in ("fusion", "cnn", "stack", "seeds"):
        return
    bst, p6 = fg.bst, fg.p6
    w_all = float(bst.sel[bst.prim][p6.cfg["probes"]["champion"]]["params_all"]["w"])
    z = Z[(Z.name == ms["winner"]) & (Z.task == bst.prim)]
    units = [u for u in bst.units() if u.task == bst.prim]
    fig, ax = plt.subplots(1, len(units), figsize=(6 * len(units), 6))
    ax = np.atleast_1d(ax)
    rows = []
    for i, u in enumerate(units):
        zr = z[z.zone == u.zone]
        if not len(zr):
            continue
        zr = zr.iloc[0]
        cell = {"candidate": zr.cell_candidate, "blend": float(zr.blend), "rep": zr.rep, "frac": float(zr.frac)}
        src_name, orig = FS.split_cell(cell)
        sd = FS.stack_dir(fg, src_name, orig) if kind == "stack" else None
        sc = json.loads((p6.ws.root / "seeds" / "candidate.json").read_text(encoding="utf-8")) if kind == "seeds" \
            else None
        s = FS.eval_score(fg, u, kind, cell, FS.zone_sources(fg, u, kind, cell, w_all, sd, sc), sc)
        q = bst.binary(u, s, cell["rep"], cell["frac"])
        truth = u.eval_truth()
        dq = ndimage.distance_transform_edt(~q) if q.any() else np.full(q.shape, np.inf)
        dt = ndimage.distance_transform_edt(~truth) if truth.any() else np.full(q.shape, np.inf)
        ex = u.exclude if u.exclude is not None else np.zeros(q.shape, bool)
        hit, miss, fp = truth & (dq <= 3), truth & (dq > 3), q & (dt > 3) & ~ex
        img = np.ones(q.shape + (3,), np.float32)
        img[u.ev] = 0.95
        img[ex] = (0.75, 0.75, 0.75)
        img[fp], img[miss], img[hit] = (0.95, 0.55, 0.1), (0.2, 0.35, 0.9), (0.1, 0.65, 0.2)
        ax[i].imshow(img, interpolation="nearest")
        ax[i].set(title=f"{u.key}: {cell['rep']} @ {cell['frac']:g} (held-out DTI {float(zr.dti):.3f})",
                  xticks=[], yticks=[])
        rows.append({"unit": u.key, "dti": float(zr.dti), "truth_px": int(truth.sum()), "hit_px": int(hit.sum()),
                     "miss_px": int(miss.sum()), "pred_px": int(q.sum()), "fp_far_px": int(fp.sum()), **cell})
        u.release()
    fig.suptitle("held-out errors: green hit, blue miss, orange FP > 300 m, grey known (masked)")
    fg.save_fig(fig, "heldout_errors", pd.DataFrame(rows), "figures")


def zone_dti(fg, top: int = 8) -> None:
    import matplotlib.pyplot as plt

    R = fg.load_table("candidate_rankings", "benchmark")
    Z = fg.load_table("zone_metrics", "benchmark")
    if not len(R) or not len(Z):
        return
    names = [n for n in R.name if n in set(Z.name)][:top]
    D = Z[Z.name.isin(names) & (Z.task == fg.bst.prim)].pivot_table(index="name", columns="zone", values="dti")
    D = D.reindex(names)
    fig, ax = plt.subplots(figsize=(max(7, 1.1 * len(names)), 4.8))
    x = np.arange(len(D))
    k = len(D.columns)
    for j, c in enumerate(D.columns):
        ax.bar(x + (j - (k - 1) / 2) * 0.8 / max(k, 1), D[c], width=0.8 / max(k, 1), label=f"zone {c}")
    ax.axhline(float(fg.cfg["selection"]["worst_floor"]), ls="--", c="k", lw=0.8, label="worst-zone floor")
    ax.set_xticks(x, [n[:18] for n in D.index], rotation=35, ha="right", fontsize=7)
    ax.set(ylabel="held-out disc_official DTI", title="per-zone discovery DTI (nested, single policy)")
    ax.legend(fontsize=7)
    fg.save_fig(fig, "zone_dti", D.reset_index(), "figures")


def cost(fg) -> None:
    """Nested discovery DTI against the producing stage's recorded seconds (measured stage time, not per model)."""
    import matplotlib.pyplot as plt

    M = fg.load_table("candidate_metrics", "benchmark")
    man = fg.read_json("benchmark", "experiment_manifest") or {}
    if not len(M):
        return
    sec = {k: v.get("stage_seconds") for k, v in (man.get("grids") or {}).items()}
    E = M[M.status == "EVALUATED"].assign(stage_seconds=lambda d: d.grid.map(sec))
    fig, ax = plt.subplots(figsize=(7, 5))
    for r in E.itertuples():
        if r.stage_seconds is None or not np.isfinite(float(r.stage_seconds or np.nan)):
            continue
        ax.scatter(float(r.stage_seconds), float(r.disc_dti), s=30)
        ax.annotate(str(r.name)[:16], (float(r.stage_seconds), float(r.disc_dti)), fontsize=6)
    ax.set_xscale("log")
    ax.set(xlabel="producing stage wall time, s (recorded manifest)", ylabel="nested disc_official DTI",
           title="performance vs recorded runtime")
    fg.save_fig(fig, "performance_vs_runtime", E[["name", "grid", "stage_seconds", "disc_dti"]], "figures")


def continuity(fg) -> None:
    """Segment lengths of the winner's full-region map (8-connected) + per-zone fragmentation of its held-out cells."""
    import matplotlib.pyplot as plt
    from scipy import ndimage

    idx = fg.load_table("submission_index", "submissions")
    Z = fg.load_table("zone_metrics", "benchmark")
    ms = fg.read_json("benchmark", "model_selection") or {}
    if not len(idx):
        return
    w = idx[idx.role == "winner"].iloc[0]
    p = fg.root / "submissions" / "rasters" / f"{w.submission.rsplit('.', 1)[0]}_maps.tif"
    if not p.exists():
        return
    import rasterio

    with rasterio.open(p) as s:
        q = s.read(2) > 0
    lab, n = ndimage.label(q, structure=np.ones((3, 3), int))
    sizes = np.bincount(lab.ravel())[1:] * PX_KM if n else np.array([])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    if sizes.size:
        ax[0].hist(sizes, bins=np.logspace(np.log10(max(sizes.min(), 0.1)), np.log10(sizes.max() + 0.1), 40))
        ax[0].set_xscale("log")
    ax[0].set(xlabel="segment length, km (pixel count x 0.1 km)", ylabel="segments",
              title=f"winner full map: {n} segments, {sizes.sum():.0f} km")
    z = Z[(Z.name == ms.get("winner")) & (Z.task == fg.bst.prim)] if len(Z) else Z
    if len(z) and "short_share" in z:
        ax[1].bar(z.zone.astype(str), z.short_share)
        ax[1].set(xlabel="held-out zone", ylabel="share of short segments (< 5 px)", title="fragmentation per zone")
    data = pd.DataFrame({"segment_km": sizes})
    fg.save_fig(fig, "continuity", data, "figures")
    fg.save_parquet(data, "winner_segments", "figures")


def run(fg) -> list[str]:
    done = []
    for f in (inputs, predictions, heldout_errors, zone_dti, cost, continuity):
        try:
            f(fg)
            done.append(f.__name__)
        except Exception as e:                                  # a figure never blocks the evidence tables
            fg.say(f"figure {f.__name__} skipped: {type(e).__name__}: {e}")
    return done
