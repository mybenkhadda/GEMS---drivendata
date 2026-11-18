"""Geological failure atlas (Phase I): automatically selected windows, 18 aligned layers, full-region maps, index.

Every panel of a window shares one extent in projected coordinates (EPSG:32611 metres from the affine transform),
one scale bar and, per layer, one colour scale across windows (robust 2-98 % limits computed once on the valid
footprint). Window statistics and the layer table are saved next to the figures.

Window categories (each chosen from per-tile statistics of one evaluation unit, label use = held-out evaluation
only, never fed back into a model):

best_detection, worst_detection, g14_strong_champion_weak, g14_weak_champion_strong, high_conf_false_positive,
missed_system, fragmented_system, isolated_structure, acquisition_artifact, regime_boundary

UNTESTED: written without execution.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

LAYERS = ("magnetic_anomaly", "rtp", "magnetic_gradient", "gravity_anomaly", "gravity_gradient", "terrain",
          "conductivity", "structural_index", "known_context", "withheld_faults", "g14_raw", "g14_representation",
          "champion", "false_positives", "false_negatives", "distance_to_withheld", "disagreement", "regime")
RAW_FOR = {"magnetic_anomaly": "mag_anom", "rtp": "rtp", "magnetic_gradient": "tmi_hg", "gravity_anomaly": "iso_grav_anom",
           "gravity_gradient": "iso_grav_anom_hg", "terrain": "det_elev", "conductivity": "cond_surf"}
CATEGORIES = ("best_detection", "worst_detection", "g14_strong_champion_weak", "g14_weak_champion_strong",
              "high_conf_false_positive", "missed_system", "fragmented_system", "isolated_structure",
              "acquisition_artifact", "regime_boundary")


def tile_stats(u, g14_q: np.ndarray, g14_s: np.ndarray, champ_q: np.ndarray | None, regimes: np.ndarray | None,
               coverage_edge: np.ndarray | None, tile_px: int = 128, pixel_m: float = 100.0) -> pd.DataFrame:
    """Per-tile statistics on one unit crop (all arrays cropped to the unit)."""
    sc = u.scorer
    supg = sc.support(np.asarray(g14_q, np.float64))
    supc = sc.support(np.asarray(champ_q, np.float64)) if champ_q is not None else None
    gr, gc = sc.g_rc
    truth = u.eval_truth()
    far = ndimage.distance_transform_edt(~truth) * pixel_m > 300 if truth.any() else np.ones(truth.shape, bool)
    fp = (g14_q > 0) & far & u.ev
    lab, _ = ndimage.label(g14_q > 0, structure=np.ones((3, 3), bool))
    tl, _ = ndimage.label(truth, structure=np.ones((3, 3), bool))
    H, W = truth.shape
    rows = []
    for t in range(0, H - tile_px + 1, tile_px // 2):
        for l in range(0, W - tile_px + 1, tile_px // 2):
            sl = (slice(t, t + tile_px), slice(l, l + tile_px))
            if u.ev[sl].mean() < 0.5:
                continue
            m = (gr >= t) & (gr < t + tile_px) & (gc >= l) & (gc < l + tile_px)
            r = {"top": t, "left": l, "truth_px": int(m.sum()),
                 "g14_recall": float(supg[m].mean()) if m.any() else np.nan,
                 "champ_recall": float(supc[m].mean()) if supc is not None and m.any() else np.nan,
                 "fp_px": int(fp[sl].sum()), "fp_conf": float(np.nan_to_num(g14_s[sl][fp[sl]]).mean()) if fp[sl].any() else 0.0,
                 "pred_components": int(len(np.unique(lab[sl])) - 1), "truth_components": int(len(np.unique(tl[sl])) - 1),
                 "edge_share": float(coverage_edge[sl].mean()) if coverage_edge is not None else np.nan,
                 "n_regimes": int(len(np.unique(regimes[sl][u.ev[sl]]))) if regimes is not None else 0}
            rows.append(r)
    u.release()
    return pd.DataFrame(rows)


def select_windows(ts: pd.DataFrame, per_category: int = 2) -> pd.DataFrame:
    """Rule per category; ties broken by position. Windows without evidence for a category are not forced in."""
    t = ts.copy()
    sel = []

    def take(cat, df, by, asc):
        df = df.dropna(subset=[by]) if by in df else df.iloc[0:0]
        for _, r in df.sort_values([by, "top", "left"], ascending=[asc, True, True]).head(per_category).iterrows():
            sel.append({**r.to_dict(), "category": cat, "criterion": f"{by} {'min' if asc else 'max'}"})

    has_t = t[t.truth_px >= 20]
    take("best_detection", has_t, "g14_recall", False)
    take("worst_detection", has_t, "g14_recall", True)
    if has_t.champ_recall.notna().any():
        has_t = has_t.assign(diff=has_t.g14_recall - has_t.champ_recall)
        take("g14_strong_champion_weak", has_t, "diff", False)
        take("g14_weak_champion_strong", has_t, "diff", True)
    take("high_conf_false_positive", t[t.fp_px >= 20], "fp_conf", False)
    take("missed_system", has_t[has_t.g14_recall < 0.05], "truth_px", False)
    take("fragmented_system", t[t.truth_components >= 1], "pred_components", False)
    take("isolated_structure", t[(t.truth_components == 1) & (t.truth_px >= 20)], "g14_recall", True)
    if t.edge_share.notna().any():
        take("acquisition_artifact", t[t.fp_px >= 10], "edge_share", False)
    take("regime_boundary", t[t.n_regimes >= 2], "n_regimes", False)
    return pd.DataFrame(sel)


def layer_stack(data, u, maps: dict, regimes: np.ndarray | None, structural: np.ndarray | None,
                context: np.ndarray, withheld: np.ndarray) -> dict:
    """Full-grid layers (None when unavailable). ``maps``: g14_raw, g14_representation, champion (full grids)."""
    L = {}
    for k, b in RAW_FOR.items():
        L[k] = data.band(b) if data.has(b) else None
    L["structural_index"] = structural
    L["known_context"] = context.astype(np.float32)
    L["withheld_faults"] = withheld.astype(np.float32)
    L["g14_raw"] = maps.get("g14_raw")
    L["g14_representation"] = maps.get("g14_representation")
    L["champion"] = maps.get("champion")
    rep = maps.get("g14_representation")
    if rep is not None:
        d = ndimage.distance_transform_edt(~withheld) if withheld.any() else np.full(withheld.shape, np.inf)
        L["false_positives"] = ((np.nan_to_num(rep) > 0) & (d > 3)).astype(np.float32)
        near = ndimage.distance_transform_edt(~(np.nan_to_num(rep) > 0)) <= 3 if (np.nan_to_num(rep) > 0).any() else \
            np.zeros(withheld.shape, bool)
        L["false_negatives"] = (withheld & ~near).astype(np.float32)
        L["distance_to_withheld"] = (np.minimum(d, 50) * abs(data.transform.a) / 1000).astype(np.float32)
    if maps.get("g14_raw") is not None and maps.get("champion") is not None:
        L["disagreement"] = (np.nan_to_num(maps["g14_raw"]) - np.nan_to_num(maps["champion"])).astype(np.float32)
    L["regime"] = None if regimes is None else regimes.astype(np.float32)
    return L


def color_limits(layers: dict, valid: np.ndarray) -> dict:
    out = {}
    for k, a in layers.items():
        if a is None:
            continue
        v = np.asarray(a, np.float32)[valid]
        v = v[np.isfinite(v)]
        out[k] = (float(np.quantile(v, 0.02)), float(np.quantile(v, 0.98))) if v.size else (0.0, 1.0)
    return out


def render_window(layers: dict, limits: dict, data, top: int, left: int, size: int, title: str, out: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = data.transform
    x0, y0 = t.c + left * t.a, t.f + top * t.e
    extent = (x0 / 1000, (x0 + size * t.a) / 1000, (y0 + size * t.e) / 1000, y0 / 1000)
    names = [k for k in LAYERS if layers.get(k) is not None]
    nc = 6
    nr = -(-len(names) // nc)
    fig, axes = plt.subplots(nr, nc, figsize=(3 * nc, 3 * nr), squeeze=False)
    sl = (slice(max(top, 0), top + size), slice(max(left, 0), left + size))
    for ax, k in zip(axes.ravel(), names):
        a = np.asarray(layers[k], np.float32)[sl]
        lo, hi = limits.get(k, (None, None))
        cmap = {"known_context": "Greys", "withheld_faults": "Reds", "false_positives": "Oranges",
                "false_negatives": "Blues", "disagreement": "RdBu_r", "regime": "tab10"}.get(k, "viridis")
        im = ax.imshow(a, extent=extent, vmin=lo, vmax=hi, cmap=cmap, interpolation="nearest")
        ax.set_title(k, fontsize=8)
        ax.tick_params(labelsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046)
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    axes[0, 0].set_xlabel("easting (km, EPSG:32611)", fontsize=6)
    axes[0, 0].set_ylabel("northing (km)", fontsize=6)
    fig.suptitle(f"{title} | window {size * abs(t.a) / 1000:.1f} km | top-left ({top}, {left}) px", fontsize=10)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def full_region(layers: dict, limits: dict, data, out_dir: Path, names=("g14_raw", "g14_representation", "champion",
                                                                       "false_positives", "false_negatives")) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = data.transform
    H, W = data.shape
    ext = (t.c / 1000, (t.c + W * t.a) / 1000, (t.f + H * t.e) / 1000, t.f / 1000)
    paths = []
    for k in names:
        if layers.get(k) is None:
            continue
        fig, ax = plt.subplots(figsize=(8, 9))
        lo, hi = limits.get(k, (None, None))
        a = np.where(data.valid, np.asarray(layers[k], np.float32), np.nan)
        im = ax.imshow(a[::4, ::4], extent=ext, vmin=lo, vmax=hi, cmap="viridis")
        fig.colorbar(im, ax=ax, fraction=0.046)
        ax.set(title=f"{k} (full region, 400 m display)", xlabel="easting (km)", ylabel="northing (km)")
        p = out_dir / f"full__{k}.png"
        fig.tight_layout()
        fig.savefig(p, dpi=110)
        plt.close(fig)
        paths.append(p)
    return paths


def build_atlas(data, units_layers: list[tuple], out_dir: str | Path, tile_px: int = 128, per_category: int = 2) -> Path:
    """``units_layers``: [(unit, layers dict (full grids), tile statistics DataFrame)]. Writes figures, tables and
    ``index.html`` / ``index.md``; returns the index path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    entries = []
    for u, layers, ts in units_layers:
        limits = color_limits(layers, data.valid)
        sel = select_windows(ts, per_category)
        sel.to_csv(out / f"windows_{u.task}_{u.zone}.csv", index=False)
        ts.to_csv(out / f"tiles_{u.task}_{u.zone}.csv", index=False)
        r0, c0 = u.sl[0].start, u.sl[1].start
        for _, r in sel.iterrows():
            p = out / f"{u.task}_{u.zone}__{r.category}__{int(r.top)}_{int(r.left)}.png"
            render_window(layers, limits, data, r0 + int(r.top), c0 + int(r.left), tile_px,
                          f"{u.task} {u.zone}: {r.category} ({r.criterion})", p)
            entries.append({"unit": f"{u.task}_{u.zone}", "category": r.category, "figure": p.name,
                            "criterion": r.criterion, "truth_px": r.truth_px, "g14_recall": r.g14_recall,
                            "champ_recall": r.get("champ_recall"), "fp_px": r.fp_px})
        for p in full_region(layers, limits, data, out):
            entries.append({"unit": f"{u.task}_{u.zone}", "category": "full_region", "figure": p.name})
    idx = pd.DataFrame(entries)
    idx.to_csv(out / "atlas_index.csv", index=False)
    md = ["# G14.1 geological failure atlas", "", "Generated from real artifacts by the G14.1 pipeline. Offline values "
          "are proxies on held-out units, never leaderboard scores.", ""]
    html = ["<html><head><meta charset='utf-8'><title>G14.1 failure atlas</title></head><body>",
            "<h1>G14.1 geological failure atlas</h1>"]
    for cat, g in idx.groupby("category") if len(idx) else []:
        md.append(f"## {cat}\n")
        html.append(f"<h2>{cat}</h2>")
        for _, e in g.iterrows():
            md.append(f"* [{e.figure}]({e.figure}) ({e.unit})")
            html.append(f"<figure><img src='{e.figure}' width='100%'><figcaption>{e.unit}</figcaption></figure>")
    (out / "index.md").write_text("\n".join(md))
    (out / "index.html").write_text("\n".join(html + ["</body></html>"]))
    return out / "index.html"
