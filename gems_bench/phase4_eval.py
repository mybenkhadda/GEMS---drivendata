"""Phase-4 evaluation: one scorecard for every model, built only from the Phase-3 scoring code.

* Geographic: per-fold binary top-a masks on the Phase-2 folds, cross-fitted area (budget.crossfit_area).
* Pseudo-discovery: discovery3 (withheld systems, exact n_px, policies ``excl`` and ``retained_fp``),
  cross-fitted area over the 3 splits. Geographic and discovery numbers are never mixed in one column.
* Thin structures: every OOF label pixel's distance to the model's per-fold top-a mask (exact TP support of
  a binary mask: k = max(0, 1 - d / 300 m)), stratified by Phase-1 segment length / sinuosity / junction
  attributes and by skeleton endpoint / branch-point proximity.
* Complementarity: rank correlation, mask Jaccard, unique recovered km, recall on faults missed by a
  reference, disagreement far from labels, false-positive (far) area, mask shape (lineament coherence).
"""
from __future__ import annotations

import time
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import ndimage

from .p3 import budget, discovery3
from .p3.context import model_color

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948",
           "#0099a8", "#9b5d3a", "#5c6f2e", "#b0479b"]


def colors_for(names) -> dict[str, str]:
    """Known Phase-2/3 models keep their colour; new models take the remaining palette slots in order."""
    out, used = {}, set()
    for n in names:
        c = model_color(n)
        if c != "#8a8985":
            out[n] = c
            used.add(c)
    free = [c for c in PALETTE if c not in used]
    for n in names:
        if n not in out:
            out[n] = free.pop(0) if free else "#8a8985"
    return out


# ---------------------------------------------------------------------------
# label-pixel table (scoring units) with segment and structure attributes
# ---------------------------------------------------------------------------
def label_pixel_table(ctx, near_m: float = 500.0) -> pd.DataFrame:
    """One row per mapped-fault pixel inside a geographic fold zone, with segment and skeleton attributes."""
    if "_lpt" in ctx._maps:
        return ctx._maps["_lpt"]
    from .folds import skeleton_px

    data, fm = ctx.data, ctx.folds.fold_map
    lab = data.labels & data.valid & (fm >= 0)
    rr, cc = np.nonzero(lab)
    df = pd.DataFrame({"r": rr, "c": cc, "fold": fm[rr, cc].astype(int)})
    df["seg_id"] = ctx.seg[rr, cc] if ctx.seg is not None else -1
    if ctx.seg_tab is not None:
        cols = [c for c in ("seg_id", "length_km", "length_bin", "sinuosity_bin", "junction", "primary_archetype", "region")
                if c in ctx.seg_tab]
        df = df.merge(ctx.seg_tab[cols], on="seg_id", how="left")
    sk = skeleton_px(data.labels)
    nb = ndimage.convolve(sk.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant") - 1
    px = (data.grid.pixel_km_y * 1000, data.grid.pixel_km_x * 1000)
    d_end = ndimage.distance_transform_edt(~(sk & (nb == 1)), sampling=px)
    d_br = ndimage.distance_transform_edt(~(sk & (nb >= 3)), sampling=px)
    df["near_endpoint"] = d_end[df.r, df.c] <= near_m
    df["near_branch"] = d_br[df.r, df.c] <= near_m
    df["structure"] = np.where(df.near_branch, "intersection", np.where(df.near_endpoint, "endpoint", "interior"))
    if "length_km" in df:
        df["short_fault"] = np.where(df.length_km < 2.0, "<2 km", np.where(df.length_km.notna(), ">=2 km", "unassigned"))
    if "sinuosity_bin" in df:
        df["curved"] = df.sinuosity_bin.astype(str).isin(["curved", "very_curved"]).map({True: "curved", False: "straight_or_gentle"})
    ctx._maps["_lpt"] = df
    return df


def fold_masks(ctx, p: np.ndarray, frac: float) -> np.ndarray:
    """Union of per-fold top-frac binary masks (the scored object of the geographic curves)."""
    return budget.binary_mask(p, ctx.data.valid, ctx.folds.fold_map, frac)


def mask_distance(ctx, mask: np.ndarray) -> np.ndarray:
    """Distance (m) from each label pixel to the mask restricted to the pixel's own fold zone."""
    lpt = label_pixel_table(ctx)
    g = ctx.data.grid
    px = (g.pixel_km_y * 1000, g.pixel_km_x * 1000)
    d = np.full(len(lpt), np.inf)
    for k in ctx.fold_ids:
        sel = (lpt.fold == k).to_numpy()
        mk = mask & (ctx.folds.fold_map == k)
        if sel.any() and mk.any():
            dist = ndimage.distance_transform_edt(~mk, sampling=px)
            d[sel] = dist[lpt.r.to_numpy()[sel], lpt.c.to_numpy()[sel]]
    return d


STRATA = ("all", "length_bin", "short_fault", "curved", "junction", "structure", "primary_archetype", "fold")


def recovery_table(ctx, maps: dict[str, np.ndarray], frac: float, strata=STRATA, masks: dict | None = None,
                   radius_m: float = 300.0) -> tuple[pd.DataFrame, dict]:
    """Long table (model, stratum, value): n_px, trace_km, weighted recall, recall@100/200/300 at a matched budget."""
    lpt = label_pixel_table(ctx)
    masks = masks or {m: fold_masks(ctx, p, frac) for m, p in maps.items()}
    K, rows = {}, []
    for m in maps:
        d = mask_distance(ctx, masks[m])
        k = np.clip(1 - d / radius_m, 0, 1)
        K[m] = k
        t = lpt.assign(k=k, h100=d <= 100, h200=d <= 200, h300=d <= 300, all="all")
        for col in strata:
            if col not in t:
                continue
            for val, g in t.groupby(col, observed=True, dropna=False):
                km = g.drop_duplicates("seg_id").length_km.sum() if "length_km" in g else np.nan
                rows.append({"model": m, "frac": frac, "stratum": col, "value": str(val), "n_px": len(g), "trace_km": km,
                             "weighted_recall": float(g.k.mean()), "recall@100m": float(g.h100.mean()),
                             "recall@200m": float(g.h200.mean()), "recall@300m": float(g.h300.mean())})
    return pd.DataFrame(rows), K


# ---------------------------------------------------------------------------
# scorecard
# ---------------------------------------------------------------------------
def geo_score(ctx, maps: dict[str, np.ndarray], frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    curves = budget.geo_curves(ctx, list(maps), maps=maps)
    cf = budget.crossfit_area(curves).set_index("model")
    pool = budget.pool_models(curves)
    rows = []
    for m in maps:
        at = pool[(pool.model == m) & np.isclose(pool.target_frac, frac)]
        at = at.iloc[0] if len(at) else None
        r = {"model": m, "geo_dti": cf.loc[m, "dti_crossfit_area"], "geo_fold_min": cf.loc[m, "fold_min"],
             "geo_fold_max": cf.loc[m, "fold_max"], "geo_frac": cf.loc[m, "frac_all_folds"],
             **{c: cf.loc[m, c] for c in cf.columns if c.startswith("dti_fold")}}
        if at is not None:
            r.update({"geo_recall300_at_match": at["recall@300m"], "geo_far_share_at_match": at["far_area_km2"] / max(at["area_km2"], 1e-9),
                      "geo_dti_at_match": at["dti"]})
        rows.append(r)
    return pd.DataFrame(rows), curves


def discovery_score(ctx, models: list[str], frac: float, policies=("excl", "retained_fp")):
    models = [m for m in models if ctx.has_discovery(m)]
    if not models:
        return pd.DataFrame(), pd.DataFrame(), {}
    dc, objs = discovery3.discovery_curves(ctx, models, policies=policies, variants=("model",))
    ds = discovery3.discovery_summary(dc)
    cf = ds[ds.row == "crossfit_area"]
    rows = []
    for m in models:
        r = {"model": m}
        for pol in policies:
            s = cf[(cf.model == m) & (cf.policy == pol)]
            if len(s):
                s = s.iloc[0]
                r.update({f"disc_dti_{pol}": s.dti, f"disc_fold_min_{pol}": s.fold_min, f"disc_frac_{pol}": s.target_frac})
            mt = ds[(ds.row == "matched_budget") & (ds.model == m) & (ds.policy == pol) & np.isclose(ds.target_frac, frac)]
            if len(mt):
                r[f"disc_recall300_at_match_{pol}"] = float(mt.iloc[0]["recall@300m"])
        rows.append(r)
    return pd.DataFrame(rows), ds, objs


def scorecard(ctx, maps: dict[str, np.ndarray], disc_models: list[str], frac: float) -> dict:
    """Wide one-row-per-model table + long tables; geographic and discovery columns kept separate."""
    t0 = time.perf_counter()
    geo, curves = geo_score(ctx, maps, frac)
    disc, ds, objs = discovery_score(ctx, disc_models, frac)
    rec, K = recovery_table(ctx, maps, frac)
    short = rec[(rec.stratum == "short_fault") & (rec.value == "<2 km")].set_index("model").weighted_recall.rename("short_recall_w")
    lb = rec[rec.stratum == "length_bin"].pivot_table(index="model", columns="value", values="weighted_recall")
    lb.columns = [f"recall_w_len_{c}" for c in lb.columns]
    st = rec[rec.stratum == "structure"].pivot_table(index="model", columns="value", values="weighted_recall")
    st.columns = [f"recall_w_{c}" for c in st.columns]
    cu = rec[(rec.stratum == "curved") & (rec.value == "curved")].set_index("model").weighted_recall.rename("recall_w_curved")
    wide = geo.set_index("model").join([short, lb, st, cu])
    if len(disc):
        wide = wide.join(disc.set_index("model"), how="left")
    print(f"scorecard: {len(maps)} maps, {len(disc)} discovery models in {time.perf_counter() - t0:.0f}s")
    return {"score": wide.reset_index(), "geo_curves": curves, "disc_summary": ds, "disc_objs": objs,
            "recovery": rec, "support": K}


def paired_folds(score: pd.DataFrame, a: str, b: str) -> dict:
    s = score.set_index("model")
    fc = [c for c in s.columns if c.startswith("dti_fold")]
    d = (s.loc[a, fc] - s.loc[b, fc]).astype(float)
    return {"folds_beating": int((d > 0).sum()), "fold_deltas": d.round(4).tolist()}


def comparison_rows(score: pd.DataFrame, pairs, label: str = "") -> pd.DataFrame:
    """Candidate - control deltas on every scorecard metric present for both."""
    s = score.set_index("model")
    rows = []
    for a, b in pairs:
        if a not in s.index or b not in s.index:
            continue
        r = {"candidate": a, "control": b, "set": label, **paired_folds(score, a, b)}
        for c in ("geo_dti", "geo_fold_min", "geo_recall300_at_match", "short_recall_w", "recall_w_len_<1", "recall_w_len_1-2",
                  "recall_w_len_5-10", "recall_w_len_>10", "recall_w_curved", "recall_w_endpoint", "recall_w_intersection",
                  "disc_dti_excl", "disc_dti_retained_fp", "geo_far_share_at_match"):
            if c in s.columns:
                r[f"{c}_cand"], r[f"{c}_ctrl"] = s.loc[a, c], s.loc[b, c]
                r[f"delta_{c}"] = s.loc[a, c] - s.loc[b, c]
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# complementarity
# ---------------------------------------------------------------------------
def mask_shape_stats(mask: np.ndarray, min_px: int = 5, elong_thr: float = 3.0, pixel_km2: float = 0.01) -> dict:
    """Connected components of a top mask: count, share in the largest, elongated-area share (lineament coherence)."""
    lab, n = ndimage.label(mask, structure=np.ones((3, 3), bool))
    if n == 0:
        return {"n_components": 0}
    rr, cc = np.nonzero(lab)
    L = lab[rr, cc]
    cnt = np.bincount(L, minlength=n + 1)[1:].astype(float)
    sr, sc = np.bincount(L, rr, n + 1)[1:], np.bincount(L, cc, n + 1)[1:]
    srr, scc, src = np.bincount(L, rr * rr, n + 1)[1:], np.bincount(L, cc * cc, n + 1)[1:], np.bincount(L, rr * cc, n + 1)[1:]
    mr, mc = sr / cnt, sc / cnt
    vrr, vcc, vrc = srr / cnt - mr ** 2, scc / cnt - mc ** 2, src / cnt - mr * mc
    tr, det = vrr + vcc, vrr * vcc - vrc ** 2
    disc = np.sqrt(np.clip(tr ** 2 / 4 - det, 0, None))
    l1, l2 = tr / 2 + disc, np.clip(tr / 2 - disc, 1e-3, None)
    elong = np.sqrt(l1 / l2)
    big = cnt >= min_px
    area = cnt * pixel_km2
    return {"n_components": int(n), "n_components_ge_min": int(big.sum()),
            "largest_share": float(area.max() / area.sum()),
            "elongated_area_share": float(area[big & (elong >= elong_thr)].sum() / max(area.sum(), 1e-9)),
            "median_elongation_area_w": float(np.median(np.repeat(elong[big], np.maximum(cnt[big].astype(int), 1)))) if big.any() else np.nan}


def complementarity(ctx, maps: dict[str, np.ndarray], frac: float, refs=("G1", "CNN_A_dti"), sample: int = 200_000,
                    support: dict | None = None, masks: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(pairwise table, per-model table). ``support`` = per-label-pixel TP support from ``recovery_table``."""
    from scipy.stats import spearmanr

    v = ctx.data.valid
    lpt = label_pixel_table(ctx)
    masks = masks or {m: fold_masks(ctx, p, frac) for m, p in maps.items()}
    if support is None:
        support = {m: np.clip(1 - mask_distance(ctx, masks[m]) / 300.0, 0, 1) for m in maps}
    far = ctx.dist_to_labels_km() > 1.0
    idx = np.flatnonzero((v & (ctx.folds.fold_map >= 0)).reshape(-1))
    idx = np.random.default_rng(ctx.cfg.seed).choice(idx, min(sample, idx.size), replace=False)
    seg = lpt.assign(**{f"k_{m}": support[m] for m in maps}).groupby("seg_id").agg(
        length_km=("length_km", "first"), **{f"k_{m}": (f"k_{m}", "mean") for m in maps}) if "length_km" in lpt else None
    km2 = ctx.data.grid.pixel_km_x * ctx.data.grid.pixel_km_y
    pair_rows = []
    for a, b in combinations(list(maps), 2):
        pa, pb = maps[a].reshape(-1)[idx], maps[b].reshape(-1)[idx]
        ok = np.isfinite(pa) & np.isfinite(pb)
        ma, mb = masks[a], masks[b]
        r = {"model_a": a, "model_b": b, "spearman_oof": float(spearmanr(pa[ok], pb[ok]).correlation),
             "jaccard_top_mask": float((ma & mb).sum() / max((ma | mb).sum(), 1)),
             "far_only_a_km2": float((ma & ~mb & far).sum() * km2), "far_only_b_km2": float((mb & ~ma & far).sum() * km2),
             "far_both_km2": float((ma & mb & far).sum() * km2)}
        if seg is not None:
            ra, rb = seg[f"k_{a}"] >= 0.5, seg[f"k_{b}"] >= 0.5
            r.update(unique_km_a=float(seg.length_km[ra & ~rb].sum()), unique_km_b=float(seg.length_km[rb & ~ra].sum()),
                     both_km=float(seg.length_km[ra & rb].sum()), neither_km=float(seg.length_km[~ra & ~rb].sum()))
        pair_rows.append(r)
    model_rows = []
    for m in maps:
        r = {"model": m, "far_area_km2_at_match": float((masks[m] & far).sum() * km2),
             "mask_area_km2": float(masks[m].sum() * km2), **mask_shape_stats(masks[m])}
        r["far_share_at_match"] = r["far_area_km2_at_match"] / max(r["mask_area_km2"], 1e-9)
        for ref in refs:
            if ref not in support or ref == m:
                continue
            missed = support[ref] < 0.5
            r[f"recall_w_on_missed_by_{ref}"] = float(support[m][missed].mean()) if missed.any() else np.nan
            if seg is not None:
                sm = seg[f"k_{ref}"] < 0.5
                r[f"km_recovered_of_missed_by_{ref}"] = float(seg.length_km[sm & (seg[f"k_{m}"] >= 0.5)].sum())
                r[f"km_missed_by_{ref}"] = float(seg.length_km[sm].sum())
        model_rows.append(r)
    return pd.DataFrame(pair_rows), pd.DataFrame(model_rows)


def unique_discovery_systems(ctx, objs: dict, models: list[str], refs=("G1", "G3_D2"), frac: float = 0.05,
                             policy: str = "excl") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Withheld systems a model recovers (recall >= 0.5) that a reference misses by >= 0.3, at a matched budget."""
    if not objs:
        return pd.DataFrame(), pd.DataFrame()
    per = discovery3.systems_at_budget(ctx, objs, frac, policy=policy)
    rows, lists = [], []
    for m in models:
        for ref in refs:
            if m == ref or ref not in set(per.model) or m not in set(per.model):
                continue
            u = discovery3.unique_recoveries(per, m, ref)
            rows.append({"model": m, "ref": ref, "policy": policy, "frac": frac, "n_unique_systems": len(u),
                         "unique_trace_km": float(u.trace_km.sum()) if len(u) else 0.0})
            if len(u):
                lists.append(u.assign(model=m, ref=ref))
    return pd.DataFrame(rows), (pd.concat(lists, ignore_index=True) if lists else pd.DataFrame())


# ---------------------------------------------------------------------------
# promotion (section 12)
# ---------------------------------------------------------------------------
def promotion(score: pd.DataFrame, cand: str, ctrl: str, rules: dict, unique: pd.DataFrame | None = None,
              ensemble_gain: float | None = None) -> dict:
    s = score.set_index("model")
    if cand not in s.index or ctrl not in s.index:
        return {"candidate": cand, "control": ctrl, "promote": False, "reason": "missing scorecard rows"}
    a, b = s.loc[cand], s.loc[ctrl]
    pf = paired_folds(score, cand, ctrl)
    met, notes = [], []
    g = a.geo_dti - b.geo_dti
    if g >= rules["min_geo_gain"] and pf["folds_beating"] >= rules["min_folds"] and \
            a.geo_fold_min >= b.geo_fold_min - rules["max_fold_min_drop"]:
        met.append("geo_dti")
    notes.append(f"geo {g:+.4f} ({pf['folds_beating']}/3 folds)")
    if "disc_dti_excl" in s.columns and pd.notna(a.get("disc_dti_excl")) and pd.notna(b.get("disc_dti_excl")):
        de = a.disc_dti_excl - b.disc_dti_excl
        dr = a.get("disc_dti_retained_fp", np.nan) - b.get("disc_dti_retained_fp", np.nan)
        ratio = a.get("disc_frac_excl", np.nan) / max(b.get("disc_frac_excl", np.nan), 1e-9)
        if de >= rules["min_disc_gain"] and (np.isnan(dr) or dr >= 0) and (np.isnan(ratio) or ratio <= rules["max_area_ratio"]):
            met.append("discovery")
        notes.append(f"disc excl {de:+.4f} / retained_fp {dr:+.4f} / area x{ratio:.2f}")
    if "short_recall_w" in s.columns:
        ds = a.short_recall_w - b.short_recall_w
        if ds >= rules["min_short_recall_gain"] and g >= -rules["max_fold_min_drop"]:
            met.append("short_faults")
        notes.append(f"<2 km recall {ds:+.4f}")
    if unique is not None and len(unique):
        u = unique[(unique.model == cand)]
        n_u = int(u.n_unique_systems.max()) if len(u) else 0
        if n_u >= rules["min_unique_systems"]:
            met.append("unique_systems")
        notes.append(f"unique systems {n_u}")
    if ensemble_gain is not None:
        if ensemble_gain >= rules.get("min_geo_gain", 0.005):
            met.append("ensemble")
        notes.append(f"ensemble gain {ensemble_gain:+.4f}")
    need = rules["expensive_min_criteria"] if cand in set(rules.get("expensive", [])) else 1
    return {"candidate": cand, "control": ctrl, "criteria_met": ",".join(met), "n_met": len(met), "needed": need,
            "promote": len(met) >= need, "notes": "; ".join(notes)}


# ---------------------------------------------------------------------------
# figures (one quantity per axis; colour follows the model)
# ---------------------------------------------------------------------------
def _style(ax, xlabel="", ylabel="", title=""):
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, loc="left")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def fig_metric_bars(score: pd.DataFrame, metric: str, title: str, fold_cols: list[str] | None = None, ref: str | None = None):
    import matplotlib.pyplot as plt

    d = score.dropna(subset=[metric]).sort_values(metric)
    col = colors_for(list(d.model))
    fig, ax = plt.subplots(figsize=(6.4, 0.32 * len(d) + 1.2), constrained_layout=True)
    y = np.arange(len(d))
    ax.barh(y, d[metric], color=[col[m] for m in d.model], height=0.62)
    if fold_cols:
        for fc in fold_cols:
            if fc in d:
                ax.plot(d[fc], y, "o", color="#3a3a38", markersize=2.6, alpha=0.8)
    for yi, v in zip(y, d[metric]):
        ax.text(v, yi, f" {v:.4f}", va="center", fontsize=7)
    if ref is not None and ref in set(d.model):
        ax.axvline(float(d.set_index("model").loc[ref, metric]), color="#8a8985", linewidth=0.8, linestyle="--")
    ax.set_yticks(y, d.model, fontsize=8)
    _style(ax, metric, "", title)
    return fig


def fig_training_curves(histories: dict[str, pd.DataFrame], metric: str = "inner_dti_binary_best"):
    import matplotlib.pyplot as plt

    names = [n for n, h in histories.items() if h is not None and metric in h]
    if not names:
        return None
    col = colors_for(names)
    folds = sorted({int(f) for h in histories.values() if h is not None and "fold" in h for f in h.fold.dropna().unique()})
    fig, axs = plt.subplots(2, max(len(folds), 1), figsize=(3.4 * max(len(folds), 1), 5.2), squeeze=False, constrained_layout=True)
    for j, f in enumerate(folds):
        for n in names:
            h = histories[n]
            h = h[h.fold == f] if "fold" in h else h
            axs[0, j].plot(h.epoch, h.train_loss, color=col[n], linewidth=1.1, label=n)
            axs[1, j].plot(h.epoch, h[metric], color=col[n], linewidth=1.1, label=n)
        _style(axs[0, j], "", "train loss", f"fold {f}")
        _style(axs[1, j], "epoch", "inner-val binary-area DTI", "")
    axs[0, 0].legend(fontsize=6.5, frameon=False)
    return fig


def fig_disagreement(ctx, maps: dict[str, np.ndarray], a: str, b: str, frac: float, step: int = 4):
    """Region map: only a / only b / both at the matched budget (labels in black). Down-sampled for display."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    col = colors_for([a, b])
    ma, mb = fold_masks(ctx, maps[a], frac), fold_masks(ctx, maps[b], frac)
    code = np.zeros(ma.shape, np.int8)
    code[ctx.data.valid] = 1
    code[ma & ~mb] = 2
    code[mb & ~ma] = 3
    code[ma & mb] = 4
    code[ctx.data.labels] = 5
    sub = ndimage.maximum_filter(code, size=step)[::step, ::step]
    cmap = ListedColormap(["#ffffff", "#ecebe7", col[a], col[b], "#3a3a38", "#000000"])
    fig, ax = plt.subplots(figsize=(8, 8 * sub.shape[0] / sub.shape[1]), constrained_layout=True)
    ax.imshow(sub, cmap=cmap, vmin=0, vmax=5, interpolation="nearest")
    ax.set_xticks([]), ax.set_yticks([])
    from matplotlib.patches import Patch

    ax.legend(handles=[Patch(color=col[a], label=f"only {a}"), Patch(color=col[b], label=f"only {b}"),
                       Patch(color="#3a3a38", label="both"), Patch(color="#000000", label="mapped faults")],
              fontsize=7, frameon=False, loc="lower left")
    ax.set_title(f"Top-{frac:.0%} masks per fold: {a} vs {b}", fontsize=10, loc="left")
    return fig


def atlas_windows(ctx, support: dict, a: str, b: str, n: int = 4, half_px: int = 60) -> pd.DataFrame:
    """Windows centred on segments recovered by a and missed by b, the reverse, and missed by both (longest first)."""
    lpt = label_pixel_table(ctx).assign(ka=support[a], kb=support[b])
    seg = lpt.groupby("seg_id").agg(r=("r", "median"), c=("c", "median"), length_km=("length_km", "first"),
                                    ka=("ka", "mean"), kb=("kb", "mean")).reset_index()
    seg = seg[seg.seg_id >= 0]
    sets = {f"{a} only": seg[(seg.ka >= 0.5) & (seg.kb < 0.5)], f"{b} only": seg[(seg.kb >= 0.5) & (seg.ka < 0.5)],
            "missed by both": seg[(seg.ka < 0.5) & (seg.kb < 0.5)]}
    rows = []
    H, W = ctx.data.shape
    for kind, s in sets.items():
        for r in s.sort_values("length_km", ascending=False).head(n).itertuples():
            r0 = int(np.clip(r.r - half_px, 0, H - 2 * half_px))
            c0 = int(np.clip(r.c - half_px, 0, W - 2 * half_px))
            rows.append({"kind": kind, "seg_id": int(r.seg_id), "length_km": r.length_km, "r0": r0, "c0": c0,
                         "size": 2 * half_px, "k_a": r.ka, "k_b": r.kb})
    return pd.DataFrame(rows)


def fig_error_atlas(ctx, maps: dict[str, np.ndarray], a: str, b: str, frac: float, windows: pd.DataFrame):
    """Hillshaded detrended elevation, mapped faults (black), mask a (fill), mask b (outline)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    from .geo_eval import _hillshade

    if not len(windows):
        return None
    col = colors_for([a, b])
    ma, mb = fold_masks(ctx, maps[a], frac), fold_masks(ctx, maps[b], frac)
    elev = ctx.data.read_band("det_elev")
    n = len(windows)
    ncol = min(4, n)
    nrow = int(np.ceil(n / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 3.3 * nrow), squeeze=False, constrained_layout=True)
    for ax, w in zip(axs.ravel(), windows.itertuples()):
        sl = (slice(w.r0, w.r0 + w.size), slice(w.c0, w.c0 + w.size))
        ax.imshow(_hillshade(np.nan_to_num(elev[sl])), cmap="gray", interpolation="nearest")
        ax.imshow(np.ma.masked_where(~ma[sl], ma[sl]), cmap=ListedColormap([col[a]]), alpha=0.45,
                  interpolation="nearest")
        ax.contour(mb[sl].astype(float), levels=[0.5], colors=[col[b]], linewidths=0.9)
        ax.contour(ctx.data.labels[sl].astype(float), levels=[0.5], colors="#000000", linewidths=0.6)
        ax.set_title(f"{w.kind} | seg {w.seg_id} ({w.length_km:.1f} km)", fontsize=7.5, loc="left")
        ax.set_xticks([]), ax.set_yticks([])
    for ax in axs.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(f"Recovered / missed fault atlas: fill = {a}, outline = {b}, black = mapped (top-{frac:.0%} per fold)",
                 fontsize=9, x=0.01, ha="left")
    return fig


def fig_architecture():
    """Schematic of the Phase-4 encoder-decoders (A single, B dual + fusion, C context branch, D aux heads)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(10, 4.6), constrained_layout=True)
    ax.set_xlim(0, 10), ax.set_ylim(0, 5), ax.axis("off")

    def box(x, y, w, h, text, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03", fc=fc, ec="#3a3a38", lw=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.5)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="->", lw=0.8, color="#3a3a38"))
    box(0.1, 3.6, 1.6, 0.8, "terrain channels\n(det_elev + RS terrain)", "#f3e3c3")
    box(0.1, 2.2, 1.6, 0.8, "subsurface channels\n(mag, grav, basement)", "#d9e4f5")
    box(0.1, 0.6, 1.6, 0.8, "context crop\n4x extent @ 1/4 res", "#e6e6e6")
    box(2.2, 3.6, 1.6, 0.8, "terrain encoder\n5 levels", "#f3e3c3")
    box(2.2, 2.2, 1.6, 0.8, "subsurface encoder\n5 levels", "#d9e4f5")
    box(2.2, 0.6, 1.6, 0.8, "context encoder\n(C only)", "#e6e6e6")
    box(4.3, 2.6, 1.6, 1.4, "per-level fusion\nconcat + 1x1\nor softmax gate", "#ecebe7")
    box(4.3, 0.6, 1.6, 0.8, "bottleneck fuse\n(centre + global)", "#e6e6e6")
    box(6.4, 2.2, 1.6, 1.8, "shared U-Net\ndecoder\n(skip = fused)", "#ecebe7")
    box(8.5, 3.5, 1.4, 0.6, "fault logit\n(DTI surrogate)", "#fbe0d4")
    box(8.5, 2.6, 1.4, 0.6, "proximity (D)\ntrain only", "#e0f0e8")
    box(8.5, 1.7, 1.4, 0.6, "cos/sin 2θ (D)\ntrain only", "#e0f0e8")
    for (x0, y0, x1, y1) in ((1.7, 4.0, 2.2, 4.0), (1.7, 2.6, 2.2, 2.6), (1.7, 1.0, 2.2, 1.0), (3.8, 4.0, 4.3, 3.5),
                             (3.8, 2.6, 4.3, 3.1), (3.8, 1.0, 4.3, 1.0), (5.9, 3.3, 6.4, 3.3), (5.1, 1.4, 6.6, 2.2),
                             (8.0, 3.8, 8.5, 3.8), (8.0, 2.9, 8.5, 2.9), (8.0, 2.4, 8.5, 2.0)):
        arrow(x0, y0, x1, y1)
    ax.text(0.1, 4.75, "A = one encoder on all channels (Phase-3 U-Net).  B = two encoders + fusion; matched-capacity "
                       "single-encoder control.  C = A/B-ctrl + context branch.  D = auxiliary geometry heads.",
            fontsize=8)
    return fig


def write_findings(path, blocks: list[tuple[str, object, str]], header: str = "") -> None:
    """Markdown report: (heading, DataFrame | str | None, rule text) blocks. Missing results say so."""
    lines = ["# GEMS — Phase 4 findings", "",
             f"_Generated {time.strftime('%Y-%m-%d %H:%M')} from saved artifacts in outputs/phase4. "
             "Local validation only; no number here is a leaderboard score. Geographic DTI and pseudo-discovery DTI "
             "are different tasks and are never compared with each other._", ""]
    if header:
        lines += [header, ""]
    for title, obj, rule in blocks:
        lines += [f"## {title}", ""]
        if obj is None or (isinstance(obj, pd.DataFrame) and not len(obj)):
            lines += ["_not run yet (enable the section's RUN_ flag)._", ""]
        elif isinstance(obj, pd.DataFrame):
            lines += [obj.round(4).to_markdown(index=False), ""]
        else:
            lines += [str(obj), ""]
        if rule:
            lines += [f"Rule: {rule}", ""]
    from pathlib import Path

    Path(path).write_text("\n".join(lines), encoding="utf-8")
