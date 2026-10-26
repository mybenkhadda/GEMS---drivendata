"""Section 02: which geological faults are invisible to B0 / B1, and why?

Unit of analysis = a fault SYSTEM (label pixels linked within 1 km, ``ctx.splits.system_map``). On ``disc_official``
the systems are the withheld ones (truth of the hidden-test analogue); on ``geo`` every mapped system of the held-out
fold (rediscovery, reported for contrast). A system is recovered by a method when >= ``system_recovered_frac`` of
its scored pixels receive TP support >= ``system_recovered_support`` from the method's held-out map (section 01
choice for that zone, or a common area fraction).

Attributes are descriptive (not causal): length, strike, linearity / sinuosity, topology (isolated, branched,
multi-segment, endpoints), near-parallel neighbours, distance to known faults, region, magnetic / gravity gradient
strength, terrain morphology, cross-modal agreement and DEM coverage, each also as a percentile of the valid area.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import ndimage

from .protocol import system_hits

LENGTH_BINS = [0, 2, 5, 10, 25, np.inf]
LENGTH_LABELS = ["<2 km", "2-5 km", "5-10 km", "10-25 km", ">25 km"]
CLASSES = ["both", "b0_only", "b1_only", "neither"]


def strike_sector(az: float) -> str:
    a = az % 180.0
    if a < 22.5 or a >= 157.5:
        return "N-S"
    if a < 67.5:
        return "NE-SW"
    if a < 112.5:
        return "E-W"
    return "NW-SE"


def _layer(ctx, name: str | None, band: str | None = None, sigma_px: float = 5.0) -> np.ndarray | None:
    """Feature-store layer, else |gradient| of a raw band (smoothed), else None. Read-only."""
    if name and ctx.store.has(name):
        return np.asarray(ctx.store.get(name), np.float32)
    if band and band in ctx.data.band_names:
        from gems_eda.enhance import nan_gaussian

        a = nan_gaussian(ctx.data.read_band(band), sigma_px)
        gr, gc = np.gradient(np.nan_to_num(a, nan=np.nanmedian(a)))
        g = np.hypot(gr, gc).astype(np.float32)
        g[~np.isfinite(a)] = np.nan
        return g
    return None


def _percentile_fn(a: np.ndarray, valid: np.ndarray, rng, n: int = 200_000):
    v = np.abs(a[valid & np.isfinite(a)])
    if not v.size:
        return lambda x: np.full(np.shape(x), np.nan)
    v = np.sort(v[rng.choice(v.size, min(n, v.size), replace=False)])
    return lambda x: np.searchsorted(v, np.abs(np.asarray(x, float))) / v.size


def system_attributes(m7) -> pd.DataFrame:
    """One row per fault system (all systems; tasks select theirs later)."""
    from ..geology.skeleton import branch_points, endpoints, skeletonize
    from ..postprocess.topology import orientation_field

    ctx, cfg = m7.ctx, m7.cfg
    fc = cfg["failure"]
    sm = np.asarray(ctx.splits.system_map)
    lab = np.asarray(ctx.data.labels, bool)
    rr, cc = np.nonzero(sm >= 0)
    sid = sm[rr, cc]
    tab = ctx.splits.systems.copy()
    n = int(sid.max()) + 1 if sid.size else 0
    cnt = np.bincount(sid, minlength=n).astype(float)
    cnt[cnt == 0] = np.nan
    x, y = cc.astype(float), -rr.astype(float)
    mx, my = np.bincount(sid, x, n) / cnt, np.bincount(sid, y, n) / cnt
    sxx = np.bincount(sid, x * x, n) / cnt - mx ** 2
    syy = np.bincount(sid, y * y, n) / cnt - my ** 2
    sxy = np.bincount(sid, x * y, n) / cnt - mx * my
    lam = np.sqrt((sxx - syy) ** 2 + 4 * sxy ** 2)
    l1, l2 = 0.5 * (sxx + syy + lam), 0.5 * (sxx + syy - lam)
    theta = 0.5 * np.degrees(np.arctan2(2 * sxy, sxx - syy))         # CCW from east
    geo = pd.DataFrame({"system_id": np.arange(n), "strike_deg": (90.0 - theta) % 180.0,
                        "linearity": np.where(l1 + l2 > 1e-9, (l1 - l2) / (l1 + l2 + 1e-9), 1.0),
                        "principal_extent_km": np.sqrt(12.0 * np.clip(l1, 0, None)) * ctx.data.grid.pixel_km})
    tab = tab.merge(geo, on="system_id", how="left")
    tab["sinuosity_proxy"] = tab.trace_km / tab.principal_extent_km.clip(lower=0.1)
    tab["strike_sector"] = tab.strike_deg.map(strike_sector)
    tab["length_bin"] = pd.cut(tab.trace_km, LENGTH_BINS, labels=LENGTH_LABELS, right=False).astype(str)
    # topology
    sk = skeletonize(lab)
    ep, bp = endpoints(sk), branch_points(sk)
    comp, _ = ndimage.label(lab, structure=np.ones((3, 3), bool))
    tab = tab.set_index("system_id")
    tab["n_endpoints"] = pd.Series(np.bincount(sm[ep & (sm >= 0)], minlength=n)).reindex(tab.index).fillna(0)
    tab["n_branch_points"] = pd.Series(np.bincount(sm[bp & (sm >= 0)], minlength=n)).reindex(tab.index).fillna(0)
    nc = pd.DataFrame({"s": sid, "c": comp[rr, cc]}).groupby("s").c.nunique()
    tab["n_segments"] = nc.reindex(tab.index).fillna(1)
    tab["topology"] = np.select([(tab.n_branch_points > 0) & (tab.n_segments > 1), tab.n_branch_points > 0,
                                 tab.n_segments > 1], ["branched_multi", "branched", "multi_segment"], "isolated_simple")
    # near-parallel neighbours + distance to other mapped systems (cropped EDT per system)
    ang, _ = orientation_field(lab, 2.0)
    az_px = (90.0 - ang) % 180.0
    pad = int(np.ceil(fc["parallel_max_km"] / ctx.data.grid.pixel_km)) + 2
    order = np.argsort(sid, kind="stable")
    bounds = np.r_[0, np.cumsum(np.bincount(sid, minlength=n))]
    par, dist_other = {}, {}
    H, W = lab.shape
    for s in tab.index:
        i0, i1 = bounds[s], bounds[s + 1]
        if i1 <= i0:
            continue
        idx = order[i0:i1]
        r, c = rr[idx], cc[idx]
        r0, r1, c0, c1 = max(r.min() - pad, 0), min(r.max() + pad + 1, H), max(c.min() - pad, 0), min(c.max() + pad + 1, W)
        other = lab[r0:r1, c0:c1] & (sm[r0:r1, c0:c1] != s)
        if not other.any():
            dist_other[s], par[s] = np.inf, 0.0
            continue
        d, (ir, ic) = ndimage.distance_transform_edt(~other, sampling=(0.1, 0.1), return_indices=True)
        dd = d[r - r0, c - c0]
        dist_other[s] = float(np.median(dd))
        near = dd <= fc["parallel_max_km"]
        da = np.abs(((az_px[r, c] - az_px[ir[r - r0, c - c0] + r0, ic[r - r0, c - c0] + c0]) + 90.0) % 180.0 - 90.0)
        par[s] = float((near & (da <= fc["parallel_max_angle_deg"])).mean())
    tab["dist_other_system_km"] = pd.Series(dist_other).reindex(tab.index)
    tab["near_parallel_share"] = pd.Series(par).reindex(tab.index).fillna(0.0)
    tab["near_parallel"] = tab.near_parallel_share >= 0.3
    # evidence along the system, as values and as percentiles of the valid area
    rng = np.random.default_rng(int(cfg["seed"]))
    valid = ctx.data.valid
    layers = {}
    for k, f in (fc.get("gradient_features") or {}).items():
        layers[f"{k}_gradient"] = _layer(ctx, f, (fc.get("context_bands") or {}).get(k))
    for k, f in (fc.get("morphology_features") or {}).items():
        layers[f"terrain_{k}"] = _layer(ctx, f)
    if fc.get("agreement_feature"):
        layers["crossmodal_agreement"] = _layer(ctx, fc["agreement_feature"])
    if "dem:coverage" in getattr(ctx.store, "specs", {}) and ctx.store.has("dem:coverage"):
        layers["dem_coverage"] = np.asarray(ctx.store.get("dem:coverage"), np.float32)
    for name, a in layers.items():
        if a is None:
            tab[name] = np.nan
            continue
        v = pd.Series(np.abs(a[rr, cc])).groupby(sid).mean()
        tab[name] = v.reindex(tab.index)
        if name != "dem_coverage":
            pf = _percentile_fn(a, valid, rng)
            tab[f"{name}_pct"] = pf(tab[name].to_numpy())
    for name in [c for c in tab.columns if c.endswith("_pct")]:
        ok = tab[name].notna()
        if ok.sum() >= 6:
            tab.loc[ok, name.replace("_pct", "_tercile")] = pd.qcut(tab.loc[ok, name].rank(method="first"), 3,
                                                                    labels=["low", "mid", "high"]).astype(str)
    if {"magnetic_gradient_pct", "gravity_gradient_pct"} <= set(tab.columns):
        tab["low_geophysical_contrast"] = (tab.magnetic_gradient_pct < 0.5) & (tab.gravity_gradient_pct < 0.5)
    tab["dem_covered"] = (tab.dem_coverage >= cfg["dem"]["min_coverage"]) if "dem_coverage" in tab else False
    return tab.reset_index()


def known_distance(m7, tab: pd.DataFrame) -> pd.DataFrame:
    """Per (split, withheld system): median / min distance to the RETAINED faults of that split (disc truth context)."""
    ctx = m7.ctx
    rows = []
    sm = np.asarray(ctx.splits.system_map)
    lab = np.asarray(ctx.data.labels, bool)
    for d in m7.zones("disc_official"):
        G = ctx.splits.withheld(d)
        dk = ndimage.distance_transform_edt(~(lab & ~G), sampling=(0.1, 0.1))
        rr, cc = np.nonzero(G & (sm >= 0))
        g = pd.DataFrame({"system_id": sm[rr, cc], "d": dk[rr, cc]}).groupby("system_id").d
        rows.append(pd.DataFrame({"zone": d, "dist_known_median_km": g.median(), "dist_known_min_km": g.min()})
                    .reset_index())
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def methods_for(task: str) -> list[str]:
    base = ["b0_binary", "b1_binary", "fusion_binary", "fusion_thin"]
    return base + (["fusion_thin_ring300"] if task.startswith("disc") else [])


def detection(m7) -> pd.DataFrame:
    """Per (task, zone, method, system): support statistics and the recovered flag."""
    from .baseline import MemberMaps, heldout_map, predict

    cfg = m7.cfg
    budget = cfg["failure"]["method_budget"]
    mm = MemberMaps(m7.ctx, m7.models)
    rows = []
    for task in ("geo", cfg["evaluation"]["primary_task"]):
        sel = (m7.baseline_selection() or {}).get(task, {})
        for u in m7.units_of(task):
            for m in methods_for(task):
                if m not in sel:
                    continue
                if budget == "selected":
                    q = heldout_map(m7, task, m, u.zone, mm)
                else:
                    z = next(r for r in sel[m]["per_zone"] if r["zone"] == u.zone)
                    q = predict(u, mm.get(u), {**z, "frac": float(budget)}, cfg["baseline"]["representations"],
                                m7.elig, float(cfg["evaluation"]["ridge_sigma_px"]))
                if q is None:
                    continue
                h = system_hits(m7.ctx, u, q, cfg)
                rows.append(h.assign(task=task, zone=u.zone, method=m))
            mm.release(u)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def classes(det: pd.DataFrame) -> pd.DataFrame:
    """both / b0_only / b1_only / neither per (task, zone, system), plus the fusion flags."""
    if not len(det):
        return pd.DataFrame()
    piv = det.pivot_table(index=["task", "zone", "system_id"], columns="method", values="recovered",
                          aggfunc="first").reset_index()
    piv.columns.name = None
    flag = (lambda c: piv[c].fillna(False).astype(bool) if c in piv else pd.Series(False, index=piv.index))
    b0, b1 = flag("b0_binary"), flag("b1_binary")
    piv["class"] = np.select([b0 & b1, b0 & ~b1, ~b0 & b1], ["both", "b0_only", "b1_only"], "neither")
    sp = det.pivot_table(index=["task", "zone", "system_id"], columns="method", values="support_mean").add_prefix("support_")
    npx = det.groupby(["task", "zone", "system_id"]).n_px.first().rename("scored_px")
    return piv.merge(sp.reset_index(), on=["task", "zone", "system_id"]).merge(npx.reset_index(),
                                                                               on=["task", "zone", "system_id"])


STRATA = ["length_bin", "strike_sector", "topology", "near_parallel", "region_name", "dist_known_band",
          "magnetic_gradient_tercile", "gravity_gradient_tercile", "terrain_slope_tercile", "terrain_relief_tercile",
          "crossmodal_agreement_tercile", "low_geophysical_contrast", "dem_covered", "linearity_band",
          "sinuosity_band"]


def stratified(cls: pd.DataFrame, attrs: pd.DataFrame, kd: pd.DataFrame, methods: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(system table with attributes, recall by stratum: system- and length-weighted, class shares)."""
    t = cls.merge(attrs, on="system_id", how="left")
    if len(kd):
        t = t.merge(kd, on=["zone", "system_id"], how="left")
    if "dist_known_median_km" in t:
        t["dist_known_band"] = pd.cut(t.dist_known_median_km, [-np.inf, 1, 3, 10, np.inf],
                                      labels=["<1 km", "1-3 km", "3-10 km", ">10 km"]).astype(str)
    t["linearity_band"] = pd.cut(t.linearity, [-np.inf, 0.8, 0.95, np.inf], labels=["curved", "mixed", "straight"]).astype(str)
    t["sinuosity_band"] = pd.cut(t.sinuosity_proxy, [-np.inf, 1.2, 2.0, np.inf], labels=["<1.2", "1.2-2", ">2"]).astype(str)
    rows = []
    for task, g in t.groupby("task"):
        for st in ["all"] + [s for s in STRATA if s in g]:
            groups = [("all", g)] if st == "all" else list(g.groupby(st))
            for val, gg in groups:
                r = {"task": task, "stratum": st, "value": str(val), "n_systems": int(len(gg)),
                     "trace_km": float(gg.trace_km.sum())}
                for m in methods:
                    if m in gg:
                        rec = gg[m].fillna(False).astype(bool)
                        r[f"system_recall_{m}"] = float(rec.mean()) if len(gg) else np.nan
                        r[f"length_recall_{m}"] = float(gg.trace_km[rec].sum() / max(gg.trace_km.sum(), 1e-9))
                for c in CLASSES:
                    r[f"share_{c}"] = float((gg["class"] == c).mean())
                rows.append(r)
    return t, pd.DataFrame(rows)


def region_density(m7) -> pd.DataFrame:
    """Predicted area share per 3 x 3 region and method on the held-out maps (geo stitched; disc per split)."""
    from ..geo_eval import region_map
    from .baseline import MemberMaps, heldout_map

    ctx = m7.ctx
    reg = region_map(ctx.data)
    from gems_eda.morphology import region_name

    mm = MemberMaps(ctx, m7.models)
    rows = []
    for task in ("geo", m7.cfg["evaluation"]["primary_task"]):
        for u in m7.units_of(task):
            rg = reg[u.sl]
            for m in methods_for(task):
                q = heldout_map(m7, task, m, u.zone, mm)
                if q is None:
                    continue
                pos = (q > 0) & u.ev
                for r in np.unique(rg[u.ev & (rg >= 0)]):
                    z = u.ev & (rg == r)
                    rows.append({"task": task, "zone": u.zone, "method": m, "region": region_name(int(r), (3, 3)),
                                 "eval_px": int(z.sum()), "positive_px": int((pos & z).sum()),
                                 "truth_px": int((u.eval_truth() & z).sum())})
            mm.release(u)
    d = pd.DataFrame(rows)
    if len(d):
        d = d.groupby(["task", "method", "region"])[["eval_px", "positive_px", "truth_px"]].sum().reset_index()
        d["predicted_share"] = d.positive_px / d.eval_px.clip(lower=1)
        d["truth_share"] = d.truth_px / d.eval_px.clip(lower=1)
        d["over_prediction_ratio"] = d.predicted_share / d.truth_share.replace(0, np.nan)
    return d


def geo_vs_disc(strat: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    """Recall per stratum on geo (rediscovery) vs disc (discovery analogue) side by side."""
    if not len(strat):
        return pd.DataFrame()
    cols = [c for c in strat.columns if c.startswith("system_recall_")]
    p = strat.pivot_table(index=["stratum", "value"], columns="task", values=cols)
    p.columns = [f"{a}@{b}" for a, b in p.columns]
    p = p.reset_index()
    for m in methods:
        a, b = f"system_recall_{m}@geo", [c for c in p.columns if c.startswith(f"system_recall_{m}@disc")]
        if a in p and b:
            p[f"geo_minus_disc_{m}"] = p[a] - p[b[0]]
    return p


# =============================================================================== case studies
def pick_cases(systab: pd.DataFrame, per_class: int, seed: int) -> pd.DataFrame:
    """Representative systems per class: spread over regions and lengths (round robin), never 'best looking'."""
    rng = np.random.default_rng(seed)
    out = []
    for (task, k), g in systab.groupby(["task", "class"]):
        g = g.sample(frac=1.0, random_state=int(rng.integers(1 << 30)))
        g = g.sort_values(["region_name", "trace_km"], ascending=[True, False])
        picked, seen = [], set()
        for _, r in g.iterrows():
            key = (r.region_name, r.length_bin)
            if key in seen:
                continue
            seen.add(key)
            picked.append(r)
            if len(picked) >= per_class:
                break
        out += picked
    return pd.DataFrame(out).reset_index(drop=True)


def case_window(ctx, system_id: int, margin_km: float, min_half_km: float) -> tuple[int, int, int, int]:
    sm = np.asarray(ctx.splits.system_map)
    rr, cc = np.nonzero(sm == system_id)
    H, W = sm.shape
    pk = ctx.data.grid.pixel_km
    rc, ccn = (rr.min() + rr.max()) // 2, (cc.min() + cc.max()) // 2
    half = int(max((rr.max() - rr.min()) / 2, (cc.max() - cc.min()) / 2) + margin_km / pk)
    half = max(half, int(min_half_km / pk))
    r0, r1 = max(rc - half, 0), min(rc + half, H)
    c0, c1 = max(ccn - half, 0), min(ccn + half, W)
    return int(r0), int(r1), int(c0), int(c1)


def case_study(m7, row: dict, cand_full: np.ndarray | None = None) -> tuple[object, dict]:
    """Nine aligned panels for one system (labels vs predictions clearly separated) + their underlying arrays."""
    from .baseline import MemberMaps, heldout_map
    from .figures import hillshade, map_panels

    ctx, cfg = m7.ctx, m7.cfg
    fc = cfg["failure"]
    task, zone, s = row["task"], int(row["zone"]), int(row["system_id"])
    r0, r1, c0, c1 = case_window(ctx, s, fc["case_margin_km"], fc["case_min_half_km"])
    win = (slice(r0, r1), slice(c0, c1))
    lab = np.asarray(ctx.data.labels, bool)[win]
    sm = np.asarray(ctx.splits.system_map)[win]
    if task == "geo":
        truth, known = lab, None
        maps = {m: np.asarray(ctx.oof(m))[win] for m in m7.models}
    else:
        G = ctx.splits.withheld(zone)[win]
        truth, known = G, lab & ~G
        maps = {m: (np.asarray(ctx.discovery_pred(m, zone))[win] if ctx.discovery_pred(m, zone) is not None
                    else np.full(lab.shape, np.nan, np.float32)) for m in m7.models}
    this = sm == s
    bands = fc.get("context_bands") or {}
    rd = {k: ctx.data.read_window(b, r0, r1, c0, c1) if b in ctx.data.band_names else np.full(lab.shape, np.nan)
          for k, b in bands.items()}
    hs = hillshade(rd.get("terrain", np.full(lab.shape, np.nan)), pixel_m=ctx.data.grid.pixel_km * 1000)
    u = m7.unit(f"{task}:{zone}")
    mm = MemberMaps(ctx, m7.models)
    fus = np.zeros(lab.shape, np.float32)
    if u is not None:
        q = heldout_map(m7, task, "fusion_thin", zone, mm)
        if q is not None:
            full = np.zeros(ctx.data.shape, np.float32)
            full[u.sl] = q
            fus = full[win]
    cand = cand_full[win] > 0 if cand_full is not None else None
    m0, m1 = m7.models
    layers = [
        {"title": f"labels: system {s} (red) + {'known faults (black)' if known is not None else 'other mapped faults'}",
         "raster": hs, "cmap": "Greys_r", "vmin": 0, "vmax": 1, "truth": this, "known": (known if known is not None
                                                                                        else truth & ~this), "legend": True},
        {"title": f"magnetic ({bands.get('magnetic', '-')})", "raster": rd.get("magnetic"), "cmap": "RdBu_r", "truth": this},
        {"title": f"gravity ({bands.get('gravity', '-')})", "raster": rd.get("gravity"), "cmap": "PuOr_r", "truth": this},
        {"title": "terrain hillshade", "raster": hs, "cmap": "Greys_r", "vmin": 0, "vmax": 1, "truth": this},
        {"title": f"B0 score ({m0})", "raster": maps[m0], "cmap": "viridis", "vmin": 0, "vmax": 1, "truth": this},
        {"title": f"B1 score ({m1})", "raster": maps[m1], "cmap": "viridis", "vmin": 0, "vmax": 1, "truth": this},
        {"title": "fusion + thinning (held-out map, orange)", "raster": hs, "cmap": "Greys_r", "vmin": 0, "vmax": 1,
         "pred_mask": fus, "truth": this, "legend": True},
        {"title": "candidate lineaments (blue; section 03)" if cand is not None else "candidate lineaments: not built yet",
         "raster": hs, "cmap": "Greys_r", "vmin": 0, "vmax": 1, "cand": cand, "truth": this, "legend": cand is not None},
        {"title": "existing fault network (black) + system (red)", "raster": None, "known": lab & ~this, "truth": this,
         "legend": True},
    ]
    title = (f"{task} zone {zone} | system {s} | class {row.get('class')} | {row.get('trace_km', np.nan):.1f} km | "
             f"{row.get('region_name', '')} | strike {row.get('strike_deg', np.nan):.0f} deg")
    fig = map_panels(ctx.data.grid, (r0, r1, c0, c1), layers, title=title, ncols=3)
    data = {"window": np.array([r0, r1, c0, c1]), "truth_system": this, "labels": lab, "fusion_thin": fus,
            **{f"band_{k}": v for k, v in rd.items()}, **{f"score_{k}": v for k, v in maps.items()}}
    if cand is not None:
        data["candidates"] = cand
    return fig, data


def run(m7) -> dict:
    attrs = system_attributes(m7)
    kd = known_distance(m7, attrs)
    det = detection(m7)
    cls = classes(det)
    methods = sorted({m for t in ("geo", m7.cfg["evaluation"]["primary_task"]) for m in methods_for(t)})
    systab, strat = stratified(cls, attrs, kd, methods) if len(cls) else (pd.DataFrame(), pd.DataFrame())
    dens = region_density(m7)
    gvd = geo_vs_disc(strat, methods)
    summary = {}
    for task, g in systab.groupby("task") if len(systab) else []:
        summary[task] = {"n_systems": int(len(g)), **{f"share_{c}": float((g["class"] == c).mean()) for c in CLASSES},
                         "missed_by_both_trace_km": float(g.trace_km[g["class"] == "neither"].sum())}
    return {"attributes": attrs, "detection": det, "systems": systab, "stratified": strat, "density": dens,
            "geo_vs_disc": gvd, "summary": summary}


def figures(m7, out: dict) -> None:
    from .figures import bar_panels16, overview_map, stacked_shares16

    ws, ctx = m7.ws, m7.ctx
    st = out["stratified"]
    sub = "failure_analysis/figures"
    for stratum in ("length_bin", "strike_sector", "topology", "dist_known_band", "magnetic_gradient_tercile",
                    "gravity_gradient_tercile", "terrain_slope_tercile", "region_name"):
        d = st[(st.stratum == stratum)] if len(st) else st
        for task, g in d.groupby("task") if len(d) else []:
            g = g.assign(label=g.value + " (n=" + g.n_systems.astype(str) + ")")
            fig = stacked_shares16(g.rename(columns={f"share_{c}": c for c in CLASSES}), "label", CLASSES,
                                   title=f"{task}: who recovers the systems, by {stratum}")
            ws.save_fig(fig, f"classes_{task}_{stratum}", g, sub)
    if len(st):
        d = st[st.stratum == "length_bin"].melt(id_vars=["task", "value"], value_vars=[c for c in st.columns
                                                if c.startswith("length_recall_")], var_name="method", value_name="recall")
        d["method"] = d.method.str.replace("length_recall_", "")
        ws.save_fig(bar_panels16(d.assign(cat=d.value + " | " + d.method), "cat", "recall", "task",
                                 xlabel="length-weighted system recall", title="Recall by fault length"),
                    "length_recall_by_length_bin", d, sub)
    sy = out["systems"]
    if len(sy):
        prim = m7.cfg["evaluation"]["primary_task"]
        p = sy[sy.task == prim].rename(columns={"r_cent": "r", "c_cent": "c"})
        if len(p):
            fig = overview_map(ctx.data.grid, ctx.data.valid, p.assign(size=p.trace_km), color_col="class",
                               title=f"{prim}: withheld systems by recovery class (size ~ trace length)")
            ws.save_fig(fig, f"overview_classes_{prim}", p[["zone", "system_id", "class", "r", "c", "trace_km"]], sub)
    dn = out["density"]
    if len(dn):
        ws.save_fig(bar_panels16(dn.assign(cat=dn.region + " | " + dn.method), "cat", "over_prediction_ratio", "task",
                                 xlabel="predicted share / truth share (per region)",
                                 title="Prediction density by region (1 = as dense as the scored truth)"),
                    "density_by_region", dn, sub)


def export_case_studies(m7, systab: pd.DataFrame, cand_rasters: dict | None = None) -> pd.DataFrame:
    """Figures (PNG + SVG), NPZ of every panel array, and an index CSV; cand_rasters = {unit key: full raster}."""
    ws = m7.ws
    fc = m7.cfg["failure"]
    cases = pick_cases(systab, int(fc["case_studies_per_class"]), int(m7.cfg["seed"]))
    idx = []
    for r in cases.to_dict("records"):
        key = f"{r['task']}:{int(r['zone'])}"
        fig, data = case_study(m7, r, (cand_rasters or {}).get(key))
        name = f"{r['task']}_z{int(r['zone'])}_sys{int(r['system_id'])}_{r['class']}"
        ws.save_fig(fig, name, None, "failure_analysis/case_studies")
        ws.save_npz({k: np.asarray(v) for k, v in data.items()}, f"failure_analysis/case_studies/{name}.npz")
        meta = {k: (v if isinstance(v, (int, float, str, bool)) or v is None else str(v)) for k, v in r.items()}
        ws.save_json(json.loads(json.dumps(meta, default=str)), f"failure_analysis/case_studies/{name}.json")
        idx.append({**meta, "figure": f"{name}.png", "arrays": f"{name}.npz"})
    t = pd.DataFrame(idx)
    ws.save_table(t, "case_study_index", "failure_analysis/case_studies")
    return t
