"""P6.1 X3: segment-level fault recovery at matched area (P6_RECOVERY_HANDOFF.md).

Lineament inventory = 1-px line pools of the configured label-free sources (W2, fallback fusion, optional G02-C),
split into branches at junctions, deduplicated across sources (a lower-priority segment is a duplicate only when most
of its pixels lie on an already kept line with a similar orientation: crossing faults are never merged). A graph view
(endpoints, junction clusters, branches) gives connectivity attributes. Three predeclared strategies, all at the area
of their H0 cell (never above):

* ``H0`` pure W2 at the cell (the reference).
* ``H1`` W2 core (``core_share`` of the budget) + fallback-only segments on which BOTH CNNs are confident
  (mean rank >= ``h1.min_cnn_rank``), by mean fusion score; then the rest of the W2 lines.
* ``H2`` W2 core + inventory segments connected to the core (endpoint gap <= ``connect_radius_px`` with a consistent
  orientation) or supported by >= ``min_sources`` sources, ordered by a fixed weighted rank; then the rest of W2.

``strategy_map`` is the ONLY implementation: unit evaluation and full-region export call it with different inputs
(training construction vs inference construction), so the selection rule cannot drift between the two.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

from . import boost as B
from . import stack as ST

STRATEGIES = ("H0", "H1", "H2")
S8 = np.ones((3, 3), bool)


# =============================================================================== pure: graph
def neighbour_count(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, bool)
    return np.where(m, ndimage.convolve(m.astype(np.int16), S8.astype(np.int16), mode="constant") - 1, 0)


def skeleton_graph(mask: np.ndarray) -> tuple[np.ndarray, pd.DataFrame, dict]:
    """(branch labels, per-branch topology, diagnostics) of a 1-px line map. Nodes: endpoints (1 neighbour) and
    junction clusters (>= 3 neighbours, 8-connected clusters count once). Edges: branches (``boost.branches``).
    Isolated short pieces are kept as edges with two endpoint nodes; nothing is removed."""
    m = np.asarray(mask, bool)
    lab = B.branches(m)
    n = int(lab.max())
    deg = neighbour_count(m)
    jn = m & (deg >= 3)
    jl, nj = ndimage.label(jn, structure=S8)
    ep = m & (deg == 1)
    if not n:
        return lab, pd.DataFrame(columns=["label", "n_px", "n_endpoints", "n_junctions", "degree_class"]), \
            {"n_nodes": int(nj + ep.sum()), "n_endpoints": int(ep.sum()), "n_junction_clusters": int(nj),
             "n_edges": 0, "n_components": int(ndimage.label(m, structure=S8)[1])}
    size = np.bincount(lab.ravel(), minlength=n + 1)[1:]
    epn = np.bincount(lab[ep & (lab > 0)], minlength=n + 1)[1:]
    touch = []
    if nj:
        jd = ndimage.grey_dilation(jl, footprint=S8)          # junction id in the 3x3 neighbourhood
        rr, cc = np.nonzero((lab > 0) & (jd > 0))
        pairs = pd.DataFrame({"label": lab[rr, cc], "j": jd[rr, cc]}).drop_duplicates()
        touch = pairs.groupby("label").j.nunique()
    nj_b = np.zeros(n, int)
    if len(touch):
        nj_b[touch.index.to_numpy() - 1] = touch.to_numpy()
    T = pd.DataFrame({"label": np.arange(1, n + 1), "n_px": size, "n_endpoints": epn, "n_junctions": nj_b})
    T["degree_class"] = np.select([T.n_junctions == 0, T.n_junctions == 1], ["isolated", "spur"], "bridge")
    _, ncomp = ndimage.label(m, structure=S8)
    return lab, T, {"n_nodes": int(nj + ep.sum()), "n_endpoints": int(ep.sum()), "n_junction_clusters": int(nj),
                    "n_edges": n, "n_components": int(ncomp), "mean_edge_px": float(size.mean()),
                    "n_isolated_edges": int((T.degree_class == "isolated").sum()),
                    "n_short_edges_lt5": int((size < 5).sum())}


def label_geometry(lab: np.ndarray) -> pd.DataFrame:
    """Per label: size, centroid, axial orientation (rad, from +column toward +row), linearity, straightness
    (extent along the major axis / pixel count, clipped to 1; approximate on diagonals)."""
    n = int(lab.max())
    if not n:
        return pd.DataFrame(columns=["label", "n_px", "theta", "linearity", "straightness"])
    rr, cc = np.nonzero(lab)
    lb = lab[rr, cc]
    cnt = np.bincount(lb, minlength=n + 1)[1:].astype(float)

    def mean(v):
        return np.bincount(lb, weights=np.asarray(v, float), minlength=n + 1)[1:] / np.maximum(cnt, 1)

    r, c = rr.astype(float), cc.astype(float)
    mr, mc = mean(r), mean(c)
    vrr, vcc, vrc = mean(r * r) - mr ** 2, mean(c * c) - mc ** 2, mean(r * c) - mr * mc
    th = np.mod(0.5 * np.arctan2(2 * vrc, vcc - vrr), np.pi)
    tr, det = vrr + vcc, vrr * vcc - vrc ** 2
    disc = np.sqrt(np.maximum(tr * tr / 4 - det, 0))
    l1, l2 = tr / 2 + disc, np.maximum(tr / 2 - disc, 0)
    lin = np.where(cnt >= 3, 1 - l2 / np.maximum(l1, 1e-9), 1.0)
    proj = (c - mc[lb - 1]) * np.cos(th[lb - 1]) + (r - mr[lb - 1]) * np.sin(th[lb - 1])
    ids = np.arange(1, n + 1)
    ext = np.asarray(ndimage.maximum(proj, lb, ids)) - np.asarray(ndimage.minimum(proj, lb, ids)) + 1
    return pd.DataFrame({"label": ids, "n_px": cnt.astype(int), "row": mr, "col": mc, "theta": th, "linearity": lin,
                         "straightness": np.clip(ext / np.maximum(cnt, 1), 0, 1)})


def dedup_inventory(pools: dict, priority: list[str], tol_px: float = 2, min_share: float = 0.6,
                    max_angle_deg: float = 20) -> tuple[np.ndarray, pd.DataFrame]:
    """One label raster of every source's branches in ``priority`` order. A segment is a duplicate (dropped) when
    >= ``min_share`` of its pixels lie within ``tol_px`` of already kept lines AND the mean axial agreement
    cos(2 dtheta) with those lines is >= cos(2 ``max_angle_deg``). Crossing lines (large angle) are always kept;
    short segments are never removed for being short. Returns (labels, table with source / duplicate flags)."""
    shape = next(iter(pools.values())).shape
    out = np.zeros(shape, np.int32)
    kept = np.zeros(shape, bool)
    kth = np.full(shape, np.nan, np.float32)
    rows, nxt = [], 1
    cmin = np.cos(2 * np.radians(max_angle_deg))
    for src in priority:
        if src not in pools:
            continue
        lab = B.branches(pools[src])
        n = int(lab.max())
        if not n:
            continue
        geo = label_geometry(lab)
        th = geo.theta.to_numpy()
        share = np.zeros(n)
        agree = np.zeros(n)
        if kept.any():
            d, (ir, ic) = ndimage.distance_transform_edt(~kept, return_indices=True)
            rr, cc = np.nonzero(lab)
            lb = lab[rr, cc]
            near = d[rr, cc] <= tol_px
            cnt = np.bincount(lb, minlength=n + 1)[1:]
            share = np.bincount(lb[near], minlength=n + 1)[1:] / np.maximum(cnt, 1)
            c2 = np.cos(2 * (th[lb[near] - 1] - kth[ir[rr[near], cc[near]], ic[rr[near], cc[near]]]))
            agree = np.bincount(lb[near], weights=np.nan_to_num(c2, nan=0.0), minlength=n + 1)[1:] / \
                np.maximum(np.bincount(lb[near], minlength=n + 1)[1:], 1)
        dup = (share >= min_share) & (agree >= cmin)
        crossing = (share > 0) & ~dup & (agree < cmin)
        for i in range(n):
            rows.append({"source": src, "source_label": i + 1, "n_px": int(geo.n_px.iloc[i]), "theta": float(th[i]),
                         "near_kept_share": float(share[i]), "kept_agreement": float(agree[i]),
                         "duplicate": bool(dup[i]), "crossing_kept_line": bool(crossing[i]),
                         "label": 0 if dup[i] else nxt + int((~dup[:i]).sum())})
        keep_ids = np.flatnonzero(~dup) + 1
        if len(keep_ids):
            remap = np.zeros(n + 1, np.int32)
            remap[keep_ids] = np.arange(nxt, nxt + len(keep_ids))
            new = remap[lab]
            put = (new > 0) & (out == 0)
            out[put] = new[put]
            kept |= new > 0
            kth[new > 0] = th[lab[new > 0] - 1]
            nxt += len(keep_ids)
    return out, pd.DataFrame(rows)


def segment_features(lab: np.ndarray, layers: dict, sources: dict, core: np.ndarray, tol_px: float,
                     ep_radius: int = 5, connect_radius: float = 5) -> pd.DataFrame:
    """Label-free features of every inventory segment: geometry, mean of each layer, number of sources with a
    line within ``tol_px`` on most of the segment, gap from the segment tips to the W2 core, orientation agreement
    with the nearest core pixels and endpoint continuation support (``boost.seg_table``)."""
    n = int(lab.max())
    if not n:
        return pd.DataFrame(columns=["label"])
    T = label_geometry(lab)
    st = B.seg_table(lab, layers.get("w2", np.zeros(lab.shape)), {k: v for k, v in layers.items() if k != "w2"},
                     lab > 0, ep_radius)
    T = T.merge(st.drop(columns=["n_px", "linearity"]), on="label", how="left")
    rr, cc = np.nonzero(lab)
    lb = lab[rr, cc]
    cnt = np.bincount(lb, minlength=n + 1)[1:]
    ns = np.zeros(n, int)
    for s, m in sources.items():
        if np.asarray(m, bool).any():
            d = ndimage.distance_transform_edt(~np.asarray(m, bool))
            sh = np.bincount(lb[d[rr, cc] <= tol_px], minlength=n + 1)[1:] / np.maximum(cnt, 1)
            T[f"on_{s}"] = sh
            ns += (sh >= 0.5).astype(int)
    T["n_sources"] = ns
    core = np.asarray(core, bool)
    tips = ST.endpoints(lab > 0)
    if core.any():
        d, (ir, ic) = ndimage.distance_transform_edt(~core, return_indices=True)
        th_core = B.axial_orientation(core)
        er, ec = np.nonzero(tips)
        gap = np.full(n, np.inf)
        if er.size:
            g = pd.DataFrame({"label": lab[er, ec], "gap": d[er, ec]}).groupby("label").gap.min()
            gap[g.index.to_numpy() - 1] = g.to_numpy()
        T["gap_to_core_px"] = gap
        T["on_core_share"] = np.bincount(lb[d[rr, cc] <= tol_px], minlength=n + 1)[1:] / np.maximum(cnt, 1)
        near = d[rr, cc] <= connect_radius + tol_px
        c2 = np.cos(2 * (T.theta.to_numpy()[lb[near] - 1] - th_core[ir[rr[near], cc[near]], ic[rr[near], cc[near]]]))
        k = np.bincount(lb[near], minlength=n + 1)[1:]
        T["core_orientation_cos2"] = np.where(k > 0, np.bincount(lb[near], weights=np.nan_to_num(c2), minlength=n + 1)
                                              [1:] / np.maximum(k, 1), np.nan)
    else:
        T["gap_to_core_px"], T["on_core_share"], T["core_orientation_cos2"] = np.inf, 0.0, np.nan
    return T


def h2_score(T: pd.DataFrame, weights: dict, short_px: int, short_penalty: float) -> np.ndarray:
    """Predeclared H2 order: weighted mean of within-pool ranks (W2 score, fallback score, number of sources,
    continuity, length, linearity); short segments down-weighted, never removed."""
    cols = {"w2": "mean_score", "fallback": "mean_fallback", "sources": "n_sources", "continuity": "endpoint_support",
            "length": "n_px", "linearity": "linearity", "geology": "mean_g02c"}
    z, ws = np.zeros(len(T)), 0.0
    for k, w in weights.items():
        c = cols.get(k)
        if c in T and float(w) > 0 and T[c].notna().any():
            z += float(w) * T[c].rank(pct=True).fillna(0).to_numpy()
            ws += float(w)
    return z / max(ws, 1e-9) * np.where(T.n_px.to_numpy() < int(short_px), float(short_penalty), 1.0)


def strategy_map(name: str, inputs: dict, apply_rep, frac: float, cfg: dict) -> tuple[np.ndarray, pd.DataFrame, dict]:
    """One strategy at one (rep, frac): ``inputs`` = label-free rank maps {w2, fallback_fusion, cnn_b0, cnn_b1
    [, g02c]}; ``apply_rep(score, frac) -> map`` is the rep operator of the domain (unit or full region). Returns
    (bool map with area <= H0's, segment table, diagnostics)."""
    full = np.asarray(apply_rep(inputs["w2"], frac)) > 0
    k = int(full.sum())
    diag = {"strategy": name, "h0_px": k}
    if name == "H0":
        return full, pd.DataFrame(), {**diag, "px": k}
    core = (np.asarray(apply_rep(inputs["w2"], frac * float(cfg["core_share"]))) > 0) & full
    pf = min(frac * float(cfg["pool_multiplier"]), float(cfg["pool_frac_cap"]))
    pools = {s: np.asarray(apply_rep(inputs[s], pf)) > 0 for s in cfg["pool_sources"] if s in inputs}
    dd = cfg["dedup"]
    lab, D = dedup_inventory(pools, [s for s in cfg["pool_sources"] if s in pools], float(dd["tol_px"]),
                             float(dd["min_share"]), float(dd["max_angle_deg"]))
    layers = {"w2": inputs["w2"], "fallback": inputs["fallback_fusion"], "cnn_b0": inputs["cnn_b0"],
              "cnn_b1": inputs["cnn_b1"], **({"g02c": inputs["g02c"]} if "g02c" in inputs else {})}
    T = segment_features(lab, layers, pools, core, float(cfg["tol_px"]), int(cfg["endpoint_radius_px"]),
                         float(cfg["h2"]["connect_radius_px"]))
    if len(T):
        T = T.merge(D[D.label > 0][["label", "source", "crossing_kept_line"]], on="label", how="left")
    tol = float(cfg["tol_px"])
    near_full = ndimage.distance_transform_edt(~full) <= tol if full.any() else np.zeros(full.shape, bool)
    if len(T):
        rr, cc = np.nonzero(lab)
        T["on_w2_full_share"] = np.bincount(lab[rr, cc][near_full[rr, cc]], minlength=int(lab.max()) + 1)[1:] / \
            np.maximum(T.n_px.to_numpy(), 1)
    if name == "H1":
        c = cfg["h1"]
        cand = T[(T.source != "w2") & (T.on_w2_full_share < 0.5) & (T.mean_cnn_b0 >= float(c["min_cnn_rank"])) &
                 (T.mean_cnn_b1 >= float(c["min_cnn_rank"]))] if len(T) else T
        order = cand.sort_values(["mean_fallback", "label"], ascending=[False, True]).label.to_numpy() if len(cand) \
            else []
    elif name == "H2":
        c = cfg["h2"]
        ok = (T.on_core_share < float(cfg["dedup"]["min_share"])) & (
            ((T.gap_to_core_px <= float(c["connect_radius_px"])) &
             (T.core_orientation_cos2.fillna(-1) >= float(c["min_orientation_cos2"]))) |
            (T.n_sources >= int(c["min_sources"]))) if len(T) else pd.Series(dtype=bool)
        cand = T[ok] if len(T) else T
        if len(cand):
            cand = cand.assign(z=h2_score(cand, c["weights"], int(c["short_px"]), float(c["short_penalty"])))
            order = cand.sort_values(["z", "label"], ascending=[False, True]).label.to_numpy()
        else:
            order = []
    else:
        raise KeyError(f"unknown strategy {name!r}")
    q = B.fill_budget(lab, order, k, base=core)
    rest, nr = ndimage.label(full & ~q, structure=S8)
    if nr:
        ms = np.asarray(ndimage.mean(np.nan_to_num(inputs["w2"]), rest, np.arange(1, nr + 1)))
        q = B.fill_budget(rest, np.arange(1, nr + 1)[np.argsort(-ms, kind="stable")], k, base=q)
    added = np.unique(lab[q & ~full & (lab > 0)])
    if len(T):
        T["candidate"] = T.label.isin(set(np.asarray(order).tolist()))
        T["added"] = T.label.isin(set(added.tolist()))
    diag.update(px=int(q.sum()), core_px=int(core.sum()), pool_frac=pf, pool_segments=int(len(D)),
                dedup_removed=int(D.duplicate.sum()) if len(D) else 0,
                crossing_preserved=int(D.crossing_kept_line.sum()) if len(D) else 0,
                candidates=int(len(order)), added_segments=int(len(added)), added_px=int((q & ~full).sum()),
                w2_px_dropped=int((full & ~q).sum()))
    if q.sum() > k:
        raise AssertionError(f"{name}: area {int(q.sum())} > H0 area {k}")
    return q, T, diag


# =============================================================================== Colab side
def unit_inputs(bst, u, use_g02c: bool) -> dict:
    s = bst.unit_sources(u)
    if use_g02c:
        from ..gems3 import representation as R

        raw = bst.raw(u)
        if "g02c" not in raw:
            raise KeyError(f"{u.key}: g02c requested but absent (never substituted)")
        s["g02c"] = R.rank01(np.nan_to_num(raw["g02c"], nan=0.0), u.ev)
    return s


def full_inputs(rc, use_g02c: bool) -> tuple[dict, np.ndarray]:
    """The inference construction of the same inputs (full-region CNN maps, frozen W2 full score, champion w)."""
    from ..gems3 import representation as R
    from .export_parity import read_band

    bst, p6 = rc.bst, rc.p6
    valid = np.asarray(p6.ctx.data.valid, bool)
    maps, _ = p6.view.full_maps()
    w = float(bst.sel[bst.prim][p6.cfg["probes"]["champion"]]["params_all"]["w"])
    r0, r1 = R.rank01(maps[0], valid), R.rank01(maps[1], valid)
    score = read_band(p6.ws.root / "stack" / "maps" / "full_score.tif")
    out = {"cnn_b0": r0, "cnn_b1": r1, "fallback_fusion": R.fuse(r0, r1, w),
           "w2": R.rank01(np.nan_to_num(score, nan=0.0), valid)}
    if use_g02c:
        g = ST.g02_full(p6, {"g02c": bst.cfg["geology"]["keys"]["g02c"]})
        if "g02c" not in g:
            raise FileNotFoundError("full-region g02c unavailable: export refused (never zero-filled)")
        out["g02c"] = R.rank01(np.nan_to_num(g["g02c"], nan=0.0), valid)
    return out, valid


def run(rc) -> dict:
    """X3 stage (``rc.segment_recovery()``): H0 / H1 / H2 on every unit and cell, nested deployable evaluation with
    the X1 objective, matched-area and false-positive checks, decision."""
    from . import export_parity as EP

    bst = rc.bst
    cfg = rc.cfg["segments"]
    stage = "segments"
    gc = rc.cfg["candidate_recall"]
    reps, fracs = list(cfg.get("reps") or gc["reps"]), [float(f) for f in (cfg.get("fracs") or gc["budgets"])]
    pre = rc.read_json("preflight", "decision") or {}
    use_g = bool(cfg.get("use_g02c_if_complete")) and bool((pre.get("g02c_complete")))
    fp = rc.fp(stage, cfg, reps, fracs, use_g, rc.input_hashes())
    if rc._done(stage, fp):
        return rc.load(stage)
    t0 = rc._begin(stage, fp)
    rows, segs, feats, graphs, sel = [], [], [], [], []
    inv = cfg["inventory_cell"]
    for u in bst.units():
        inp = unit_inputs(bst, u, use_g)
        for rep in reps:
            def ap(x, f, _u=u, _r=rep):
                return bst.binary(_u, x, _r, f)
            for f in fracs:
                for name in cfg["strategies"]:
                    q, T, dg = strategy_map(name, inp, ap, f, cfg)
                    rows.append(bst._eval_q(u, q, candidate=name, blend=1.0, rep=rep, frac=f, role="segment_strategy"))
                    sel.append({"unit": u.key, "rep": rep, "frac": f, **dg})
                    if name == "H2" and rep == inv["rep"] and abs(f - float(inv["frac"])) < 1e-12 and len(T):
                        feats.append(T.assign(unit=u.key, rep=rep, frac=f))
                        segs.append(T[[c for c in ("label", "source", "n_px", "row", "col", "theta", "linearity",
                                                   "straightness", "crossing_kept_line", "candidate", "added")
                                       if c in T]].assign(unit=u.key, rep=rep, frac=f))
                pool = np.asarray(ap(inp["w2"], min(f * float(cfg["pool_multiplier"]), float(cfg["pool_frac_cap"])))) > 0
                gd = skeleton_graph(pool)[2]
                graphs.append({"unit": u.key, "rep": rep, "frac": f, "pool": "w2", **gd})
        bst._raw.pop(u.key, None)
        u.release()
    G = pd.DataFrame(rows)
    rc.save_parquet(pd.concat(segs, ignore_index=True) if segs else pd.DataFrame(), "segments", stage)
    rc.save_parquet(pd.concat(feats, ignore_index=True) if feats else pd.DataFrame(), "segment_features", stage)
    rc.save_table(pd.DataFrame(sel), "segment_selection", stage)
    rc.save_table(pd.DataFrame(graphs), "graph_diagnostics", stage)
    rc.save_table(G.drop(columns=["systems"], errors="ignore"), "grid", stage)
    from .boost_audit import budget_violations

    viol = budget_violations(G, float(rc.gates()["budget_tolerance"]), "H0", ("H1", "H2"))
    rc.save_table(viol, "budget_violations", stage)
    gates = rc.gates()
    dx = rc.read_json("export_parity", "decision") or {}
    obj = dx.get("objective") or rc.cfg["export_parity"]["objective"]
    summ, pers = {}, {}
    for name in cfg["strategies"]:
        s, P = EP.evaluate_policy(G, name, {"candidates": [name]}, obj, bst.prim, float(gates["geo_floor"]),
                                  rc.cfg["tasks"])
        summ[name], pers[name] = s, P
    base = summ.get("H0", {})
    comp, sysrows = [], []
    for name, s in summ.items():
        v = EP.systems_vs_reference(pers[name], pers.get("H0", pd.DataFrame()), bst.prim)
        P = pers[name][pers[name].task == bst.prim]
        P0 = pers.get("H0", pd.DataFrame())
        P0 = P0[P0.task == bst.prim] if len(P0) else P0
        fp_inc = (P.fp_w.sum() / P0.fp_w.sum() - 1.0) if len(P0) and P0.fp_w.sum() > 0 else np.nan
        c = {"candidate": name, **{k: s.get(k) for k in ("disc_dti", "disc_worst", "disc_worst_zone", "geo_dti",
                                                         "geo_worst", "disc_fp_w", "disc_precision_w",
                                                         "disc_recall_w", "deploy_cell", "deployable")},
             "objective": obj, **v, "fp_increase": fp_inc, "length_km": float(P.length_km.sum()) if len(P) else np.nan,
             "n_segments": float(P.n_segments.sum()) if len(P) else np.nan,
             "short_share_mean": float(P.short_share.mean()) if len(P) else np.nan}
        if name == "H0":
            c.update(decision="BASE", reasons="")
        else:
            ok, why = B.accept({**c, "status": "RUN"}, base, cfg["accept"], float(gates["geo_floor"]))
            if len(viol) and (viol.candidate == name).any():
                ok, why = False, why + ["area above the H0 cell (matched-budget violation)"]
            if v["systems_gained"] > 0 and not ok:
                why.append("systems gained without a nested DTI / worst-zone / FP acceptance: stop condition")
            c.update(decision="ACCEPT" if ok else "REJECT", reasons="; ".join(why))
        comp.append(c)
        for r in P.itertuples():
            r0 = P0[P0.zone == r.zone] if len(P0) else P0
            a, b0 = B.system_set(r.systems), B.system_set(r0.systems.iloc[0]) if len(r0) else set()
            sysrows.append({"candidate": name, "zone": int(r.zone), "rep": r.rep, "frac": r.frac,
                            "recovered": len(a), "h0_recovered": len(b0), "gained": len(a - b0), "lost": len(b0 - a),
                            "gained_ids": ";".join(map(str, sorted(a - b0))),
                            "lost_ids": ";".join(map(str, sorted(b0 - a)))})
    C = pd.DataFrame(comp)
    rc.save_table(pd.DataFrame([B._flat(r) for r in comp]), "hybrid_comparison", stage)
    rc.save_table(pd.DataFrame(sysrows), "system_recovery", stage)
    rc.save_table(pd.concat([p.drop(columns=["systems"], errors="ignore") for p in pers.values()], ignore_index=True),
                  "per_zone_metrics", stage)
    acc = C[C.decision == "ACCEPT"].sort_values("disc_dti", ascending=False)
    chosen = str(acc.candidate.iloc[0]) if len(acc) else "H0"
    rc.fig_segments(stage, chosen if chosen != "H0" else "H2", reps, fracs, use_g)
    dec = {"stage": stage, "chosen": chosen, "accepted": list(acc.candidate), "objective": obj,
           "use_g02c": use_g, "rules": cfg["accept"], "geo_floor": gates["geo_floor"],
           "budget_violations": int(len(viol)), "summary": {k: B._flat(v) for k, v in summ.items()},
           "stop_condition": "reject a method that adds systems without a nested DTI gain or with a worst-zone / FP "
                             "regression", "deploy_cell": (summ.get(chosen) or {}).get("deploy_cell")}
    rc.save_json(dec, stage, "decision")
    rc._finish(stage, fp, t0, {"inputs": rc.input_hashes()}, {"chosen": chosen, "accepted": dec["accepted"]},
               schema=["w2", "fallback_fusion", "cnn_b0", "cnn_b1"] + (["g02c"] if use_g else []),
               selection={"objective": obj, "space": "rep+frac at blend 1.0 per strategy"})
    return rc.load(stage)


def full_region_map(rc, rec: dict) -> np.ndarray:
    """Export path of a segment recipe: the same ``strategy_map`` on the inference construction."""
    from ..gems3 import representation as R
    from ..gems3 import submission as SUB

    p6 = rc.p6
    data = p6.ctx.data
    pol = rec["policy"]
    inp, valid = full_inputs(rc, bool(rec["preprocessing"].get("use_g02c")))
    reps = p6.m9.sh["baseline"]["representations"]
    el = SUB.eligible_mask(data, 0, "eligible")
    sigma = float(p6.m9.sh["evaluation"]["ridge_sigma_px"])

    def ap(x, f):
        return R.apply(reps[pol["rep"]], x, valid, f, known=data.labels, sigma_px=sigma, eligible=el) > 0

    q, _, _ = strategy_map(rec["model"]["strategy"], inp, ap, float(pol["frac"]), rec["preprocessing"]["segments"])
    return q.astype(np.float32)
