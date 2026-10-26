"""Section 09: synthetic fault-map corruption benchmark (map completion is NOT segmentation).

Leakage control: fault systems are assigned to geographic folds BEFORE any corruption (majority fold,
``gems3.map_completion.systems_by_fold``). For held-out fold k only fold-k systems are damaged; the completer sees the
damaged map + label-free evidence, never the intact geometry of a damaged system. Truth = the removed pixels of one
corruption class; the damaged map is the known mask (pixel-exact, its predictions ignored: official convention);
evaluation zone = valid pixels within ``eval_radius_km`` of the removed pixels.

Corruptions: ``endpoint_truncation``, ``interior_gap``, ``lateral_shift`` (a component displaced by ``shift_px``; the
target is the original trace), ``missing_branch`` (a branch < ``branch_max_km`` attached to a junction),
``incomplete_network`` (one non-largest component of a multi-component system), ``short_fault`` (whole short faults).

Deterministic completers (section 08 hypotheses on the damaged map): ``ext_straight``, ``ext_rot+-a``,
``ext_evidence``, ``gap_bridge``, and their union; the extension length L is chosen on the other folds.
Metrics: official DTI / weighted recall, median distance of proposals to the original trace, directional agreement,
added line length farther than 300 m from ANY original fault, recall by evidence tercile. A learned completer is
trained only if the deterministic baseline shows headroom (``learned_if``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

from ..gems3.map_completion import _remove_disk, systems_by_fold
from ..gems3.metric_opt import nested_select
from ..gems3.official import _unit
from .network import proposals

CLASSES = ("endpoint_truncation", "interior_gap", "lateral_shift", "missing_branch", "incomplete_network",
           "short_fault")


def corrupt_fold(labels: np.ndarray, sm: np.ndarray, sys_fold: pd.DataFrame, k: int, rng, cc: dict):
    """(damaged map, {class: removed mask}, protocol rows) with events on fold-k systems only."""
    from ..geology.skeleton import branch_points, component_length_km, endpoints, skeletonize
    from ..p3.lineaments import split_branches

    frac = float(cc["sample_frac"])
    elig_sys = set(sys_fold[sys_fold.fold == k].system.tolist())
    on = np.isin(sm, list(elig_sys)) & labels
    dmg = labels.copy()
    removed = {c: np.zeros(labels.shape, bool) for c in CLASSES}
    rows = []
    sk = skeletonize(labels)
    ep, bp = endpoints(sk), branch_points(sk)
    comp, _ = ndimage.label(labels, structure=np.ones((3, 3), bool))
    comp_km = component_length_km(sk, 0.1)
    # short faults
    ids = np.unique(comp[on & (comp_km > 0) & (comp_km < float(cc["short_fault_km"]))])
    ids = ids[ids > 0]
    pick = rng.choice(ids, int(round(frac * len(ids))), replace=False) if len(ids) else []
    m = np.isin(comp, pick) & dmg
    removed["short_fault"] |= m
    dmg &= ~m
    rows.append({"class": "short_fault", "fold": k, "n_events": len(pick)})
    # incomplete networks: one non-largest component of a multi-component system
    df = pd.DataFrame({"s": sm[on], "c": comp[on]})
    n_ev = 0
    for s, g in df.groupby("s"):
        cs = g.c.value_counts()
        if len(cs) >= 2 and rng.random() < frac:
            victim = int(rng.choice(cs.index[1:]))
            mm = (comp == victim) & dmg
            removed["incomplete_network"] |= mm
            dmg &= ~mm
            n_ev += 1
    rows.append({"class": "incomplete_network", "fold": k, "n_events": n_ev})
    # missing branches
    lab_b, _, junction = split_branches(sk & on, 2)
    jz = ndimage.binary_dilation(junction, structure=np.ones((3, 3), bool), iterations=2)
    n_ev = 0
    for i, sl in enumerate(ndimage.find_objects(lab_b), start=1):
        if sl is None or rng.random() >= frac:
            continue
        b = np.zeros(labels.shape, bool)
        b[sl] = lab_b[sl] == i
        L = b.sum() * 0.112
        touches = (ndimage.binary_dilation(b, structure=np.ones((3, 3), bool)) & jz).any()
        if not touches or L > float(cc["branch_max_km"]):
            continue
        mm = ndimage.binary_dilation(b, structure=np.ones((3, 3), bool)) & dmg & ~jz
        removed["missing_branch"] |= mm
        dmg &= ~mm
        n_ev += 1
    rows.append({"class": "missing_branch", "fold": k, "n_events": n_ev})
    # endpoint truncation
    eps = np.argwhere(ep & dmg & on)
    sel = eps[rng.choice(len(eps), int(round(frac * len(eps))), replace=False)] if len(eps) else []
    for r, c in sel:
        _remove_disk(dmg, removed["endpoint_truncation"], r, c, int(round(float(rng.choice(cc["trim_endpoint_km"])) / 0.1)))
    rows.append({"class": "endpoint_truncation", "fold": k, "n_events": len(sel)})
    # interior gaps on long traces
    interior = np.argwhere(sk & dmg & on & ~ndimage.binary_dilation(ep, iterations=5) & (comp_km >= 3.0))
    n = int(round(frac * len(interior) / 50.0))
    sel = interior[rng.choice(len(interior), min(n, len(interior)), replace=False)] if len(interior) else []
    for r, c in sel:
        _remove_disk(dmg, removed["interior_gap"], r, c, int(round(float(rng.choice(cc["delete_segment_km"])) / 0.2)))
    rows.append({"class": "interior_gap", "fold": k, "n_events": len(sel)})
    # lateral shifts
    lab2, n2 = ndimage.label(dmg & on, structure=np.ones((3, 3), bool))
    ids = rng.choice(np.arange(1, n2 + 1), int(round(frac * n2 / 3)), replace=False) if n2 else []
    s = int(cc["shift_px"])
    for i in ids:
        mm = lab2 == i
        moved = np.roll(mm, (s, s), axis=(0, 1))
        removed["lateral_shift"] |= mm & ~moved
        dmg = (dmg & ~mm) | moved
    rows.append({"class": "lateral_shift", "fold": k, "n_events": len(ids)})
    for c in CLASSES:
        removed[c] &= ~dmg
    for r in rows:
        r["removed_px"] = int(removed[r["class"]].sum())
    return dmg, removed, rows


def _rasters(cs, shape) -> dict:
    """One 1-px raster per (family, L) + union per L."""
    out = {}
    if not len(cs):
        return out
    t = cs.table
    for (src, L), g in t.groupby(["source", "L_km"]):
        fam = "gap_bridge" if src == "gap_bridge" else src
        key = (fam, float(L) if fam != "gap_bridge" else 0.0)
        q = out.setdefault(key, np.zeros(shape, np.float32))
        for i in g.index:
            p = cs.paths[i]
            q[p[:, 0], p[:, 1]] = 1.0
    for L in sorted({k[1] for k in out if k[0] != "gap_bridge"}):
        u = np.zeros(shape, np.float32)
        for (fam, LL), q in out.items():
            if fam == "gap_bridge" or LL == L:
                u = np.maximum(u, q)
        out[("union", L)] = u
    return out


def metrics(u, q: np.ndarray, truth_full_crop: np.ndarray, orig_crop: np.ndarray, ev_tercile: np.ndarray | None) -> dict:
    from ..postprocess.topology import _axial_diff, orientation_field

    c = u.components(q)
    out = c.as_dict()
    pos = (q > 0) & u.ev
    G = u.eval_truth()
    if pos.any() and G.any():
        dG = ndimage.distance_transform_edt(~G, sampling=(0.1, 0.1))
        out["median_dist_to_truth_km"] = float(np.median(dG[pos]))
        near = pos & (dG <= 0.3)
        if near.any():
            a_p, _ = orientation_field(pos, 2.0)
            a_t, _ = orientation_field(G, 2.0)
            _, (ir, ic) = ndimage.distance_transform_edt(~G, return_indices=True)
            dd = _axial_diff(a_p[near], a_t[ir[near], ic[near]])
            out["direction_agreement"] = float(np.mean(np.abs(np.cos(np.radians(2 * dd)))))
        dO = ndimage.distance_transform_edt(~orig_crop, sampling=(0.1, 0.1))
        out["added_false_line_km"] = float((pos & (dO > 0.3)).sum() * 0.1)
    else:
        out.update(median_dist_to_truth_km=np.nan, direction_agreement=np.nan,
                   added_false_line_km=float(pos.sum() * 0.1))
    if ev_tercile is not None and G.any():
        sup = u.scorer.support(q)
        gr, gc = u.scorer.g_rc
        terc = ev_tercile[gr, gc]
        for t in (0, 1, 2):
            m = terc == t
            out[f"recall_evidence_t{t}"] = float((sup[m] >= 0.5).mean()) if m.any() else np.nan
    return out


def evaluate(m7, evidence: np.ndarray) -> dict:
    ctx, cfg = m7.ctx, m7.cfg
    mc = cfg["map_completion"]
    cc = mc["corruption"]
    rng = np.random.default_rng(int(cc["seed"]))
    lab, valid = np.asarray(ctx.data.labels, bool), ctx.data.valid
    sm, fm = np.asarray(ctx.splits.system_map), ctx.folds.fold_map
    sysf = systems_by_fold(sm, fm)
    v = evidence[valid]
    q1, q2 = np.quantile(v, [1 / 3, 2 / 3]) if v.size else (0, 0)
    terc_full = np.digitize(evidence, [q1, q2]).astype(np.int8)
    dem = m7.dem_layer()
    proto, rows, props = [], [], []
    for real in range(int(cc["n_realisations"])):
        for k in m7.zones("geo"):
            dmg, removed, pr = corrupt_fold(lab, sm, sysf, k, rng, cc)
            proto += [{**r, "realisation": real} for r in pr]
            anyrem = np.zeros(lab.shape, bool)
            for c in CLASSES:
                anyrem |= removed[c]
            if not anyrem.any():
                continue
            near_full = ndimage.distance_transform_edt(~anyrem, sampling=(0.1, 0.1)) <= float(cc["eval_radius_km"])
            ev_zone = near_full & valid
            u0 = _unit("map_completion", k, ev_zone, anyrem, dmg, False, dmg)
            sl = u0.sl
            cs, _ = proposals(dmg[sl], valid[sl], evidence[sl], mc, tail_px=int(cfg["candidates"]["tip_tail_px"]),
                              near=near_full[sl], dem=None if dem is None else dem[sl], prefix=f"r{real}f{k}/")
            ras = _rasters(cs, u0.ev.shape)
            for cls in CLASSES:
                G = removed[cls]
                if not G.any():
                    continue
                zone = (ndimage.distance_transform_edt(~G, sampling=(0.1, 0.1)) <= float(cc["eval_radius_km"])) & valid
                u = _unit(f"mc_{cls}", k, zone, G, dmg, False, dmg)
                for (fam, L), q in ras.items():
                    full = np.zeros(lab.shape, np.float32)
                    full[sl] = q
                    qq = full[u.sl]
                    rows.append({"realisation": real, "class": cls, "zone": k, "completer": fam, "L_km": L,
                                 **metrics(u, qq, G[u.sl], lab[u.sl], terc_full[u.sl])})
            if len(cs):
                t = cs.table.copy()
                t["realisation"], t["zone"] = real, k
                dG = ndimage.distance_transform_edt(~anyrem[sl], sampling=(0.1, 0.1))
                kd = np.clip(1.0 - dG / 0.3, 0, None)
                t["kd_mean"] = [float(kd[p[:, 0], p[:, 1]].mean()) for p in cs.paths]   # label: reads the truth
                t["length_km"] = [len(p) * 0.112 for p in cs.paths]
                props.append(t)
            print(f"[m7] s09 realisation {real} fold {k}: {len(cs)} hypotheses")
    res = pd.DataFrame(rows)
    sel = selection(res)
    dec = decision(sel, mc)
    out = {"protocol": pd.DataFrame(proto), "results": res, "selection": sel, "decision": dec,
           "proposals": pd.concat(props, ignore_index=True) if props else pd.DataFrame()}
    if dec.decision.iloc[0] == "TRAIN_LEARNED" and len(out["proposals"]):
        out["learned"] = learned(out["proposals"], mc)
    return out


def selection(res: pd.DataFrame) -> pd.DataFrame:
    """Per (class, completer): L chosen on the other folds (pooled over realisations)."""
    rows = []
    if not len(res):
        return pd.DataFrame()
    for (cls, fam), g in res.groupby(["class", "completer"]):
        g2 = g.groupby(["zone", "L_km"])[["tp_w", "fp_w", "fn_w", "n_truth"]].sum().reset_index()
        per, s = nested_select(g2, ["L_km"])
        extra = g.groupby("L_km")[["median_dist_to_truth_km", "direction_agreement", "added_false_line_km"]].mean()
        L = s["params_all"]["L_km"] if s.get("params_all") else np.nan
        rows.append({"class": cls, "completer": fam, "dti_crossfit": s["dti_crossfit"],
                     "recall_crossfit": float(per.tp_w.sum() / max(per.n_truth.sum(), 1)) if len(per) else np.nan,
                     "fold_min": s["fold_min"], "L_all_zones": L,
                     **({k: float(extra.loc[L, k]) for k in extra.columns} if L in extra.index else {})})
    return pd.DataFrame(rows)


def decision(sel: pd.DataFrame, mc: dict) -> pd.DataFrame:
    li = mc["learned_if"]
    if not len(sel):
        return pd.DataFrame([{"decision": "NO_DATA", "reason": "no corruption events"}])
    best = sel.sort_values("dti_crossfit", ascending=False).groupby("class").head(1)
    ok = best[(best.recall_crossfit >= float(li["recall_crossfit"])) & (best.dti_crossfit >= float(li["dti_crossfit"]))]
    return pd.DataFrame([{"decision": "TRAIN_LEARNED" if len(ok) else "DETERMINISTIC_ONLY",
                          "classes_with_headroom": ";".join(ok["class"]) if len(ok) else "",
                          "best_per_class": best[["class", "completer", "dti_crossfit", "recall_crossfit"]].to_json(
                              orient="records"),
                          "rule": f"recall_crossfit >= {li['recall_crossfit']} and dti_crossfit >= {li['dti_crossfit']}"}])


FEATS = ["L_km", "angle_offset", "prior_dependency", "evidence_mean", "dem_alignment", "length_km"]


def learned(props: pd.DataFrame, mc: dict) -> pd.DataFrame:
    """Leave-one-fold-out LightGBM on completion hypotheses (label: mean k(d) to the removed truth >= 0.5).
    Trained only on other folds' damaged systems: the held-out fold's intact geometry is never seen."""
    import lightgbm as lgb
    from sklearn.metrics import average_precision_score

    prm = dict(mc["learned"]["lgbm"])
    seed = prm.pop("seed", 2026)
    d = props.copy()
    d["is_evidence"] = (d.source == "ext_evidence").astype(int)
    d["is_gap"] = (d.source == "gap_bridge").astype(int)
    feats = [f for f in FEATS if f in d] + ["is_evidence", "is_gap"]
    y = (d.kd_mean >= 0.5).astype(int)
    rows = []
    for k in sorted(d.zone.unique()):
        tr, te = d.zone != k, d.zone == k
        if y[tr].sum() < 10 or y[te].sum() < 3:
            rows.append({"zone": k, "status": "too few positives"})
            continue
        mdl = lgb.LGBMClassifier(random_state=seed, verbose=-1, **prm)
        mdl.fit(d.loc[tr, feats], y[tr])
        p = mdl.predict_proba(d.loc[te, feats])[:, 1]
        base = d.loc[te, "evidence_mean"].fillna(0).to_numpy()
        rows.append({"zone": k, "status": "ok", "ap_learned": float(average_precision_score(y[te], p)),
                     "ap_evidence_only": float(average_precision_score(y[te], base)),
                     "ap_prevalence": float(y[te].mean())})
    return pd.DataFrame(rows)
