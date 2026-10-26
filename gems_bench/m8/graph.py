"""Section 12 (optional): fragment graph over ranked candidates.

Runs only after section 11 (it needs candidate scores) and only when switched on. Nodes = candidate endpoints; an edge
links endpoints of DIFFERENT candidates closer than ``link_radius_px`` whose outward tangents point at each other
within ``max_angle_deg`` (a geometrically plausible connection). Edge features: gap length, both tangent angles,
label-free evidence along the straight bridge, both candidates' scores.

Edge probability: ``heuristic`` = exp(-gap / radius) x cos(mean angle) x (0.5 + 0.5 bridge evidence); ``lgbm`` = a
leave-one-zone-out LightGBM edge ranker (edge label: both candidates labelled 1 AND the bridge midpoint within 300 m
of the scored truth; evaluation-only labels, other zones only). Graph score of a candidate:
(1 - w) s + w max_neighbour(p_edge x s_neighbour) (support from plausible continuations), w = ``propagate_weight``.

Evaluated like any ranker (1-px centerlines, eligible budget, nested frac) plus a variant that also draws the bridges
of edges with p >= 0.5 between kept candidates. Connectivity, false connections (bridges > 300 m from truth while both
ends are near it), and recovered missing structure are reported. A graph that does not beat the section-11 score by
``graph.min_gain`` is not retained (the simpler system stays).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage


def endpoints(cs) -> pd.DataFrame:
    rows = []
    for i, p in enumerate(cs.paths):
        if len(p) < 3:
            continue
        for end, (a, b) in (("head", (p[0], p[min(4, len(p) - 1)])), ("tail", (p[-1], p[max(len(p) - 5, 0)]))):
            d = a.astype(float) - b.astype(float)
            ang = float(np.degrees(np.arctan2(-d[0], d[1])) % 360.0)      # outward heading, CCW from east
            rows.append({"cand": i, "end": end, "r": int(a[0]), "c": int(a[1]), "dir_deg": ang})
    return pd.DataFrame(rows)


def edges(ep: pd.DataFrame, radius: float, max_angle: float) -> pd.DataFrame:
    from scipy.spatial import cKDTree

    if len(ep) < 2:
        return pd.DataFrame(columns=["i", "j", "gap_px", "a1", "a2"])
    tree = cKDTree(ep[["r", "c"]].to_numpy(float))
    rows = []
    for i, j in tree.query_pairs(r=radius):
        a, b = ep.iloc[i], ep.iloc[j]
        if a.cand == b.cand:
            continue
        d12 = float(np.degrees(np.arctan2(-(b.r - a.r), b.c - a.c)) % 360.0)
        a1 = abs((d12 - a.dir_deg + 180) % 360 - 180)
        a2 = abs(((d12 + 180) % 360 - b.dir_deg + 180) % 360 - 180)
        gap = float(np.hypot(b.r - a.r, b.c - a.c))
        if (a1 <= max_angle and a2 <= max_angle) or gap <= 1.5:
            rows.append({"i": int(a.cand), "j": int(b.cand), "ri": a.r, "ci": a.c, "rj": b.r, "cj": b.c,
                         "gap_px": gap, "a1": a1, "a2": a2})
    return pd.DataFrame(rows)


def bridge(r0, c0, r1, c1):
    from skimage.draw import line as draw_line

    rr, cc = draw_line(int(r0), int(c0), int(r1), int(c1))
    return np.c_[rr, cc]


def unit_graph(m8, u, cs, score: np.ndarray, ev: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    gcfg = m8.cfg["graph"]
    E = edges(endpoints(cs), float(gcfg["link_radius_px"]), float(gcfg["max_angle_deg"]))
    if not len(E):
        return E, pd.DataFrame({"cand_id": cs.table.cand_id, "graph_degree": 0})
    E["bridge_evidence"] = [float(ev[bridge(r.ri, r.ci, r.rj, r.cj)[:, 0], bridge(r.ri, r.ci, r.rj, r.cj)[:, 1]].mean())
                            for r in E.itertuples()]
    E["s_i"], E["s_j"] = score[E.i.to_numpy()], score[E.j.to_numpy()]
    am = np.radians((E.a1 + E.a2) / 2.0)
    E["p_heuristic"] = np.exp(-E.gap_px / float(gcfg["link_radius_px"])) * np.clip(np.cos(am), 0, 1) * \
        (0.5 + 0.5 * E.bridge_evidence)
    E["task"], E["zone"] = u.task, u.zone
    deg = np.bincount(np.r_[E.i, E.j], minlength=len(cs))
    return E, pd.DataFrame({"cand_id": cs.table.cand_id.to_numpy(), "graph_degree": deg})


def edge_labels(m8, u, cs, E: pd.DataFrame, T_unit: pd.DataFrame) -> np.ndarray:
    """EVALUATION-ONLY edge label (used for training on OTHER zones only)."""
    G = u.eval_truth()
    if not G.any() or not len(E):
        return np.full(len(E), -1)
    d = ndimage.distance_transform_edt(~G) * 100.0
    lab = T_unit.sort_values("cand_id").label.to_numpy()
    mid_r = ((E.ri + E.rj) / 2).round().astype(int).to_numpy()
    mid_c = ((E.ci + E.cj) / 2).round().astype(int).to_numpy()
    both = (lab[E.i.to_numpy()] == 1) & (lab[E.j.to_numpy()] == 1)
    return np.where(both & (d[mid_r, mid_c] <= 300.0), 1, np.where((lab[E.i.to_numpy()] == 0) | (lab[E.j.to_numpy()] == 0), 0, -1))


def propagate(score: np.ndarray, E: pd.DataFrame, p: np.ndarray, w: float) -> np.ndarray:
    sup = np.zeros(len(score))
    if len(E):
        a, b = E.i.to_numpy(), E.j.to_numpy()
        np.maximum.at(sup, a, p * score[b])
        np.maximum.at(sup, b, p * score[a])
    return (1.0 - w) * score + w * sup


def run(m8, T: pd.DataFrame, sets: dict, score_col: str) -> dict:
    from ..gems3.metric_opt import nested_select
    from ..gems3.candidates import CandidateSet
    from ..m7.protocol import score_row
    from ..m7.rasterize import predict, save_heldout

    gcfg = m8.cfg["graph"]
    w = float(gcfg["propagate_weight"])
    fracs = m8.sh["rasterization"]["area_fracs"]
    from ..rs_features import prank

    lays = [nm for nm in m8.cfg["features"]["physics_layers"][:3] if m8.layers.allowed(*nm.split("@")[0].split(":"))[0]]
    valid = np.asarray(m8.ctx.data.valid, bool)
    ev_full = np.zeros(valid.shape, np.float32)
    for nm in lays:
        ev_full += np.nan_to_num(prank(np.asarray(m8.layers.get(nm), np.float32), valid), nan=0.0) / max(len(lays), 1)
    E_all, units, per_unit = [], [], {}
    for task in ("geo", m8.prim):
        for u in m8.units_of(task):
            cs = sets.get(u.key)
            if cs is None or not len(cs):
                continue
            sub = T[(T.task == u.task) & (T.zone == u.zone)].sort_values("cand_id")
            s = sub[score_col].fillna(0).to_numpy()
            s = pd.Series(s).rank(pct=True).to_numpy()
            E, deg = unit_graph(m8, u, cs, s, ev_full[u.sl])
            if len(E):
                E["label"] = edge_labels(m8, u, cs, E, sub)
            per_unit[u.key] = (u, cs, s, sub.label.to_numpy() if "label" in sub else np.full(len(cs), -1))
            E_all.append(E)
            units.append(u)
    EA = pd.concat([e for e in E_all if len(e)], ignore_index=True) if any(len(e) for e in E_all) else pd.DataFrame()
    if gcfg["edge_model"] == "lgbm" and len(EA):
        import lightgbm as lgb

        feats = ["gap_px", "a1", "a2", "bridge_evidence", "s_i", "s_j"]
        EA["p_model"] = np.nan
        for u in units:
            te = (EA.task == u.task) & (EA.zone == u.zone)
            tr = (EA.task == u.task) & (EA.zone != u.zone) & (EA.label >= 0)
            y = EA.loc[tr, "label"]
            if y.sum() < 10 or (y == 0).sum() < 10:
                EA.loc[te, "p_model"] = EA.loc[te, "p_heuristic"]
                continue
            mdl = lgb.LGBMClassifier(n_estimators=150, num_leaves=7, learning_rate=0.05, min_child_samples=20,
                                     random_state=int(m8.cfg["seed"]), verbose=-1)
            mdl.fit(EA.loc[tr, feats], y)
            EA.loc[te, "p_model"] = mdl.predict_proba(EA.loc[te, feats])[:, 1]
    pcol = "p_model" if "p_model" in EA else "p_heuristic"
    rows, met, variants = [], [], {}
    for k, (u, cs, s, lab) in per_unit.items():
        Eu = EA[(EA.task == u.task) & (EA.zone == u.zone)] if len(EA) else EA
        p = Eu[pcol].to_numpy() if len(Eu) else np.zeros(0)
        g = propagate(s, Eu, p, w) if len(Eu) else s
        strong = Eu[p >= 0.5] if len(Eu) else Eu
        br = [bridge(r.ri, r.ci, r.rj, r.cj) for r in strong.itertuples()] if len(strong) else []
        cs_b = CandidateSet(pd.concat([cs.table, pd.DataFrame({"cand_id": np.arange(len(cs), len(cs) + len(br))})],
                                      ignore_index=True), cs.paths + br)
        s_b = np.r_[g, np.minimum(g[strong.i.to_numpy()], g[strong.j.to_numpy()]) if len(strong) else np.zeros(0)]
        variants[k] = {"graph_base": (cs, s), "graph_propagated": (cs, g), "graph_bridges": (cs_b, s_b)}
        for f in fracs:
            for mth, (cset, sc) in variants[k].items():
                q, _ = predict(u, cset, sc, {"kind": "centerline"}, float(f), 0, m8.elig)
                rows.append(score_row(u, q, m8.elig(u, 0), m8.pkey(u), method=mth, frac=float(f)))
        G = u.eval_truth()
        if len(strong) and G.any():
            dG = ndimage.distance_transform_edt(~G) * 100.0
            mids = np.array([b[len(b) // 2] for b in br])
            near = dG[mids[:, 0], mids[:, 1]] <= 300.0
            both_pos = (lab[strong.i.to_numpy()] == 1) & (lab[strong.j.to_numpy()] == 1)
            met.append({"task": u.task, "zone": u.zone, "n_edges": int(len(Eu)), "n_strong_edges": int(len(strong)),
                        "bridges_near_truth": int(near.sum()),
                        "false_connections_between_true_fragments": int((both_pos & ~near).sum()),
                        "mean_degree": float(np.bincount(np.r_[Eu.i, Eu.j], minlength=len(cs)).mean()) if len(Eu) else 0.0})
    curves = pd.DataFrame(rows)
    summ = []
    for (task, mth), g in curves.groupby(["task", "method"]) if len(curves) else []:
        pz, sm = nested_select(g, ["frac"])
        summ.append({"task": task, "method": mth, "dti_crossfit": sm["dti_crossfit"], "fold_min": sm["fold_min"],
                     "frac_all_zones": sm["params_all"]["frac"]})
        if mth != "graph_base":
            for r in pz.itertuples():
                key = f"{task}:{r.zone}"
                u = per_unit[key][0]
                cset, sc = variants[key][mth]
                q, _ = predict(u, cset, sc, {"kind": "centerline"}, float(r.frac), 0, m8.elig)
                save_heldout(m8.ws, f"m8_{mth}", u, q, m8.pkey(u))
    S = pd.DataFrame(summ)
    dec = []
    if len(S):
        p = S[S.task == m8.prim].set_index("method").dti_crossfit
        if {"graph_base", "graph_propagated"} <= set(p.index):
            gain = float(max(p.get("graph_propagated", np.nan), p.get("graph_bridges", -np.inf)) - p["graph_base"])
            dec.append({"gate": "graph_value", "task": m8.prim, "gain_over_candidate_ranker": gain,
                        "decision": "RETAIN_GRAPH" if gain >= float(gcfg["min_gain"]) else "KEEP_SIMPLER_RANKER",
                        "edge_model": gcfg["edge_model"]})
    return {"edges": EA, "curves": curves, "summary": S, "metrics": pd.DataFrame(met), "decision": pd.DataFrame(dec)}
