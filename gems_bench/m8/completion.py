"""Section 08: Monte Carlo fault-network completion (stochastic, evidence-conditioned; M7 deterministic = baseline).

The known map is an incomplete graph (M7 ``network``: skeleton segments, endpoints with outward tangents, junctions,
gaps). Hypotheses:

* stochastic continuations: from every endpoint, ``completion_samples`` particles walk up to ``max_len_km``; at each
  1-px step the heading change is drawn from ``headings_deg`` with probability
  softmax[(E(next) - curvature_penalty |dh| / 15 deg + orientation_weight cos(2 (heading - local strike))) / T],
  restricted to a +/- ``cone_deg`` cone around the initial tangent. E = mean percentile of label-free evidence layers
  (physics operators + optional DEM); the local strike comes from the structure tensor of a physics layer. A particle
  stops after ``stop_patience_px`` steps below ``stop_evidence`` (evidence-terminated length) or when it reaches
  another known trace (branching / connection support);
* gap bridges: the same walker started at one endpoint of a gap pair with an attraction term to the partner endpoint;
* displaced traces: each known segment shifted by +/- ``shift_px`` along its normal when the shifted trace has
  ``shift_min_gain`` more mean evidence (a mislocated trace).

Per endpoint: path OCCUPANCY (share of particles through each pixel), the consensus path (the particle with the highest
mean occupancy), diversity (spread of particle end points, km), lateral displacement of end points, continuity (mean
walked length / max), branching support (share joining a known trace), geophysical support (mean evidence of the
consensus path), prior dependency (share within 1 px of the straight extrapolation). Proposals never read the truth.

Evaluation: (1) the M7 corruption benchmark (systems assigned to folds BEFORE damage; the completer sees only the
damaged map) per class, stochastic vs M7 deterministic completers, parameters nested over folds; (2) disc units:
oracle-style coverage of the withheld systems and System C = deterministic + stochastic hypotheses ranked by the
label-free completion score, eligible budget, nested frac (held-out maps ``validation/maps/m8_completion``).
Completion hypotheses are never promoted to full-region predictions automatically.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

from ..gems3.candidates import CandidateSet

PIXEL_KM = 0.1


# =============================================================================== evidence
def evidence(m8, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray, list]:
    """(mean percentile of the evidence layers, local strike azimuth from the orientation layer, used layers)."""
    from ..rs_features import prank

    cc = m8.cfg["completion"]
    L = m8.layers
    acc, k, used = np.zeros(valid.shape, np.float32), 0, []
    for nm in cc["evidence_layers"]:
        fam, rest = nm.split(":", 1)
        op, sc = rest.split("@")
        if not L.allowed(fam, op)[0]:
            continue
        a = np.asarray(L.get(nm), np.float32)
        acc += np.nan_to_num(prank(a, valid), nan=0.0)
        k += 1
        used.append(nm)
    dem = m8.dem_layer() if cc.get("use_dem") else None
    if dem is not None:
        acc += np.where(np.isfinite(dem), dem, acc / max(k, 1))
        k += 1
        used.append("dem")
    ev = (acc / max(k, 1)).astype(np.float32)
    strike = np.zeros(valid.shape, np.float32)
    on = cc.get("orientation_layer")
    if on:
        fam, rest = on.split(":", 1)
        sc = float(rest.split("@")[1])
        from . import potential_fields as PF

        g = PF.gradient_parts(L.field(fam), PF.sigma_px_of(sc, L.pixel_m), L.px_km)
        strike = np.nan_to_num(g["strike"], nan=0.0)
        L.release()
    return ev, strike, used


# =============================================================================== walker
def walk(r: float, c: float, theta0: float, n_steps: int, n: int, ev: np.ndarray, strike: np.ndarray,
         valid: np.ndarray, known: np.ndarray, cc: dict, rng, target: tuple | None = None) -> list[dict]:
    """``n`` particles from (r, c) with initial heading ``theta0`` (deg, CCW from east, gems convention)."""
    h, w = ev.shape
    dh = np.asarray(cc["headings_deg"], float)
    T = max(float(cc["temperature"]), 1e-3)
    y = np.full(n, float(r))
    x = np.full(n, float(c))
    th = np.full(n, float(theta0))
    alive = np.ones(n, bool)
    low = np.zeros(n, int)
    joined = np.zeros(n, bool)
    paths = [[] for _ in range(n)]
    lam_c, lam_o, cone = float(cc["curvature_penalty"]), float(cc["orientation_weight"]), float(cc["cone_deg"])
    for step in range(n_steps):
        if not alive.any():
            break
        cand = th[:, None] + dh[None, :]
        t = np.radians(cand)
        ny, nx = y[:, None] - np.sin(t), x[:, None] + np.cos(t)
        iy, ix = np.rint(ny).astype(int), np.rint(nx).astype(int)
        inside = (iy >= 0) & (iy < h) & (ix >= 0) & (ix < w)
        iy2, ix2 = np.clip(iy, 0, h - 1), np.clip(ix, 0, w - 1)
        e = ev[iy2, ix2]
        s_az = strike[iy2, ix2]
        s_math = 90.0 - s_az                                      # strike azimuth -> math angle (axial)
        orient = np.cos(np.radians(2.0 * (cand - s_math)))
        off = np.abs((cand - theta0 + 180.0) % 360.0 - 180.0)
        logit = (e - lam_c * np.abs(dh)[None, :] / 15.0 + lam_o * orient) / T
        if target is not None:
            ty, tx = target
            dist = np.hypot(ny - ty, nx - tx)
            logit -= float(cc["gap_attraction"]) * dist / max(n_steps, 1) / T
        bad = ~inside | ~valid[iy2, ix2] | (off > cone)
        logit = np.where(bad, -np.inf, logit)
        g = rng.gumbel(size=logit.shape)
        k = np.argmax(logit + g, axis=1)
        none = ~np.isfinite(logit).any(axis=1)
        go = alive & ~none
        rows = np.arange(n)
        y = np.where(go, ny[rows, k], y)
        x = np.where(go, nx[rows, k], x)
        th = np.where(go, cand[rows, k], th)
        py, px = iy2[rows, k], ix2[rows, k]
        for i in np.flatnonzero(go):
            pt = (int(py[i]), int(px[i]))
            if not paths[i] or paths[i][-1] != pt:
                paths[i].append(pt)
        ee = e[rows, k]
        low = np.where(go & (ee < float(cc["stop_evidence"])), low + 1, np.where(go, 0, low))
        hitk = go & known[py, px] & (step >= 2)
        joined |= hitk
        if target is not None:
            joined |= go & (np.hypot(y - target[0], x - target[1]) <= 1.5)
        alive &= go & (low < int(cc["stop_patience_px"])) & ~hitk
        if target is not None:
            alive &= ~(np.hypot(y - target[0], x - target[1]) <= 1.5)
    return [{"path": np.array(p, int).reshape(-1, 2), "joined": bool(j)} for p, j in zip(paths, joined)]


def endpoint_hypotheses(e, samples: list[dict], ev: np.ndarray, shape, n_steps: int, prefix: str,
                        min_px: int) -> tuple[list, list, np.ndarray]:
    occ = np.zeros(shape, np.float32)
    for s in samples:
        p = s["path"]
        if len(p):
            occ[p[:, 0], p[:, 1]] += 1.0
    occ /= max(len(samples), 1)
    good = [s for s in samples if len(s["path"]) >= min_px]
    if not good:
        return [], [], occ
    score = [float(occ[s["path"][:, 0], s["path"][:, 1]].mean()) for s in good]
    best = good[int(np.argmax(score))]["path"]
    ends = np.array([s["path"][-1] for s in good], float)
    t = np.radians(e.dir_deg)
    nrm = np.array([np.cos(t), np.sin(t)])                     # (row, col) normal to the heading (row = -north)
    lat = (ends - np.array([e.r, e.c])) @ nrm
    from skimage.draw import line as draw_line

    n = len(best)
    rr, cc_ = draw_line(int(e.r), int(e.c), int(round(e.r - n * np.sin(t))), int(round(e.c + n * np.cos(t))))
    straight = {(a + dr, b + dc) for a, b in zip(rr, cc_) for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
    row = {"uid": f"{prefix}mc_ep{int(e.endpoint)}", "source": "mc_continuation", "modality": "network",
           "kind": "completion", "mode": "stochastic", "hypothesis_group": f"{prefix}ep{int(e.endpoint)}",
           "L_km": float(n * PIXEL_KM), "angle_offset": np.nan,
           "prior_dependency": float(np.mean([tuple(q) in straight for q in best.tolist()])),
           "evidence_mean": float(ev[best[:, 0], best[:, 1]].mean()),
           "occupancy_mean": float(occ[best[:, 0], best[:, 1]].mean()),
           "diversity_km": float(np.sqrt(((ends - ends.mean(0)) ** 2).sum(1)).mean() * PIXEL_KM),
           "displacement_km": float(np.std(lat) * PIXEL_KM),
           "continuity": float(np.mean([len(s["path"]) for s in samples]) / max(n_steps, 1)),
           "join_share": float(np.mean([s["joined"] for s in samples])), "n_samples": len(samples),
           "scale_m": np.nan, "network_id": f"{prefix}seg{int(e.segment)}", "dem_alignment": np.nan}
    return [best], [row], occ


def shift_hypotheses(known: np.ndarray, valid: np.ndarray, ev: np.ndarray, cc: dict, prefix: str) -> tuple[list, list]:
    """Displaced-trace hypotheses: a segment moved along its normal when the moved trace carries more evidence."""
    from ..m7.generator import mask_to_paths
    from . import profiles as P

    paths, _ = mask_to_paths(known, {"min_len_km": 0.5, "link_gap_px": 0, "link_angle_deg": 30.0})
    h, w = known.shape
    out_p, rows = [], []
    for j, p in enumerate(paths):
        pos, nrm = P.stations(p, 1, 3, 3)
        base = float(np.nanmean(ev[p[:, 0], p[:, 1]]))
        best, bs = 0, base
        for s in cc["shift_px"]:
            for sg in (1, -1):
                q = np.rint(pos + sg * float(s) * nrm).astype(int)
                ok = (q[:, 0] >= 0) & (q[:, 0] < h) & (q[:, 1] >= 0) & (q[:, 1] < w)
                if ok.mean() < 0.9:
                    continue
                q = q[ok]
                if not valid[q[:, 0], q[:, 1]].all():
                    continue
                v = float(ev[q[:, 0], q[:, 1]].mean())
                if v > bs:
                    bs, best = v, sg * float(s)
        if best != 0 and bs - base >= float(cc["shift_min_gain"]):
            q = np.rint(pos + best * nrm).astype(int)
            q = q[(q[:, 0] >= 0) & (q[:, 0] < h) & (q[:, 1] >= 0) & (q[:, 1] < w)]
            q = q[~known[q[:, 0], q[:, 1]]]
            if len(q) >= 3:
                out_p.append(q)
                rows.append({"uid": f"{prefix}shift{j}", "source": "displaced_trace", "modality": "network",
                             "kind": "completion", "mode": "shift", "hypothesis_group": f"{prefix}seg{j}",
                             "L_km": float(len(q) * PIXEL_KM), "angle_offset": np.nan, "prior_dependency": 1.0,
                             "evidence_mean": bs, "occupancy_mean": np.nan, "shift_px": best,
                             "scale_m": np.nan, "network_id": f"{prefix}shift{j}", "dem_alignment": np.nan})
    return out_p, rows


def stochastic(m8, known: np.ndarray, valid: np.ndarray, ev: np.ndarray, strike: np.ndarray, *, near=None,
               prefix: str = "", seed: int = 0, n_samples: int | None = None) -> tuple[CandidateSet, np.ndarray, dict]:
    """All stochastic hypotheses of one known map (paths in the arrays' coordinates) + total occupancy + elements."""
    from ..m7.network import endpoint_table, gap_table, segments

    cc = m8.cfg["completion"]
    mc7 = m8.sh["map_completion"]
    rng = np.random.default_rng(seed)
    n = int(n_samples or m8.profile["completion_samples"])
    n_steps = int(round(float(cc["max_len_km"]) / PIXEL_KM))
    min_px = max(int(round(float(cc["min_len_km"]) / PIXEL_KM)), 3)
    seg, sk, _ = segments(known)
    ep = endpoint_table(sk, int(m8.cfg["generation"]["tip_tail_px"]))
    if near is not None and len(ep):
        ep = ep[near[ep.r.to_numpy(), ep.c.to_numpy()]].reset_index(drop=True)
    paths, rows = [], []
    occ_all = np.zeros(known.shape, np.float32)
    for e in ep.itertuples():
        s = walk(e.r, e.c, e.dir_deg, n_steps, n, ev, strike, valid, known, cc, rng)
        ps, rs, occ = endpoint_hypotheses(e, s, ev, known.shape, n_steps, prefix, min_px)
        occ_all = np.maximum(occ_all, occ)
        paths += ps
        rows += rs
    gaps = gap_table(ep, float(mc7["gap_max_km"]), float(mc7["gap_max_angle_deg"]))
    epi = ep.set_index("endpoint")
    for g in gaps.itertuples():
        a, b = epi.loc[g.e1], epi.loc[g.e2]
        L = int(np.ceil(np.hypot(b.r - a.r, b.c - a.c) * 1.5)) + 2
        s = walk(a.r, a.c, a.dir_deg, L, n, ev, strike, valid, known, cc, rng, target=(float(b.r), float(b.c)))
        reach = [x for x in s if x["joined"] and len(x["path"]) >= 2]
        if not reach:
            continue
        occ = np.zeros(known.shape, np.float32)
        for x in reach:
            occ[x["path"][:, 0], x["path"][:, 1]] += 1.0 / len(s)
        best = max(reach, key=lambda x: float(occ[x["path"][:, 0], x["path"][:, 1]].mean()))["path"]
        best = best[~known[best[:, 0], best[:, 1]]]
        if len(best) < 2:
            continue
        occ_all = np.maximum(occ_all, occ)
        paths.append(best)
        rows.append({"uid": f"{prefix}mc_gap{int(g.e1)}_{int(g.e2)}", "source": "mc_gap_bridge", "modality": "network",
                     "kind": "completion", "mode": "stochastic", "hypothesis_group": f"{prefix}gap",
                     "L_km": float(len(best) * PIXEL_KM), "angle_offset": np.nan, "prior_dependency": np.nan,
                     "evidence_mean": float(ev[best[:, 0], best[:, 1]].mean()),
                     "occupancy_mean": float(occ[best[:, 0], best[:, 1]].mean()), "join_share": len(reach) / len(s),
                     "n_samples": len(s), "scale_m": np.nan, "network_id": f"{prefix}gap", "dem_alignment": np.nan})
    sp, sr = shift_hypotheses(known, valid, ev, cc, prefix)
    paths += sp
    rows += sr
    t = pd.DataFrame(rows)
    return CandidateSet(t, paths), occ_all, {"segments": seg, "endpoints": ep, "gaps": gaps}


def completion_score(t: pd.DataFrame) -> np.ndarray:
    """Label-free hypothesis score: evidence x (occupancy for stochastic paths, 1 - 0.5 prior dependency otherwise)."""
    ev = t.get("evidence_mean", pd.Series(0.0, index=t.index)).fillna(0).to_numpy()
    occ = t.get("occupancy_mean", pd.Series(np.nan, index=t.index)).to_numpy()
    pri = t.get("prior_dependency", pd.Series(1.0, index=t.index)).fillna(1.0).to_numpy()
    return np.where(np.isfinite(occ), ev * (0.5 + 0.5 * occ), ev * (1.0 - 0.5 * pri)) - 1e-3 * t.get(
        "L_km", pd.Series(0.0, index=t.index)).fillna(0).to_numpy()


# =============================================================================== disc units: System C
def disc_units(m8, ev: np.ndarray, strike: np.ndarray) -> dict:
    """Per primary-task unit: stochastic + M7 deterministic hypotheses on the RETAINED map (union, provenance)."""
    from ..m7.network import proposals

    mc7 = m8.sh["map_completion"]
    dem = m8.dem_layer()
    out, occ = {}, {}
    for u in m8.units_of(m8.prim):
        cs_s, o, _ = stochastic(m8, u.known, u.ev, ev[u.sl], strike[u.sl], prefix=f"{u.key}/",
                                seed=int(m8.cfg["seed"]) + u.zone)
        cs_d, _ = proposals(u.known, u.ev, ev[u.sl], mc7, tail_px=int(m8.cfg["generation"]["tip_tail_px"]),
                            dem=None if dem is None else dem[u.sl], prefix=f"{u.key}/")
        t = pd.concat([cs_d.table.assign(origin="m7_deterministic"), cs_s.table.assign(origin="m8_stochastic")],
                      ignore_index=True)
        cs = CandidateSet(t, cs_d.paths + cs_s.paths)
        cs.table["cand_id"] = np.arange(len(cs))
        cs.table["completion_score"] = completion_score(cs.table)
        out[u.key] = cs
        occ[u.key] = o
        m8.say(f"s08 {u.key}: {len(cs_s)} stochastic + {len(cs_d)} deterministic hypotheses")
    return {"sets": out, "occupancy": occ}


def system_c(m8, sets: dict) -> dict:
    """System C held-out maps: hypotheses ranked by the label-free completion score, eligible budget, nested frac."""
    from ..gems3.metric_opt import nested_select
    from ..m7.protocol import score_row
    from ..m7.rasterize import predict, save_heldout

    fracs = m8.sh["rasterization"]["area_fracs"]
    rows = []
    for k, cs in sets.items():
        u = m8.unit(k)
        for origin in ("all", "m8_stochastic", "m7_deterministic"):
            idx = np.arange(len(cs)) if origin == "all" else np.flatnonzero(cs.table.origin.to_numpy() == origin)
            if not len(idx):
                continue
            sub = cs.subset(idx.tolist())
            for f in fracs:
                q, _ = predict(u, sub, sub.table.completion_score.to_numpy(), {"kind": "centerline"}, float(f), 0, m8.elig)
                rows.append(score_row(u, q, m8.elig(u, 0), m8.pkey(u), method=f"completion_{origin}", frac=float(f)))
    g = pd.DataFrame(rows)
    summ, per_all = [], []
    for mth, gg in g.groupby("method") if len(g) else []:
        per, s = nested_select(gg, ["frac"])
        per_all.append(per.assign(method=mth, task=m8.prim))
        summ.append({"task": m8.prim, "method": mth, "dti_crossfit": s["dti_crossfit"], "fold_min": s["fold_min"],
                     "frac_all_zones": (s.get("params_all") or {}).get("frac"),
                     **{f"dti_zone{r.zone}": r.dti_heldout for r in per.itertuples()}})
        if mth == "completion_all":
            for r in per.itertuples():
                u = m8.unit(f"{m8.prim}:{r.zone}")
                cs = sets[u.key]
                q, _ = predict(u, cs, cs.table.completion_score.to_numpy(), {"kind": "centerline"}, float(r.frac), 0, m8.elig)
                save_heldout(m8.ws, "m8_completion", u, q, m8.pkey(u))
    return {"curve": g, "summary": pd.DataFrame(summ),
            "per_zone": pd.concat(per_all, ignore_index=True) if per_all else pd.DataFrame()}


def coverage_audit(m8, sets: dict) -> pd.DataFrame:
    """EVALUATION: coverage of withheld systems per hypothesis family (oracle-style)."""
    from ..gems3.oracle import coverage
    from ..m7.coverage import _ocfg

    oo = _ocfg(m8.sh)
    rows = []
    for k, cs in sets.items():
        u = m8.unit(k)
        for fam in ["all"] + sorted(cs.table.source.astype(str).unique()):
            idx = np.arange(len(cs)) if fam == "all" else np.flatnonzero(cs.table.source.astype(str).to_numpy() == fam)
            rows.append({"unit": k, "task": u.task, "zone": u.zone, "family": fam,
                         **coverage(m8.ctx, u, cs.subset(idx.tolist()), oo)})
    return pd.DataFrame(rows)


# =============================================================================== corruption benchmark
def benchmark(m8, ev: np.ndarray, strike: np.ndarray) -> dict:
    """Stochastic vs M7 deterministic completers on synthetically damaged maps (leakage-controlled)."""
    from ..gems3.map_completion import systems_by_fold
    from ..gems3.official import _unit
    from ..m7.corruption import CLASSES, _rasters, corrupt_fold, metrics
    from ..m7.network import proposals

    ctx = m8.ctx
    mc7 = m8.sh["map_completion"]
    cc7 = mc7["corruption"]
    cc = m8.cfg["completion"]
    rng = np.random.default_rng(int(cc7["seed"]))
    lab, valid = np.asarray(ctx.data.labels, bool), np.asarray(ctx.data.valid, bool)
    sm, fm = np.asarray(ctx.splits.system_map), ctx.folds.fold_map
    sysf = systems_by_fold(sm, fm)
    v = ev[valid]
    q1, q2 = np.quantile(v, [1 / 3, 2 / 3]) if v.size else (0, 0)
    terc = np.digitize(ev, [q1, q2]).astype(np.int8)
    rows, proto = [], []
    for real in range(int(m8.profile["corruption_realisations"])):
        for k in m8.zones("geo"):
            dmg, removed, pr = corrupt_fold(lab, sm, sysf, k, rng, cc7)
            proto += [{**r, "realisation": real} for r in pr]
            anyrem = np.zeros(lab.shape, bool)
            for c in CLASSES:
                anyrem |= removed[c]
            if not anyrem.any():
                continue
            near = ndimage.distance_transform_edt(~anyrem, sampling=(0.1, 0.1)) <= float(cc7["eval_radius_km"])
            u0 = _unit("map_completion", k, near & valid, anyrem, dmg, False, dmg)
            sl = u0.sl
            det, _ = proposals(dmg[sl], valid[sl], ev[sl], mc7, tail_px=int(m8.cfg["generation"]["tip_tail_px"]),
                               near=near[sl], prefix=f"r{real}f{k}/")
            sto, occ, _ = stochastic(m8, dmg[sl], valid[sl], ev[sl], strike[sl], near=near[sl], prefix=f"r{real}f{k}/",
                                     seed=int(cc7["seed"]) + 97 * real + k)
            ras = {(f"m7_{fam}", L): q for (fam, L), q in _rasters(det, u0.ev.shape).items()}
            for thr in cc["occupancy_thresholds"]:
                ras[("m8_occupancy", float(thr))] = ((occ >= float(thr)) & ~dmg[sl]).astype(np.float32)
            if len(sto):
                cons = np.zeros(u0.ev.shape, np.float32)
                for p in sto.paths:
                    cons[p[:, 0], p[:, 1]] = 1.0
                ras[("m8_consensus", 0.0)] = cons
                for (f, L), q in list(ras.items()):
                    if f == "m7_union":
                        ras[("m8_consensus_plus_m7_union", L)] = np.maximum(cons, q)
            for cls in CLASSES:
                G = removed[cls]
                if not G.any():
                    continue
                zone = (ndimage.distance_transform_edt(~G, sampling=(0.1, 0.1)) <= float(cc7["eval_radius_km"])) & valid
                u = _unit(f"mc_{cls}", k, zone, G, dmg, False, dmg)
                for (fam, prm), q in ras.items():
                    full = np.zeros(lab.shape, np.float32)
                    full[sl] = q
                    rows.append({"realisation": real, "class": cls, "zone": k, "completer": fam, "param": prm,
                                 **metrics(u, full[u.sl], G[u.sl], lab[u.sl], terc[u.sl])})
            m8.say(f"s08 benchmark realisation {real} fold {k}: {len(det)} deterministic, {len(sto)} stochastic")
    res = pd.DataFrame(rows)
    return {"results": res, "protocol": pd.DataFrame(proto), "selection": select(res)}


def select(res: pd.DataFrame) -> pd.DataFrame:
    from ..gems3.metric_opt import nested_select

    rows = []
    if not len(res):
        return pd.DataFrame()
    for (cls, fam), g in res.groupby(["class", "completer"]):
        g2 = g.groupby(["zone", "param"])[["tp_w", "fp_w", "fn_w", "n_truth"]].sum().reset_index()
        per, s = nested_select(g2, ["param"])
        extra = g.groupby("param")[["median_dist_to_truth_km", "direction_agreement", "added_false_line_km"]].mean()
        p = (s.get("params_all") or {}).get("param", np.nan)
        rows.append({"class": cls, "completer": fam, "dti_crossfit": s["dti_crossfit"],
                     "recall_crossfit": float(per.tp_w.sum() / max(per.n_truth.sum(), 1)) if len(per) else np.nan,
                     "fold_min": s["fold_min"], "param_all_zones": p,
                     **({k: float(extra.loc[p, k]) for k in extra.columns} if p in extra.index else {})})
    t = pd.DataFrame(rows)
    if len(t):
        best_m7 = t[t.completer.str.startswith("m7_")].groupby("class").dti_crossfit.max()
        t["delta_vs_best_m7"] = t.dti_crossfit - t["class"].map(best_m7)
    return t
