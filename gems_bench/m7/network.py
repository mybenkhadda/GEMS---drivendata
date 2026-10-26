"""Section 08: the known fault map as an incomplete network, and geometric completion hypotheses.

Elements (on any "known" raster: all mapped faults for the full region, the retained faults of a discovery split,
or a synthetically damaged map in section 09)::

    segments    8-connected skeleton components: length, strike, endpoints, branch points
    endpoints   skeleton endpoints with the outward tangent (from the last ``tail_px`` pixels)
    gaps        endpoint pairs of different segments closer than ``gap_max_km`` whose tangents point at each other
    parallels   segment pairs with similar strike, a perpendicular offset in ``parallel_offset_km`` and along-strike
                overlap (en-echelon / step-over geometry)

Hypotheses (several per endpoint; none is forced)::

    ext_straight     straight continuation of the tangent, lengths ``extension_km``
    ext_rot<+/-a>    the same, rotated by +/- a degrees (``angle_hypotheses_deg``: curved / bending traces)
    ext_evidence     a walk that follows the strongest label-free evidence inside a +/- ``evidence_cone_deg`` cone
    gap_bridge       straight bridge across a gap

Each proposal records ``prior_dependency`` (share of its pixels within 1 px of the straight extrapolation: 1 =
pure geometry, lower = the evidence moved it) and ``evidence_mean`` (mean percentile of the evidence along it),
plus ``dem_alignment`` when a DEM mosaic is available. Proposals never read the truth.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

from ..gems3.candidates import CandidateSet

PIXEL_KM = 0.1


def evidence_layer(ctx, names: list[str], valid: np.ndarray, seed: int = 2026, n: int = 200_000) -> np.ndarray:
    """Mean percentile (0-1) of |layer| over the available label-free layers (full grid, float32)."""
    rng = np.random.default_rng(seed)
    acc, k = np.zeros(valid.shape, np.float32), 0
    for nm in names:
        if not ctx.store.has(nm):
            continue
        a = np.abs(np.asarray(ctx.store.get(nm), np.float32))
        v = a[valid & np.isfinite(a)]
        if not v.size:
            continue
        ref = np.sort(v[rng.choice(v.size, min(n, v.size), replace=False)])
        acc += np.where(np.isfinite(a), np.searchsorted(ref, a) / ref.size, 0.0).astype(np.float32)
        k += 1
    return (acc / max(k, 1)).astype(np.float32) if k else np.zeros(valid.shape, np.float32)


def segments(known: np.ndarray) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    from ..geology.skeleton import branch_points, endpoints, skeletonize
    from ..evaluation.geological_errors import strike_from_pixels

    sk = skeletonize(known)
    lab, n = ndimage.label(sk, structure=np.ones((3, 3), bool))
    ep, bp = endpoints(sk), branch_points(sk)
    rows = []
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None:
            continue
        m = lab[sl] == i
        rr, cc = np.nonzero(m)
        rows.append({"segment": i, "n_px": int(m.sum()), "length_km": float(m.sum() * PIXEL_KM * 1.1),
                     "strike_deg": strike_from_pixels(rr + sl[0].start, cc + sl[1].start),
                     "r_mid": float(rr.mean() + sl[0].start), "c_mid": float(cc.mean() + sl[1].start),
                     "n_endpoints": int((ep[sl] & m).sum()), "n_branch_points": int((bp[sl] & m).sum())})
    return pd.DataFrame(rows), sk, lab


def endpoint_table(sk: np.ndarray, tail_px: int) -> pd.DataFrame:
    from ..postprocess.topology import _endpoint_tangents

    e = _endpoint_tangents(sk, tail_px)
    return pd.DataFrame(e, columns=["r", "c", "dir_deg", "segment"]).assign(endpoint=lambda d: np.arange(len(d)))


def gap_table(ep: pd.DataFrame, max_km: float, max_angle: float) -> pd.DataFrame:
    from scipy.spatial import cKDTree

    if len(ep) < 2:
        return pd.DataFrame(columns=["e1", "e2", "gap_km", "angle1", "angle2"])
    tree = cKDTree(ep[["r", "c"]].to_numpy(float))
    rows = []
    for i, j in sorted(map(tuple, tree.query_pairs(r=max_km / PIXEL_KM, output_type="ndarray"))):
        a, b = ep.iloc[i], ep.iloc[j]
        if a.segment == b.segment:
            continue
        d12 = float(np.degrees(np.arctan2(-(b.r - a.r), b.c - a.c)) % 360.0)
        a1 = abs((d12 - a.dir_deg + 180) % 360 - 180)
        a2 = abs(((d12 + 180) % 360 - b.dir_deg + 180) % 360 - 180)
        if a1 <= max_angle and a2 <= max_angle:
            rows.append({"e1": int(a.endpoint), "e2": int(b.endpoint), "gap_km": float(np.hypot(b.r - a.r, b.c - a.c)
                                                                                       * PIXEL_KM),
                         "angle1": a1, "angle2": a2})
    return pd.DataFrame(rows)


def parallel_table(seg: pd.DataFrame, off_km: tuple[float, float], max_angle: float) -> pd.DataFrame:
    """Segment pairs: |strike difference| <= max_angle, perpendicular offset in [lo, hi] km, along-strike overlap."""
    from scipy.spatial import cKDTree

    s = seg[seg.length_km >= 0.5].reset_index(drop=True)
    if len(s) < 2:
        return pd.DataFrame()
    lo, hi = off_km
    tree = cKDTree(s[["r_mid", "c_mid"]].to_numpy(float))
    rows = []
    for i, j in tree.query_pairs(r=(hi + s.length_km.max() / 2) / PIXEL_KM):
        a, b = s.iloc[i], s.iloc[j]
        da = abs((a.strike_deg - b.strike_deg + 90) % 180 - 90)
        if da > max_angle:
            continue
        t = np.radians(a.strike_deg)
        along = np.array([-np.cos(t), np.sin(t)])          # (row, col) unit vector along strike (azimuth from north)
        nrm = np.array([along[1], -along[0]])
        dv = np.array([b.r_mid - a.r_mid, b.c_mid - a.c_mid]) * PIXEL_KM
        perp, par = abs(float(dv @ nrm)), abs(float(dv @ along))
        overlap = (a.length_km + b.length_km) / 2 - par
        if lo <= perp <= hi and overlap > 0:
            rows.append({"seg1": int(a.segment), "seg2": int(b.segment), "strike_diff": da, "offset_km": perp,
                         "overlap_km": overlap})
    return pd.DataFrame(rows)


def _ray(r: int, c: int, theta_deg: float, n_px: int, shape) -> np.ndarray:
    from skimage.draw import line as draw_line

    t = np.radians(theta_deg)
    r1, c1 = int(round(r - n_px * np.sin(t))), int(round(c + n_px * np.cos(t)))
    rr, cc = draw_line(int(r), int(c), r1, c1)
    ok = (rr >= 0) & (rr < shape[0]) & (cc >= 0) & (cc < shape[1])
    return np.c_[rr[ok], cc[ok]][1:]


def _walk(r: int, c: int, theta_deg: float, n_px: int, ev: np.ndarray, cone: float, valid: np.ndarray,
          straight_bias: float = 0.15) -> np.ndarray:
    """Evidence-following path: at each 1-px step choose the heading (within +/- cone of the initial tangent and
    +/- 15 deg of the current heading) with the best evidence one step ahead, mildly biased to straight."""
    h, w = ev.shape
    y, x = float(r), float(c)
    head = theta_deg
    out = []
    for _ in range(n_px):
        best, bv = head, -np.inf
        for dh in (-15.0, -7.5, 0.0, 7.5, 15.0):
            hh = head + dh
            if abs(((hh - theta_deg) + 180) % 360 - 180) > cone:
                continue
            t = np.radians(hh)
            yy, xx = y - np.sin(t), x + np.cos(t)
            iy, ix = int(round(yy)), int(round(xx))
            if not (0 <= iy < h and 0 <= ix < w) or not valid[iy, ix]:
                continue
            v = float(ev[iy, ix]) - straight_bias * abs(dh) / 15.0
            if v > bv:
                bv, best = v, hh
        if not np.isfinite(bv):
            break
        t = np.radians(best)
        y, x, head = y - np.sin(t), x + np.cos(t), best
        p = (int(round(y)), int(round(x)))
        if not out or out[-1] != p:
            out.append(p)
    return np.array(out, int).reshape(-1, 2)


def proposals(known: np.ndarray, valid: np.ndarray, evidence: np.ndarray, mc: dict, *, tail_px: int = 5,
              near: np.ndarray | None = None, dem: np.ndarray | None = None, prefix: str = "") -> tuple[CandidateSet, dict]:
    """Competing completion hypotheses of one known map (paths in the arrays' coordinates) + the element tables."""
    seg, sk, _ = segments(known)
    ep = endpoint_table(sk, tail_px)
    if near is not None and len(ep):
        ep = ep[near[ep.r.to_numpy(), ep.c.to_numpy()]].reset_index(drop=True)
    gaps = gap_table(ep, float(mc["gap_max_km"]), float(mc["gap_max_angle_deg"]))
    par = parallel_table(seg, tuple(mc["parallel_offset_km"]), float(mc["parallel_max_angle_deg"]))
    Ls = sorted(float(x) for x in mc["extension_km"])
    nmax = int(round(Ls[-1] / PIXEL_KM))
    rows, paths = [], []
    shape = known.shape

    def add(p, src, e, L, ang, straight):
        p = p[valid[p[:, 0], p[:, 1]] & ~known[p[:, 0], p[:, 1]]] if len(p) else p
        if len(p) < 3:
            return
        sset = set(map(tuple, straight.tolist())) if straight is not None and len(straight) else set()
        dil = {(a + dr, b + dc) for a, b in sset for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
        prior = float(np.mean([tuple(q) in dil for q in p.tolist()])) if dil else 1.0
        rows.append({"uid": f"{prefix}{src}:{int(e.endpoint) if e is not None else -1}:{L:g}",
                     "source": src, "modality": "network", "kind": "completion", "mode": "hypothesis",
                     "hypothesis_group": f"{prefix}ep{int(e.endpoint)}" if e is not None else f"{prefix}gap",
                     "L_km": L, "angle_offset": ang, "prior_dependency": prior,
                     "evidence_mean": float(evidence[p[:, 0], p[:, 1]].mean()),
                     "dem_alignment": float(np.nanmean(dem[p[:, 0], p[:, 1]])) if dem is not None and
                     np.isfinite(dem[p[:, 0], p[:, 1]]).any() else np.nan,
                     "scale_m": np.nan, "network_id": f"{prefix}seg{int(e.segment) if e is not None else -1}"})
        paths.append(p)

    for e in ep.itertuples():
        straight_full = _ray(e.r, e.c, e.dir_deg, nmax, shape)
        for ang in mc["angle_hypotheses_deg"]:
            full = straight_full if float(ang) == 0.0 else _ray(e.r, e.c, e.dir_deg + float(ang), nmax, shape)
            for L in Ls:
                n = int(round(L / PIXEL_KM))
                add(full[:n], "ext_straight" if float(ang) == 0.0 else f"ext_rot{float(ang):+g}", e, L, float(ang),
                    straight_full[:n])
        if mc.get("evidence_guided", True):
            wk = _walk(e.r, e.c, e.dir_deg, nmax, evidence, float(mc["evidence_cone_deg"]), valid)
            for L in Ls:
                n = int(round(L / PIXEL_KM))
                add(wk[:n], "ext_evidence", e, L, np.nan, straight_full[:n])
    from skimage.draw import line as draw_line

    epi = ep.set_index("endpoint")
    for g in gaps.itertuples():
        a, b = epi.loc[g.e1], epi.loc[g.e2]
        rr, cc = draw_line(int(a.r), int(a.c), int(b.r), int(b.c))
        p = np.c_[rr, cc]
        add(p, "gap_bridge", None, float(g.gap_km), 0.0, p)
    t = pd.DataFrame(rows)
    return CandidateSet(t, paths), {"segments": seg, "endpoints": ep, "gaps": gaps, "parallels": par}


def run(m7) -> dict:
    """Full-region elements + proposals (known = all mapped faults) and per-disc-unit proposals (known = retained)."""
    ctx, cfg = m7.ctx, m7.cfg
    mc = cfg["map_completion"]
    valid = ctx.data.valid
    ev = evidence_layer(ctx, mc["evidence_layers"], valid, int(cfg["seed"]))
    dem = m7.dem_layer()
    full, tabs = proposals(np.asarray(ctx.data.labels, bool), valid, ev, mc, tail_px=int(cfg["candidates"]["tip_tail_px"]),
                           dem=dem, prefix="full/")
    per_unit = {}
    for u in m7.units_of(cfg["evaluation"]["primary_task"]):
        cs, _ = proposals(u.known, u.ev, ev[u.sl], mc, tail_px=int(cfg["candidates"]["tip_tail_px"]),
                          dem=None if dem is None else dem[u.sl], prefix=f"{u.key}/")
        per_unit[u.key] = cs
        print(f"[m7] s08 {u.key}: {len(cs)} completion hypotheses")
    summary = pd.DataFrame([{"unit": "full", "n_segments": len(tabs["segments"]), "n_endpoints": len(tabs["endpoints"]),
                             "n_gaps": len(tabs["gaps"]), "n_parallel_pairs": len(tabs["parallels"]),
                             "n_hypotheses": len(full)}] +
                           [{"unit": k, "n_hypotheses": len(v)} for k, v in per_unit.items()])
    return {"full": full, "tables": tabs, "per_unit": per_unit, "summary": summary, "evidence": ev}


def completion_baseline(m7, per_unit: dict) -> dict:
    """Deterministic map-completion baseline on the disc units: hypotheses ranked by their label-free evidence
    (ties: shorter first, i.e. closer to the known tip), 1-px lines, eligible budget, frac chosen on other splits.
    Geo folds have no visible known map (their truth IS the mapped faults), so the method is disc-only."""
    from ..gems3.metric_opt import nested_select
    from .protocol import score_row
    from .rasterize import predict, save_heldout

    fracs = m7.cfg["rasterization"]["area_fracs"]
    rows = []
    for k, cs in per_unit.items():
        u = m7.unit(k)
        if u is None or not len(cs):
            continue
        s = cs.table.evidence_mean.fillna(0).to_numpy() - 1e-3 * cs.table.L_km.fillna(0).to_numpy()
        for f in fracs:
            q, _ = predict(u, cs, s, {"kind": "centerline"}, float(f), 0, m7.elig)
            rows.append(score_row(u, q, m7.elig(u, 0), m7.pkey(u), method="mc_completion_baseline", frac=float(f)))
    g = pd.DataFrame(rows)
    if not len(g):
        return {"curve": g, "summary": pd.DataFrame()}
    per, s = nested_select(g, ["frac"])
    for r in per.itertuples():
        u = m7.unit(f"{g.task.iloc[0]}:{r.zone}")
        cs = per_unit[u.key]
        sc = cs.table.evidence_mean.fillna(0).to_numpy() - 1e-3 * cs.table.L_km.fillna(0).to_numpy()
        q, _ = predict(u, cs, sc, {"kind": "centerline"}, float(r.frac), 0, m7.elig)
        save_heldout(m7.ws, "mc_completion_baseline", u, q, m7.pkey(u))
    summ = pd.DataFrame([{"task": g.task.iloc[0], "method": "mc_completion_baseline", "dti_crossfit": s["dti_crossfit"],
                          "fold_min": s["fold_min"], "frac_all_zones": s["params_all"]["frac"],
                          **{f"dti_zone{r.zone}": r.dti_heldout for r in per.itertuples()}}])
    return {"curve": g, "summary": summ}


def unit_completion_audit(m7, per_unit: dict) -> pd.DataFrame:
    """Coverage of the withheld systems by the completion hypotheses of each disc unit (oracle-style, diagnostic),
    per hypothesis family: how much new truth is reachable by extending / bridging the RETAINED map."""
    from ..gems3.oracle import coverage
    from .coverage import _ocfg

    rows = []
    oo = _ocfg(m7.cfg)
    for k, cs in per_unit.items():
        u = m7.unit(k)
        if u is None or not len(cs):
            continue
        for src in ["all"] + sorted(cs.table.source.unique()):
            idx = np.arange(len(cs)) if src == "all" else np.flatnonzero(cs.table.source.to_numpy() == src)
            rows.append({"unit": k, "task": u.task, "zone": u.zone, "family": src,
                         **coverage(m7.ctx, u, cs.subset(idx.tolist()), oo)})
    return pd.DataFrame(rows)
