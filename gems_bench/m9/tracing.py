"""Section 11: structural fault tracing on the dense model's held-out predictions, oracle coverage (Gate 4).

Inputs per unit: rank-normalised dense evidence E (the unit's own held-out seg map), the predicted axial strike
field theta (orientation head, doubled-angle blend), endpoint / junction heads. Label-free: no truth is read until
the oracle.

Tracers
    crest_mcp     deterministic baseline: crest pixels of E above a per-unit quantile -> skeleton -> linked paths
                  (``m7.generator.mask_to_paths``) + minimum-cost bridges (``skimage.graph.route_through_array``,
                  cost 1 / (0.05 + E)) between fragment endpoints closer than ``gap_px`` whose tangents agree.
    beam          orientation-aware beam search, vectorised over every seed: seeds = crest maxima of E above
                  ``seed_quantile`` (spatial NMS), started in both strike directions; per step every beam tries
                  headings phi + k x turn (k = -n..n), score += w_e log(E) + w_o (|cos(phi - theta)| - 1) - w_c |k|;
                  steps with E below the ``gap_quantile`` count as gaps (``max_gap_px`` consecutive gaps end a beam);
                  the best ``beam_width`` beams survive. Paths end where the best beam dies (trailing gaps trimmed).
    beam_branch   where two surviving beams of one trace diverge by >= ``branch.min_angle_deg`` with comparable
                  scores, the divergence point is re-traced along the second heading (branch hypothesis).
    beam_stoch    the same with Gumbel noise of ``stochastic.temperature`` on every step (diverse alternatives;
                  Monte Carlo uses it for path occupancy).
Nothing is thresholded away permanently: every path keeps its evidence statistics for the ranker, and the dense map
remains the fallback channel.

Oracle (diagnostic upper bound, reads the truth): candidates ordered by label-free confidence (mean E x length)
are added until a density cap (km per 100 km^2); system recall (official support >= 0.5 on >= half the pixels), and
systems recovered by the traces but NOT by the dense map at the same number of pixels. Gate 4: CONTINUE when the
recall at ``oracle.gate_density`` >= ``g4.min_system_recall`` AND additional systems >= ``g4.min_additional_systems``.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from scipy import ndimage

PX_KM = 0.1
CHUNK = 2048                                                              # seeds traced together


# =============================================================================== fields
def fields(pred: dict, ev: np.ndarray, sigma_px: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(E rank in [0, 1] inside ev, strike rad clockwise from north, orientation confidence)."""
    from ..gems3.representation import rank01

    s = np.nan_to_num(pred["seg"])
    if sigma_px > 0:
        s = ndimage.gaussian_filter(s, sigma_px)
    E = rank01(s, ev)
    c, sn = np.nan_to_num(pred["ocos"]), np.nan_to_num(pred["osin"])
    theta = 0.5 * np.arctan2(sn, c)                                       # strike (rad), axial
    return E.astype(np.float32), theta.astype(np.float32), np.hypot(c, sn).astype(np.float32)


def seeds(E: np.ndarray, ev: np.ndarray, q: float, nms_px: int, cap: int, sigma_px: float = 1.0) -> np.ndarray:
    from ..gems3.representation import crest_map

    cr = crest_map(E, ev, max(sigma_px, 0.7))
    v = E[ev]
    thr = float(np.quantile(v, q)) if v.size else 1.0
    cand = cr & (E >= thr)
    loc = ndimage.maximum_filter(np.where(cand, E, 0.0), size=2 * int(nms_px) + 1)
    pk = np.argwhere(cand & (E >= loc - 1e-7))
    if len(pk) > cap:
        o = np.argsort(-E[pk[:, 0], pk[:, 1]], kind="stable")[:cap]
        pk = pk[o]
    return pk


# =============================================================================== beam search
def beam_trace(E: np.ndarray, theta: np.ndarray, valid: np.ndarray, starts: np.ndarray, headings: np.ndarray,
               tc: dict, *, temperature: float = 0.0, rng=None, max_steps: int | None = None,
               gap_thr: float = 0.0) -> tuple[list[np.ndarray], np.ndarray, list[tuple]]:
    """Batched beam search. ``starts`` (S, 2) float rows / cols, ``headings`` (S,) rad (clockwise from north).
    Returns (paths as (n, 2) float arrays incl. the start, total score per path, branch events (s, step, r, c, phi))."""
    S = len(starts)
    if S == 0:
        return [], np.zeros(0), []
    B, n = int(tc["beam_width"]), int(tc["n_turns"])
    turns = np.radians(float(tc["turn_deg"])) * np.arange(-n, n + 1)
    K = len(turns)
    step = float(tc["step_px"])
    T = int(max_steps or round(float(tc["max_len_km"]) / PX_KM / step))
    we, wo, wc = float(tc["w_evidence"]), float(tc["w_orient"]), float(tc["w_curvature"])
    mg = int(tc["max_gap_px"])
    br = tc["branch"]
    H, W = E.shape
    r = np.repeat(starts[:, :1].astype(np.float32), B, 1)
    c = np.repeat(starts[:, 1:].astype(np.float32), B, 1)
    phi = np.repeat(headings[:, None].astype(np.float32), B, 1)
    score = np.full((S, B), -np.inf, np.float32)
    score[:, 0] = 0.0
    gaps = np.zeros((S, B), np.int16)
    pos = np.zeros((T + 1, S, B, 2), np.float32)
    par = np.zeros((T + 1, S, B), np.int16)
    pos[0, :, :, 0], pos[0, :, :, 1] = r, c
    last = np.zeros(S, np.int32)
    final_b = np.zeros(S, np.int32)
    alive_seed = np.ones(S, bool)
    events = []
    for t in range(1, T + 1):
        ph = phi[:, :, None] + turns[None, None, :]                       # (S, B, K)
        rr = r[:, :, None] - np.cos(ph) * step
        cc = c[:, :, None] + np.sin(ph) * step
        inside = (rr >= 0) & (rr <= H - 1) & (cc >= 0) & (cc <= W - 1)
        ri, ci = np.clip(np.rint(rr).astype(int), 0, H - 1), np.clip(np.rint(cc).astype(int), 0, W - 1)
        e = ndimage.map_coordinates(E, [rr.ravel(), cc.ravel()], order=1, mode="constant", cval=0.0).reshape(rr.shape)
        th = theta[ri, ci]
        al = np.abs(np.cos(ph - th))
        inside &= valid[ri, ci]
        g = np.where(e < gap_thr, gaps[:, :, None] + 1, 0)
        inc = we * np.log(e + 1e-3) + wo * (al - 1.0) - wc * np.abs(turns)[None, None, :]
        if temperature > 0 and rng is not None:
            inc = inc + temperature * rng.gumbel(size=inc.shape).astype(np.float32)
        new = np.where(inside & (g <= mg) & np.isfinite(score[:, :, None]), score[:, :, None] + inc, -np.inf)
        flat = new.reshape(S, B * K)
        k = min(B, B * K)
        top = np.argpartition(-flat, k - 1, axis=1)[:, :k]
        ts = np.take_along_axis(flat, top, 1)
        order = np.argsort(-ts, axis=1)
        top, ts = np.take_along_axis(top, order, 1), np.take_along_axis(ts, order, 1)
        pb, pk = top // K, top % K
        ok = np.isfinite(ts).any(1) & alive_seed
        if br.get("enabled") and B > 1:
            # branch: the runner-up beam turns away (>= min angle) while its score is within -log(ratio) of the best
            hd = np.take_along_axis(ph.reshape(S, -1), top[:, :2], 1)
            d = np.abs(np.angle(np.exp(1j * (hd[:, 0] - hd[:, 1]))))
            close = (ts[:, 0] - ts[:, 1]) <= -np.log(max(float(br["min_score_ratio"]), 1e-6))
            cond = ok & np.isfinite(ts[:, 1]) & (d >= np.radians(float(br["min_angle_deg"]))) & close
            for s in np.flatnonzero(cond)[:64]:
                j = top[s, 1]
                events.append((int(s), t, float(rr.reshape(S, -1)[s, j]), float(cc.reshape(S, -1)[s, j]),
                               float(ph.reshape(S, -1)[s, j])))
        last[ok] = t
        final_b[ok] = 0
        alive_seed &= ok
        if not alive_seed.any():
            break
        sel = lambda a: np.take_along_axis(a.reshape(S, -1), top, 1)
        r = np.where(alive_seed[:, None], sel(rr), r)
        c = np.where(alive_seed[:, None], sel(cc), c)
        phi = np.where(alive_seed[:, None], sel(ph), phi)
        gaps = np.where(alive_seed[:, None], sel(np.broadcast_to(g, rr.shape)), gaps).astype(np.int16)
        score = np.where(alive_seed[:, None], ts, score)
        pos[t, :, :, 0], pos[t, :, :, 1] = r, c
        par[t] = np.where(alive_seed[:, None], pb, np.arange(B)[None, :])
    paths, scores = [], np.zeros(S, np.float32)
    for s in range(S):
        L, b = int(last[s]), int(final_b[s])
        pts = np.empty((L + 1, 2), np.float32)
        for t in range(L, -1, -1):
            pts[t] = pos[t, s, b]
            b = int(par[t, s, b]) if t > 0 else b
        paths.append(pts)
        scores[s] = score[s, 0] if np.isfinite(score[s, 0]) else 0.0
    return paths, scores, events


def _trim(path: np.ndarray, E: np.ndarray, gap_thr: float) -> np.ndarray:
    """Drop trailing low-evidence (gap) steps."""
    if len(path) < 2:
        return path
    ri = np.clip(np.rint(path[:, 0]).astype(int), 0, E.shape[0] - 1)
    ci = np.clip(np.rint(path[:, 1]).astype(int), 0, E.shape[1] - 1)
    ok = np.flatnonzero(E[ri, ci] >= gap_thr)
    return path[: ok[-1] + 1] if len(ok) else path[:1]


def to_pixels(path: np.ndarray, shape) -> np.ndarray:
    """Float path -> unique, 8-connected (n, 2) int pixel path (consecutive duplicates removed)."""
    if not len(path):
        return np.zeros((0, 2), np.int32)
    seg = [path[0]]
    for a, b in zip(path[:-1], path[1:]):
        n = int(np.ceil(np.max(np.abs(b - a)))) or 1
        seg.extend(a + (b - a) * (np.arange(1, n + 1)[:, None] / n))
    p = np.rint(np.asarray(seg)).astype(np.int32)
    p[:, 0] = np.clip(p[:, 0], 0, shape[0] - 1)
    p[:, 1] = np.clip(p[:, 1], 0, shape[1] - 1)
    keep = np.r_[True, np.any(np.diff(p, axis=0) != 0, axis=1)]
    return p[keep]


def trace_unit(E, theta, valid, tc: dict, cap: int, *, temperature: float = 0.0, rng=None, source: str = "beam"):
    """Both-direction beam traces from every seed (+ branch re-traces). Returns (list of int paths, rows)."""
    v = E[valid]
    gap_thr = float(np.quantile(v, float(tc["gap_quantile"]))) if v.size else 0.0
    sd = seeds(E, valid, float(tc["seed_quantile"]), int(tc["seed_nms_px"]), cap)
    if not len(sd):
        return [], []
    th0 = theta[sd[:, 0], sd[:, 1]]
    st = np.r_[sd, sd].astype(np.float32)
    hd = np.r_[th0, th0 + np.pi]
    paths, sc, ev = [], [], []
    for a in range(0, len(st), CHUNK):                                    # bounded memory: (T, chunk, B, 2) floats
        p, s, e = beam_trace(E, theta, valid, st[a:a + CHUNK], hd[a:a + CHUNK], tc, temperature=temperature, rng=rng,
                             gap_thr=gap_thr)
        paths += p
        sc.append(s)
        ev += [(i + a, t, r, c, ph) for i, t, r, c, ph in e]
    sc = np.concatenate(sc) if sc else np.zeros(0)
    n = len(sd)
    out, rows = [], []
    for i in range(n):
        fwd, bwd = _trim(paths[i], E, gap_thr), _trim(paths[n + i], E, gap_thr)
        full = np.r_[bwd[::-1], fwd[1:]] if len(fwd) > 1 else bwd[::-1]
        out.append(to_pixels(full, E.shape))
        rows.append({"source": source, "seed_r": int(sd[i, 0]), "seed_c": int(sd[i, 1]), "beam_score": float(sc[i] + sc[n + i]),
                     "parent": -1})
    if tc["branch"].get("enabled") and ev:
        per = {}
        for s, t, r, c, ph in ev:
            per.setdefault(s % n, []).append((t, r, c, ph))
        bst, bhd, bpar = [], [], []
        for s, lst in per.items():
            lst = sorted(lst)[: int(tc["branch"]["max_per_trace"])]
            for t, r, c, ph in lst:
                bst.append((r, c))
                bhd.append(ph)
                bpar.append(s)
        if bst:
            bp, bs, _ = beam_trace(E, theta, valid, np.asarray(bst, np.float32), np.asarray(bhd, np.float32),
                                   {**tc, "branch": {"enabled": False}}, temperature=temperature, rng=rng, gap_thr=gap_thr)
            for p, s_, parent in zip(bp, bs, bpar):
                p = _trim(p, E, gap_thr)
                out.append(to_pixels(p, E.shape))
                rows.append({"source": f"{source}_branch", "seed_r": int(round(p[0, 0])), "seed_c": int(round(p[0, 1])),
                             "beam_score": float(s_), "parent": int(parent)})
    return out, rows


# =============================================================================== deterministic baseline
def crest_mcp(E: np.ndarray, valid: np.ndarray, tc: dict) -> tuple[list[np.ndarray], list[dict]]:
    from skimage.graph import route_through_array

    from ..gems3.representation import crest_map
    from ..m7.generator import mask_to_paths
    from ..postprocess.topology import _endpoint_tangents

    mc = tc["mcp_baseline"]
    v = E[valid]
    thr = float(np.quantile(v, float(mc["crest_quantile"]))) if v.size else 1.0
    mask = crest_map(E, valid, 1.0) & (E >= thr)
    frag, meta = mask_to_paths(mask, {"min_len_km": float(tc["min_len_km"]) / 2, "link_gap_px": 0, "link_angle_deg": 30.0})
    rows = [{"source": "crest", "seed_r": int(p[0, 0]), "seed_c": int(p[0, 1]), "beam_score": np.nan, "parent": -1}
            for p in frag]
    sk = np.zeros(E.shape, bool)
    for p in frag:
        sk[p[:, 0], p[:, 1]] = True
    tips = _endpoint_tangents(sk, 5)
    if len(tips) < 2:
        return frag, rows
    from scipy.spatial import cKDTree

    pts = np.array([(r, c) for r, c, _, _ in tips], float)
    ang = np.array([d for _, _, d, _ in tips], float)                    # outward direction (deg CCW from +col)
    comp = np.array([k for _, _, _, k in tips])
    tree = cKDTree(pts)
    cost = (1.0 / (0.05 + E)).astype(np.float64)
    cost[~valid] = 1e6
    pad = int(mc["pad_px"])
    done = set()
    for i, j in tree.query_pairs(float(mc["gap_px"])):
        if comp[i] == comp[j] or (j, i) in done:
            continue
        v_ij = pts[j] - pts[i]
        a_ij = np.degrees(np.arctan2(-v_ij[0], v_ij[1])) % 360
        d1 = abs((ang[i] - a_ij + 180) % 360 - 180)
        d2 = abs((ang[j] - (a_ij + 180)) % 360 - 180)
        if max(d1, d2) > float(mc["max_angle_deg"]):
            continue
        r0, r1 = int(min(pts[i, 0], pts[j, 0])) - pad, int(max(pts[i, 0], pts[j, 0])) + pad + 1
        c0, c1 = int(min(pts[i, 1], pts[j, 1])) - pad, int(max(pts[i, 1], pts[j, 1])) + pad + 1
        r0, c0 = max(r0, 0), max(c0, 0)
        win = cost[r0:r1, c0:c1]
        a = (int(pts[i, 0]) - r0, int(pts[i, 1]) - c0)
        b = (int(pts[j, 0]) - r0, int(pts[j, 1]) - c0)
        try:
            idx, _ = route_through_array(win, a, b, fully_connected=True, geometric=True)
        except Exception:
            continue
        p = np.asarray(idx, np.int32) + np.array([r0, c0])
        frag.append(p)
        rows.append({"source": "mcp_bridge", "seed_r": int(p[0, 0]), "seed_c": int(p[0, 1]), "beam_score": np.nan,
                     "parent": -1})
        done.add((i, j))
    return frag, rows


# =============================================================================== describe / dedup
def describe(paths: list[np.ndarray], rows: list[dict], E, theta, conf, ep, jn, unit_key: str, r0c0=(0, 0),
             gap_thr: float = 0.0) -> pd.DataFrame:
    out = []
    for p, r in zip(paths, rows):
        if len(p) < 2:
            out.append({**r, "n_px": len(p), "length_km": 0.0})
            continue
        e = E[p[:, 0], p[:, 1]]
        d = np.diff(p.astype(float), axis=0)
        phi = np.arctan2(d[:, 1], -d[:, 0])                               # path direction, clockwise from north
        al = np.abs(np.cos(phi - theta[p[1:, 0], p[1:, 1]]))
        turn = np.abs(np.angle(np.exp(1j * np.diff(phi)))) if len(phi) > 1 else np.zeros(1)
        L = float(np.hypot(d[:, 0], d[:, 1]).sum() * PX_KM)
        key = f"{unit_key}|{r['source']}|{int(p[0, 0]) + r0c0[0]},{int(p[0, 1]) + r0c0[1]}|{int(p[-1, 0]) + r0c0[0]}," \
              f"{int(p[-1, 1]) + r0c0[1]}|{len(p)}"
        out.append({**r, "uid": hashlib.sha1(key.encode()).hexdigest()[:12], "n_px": int(len(p)), "length_km": L,
                    "e_mean": float(e.mean()), "e_min": float(e.min()), "e_p10": float(np.quantile(e, 0.1)),
                    "e_p90": float(np.quantile(e, 0.9)), "align_mean": float(al.mean()) if len(al) else np.nan,
                    "orient_conf_mean": float(conf[p[:, 0], p[:, 1]].mean()),
                    "turn_mean_deg": float(np.degrees(turn.mean())), "gap_frac": float((e < gap_thr).mean()),
                    "ep_support": float(max(ep[p[0, 0], p[0, 1]], ep[p[-1, 0], p[-1, 1]])),
                    "jn_support": float(jn[p[:, 0], p[:, 1]].max()), "confidence_lf": float(e.mean() * np.sqrt(max(L, 1e-3)))})
    return pd.DataFrame(out)


def dedup(paths: list[np.ndarray], table: pd.DataFrame, shape, radius_px: int, overlap: float, min_px: int):
    """Greedy (highest label-free confidence first) removal of paths mostly inside already kept ones."""
    from ..gems3.candidates import CandidateSet

    taken = np.zeros(shape, bool)
    offs = [(dr, dc) for dr in range(-radius_px, radius_px + 1) for dc in range(-radius_px, radius_px + 1)
            if dr * dr + dc * dc <= radius_px * radius_px]
    keep = []
    for i in np.argsort(-table.confidence_lf.fillna(0).to_numpy(), kind="stable"):
        p = paths[i]
        if len(p) < min_px:
            continue
        if taken[p[:, 0], p[:, 1]].mean() >= overlap:
            continue
        keep.append(i)
        for dr, dc in offs:
            rr = np.clip(p[:, 0] + dr, 0, shape[0] - 1)
            cc = np.clip(p[:, 1] + dc, 0, shape[1] - 1)
            taken[rr, cc] = True
    keep = sorted(keep)
    t = table.iloc[keep].reset_index(drop=True).copy()
    t["cand_id"] = np.arange(len(t))
    return CandidateSet(t, [paths[i] for i in keep])


def unit_candidates(m9, u, pred: dict, *, stochastic: bool = False, seed: int = 0, extra_sources: bool = True):
    """Every tracer on one unit -> deduplicated CandidateSet (crop coordinates) + per-source log."""
    tc = m9.cfg["tracing"]
    ev = u.scored_mask()
    E, theta, conf = fields(pred, ev, float(tc["evidence_sigma_px"]))
    gap_thr = float(np.quantile(E[ev], float(tc["gap_quantile"]))) if ev.any() else 0.0
    ep, jn = np.nan_to_num(pred["ep"]), np.nan_to_num(pred["jn"])
    cap = int(m9.profile["max_seeds_per_unit"])
    paths, rows, log = [], [], []
    p, r = trace_unit(E, theta, ev, tc, cap)
    paths += p
    rows += r
    log.append({"unit": u.key, "source": "beam", "n": len(p)})
    if stochastic and tc["stochastic"].get("enabled"):
        rng = np.random.default_rng(seed)
        p, r = trace_unit(E, theta, ev, tc, cap, temperature=float(tc["stochastic"]["temperature"]), rng=rng,
                          source="beam_stoch")
        paths += p
        rows += r
        log.append({"unit": u.key, "source": "beam_stoch", "n": len(p)})
    if extra_sources and tc["mcp_baseline"].get("enabled"):
        p, r = crest_mcp(E, ev, tc)
        paths += p
        rows += r
        log.append({"unit": u.key, "source": "crest_mcp", "n": len(p)})
    T = describe(paths, rows, E, theta, conf, ep, jn, u.key, (u.sl[0].start, u.sl[1].start), gap_thr)
    min_px = max(int(round(float(tc["min_len_km"]) / (PX_KM * 1.12))), 2)
    cs = dedup(paths, T, E.shape, int(tc["dedup"]["radius_px"]), float(tc["dedup"]["overlap"]), min_px)
    cs.table["unit"], cs.table["task"], cs.table["zone"] = u.key, u.task, u.zone
    cs.table["r_full"] = [float(p[len(p) // 2, 0] + u.sl[0].start) for p in cs.paths]
    cs.table["c_full"] = [float(p[len(p) // 2, 1] + u.sl[1].start) for p in cs.paths]
    cs.table["network_id"] = cs.table.uid
    log.append({"unit": u.key, "source": "deduplicated", "n": len(cs)})
    return cs, pd.DataFrame(log), {"E": E, "theta": theta}


# =============================================================================== oracle (Gate 4)
def oracle_unit(m9, u, cs, dense_rank: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recall-density curve of the traces + systems they add over the dense map at the same pixel count."""
    from ..gems3.representation import top_mask
    from ..m7.protocol import system_hits

    caps = m9.cfg["tracing"]["oracle"]["density_caps_km_per_100km2"]
    area_100km2 = max(u.ev.sum() * 0.01 / 100.0, 1e-9)
    order = np.argsort(-cs.table.confidence_lf.fillna(0).to_numpy(), kind="stable")
    cum = np.cumsum(cs.table.length_km.to_numpy()[order])
    rows, hits = [], []
    el = m9.elig(u, 0)
    for cap in caps:
        k = int(np.searchsorted(cum, cap * area_100km2, side="right"))
        q = cs.subset(order[:k]).raster(u.ev.shape) if k else np.zeros(u.ev.shape, np.float32)
        q = np.where(u.ev, q, 0.0)
        h = system_hits(m9.ctx, u, q, m9.sh)
        n_px = int(((q > 0) & el).sum())
        qd = top_mask(dense_rank, el, n_px / max(int(el.sum()), 1)).astype(np.float32) if n_px else np.zeros_like(q)
        hd = system_hits(m9.ctx, u, qd, m9.sh)
        add = 0
        if len(h) and len(hd):
            m = h.merge(hd, on="system_id", suffixes=("", "_dense"))
            add = int((m.recovered & ~m.recovered_dense).sum())
        rows.append({"task": u.task, "zone": u.zone, "density_cap": cap, "n_candidates": k, "pred_px": n_px,
                     "system_recall": float(h.recovered.mean()) if len(h) else np.nan, "n_systems": int(len(h)),
                     "systems_recovered": int(h.recovered.sum()) if len(h) else 0,
                     "dense_systems_recovered": int(hd.recovered.sum()) if len(hd) else 0, "additional_systems": add})
        if cap == max(caps) and len(h):
            hits.append(h.assign(task=u.task, zone=u.zone))
    return pd.DataFrame(rows), (pd.concat(hits, ignore_index=True) if hits else pd.DataFrame())


def pooled_oracle(curve: pd.DataFrame) -> pd.DataFrame:
    if not len(curve):
        return curve
    g = curve.groupby(["task", "density_cap"]).agg(systems_recovered=("systems_recovered", "sum"),
                                                     n_systems=("n_systems", "sum"),
                                                     additional_systems=("additional_systems", "sum"),
                                                     dense_systems_recovered=("dense_systems_recovered", "sum")).reset_index()
    g["system_recall"] = g.systems_recovered / g.n_systems.clip(lower=1)
    return g


def gate4(m9, pooled: pd.DataFrame) -> pd.DataFrame:
    g = m9.cfg["gates"]["g4_tracing"]
    dens = float(m9.cfg["tracing"]["oracle"]["gate_density"])
    r = pooled[(pooled.task == m9.prim) & (pooled.density_cap == dens)] if len(pooled) else pooled
    base = {"gate": "G4_tracing", "task": m9.prim, "gate_density": dens, "profile": m9.profile["name"],
            "provisional": bool(m9.profile["provisional"])}
    if not len(r):
        return pd.DataFrame([{**base, "decision": "NOT_EVALUATED", "reason": "no traced primary-task unit"}])
    rec, add = float(r.system_recall.iloc[0]), int(r.additional_systems.iloc[0])
    ok = rec >= float(g["min_system_recall"]) and add >= int(g["min_additional_systems"])
    return pd.DataFrame([{**base, "decision": "CONTINUE" if ok else "STOP", "system_recall": rec,
                          "additional_systems_vs_dense": add,
                          "rule": f"system recall >= {g['min_system_recall']} AND additional systems >= "
                                  f"{g['min_additional_systems']} at {dens:g} km / 100 km^2 (oracle: diagnostic only)"}])
