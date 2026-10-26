"""Section 12 (P2, off in production unless it earns its place): Monte Carlo uncertainty of the structured channel.

MC-A  structural-evidence uncertainty. Per realisation (seeded): spatially correlated Gaussian noise (white noise
      smoothed at ``noise_corr_px``; amplitude ``noise_frac`` x robust std of the dense map) added to the dense
      evidence, smoothing-scale jitter, ONE rigid sub-pixel registration shift of the whole evidence / orientation
      pair (the shared source is perturbed once, never per channel), orientation jitter, seed-threshold jitter ->
      deterministic beam tracing. These distributions are ASSUMPTIONS (recorded in ``assumptions.json``): a
      heuristic sensitivity analysis of the processing chain, not a measured noise model.
MC-B  alternative paths: stochastic beam search (Gumbel temperature) on the unperturbed evidence -> path occupancy,
      branch frequency, positional variability, MC-only structures.
MC-C  model robustness: independently seeded training variants (``core@seed<i>``, profile ``n_seeds``) whose
      held-out maps are averaged into ``m9_dense_core_ens`` by section 09; grouped (block) bootstrap in comparisons.

Per section-11 candidate: detection frequency (>= half of its pixels within ``match_radius_px`` of a realisation
path), positional variability (mean distance to the matched realisation pixels), length ratio, fragmentation
(distinct realisation paths touching it), branch frequency. Occupancy = share of realisations covering a pixel.
Convergence: the max change of the mean detection frequency between R/2 and R realisations; when it exceeds
``convergence_tol`` the table says so and the profile's ``mc_realizations`` may be raised (only after Gate 5 shows
value). Occupancy is NOT a calibrated fault probability.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage


def draw(rng, mc: dict) -> dict:
    a = mc["mc_a"]
    return {"noise_frac": float(rng.uniform(*a["noise_frac"])), "noise_corr_px": float(rng.choice(a["noise_corr_px"])),
            "smooth_sigma_px": float(rng.uniform(*a["smooth_sigma_px"])),
            "shift": rng.normal(0.0, float(a["registration_px_sigma"]), 2).tolist(),
            "seed_q_delta": float(rng.uniform(-a["seed_quantile_jitter"], a["seed_quantile_jitter"])),
            "orient_jitter_deg": float(rng.normal(0.0, a["orient_tol_jitter_deg"]))}


def perturb(pred: dict, prm: dict, rng) -> dict:
    s = np.nan_to_num(pred["seg"]).astype(np.float32)
    v = s[s > 0]
    sd = float(np.subtract(*np.percentile(v, [75, 25])) / 1.349) if v.size else 0.0
    n = ndimage.gaussian_filter(rng.standard_normal(s.shape).astype(np.float32), prm["noise_corr_px"])
    n *= prm["noise_frac"] * sd / max(float(n.std()), 1e-9)
    s = ndimage.gaussian_filter(s + n, prm["smooth_sigma_px"])
    dr, dc = prm["shift"]
    out = {"seg": ndimage.shift(s, (dr, dc), order=1, mode="nearest")}
    th = 0.5 * np.arctan2(np.nan_to_num(pred["osin"]), np.nan_to_num(pred["ocos"])) + np.radians(prm["orient_jitter_deg"])
    conf = np.hypot(np.nan_to_num(pred["ocos"]), np.nan_to_num(pred["osin"]))
    out["ocos"] = ndimage.shift(conf * np.cos(2 * th), (dr, dc), order=1, mode="nearest")
    out["osin"] = ndimage.shift(conf * np.sin(2 * th), (dr, dc), order=1, mode="nearest")
    out["ep"], out["jn"] = pred["ep"], pred["jn"]
    return out


def _match(cs, paths: list[np.ndarray], shape, radius: int) -> pd.DataFrame:
    occ = np.zeros(shape, bool)
    lab = np.zeros(shape, np.int32)
    for i, p in enumerate(paths, start=1):
        occ[p[:, 0], p[:, 1]] = True
        lab[p[:, 0], p[:, 1]] = i
    if not occ.any():
        return pd.DataFrame({"detected": np.zeros(len(cs), bool), "dist_px": np.nan, "n_frag": 0, "len_ratio": 0.0})
    d, (ir, ic) = ndimage.distance_transform_edt(~occ, return_indices=True)
    rows = []
    for p in cs.paths:
        dd = d[p[:, 0], p[:, 1]]
        near = dd <= radius
        ids = np.unique(lab[ir[p[:, 0], p[:, 1]][near], ic[p[:, 0], p[:, 1]][near]]) if near.any() else []
        rows.append({"detected": bool(near.mean() >= 0.5), "dist_px": float(dd[near].mean()) if near.any() else np.nan,
                     "n_frag": int(len(ids)), "len_ratio": float(near.mean())})
    return pd.DataFrame(rows)


def run_unit(m9, u, pred: dict, cs) -> dict:
    """MC-A + MC-B on one unit; returns stability table, occupancy rasters and convergence rows."""
    from .tracing import fields, trace_unit

    mc, tc = m9.cfg["monte_carlo"], m9.cfg["tracing"]
    R = int(m9.profile["mc_realizations"])
    rad = int(mc["match_radius_px"])
    ev = u.scored_mask()
    shape = u.ev.shape
    rng = np.random.default_rng(int(m9.cfg["seed"]) + 1200 + u.zone)
    occA = np.zeros(shape, np.uint16)
    occB = np.zeros(shape, np.uint16)
    det = np.zeros((R, len(cs)), bool)
    dist = np.full((R, len(cs)), np.nan)
    frag = np.zeros((R, len(cs)), np.int16)
    branchB = np.zeros(len(cs), np.int32)
    detB = np.zeros(len(cs), np.int32)
    prm_log, conv = [], []
    stoch_paths = []
    E0, th0, _ = fields(pred, ev, float(tc["evidence_sigma_px"]))         # MC-B: unperturbed evidence
    for r in range(R):
        prm = draw(rng, mc)
        pp = perturb(pred, prm, rng)
        E, theta, _ = fields(pp, ev, float(tc["evidence_sigma_px"]))
        tcr = {**tc, "seed_quantile": float(np.clip(tc["seed_quantile"] + prm["seed_q_delta"], 0.5, 0.999))}
        paths, _ = trace_unit(E, theta, ev, tcr, int(m9.profile["max_seeds_per_unit"]))
        m = _match(cs, paths, shape, rad)
        det[r], dist[r], frag[r] = m.detected, m.dist_px, m.n_frag
        for p in paths:
            occA[p[:, 0], p[:, 1]] += 1
        prm_log.append({"unit": u.key, "realisation": r, "family": "MC-A", **{k: (v if not isinstance(v, list) else
                                                                               str(v)) for k, v in prm.items()}})
        sp, srows = trace_unit(E0, th0, ev, tc, int(m9.profile["max_seeds_per_unit"]),
                               temperature=float(tc["stochastic"]["temperature"]), rng=rng, source="mc_b")
        mb = _match(cs, sp, shape, rad)
        detB += mb.detected.to_numpy().astype(int)
        br = [p for p, rw in zip(sp, srows) if str(rw["source"]).endswith("_branch")]
        if br:
            branchB += _match(cs, br, shape, rad).detected.to_numpy().astype(int)
        for p in sp:
            occB[p[:, 0], p[:, 1]] += 1
        stoch_paths += [(r, p, rw) for p, rw in zip(sp, srows)]
        if r + 1 >= 8 and (r + 1) % 4 == 0:
            h = (r + 1) // 2
            conv.append({"unit": u.key, "realisations": r + 1,
                         "max_change_mean_detection": float(np.nanmax(np.abs(det[:h].mean(0) - det[:r + 1].mean(0))))
                         if len(cs) else 0.0})
    st = cs.table[["uid", "cand_id"]].copy()
    st["unit"] = u.key
    st["mc_a_detection"] = det.mean(0) if len(cs) else []
    st["mc_a_pos_var_px"] = np.nanmean(dist, 0) if len(cs) else []
    st["mc_a_fragmentation"] = frag.mean(0) if len(cs) else []
    st["mc_b_occupancy"] = detB / max(R, 1)
    st["mc_b_branch_freq"] = branchB / max(R, 1)
    return {"stability": st, "occA": occA.astype(np.float32) / max(R, 1), "occB": occB.astype(np.float32) / max(R, 1),
            "params": pd.DataFrame(prm_log), "convergence": pd.DataFrame(conv), "stoch_paths": stoch_paths}


def mc_only_pool(m9, u, occB: np.ndarray, cs, min_occ: float = 0.5):
    """Structures of the MC-B occupancy (crests >= ``min_occ``) not covered by the deterministic pool."""
    from ..gems3.candidates import CandidateSet
    from ..m7.generator import mask_to_paths

    covered = cs.raster(occB.shape) > 0 if len(cs) else np.zeros(occB.shape, bool)
    covered = ndimage.binary_dilation(covered, iterations=2)
    mask = (occB >= min_occ) & ~covered & u.ev
    paths, _ = mask_to_paths(mask, {"min_len_km": float(m9.cfg["tracing"]["min_len_km"]), "link_gap_px": 2,
                                    "link_angle_deg": 30.0})
    t = pd.DataFrame({"source": "mc_only", "length_km": [len(p) * 0.112 for p in paths],
                      "confidence_lf": [float(occB[p[:, 0], p[:, 1]].mean()) * np.sqrt(len(p) * 0.112) for p in paths]})
    return CandidateSet(t, paths)


def gate5_mc(m9, extra: pd.DataFrame, ranking_gain: float | None) -> dict:
    g = m9.cfg["gates"]["g5_extensions"]
    n = int(extra.additional_systems.sum()) if len(extra) and "additional_systems" in extra else 0
    ok = n >= int(g["min_unique_systems"]) or (ranking_gain is not None and ranking_gain >= float(g["min_dti_gain"]))
    return {"gate": "G5_extensions", "component": "monte_carlo", "task": m9.prim,
            "decision": "INCLUDE" if ok else "UNCERTAINTY_REPORTING_ONLY", "mc_only_additional_systems": n,
            "ranking_gain_with_mc_features": ranking_gain, "profile": m9.profile["name"],
            "provisional": bool(m9.profile["provisional"]),
            "rule": f">= {g['min_unique_systems']} systems covered only by MC structures OR ranking gain >= {g['min_dti_gain']}"}
