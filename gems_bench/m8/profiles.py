"""Section 05: candidate-oriented fault-normal geophysical profiles (label-free features; gallery = evaluation).

For a polyline (row, col pixels, any curvature):
1. the path is smoothed (running mean over ``smooth_px``) and stations are placed every ``station_step_px``;
2. the local tangent is the central difference over +/- ``tangent_px`` stations, the normal its 90 deg rotation,
   oriented CONSISTENTLY along the path (so a sign of the contrast is comparable from station to station);
3. every field is sampled (bilinear) at +/- each offset of ``offsets_m`` and on the trace; a sample is kept only when
   its 4-neighbour validity is >= ``min_valid_share`` AND it lies inside the ``allowed`` mask (the unit's evaluation
   zone for held-out units: transects never reach across a fold / split boundary or into nodata).

Per field and candidate (contrast c(o) = F(+o) - F(-o), scaled by the field's robust IQR)::

    {f}_contrast_med_<o>m   median |c(o)| / IQR over stations           {f}_contrast_max   max over stations and offsets
    {f}_contrast_q25/q75/q90  robust quantiles of |c(o_ref)| / IQR     {f}_sign_consistency  |mean(sign c(o_ref))|
    {f}_continuity          share of stations with |c(o_ref)| / IQR >= continuity_threshold
    {f}_longest_run         longest contiguous run of such stations / n stations   (along-strike continuity)
    {f}_gradient            median |c(o_min)| / (2 o_min) / IQR per km  (cross-fault gradient strength)
    {f}_asymmetry           median (|F(+o)-F0| - |F(-o)-F0|) / (sum)   (left / right profile asymmetry)
    {f}_localization        median |g(o_min)| / max_o |g(o)| with g = c / 2o (1 = step localised on the trace)
    {f}_variability         IQR / median of |c(o_ref)| along the trace
    xmod_contrast_agreement number of fields with continuity >= 0.5;  xmod_contrast_corr_mag_grav  Spearman of the
                            station contrasts of magnetic vs gravity along the trace (sign-aware co-variation)

o_ref = 300 m (or the nearest configured offset). Missing fields leave NaN (a missingness indicator is added later).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage


def stations(path: np.ndarray, step: int, smooth_px: int, tangent: int) -> tuple[np.ndarray, np.ndarray]:
    """(station positions (n, 2) float, unit normals (n, 2) in (row, col)), consistently oriented."""
    p = np.asarray(path, np.float64)
    if len(p) >= smooth_px >= 3:
        k = np.ones(smooth_px) / smooth_px
        pad = smooth_px // 2
        q = np.vstack([np.repeat(p[:1], pad, 0), p, np.repeat(p[-1:], pad, 0)])
        p = np.c_[np.convolve(q[:, 0], k, "valid"), np.convolve(q[:, 1], k, "valid")][:len(path)]
    idx = np.arange(0, len(p), max(int(step), 1))
    if len(idx) < 2:
        idx = np.arange(len(p))
    a = np.clip(idx - tangent, 0, len(p) - 1)
    b = np.clip(idx + tangent, 0, len(p) - 1)
    t = p[b] - p[a]
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    n = np.c_[-t[:, 1], t[:, 0]]
    for i in range(1, len(n)):                       # keep the normal on the same side along a curved path
        if n[i] @ n[i - 1] < 0:
            n[i] = -n[i]
    return p[idx], n


def sample(F: np.ndarray, valid: np.ndarray, pts: np.ndarray, min_valid: float) -> np.ndarray:
    h, w = F.shape
    r, c = pts[..., 0], pts[..., 1]
    inside = (r >= 0) & (r <= h - 1) & (c >= 0) & (c <= w - 1)
    rr, cc = np.clip(r, 0, h - 1), np.clip(c, 0, w - 1)
    v = ndimage.map_coordinates(F, [rr.ravel(), cc.ravel()], order=1, mode="nearest").reshape(r.shape)
    ok = ndimage.map_coordinates(valid.astype(np.float32), [rr.ravel(), cc.ravel()], order=1,
                                 mode="nearest").reshape(r.shape) >= min_valid
    return np.where(inside & ok, v, np.nan)


def transect(F: np.ndarray, valid: np.ndarray, path: np.ndarray, pc: dict, pixel_m: float = 100.0) -> dict:
    """Station x offset samples of one field: {'plus': (n, k), 'minus': (n, k), 'center': (n,), 'offsets_m': (k,)}."""
    pos, nrm = stations(path, int(pc["station_step_px"]), int(pc["smooth_px"]), int(pc["tangent_px"]))
    offs = np.asarray(pc["offsets_m"], float)
    o_px = offs / pixel_m
    plus = pos[:, None, :] + nrm[:, None, :] * o_px[None, :, None]
    minus = pos[:, None, :] - nrm[:, None, :] * o_px[None, :, None]
    mv = float(pc["min_valid_share"])
    return {"plus": sample(F, valid, plus, mv), "minus": sample(F, valid, minus, mv),
            "center": sample(F, valid, pos, mv), "offsets_m": offs, "pos": pos}


def summarize(tr: dict, iqr: float, pc: dict, prefix: str) -> dict:
    offs = tr["offsets_m"]
    c = (tr["plus"] - tr["minus"]) / max(iqr, 1e-12)
    ac = np.abs(c)
    out = {}
    j_ref = int(np.argmin(np.abs(offs - 300.0)))
    for j, o in enumerate(offs):
        out[f"{prefix}_contrast_med_{int(o)}m"] = float(np.nanmedian(ac[:, j])) if np.isfinite(ac[:, j]).any() else np.nan
    a = ac[:, j_ref]
    fin = np.isfinite(a)
    if not fin.any():
        return out
    out[f"{prefix}_contrast_max"] = float(np.nanmax(ac))
    for qq in (25, 75, 90):
        out[f"{prefix}_contrast_q{qq}"] = float(np.nanpercentile(a, qq))
    out[f"{prefix}_sign_consistency"] = float(abs(np.nanmean(np.sign(c[fin, j_ref]))))
    hit = np.where(fin, a >= float(pc["continuity_threshold"]), False)
    out[f"{prefix}_continuity"] = float(hit[fin].mean())
    best = run = 0
    for h in hit:
        run = run + 1 if h else 0
        best = max(best, run)
    out[f"{prefix}_longest_run"] = best / max(len(hit), 1)
    g = ac / (2.0 * offs[None, :] / 1000.0)
    out[f"{prefix}_gradient"] = float(np.nanmedian(g[:, 0]))
    d0 = np.abs(tr["plus"][:, j_ref] - tr["center"]) - np.abs(tr["minus"][:, j_ref] - tr["center"])
    s0 = np.abs(tr["plus"][:, j_ref] - tr["center"]) + np.abs(tr["minus"][:, j_ref] - tr["center"])
    out[f"{prefix}_asymmetry"] = float(np.nanmedian(d0 / np.maximum(s0, 1e-12)))
    gmax = np.nanmax(g, axis=1)
    out[f"{prefix}_localization"] = float(np.nanmedian(g[:, 0] / np.maximum(gmax, 1e-12)))
    med = np.nanmedian(a)
    out[f"{prefix}_variability"] = float((np.nanpercentile(a, 75) - np.nanpercentile(a, 25)) / max(med, 1e-12))
    out[f"_{prefix}_station_contrast"] = c[:, j_ref]
    return out


class FieldCrops:
    """Field crops + robust IQR for one unit (or the full grid)."""

    def __init__(self, m8, u, allowed: np.ndarray):
        self.m8, self.u = m8, u
        self.allowed = allowed
        self.fields: dict = {}
        for k, b in m8.cfg["profiles_fn"]["fields"].items():
            if b not in m8.ctx.data.band_names:
                continue
            F = m8.ctx.data.read_window(b, u.sl[0].start, u.sl[0].stop, u.sl[1].start, u.sl[1].stop)
            v = F[np.isfinite(F) & allowed]
            iqr = float(np.subtract(*np.percentile(v, [75, 25]))) if v.size else 1.0
            from .potential_fields import fill

            Ff, m = fill(F)
            self.fields[k] = (Ff.astype(np.float32), m & allowed, iqr if iqr > 0 else 1.0)


def candidate_profiles(m8, u, cs, allowed: np.ndarray | None = None, keep_transects: int = 0) -> tuple[pd.DataFrame, list]:
    """Per-candidate profile features of one unit's candidate set (+ up to ``keep_transects`` raw transects)."""
    pc = m8.cfg["profiles_fn"]
    allowed = u.ev if allowed is None else allowed
    fc = FieldCrops(m8, u, allowed)
    rows, keep = [], []
    for i, p in enumerate(cs.paths):
        r = {"cand_id": int(cs.table.cand_id.iloc[i]) if "cand_id" in cs.table else i}
        sc = {}
        for k, (F, m, iqr) in fc.fields.items():
            if len(p) < 3:
                continue
            tr = transect(F, m, p, pc)
            s = summarize(tr, iqr, pc, f"prof_{k}")
            sc[k] = s.pop(f"_prof_{k}_station_contrast", None)
            r.update(s)
            if len(keep) < keep_transects:
                keep.append({"cand_id": r["cand_id"], "field": k, "plus": tr["plus"], "minus": tr["minus"],
                             "center": tr["center"], "offsets_m": tr["offsets_m"]})
        r["xmod_contrast_agreement"] = int(sum((r.get(f"prof_{k}_continuity") or 0) >= 0.5 for k in fc.fields))
        a, b = sc.get("magnetic"), sc.get("gravity")
        if a is not None and b is not None:
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() >= 5:
                from scipy.stats import spearmanr

                r["xmod_contrast_corr_mag_grav"] = float(spearmanr(a[ok], b[ok]).correlation)
        rows.append(r)
    return pd.DataFrame(rows), keep


def transects_long(keep: list, unit: str) -> pd.DataFrame:
    rows = []
    for t in keep:
        for j, o in enumerate(t["offsets_m"]):
            for s in range(len(t["center"])):
                rows.append({"unit": unit, "cand_id": t["cand_id"], "field": t["field"], "station": s,
                             "offset_m": float(o), "plus": float(t["plus"][s, j]), "minus": float(t["minus"][s, j]),
                             "center": float(t["center"][s])})
    return pd.DataFrame(rows)


# =============================================================================== gallery (evaluation)
def gallery_sets(m8, u, pool, n: int, rng):
    """Four groups on one unit: mapped faults (truth traces), unlabelled geophysical boundaries (pool candidates
    > unlabelled_min_dist_km from ANY mapped fault: not negatives, just unlabelled), random straight controls with the
    same length distribution, and pool candidates. Paths in crop coordinates."""
    from ..gems3.candidates import CandidateSet
    from ..m7.generator import mask_to_paths

    pc = m8.cfg["profiles_fn"]
    min_px = int(float(pc["gallery_min_len_km"]) / 0.1)
    truth_paths, _ = mask_to_paths(u.eval_truth(), {"min_len_km": pc["gallery_min_len_km"], "link_gap_px": 0,
                                                    "link_angle_deg": 30.0})
    lab = np.asarray(m8.ctx.data.labels, bool)[u.sl]
    dk = ndimage.distance_transform_edt(~lab) * 0.1
    groups = {"mapped_fault": truth_paths}
    if pool is not None and len(pool):
        far = [p for p in pool.paths if len(p) >= min_px and np.median(dk[p[:, 0], p[:, 1]]) >= float(pc["unlabelled_min_dist_km"])]
        groups["unlabelled_boundary"] = far
        groups["candidate"] = [p for p in pool.paths if len(p) >= min_px]
    lens = [len(p) for p in truth_paths] or [min_px * 2]
    ctrl = []
    ev = u.ev
    rr, cc = np.nonzero(ev)
    for _ in range(n * 4):
        if len(ctrl) >= n or not len(rr):
            break
        i = rng.integers(len(rr))
        L = int(rng.choice(lens))
        t = rng.uniform(0, np.pi)
        r = np.rint(rr[i] + np.arange(L) * np.sin(t)).astype(int)
        c = np.rint(cc[i] + np.arange(L) * np.cos(t)).astype(int)
        ok = (r >= 0) & (r < ev.shape[0]) & (c >= 0) & (c < ev.shape[1])
        if ok.mean() < 1.0 or not ev[r, c].all():
            continue
        ctrl.append(np.c_[r, c])
    groups["random_control"] = ctrl
    out = {}
    for g, ps in groups.items():
        if len(ps) > n:
            ps = [ps[i] for i in rng.choice(len(ps), n, replace=False)]
        out[g] = CandidateSet(pd.DataFrame({"cand_id": np.arange(len(ps))}), list(ps))
    return out


def gallery(m8, units, pools: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per (group, field, offset): mean / quartiles of |contrast| / IQR, and the per-candidate summaries."""
    pc = m8.cfg["profiles_fn"]
    rng = np.random.default_rng(int(m8.cfg["seed"]))
    n = int(m8.profile["gallery_per_group"])
    feats, prof = [], []
    for u in units:
        sets = gallery_sets(m8, u, pools.get(u.key), n, rng)
        fc = FieldCrops(m8, u, u.ev)
        for g, cs in sets.items():
            df, _ = candidate_profiles(m8, u, cs, u.ev, 0)
            feats.append(df.assign(group=g, unit=u.key))
            for k, (F, m, iqr) in fc.fields.items():
                for p in cs.paths:
                    if len(p) < 3:
                        continue
                    tr = transect(F, m, p, pc)
                    c = np.abs(tr["plus"] - tr["minus"]) / iqr
                    for j, o in enumerate(tr["offsets_m"]):
                        v = c[:, j]
                        if np.isfinite(v).any():
                            prof.append({"group": g, "field": k, "offset_m": float(o), "abs_contrast_med": float(np.nanmedian(v)),
                                         "unit": u.key})
    P = pd.DataFrame(prof)
    if len(P):
        P = P.groupby(["group", "field", "offset_m"]).abs_contrast_med.agg(
            mean="mean", q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75), n="size").reset_index()
    return P, pd.concat(feats, ignore_index=True) if feats else pd.DataFrame()
