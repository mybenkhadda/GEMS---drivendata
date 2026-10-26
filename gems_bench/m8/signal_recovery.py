"""Section 02: where does geological information disappear?

EVALUATION ONLY: everything here reads the withheld truth of each unit to DIAGNOSE the extraction chain; nothing is
deployable and no generator parameter is tuned from it.

For every withheld fault system (``disc_official``; ``geo`` too in larger profiles) and every modality, support is
measured at successive stages of the M7 extraction chain::

    1 raw          raw-field response, three competing definitions, each against a SYSTEM-MATCHED NULL (the same
                   trace rigidly translated 3-12 km to valid ground away from mapped faults; percentile of the
                   observed value among the null values):
                     contrast  |F(+o n) - F(-o n)| / IQR(F) across the system's own strike (o = contrast_offset_m)
                     gradient  percentile of |grad F| (sigma 0.7 px) along the trace
                     variance  local std (variance_window_m) / regional std (variance_regional_m)
                   a definition is 'present' when the observed value is >= present_percentile of its null;
                   raw_present = ANY definition (a system is called raw-absent only when absent under ALL of them)
    2 gradient     THDR@stage_scale percentile >= present_percentile on >= covered_frac of the trace (1-px tolerance)
    3 normalized   LCN / tilt-edge percentile (best available normalised operator)
    4 directional  |directional derivative across the system strike| vs the THDR distribution (uses the truth
                   orientation: a diagnostic upper bound on what an oriented operator could see)
    5 crest        non-maximum-suppressed THDR crest within tolerance (no gate)
    6 fragment     M7 gated crest (strength top-frac + coherence) at any M7 scale
    7 persistent   M7 persistence filter (>= min_persistence scales or strong)
    8 linked       M7 traced, linked, fused candidate polyline of that modality (M7 pool, read-only)
    9 ranked       inside the M7 pool's label-free density cap (structural_score order, gate density)

Stages 2-9 count a system as covered when >= covered_frac of its scored pixels lie within ``tolerance_px`` of the
stage's mask (TP kernel support >= 0.5). ``any`` = union over modalities.

Failure reason (first stage lost after the raw response)::

    raw_response_absent                 information limit UNDER THE TESTED OPERATORS (not proof of invisibility)
    <stage>_present -> <next>_missing   extraction bottleneck at that step
    linked_present -> ranked_missing    ranking / density bottleneck
    recovered                           reaches stage 9
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

from . import potential_fields as PF

STAGES = ["raw", "gradient", "normalized", "directional", "crest", "fragment", "persistent", "linked", "ranked"]
M8_STAGES = ["m8_worm", "m8_linked", "m8_ranked"]


# =============================================================================== helpers
def _pct_ref(a: np.ndarray, valid: np.ndarray, n: int = 300_000, seed: int = 2026) -> np.ndarray:
    v = np.asarray(a)[valid & np.isfinite(a)]
    if v.size > n:
        v = np.random.default_rng(seed).choice(v, n, replace=False)
    return np.sort(v)


def _pct(ref: np.ndarray, x) -> np.ndarray:
    return np.searchsorted(ref, np.asarray(x, np.float64)) / max(ref.size, 1) if ref.size else np.full(np.shape(x), np.nan)


def _sample(F: np.ndarray, valid: np.ndarray, r: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Bilinear samples of a filled field; NaN where the 4-neighbour validity is incomplete or outside the crop."""
    h, w = F.shape
    inside = (r >= 0) & (r <= h - 1) & (c >= 0) & (c <= w - 1)
    rr, cc = np.clip(r, 0, h - 1), np.clip(c, 0, w - 1)
    v = ndimage.map_coordinates(F, [rr, cc], order=1, mode="nearest")
    ok = ndimage.map_coordinates(valid.astype(np.float32), [rr, cc], order=1, mode="nearest") > 0.99
    return np.where(inside & ok, v, np.nan)


def system_orientation(truth: np.ndarray) -> np.ndarray:
    """Per-pixel strike azimuth (deg from north) of the truth traces (gems convention, as in M7 failure analysis)."""
    from ..postprocess.topology import orientation_field

    ang, _ = orientation_field(truth, 2.0)
    return ((90.0 - ang) % 180.0).astype(np.float32)


def _normal_rc(az_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(row, col) unit normal of a strike azimuth: normal azimuth az + 90 -> (dr, dc) = (-cos, sin)."""
    t = np.radians(np.asarray(az_deg, np.float64) + 90.0)
    return -np.cos(t), np.sin(t)


# =============================================================================== raw response with a matched null
def raw_metrics(F: np.ndarray, valid: np.ndarray, r, c, az, sc: dict, grad_ref: np.ndarray, G: np.ndarray,
                varratio: np.ndarray) -> dict:
    """The three raw definitions for one set of trace pixels (observed or translated)."""
    nr, nc = _normal_rc(az)
    o = float(sc["contrast_offset_px"])
    a = _sample(F, valid, r + o * nr, c + o * nc)
    b = _sample(F, valid, r - o * nr, c - o * nc)
    con = np.abs(a - b) / sc["iqr"]
    ri, ci = np.clip(np.rint(r).astype(int), 0, F.shape[0] - 1), np.clip(np.rint(c).astype(int), 0, F.shape[1] - 1)
    ok = valid[ri, ci]
    return {"contrast": float(np.nanmedian(con)) if np.isfinite(con).any() else np.nan,
            "gradient": float(np.nanmean(_pct(grad_ref, G[ri, ci][ok]))) if ok.any() else np.nan,
            "variance": float(np.nanmedian(varratio[ri, ci][ok])) if ok.any() else np.nan,
            "valid_share": float(ok.mean())}


def null_shifts(rng, n: int, km_range, pixel_km: float) -> np.ndarray:
    d = rng.uniform(km_range[0], km_range[1], n) / pixel_km
    t = rng.uniform(0, 2 * np.pi, n)
    return np.c_[d * np.sin(t), d * np.cos(t)]


# =============================================================================== stage masks
class StageMasks:
    """Full-grid M7 chain masks per modality (fragment / persistent), cached in the layer store."""

    def __init__(self, m8):
        self.m8 = m8
        self.cc = m8.sh["candidates"]
        self.key = m8.fp("m7_stage_masks", self.cc)

    def sources(self, mod: str) -> list[dict]:
        return [s for s in self.cc["sources"] if s.get("modality") == mod]

    def masks(self, mod: str) -> tuple[np.ndarray | None, np.ndarray | None]:
        from ..m7.generator import gated_crests, read_source

        L = self.m8.layers
        nf, npers = f"m7crest:{mod}:fragment", f"m7crest:{mod}:persistent"
        if L.has(nf) and L.has(npers) and L._pointer(nf).get("key") == self.key:
            return np.asarray(L.get(nf)) > 0, np.asarray(L.get(npers)) > 0
        srcs = self.sources(mod)
        if not srcs:
            return None, None
        frag = np.zeros(self.m8.ctx.data.shape, bool)
        pers = np.zeros(self.m8.ctx.data.shape, bool)
        for s in srcs:
            a = read_source(self.m8.ctx, s)
            if a is None:
                continue
            per, P = gated_crests(a, self.m8.ctx.data.valid & np.isfinite(a), s.get("scales_m") or self.cc["scales_m"],
                                  s["mode"], self.cc)
            mp = min(int(self.cc["min_persistence"]), len(per))
            for lv in per:
                if lv is None:
                    continue
                frag |= lv["mask"]
                pers |= lv["mask"] & ((P >= mp) | lv["strong"])
            del a, per, P
        L.put(nf, frag.astype(np.uint8), self.key)
        L.put(npers, pers.astype(np.uint8), self.key)
        return frag, pers


def _pool_masks(m8, u, mod: str, pool, gate_density: float):
    """(linked, ranked) rasters of one modality from a candidate pool (None when the pool is absent)."""
    from ..m7.coverage import density_subset

    if pool is None or not len(pool):
        return None, None
    t = pool.table
    src_mod = {s["name"]: s.get("modality") for s in m8.sh["candidates"]["sources"]}
    src_mod.update({s["name"]: s.get("modality") for s in m8.cfg["generation"]["families"]})

    def has_mod(i):
        if mod == "any":
            return True
        if str(t.modality.iloc[i]) == mod:
            return True
        return any(src_mod.get(x.split("@")[0]) == mod for x in str(t.sources.iloc[i] if "sources" in t else "").split(";"))

    idx = [i for i in range(len(pool)) if has_mod(i)]
    linked = pool.subset(idx).raster(u.ev.shape) > 0
    capped = density_subset(pool, u, gate_density)
    keep_uid = set(capped.table.uid.astype(str)) if "uid" in capped.table else set()
    idx2 = [i for i in idx if str(t.uid.iloc[i]) in keep_uid] if "uid" in t else []
    ranked = pool.subset(idx2).raster(u.ev.shape) > 0
    return linked, ranked


def _cover(mask: np.ndarray | None, gr, gc, tol_px: float) -> tuple[np.ndarray, np.ndarray]:
    """(within tolerance, distance m) per scored truth pixel."""
    if mask is None:
        return np.full(len(gr), np.nan), np.full(len(gr), np.nan)
    if not mask.any():
        return np.zeros(len(gr), bool), np.full(len(gr), np.inf)
    d = ndimage.distance_transform_edt(~mask)[gr, gc]
    return d <= tol_px + 1e-9, d * 100.0


# =============================================================================== per unit
def unit_matrix(m8, u, sm: StageMasks, pools: dict, rng) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Long table (system, modality, stage) + per-(modality, stage) density rows for one unit."""
    from ..m7.protocol import system_ids

    src = m8.cfg["signal_recovery"]
    fams = m8.cfg["bands"]["families"]
    L = m8.layers
    gr, gc = u.scorer.g_rc
    if not len(gr):
        return pd.DataFrame(), pd.DataFrame()
    sid = system_ids(m8.ctx, u)
    keep = sid >= 0
    gr, gc, sid = gr[keep], gc[keep], sid[keep]
    truth = u.eval_truth()
    az = system_orientation(truth)[gr, gc]
    lab_crop = np.asarray(m8.ctx.data.labels, bool)[u.sl]
    far_labels = ndimage.distance_transform_edt(~lab_crop) * 0.1 > 1.0
    valid_crop = np.asarray(m8.ctx.data.valid, bool)[u.sl]
    tol = float(src["tolerance_px"])
    q = float(src["present_percentile"])
    cf = float(src["covered_frac"])
    stage_s = float(src["stage_scale_m"])
    ev_km2 = max(float(u.ev.sum()) * 0.01, 1e-9)
    shifts = null_shifts(rng, int(src["null_shifts"]), src["null_shift_km"], 0.1)
    rows, dens = [], []
    systems = np.unique(sid)
    any_cov = {s: {st: np.zeros(int((sid == s).sum()), bool) for st in STAGES[1:]} for s in systems}
    any_raw = {s: {} for s in systems}
    for mod in src["modalities"]:
        spec = fams.get(mod)
        if spec is None or not L.has_band(spec["field"]):
            continue
        F = m8.ctx.data.read_window(spec["field"], u.sl[0].start, u.sl[0].stop, u.sl[1].start, u.sl[1].stop)
        fv = valid_crop & np.isfinite(F)
        Ff, _ = PF.fill(F)
        v = Ff[fv]
        iqr = float(np.subtract(*np.percentile(v, [75, 25]))) if v.size else 1.0
        ge, gn = PF.hgrad(Ff, 0.7)
        G = np.hypot(ge, gn)
        grad_ref = _pct_ref(G, fv)
        w = max(int(round(float(src["variance_window_m"]) / 100.0)), 3)
        mu = ndimage.uniform_filter(Ff, w)
        loc_sd = np.sqrt(np.maximum(ndimage.uniform_filter(Ff * Ff, w) - mu * mu, 0))
        reg_s = float(src["variance_regional_m"]) / 100.0 / 2.0
        mu2 = ndimage.gaussian_filter(Ff, reg_s)
        regsd = np.sqrt(np.maximum(ndimage.gaussian_filter(Ff * Ff, reg_s) - mu2 * mu2, 1e-12))
        varratio = loc_sd / regsd
        sc = {"iqr": iqr if iqr > 0 else 1.0, "contrast_offset_px": float(src["contrast_offset_m"]) / 100.0}
        # stages 2-5 (label-free layers; percentiles against the full-grid distribution)
        thdr_name = L.name(mod, "thdr", max(stage_s, L.scales(mod)[0]))
        thdr_full = L.get(thdr_name)
        ref_t = _pct_ref(thdr_full, np.asarray(m8.ctx.data.valid, bool))
        tcrop = np.asarray(thdr_full[u.sl], np.float32)
        size = int(2 * tol + 1)
        p_grad = _pct(ref_t, ndimage.maximum_filter(np.nan_to_num(tcrop, nan=0.0), size=size)[gr, gc]) >= q
        norm_ops = [op for op in ("lcn", "tilt_edge", "tdx") if L.allowed(mod, op)[0]]
        p_norm = np.zeros(len(gr), bool)
        for op in norm_ops:
            nm = L.name(mod, op, max(stage_s, L.scales(mod)[0]))
            a = L.get(nm)
            ref = _pct_ref(a, np.asarray(m8.ctx.data.valid, bool))
            p_norm |= _pct(ref, ndimage.maximum_filter(np.nan_to_num(np.asarray(a[u.sl], np.float32), nan=0.0),
                                                       size=size)[gr, gc]) >= q
        sig = PF.sigma_px_of(max(stage_s, L.scales(mod)[0]))
        dge, dgn = PF.hgrad(Ff, sig)
        tn = np.radians(az + 90.0)
        dd = np.abs(dge[gr, gc] * np.sin(tn) + dgn[gr, gc] * np.cos(tn))
        p_dir = _pct(ref_t, dd) >= q
        from ..m7.generator import response

        crest = response(F, fv, sig, "edge")["nms"] & fv
        frag_f, pers_f = sm.masks(mod)
        frag = frag_f[u.sl] if frag_f is not None else None
        pers = pers_f[u.sl] if pers_f is not None else None
        linked, ranked = _pool_masks(m8, u, mod, pools.get("m7"), float(m8.cfg["oracle"]["gate_density_km_per_100km2"]))
        stage_px = {"gradient": (p_grad, None), "normalized": (p_norm if norm_ops else None, None),
                    "directional": (p_dir, None)}
        for st, mk in (("crest", crest), ("fragment", frag), ("persistent", pers), ("linked", linked),
                       ("ranked", ranked)):
            stage_px[st] = _cover(mk, gr, gc, tol)
            if mk is not None:
                dens.append({"task": u.task, "zone": u.zone, "modality": mod, "stage": st,
                             "km_per_100km2": float((mk & u.ev).sum() * 0.1 / ev_km2 * 100.0)})
        for s in systems:
            idx = sid == s
            r0, c0 = gr[idx].astype(float), gc[idx].astype(float)
            obs = raw_metrics(Ff, fv, r0, c0, az[idx], sc, grad_ref, G, varratio)
            nulls = []
            for dr, dc in shifts:
                rr, cc = r0 + dr, c0 + dc
                ri, ci = np.rint(rr).astype(int), np.rint(cc).astype(int)
                ins = (ri >= 0) & (ri < Ff.shape[0]) & (ci >= 0) & (ci < Ff.shape[1])
                if ins.mean() < float(src["null_min_valid"]):
                    continue
                ok = np.zeros(len(ri), bool)
                ok[ins] = fv[ri[ins], ci[ins]] & far_labels[ri[ins], ci[ins]]
                if ok.mean() < float(src["null_min_valid"]):
                    continue
                nulls.append(raw_metrics(Ff, fv, rr, cc, az[idx], sc, grad_ref, G, varratio))
            base = {"task": u.task, "zone": u.zone, "system_id": int(s), "modality": mod, "n_px": int(idx.sum())}
            pres = []
            for dname in src["raw_definitions"]:
                nv = np.array([n[dname] for n in nulls if np.isfinite(n[dname])])
                pct = float((nv < obs[dname]).mean()) if len(nv) >= 5 and np.isfinite(obs[dname]) else np.nan
                ok = bool(np.isfinite(pct) and pct >= q)
                pres.append(ok)
                any_raw[s][f"{mod}:{dname}"] = ok
                rows.append({**base, "stage": "raw", "definition": dname, "observed": obs[dname],
                             "null_percentile": pct, "n_null": int(len(nv)), "present": ok})
            raw_ok = any(pres) if src["raw_consensus"] == "any" else (sum(pres) > len(pres) / 2)
            rows.append({**base, "stage": "raw", "definition": "consensus", "present": bool(raw_ok)})
            for st in STAGES[1:]:
                cov, dist = stage_px.get(st, (None, None))
                if cov is None:
                    rows.append({**base, "stage": st, "present": np.nan, "note": "stage unavailable"})
                    continue
                cv = np.asarray(cov, float)[idx]
                hit = float(np.nanmean(cv)) if np.isfinite(cv).any() else np.nan
                ok = bool(np.isfinite(hit) and hit >= cf)
                any_cov[s][st] |= np.nan_to_num(cv, nan=0.0) > 0
                loc_m = float(np.nanmedian(np.asarray(dist)[idx])) if dist is not None else np.nan
                rows.append({**base, "stage": st, "present": ok, "share": hit, "localization_m": loc_m})
        del F, Ff, G, varratio, mu, loc_sd, mu2, regsd, dge, dgn, crest
    for s in systems:
        base = {"task": u.task, "zone": u.zone, "system_id": int(s), "modality": "any", "n_px": int((sid == s).sum())}
        rows.append({**base, "stage": "raw", "definition": "consensus", "present": bool(any(any_raw[s].values()))})
        for st in STAGES[1:]:
            v = any_cov[s][st]
            rows.append({**base, "stage": st, "present": bool(v.mean() >= cf) if len(v) else False,
                         "share": float(v.mean()) if len(v) else np.nan})
    return pd.DataFrame(rows), pd.DataFrame(dens)


# =============================================================================== reasons + aggregates
def as_float(s: pd.Series) -> pd.Series:
    """bool / None / NaN object column -> 1.0 / 0.0 / NaN."""
    return s.map(lambda x: np.nan if x is None or (isinstance(x, float) and np.isnan(x)) else float(bool(x)))


def wide(long: pd.DataFrame) -> pd.DataFrame:
    """One row per (task, zone, system, modality) with one boolean column per stage (raw = consensus)."""
    d = long[(long.stage != "raw") | (long.get("definition", pd.Series("", index=long.index)) == "consensus")]
    w = d.pivot_table(index=["task", "zone", "system_id", "modality"], columns="stage", values="present",
                      aggfunc="first").reset_index()
    w.columns.name = None
    return w


def failure_reason(row: pd.Series, stages=STAGES) -> tuple[str, str]:
    def ok(st):
        v = row.get(st)
        return bool(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else None

    if ok("raw") is False:
        later = [s for s in stages[1:] if ok(s)]
        if later:
            return f"raw_response_absent but {later[0]}_present", "operator_beyond_raw_definitions"
        return "raw_response_absent", "information_limit_under_tested_operators"
    prev = "raw"
    for st in stages[1:]:
        v = ok(st)
        if v is None:
            continue
        if not v:
            cat = "ranking_bottleneck" if st == "ranked" else "extraction_bottleneck"
            return f"{prev}_present -> {st}_missing", cat
        prev = st
    return "recovered", "recovered"


def reasons(W: pd.DataFrame) -> pd.DataFrame:
    a = W[W.modality == "any"].copy()
    if not len(a):
        return a
    rr = [failure_reason(r) for _, r in a.iterrows()]
    a["failure_reason"], a["failure_category"] = [x[0] for x in rr], [x[1] for x in rr]
    return a


def aggregates(W: pd.DataFrame, systab: pd.DataFrame, attrs: pd.DataFrame, dens: pd.DataFrame) -> dict:
    sysinfo = attrs[[c for c in ("system_id", "trace_km", "length_bin", "strike_sector", "region_name") if c in attrs]] \
        if len(attrs) else pd.DataFrame(columns=["system_id"])
    cls = systab[[c for c in ("task", "zone", "system_id", "class") if c in systab]] if len(systab) else None
    X = W.merge(sysinfo, on="system_id", how="left")
    if cls is not None and "class" in cls:
        X = X.merge(cls, on=["task", "zone", "system_id"], how="left")
    stages = [s for s in STAGES + M8_STAGES if s in X]
    rows = []
    strata = [("all", None)] + [(c, c) for c in ("length_bin", "strike_sector", "region_name", "class") if c in X]
    for (task, mod), g in X.groupby(["task", "modality"]):
        for sname, col in strata:
            groups = [("all", g)] if col is None else list(g.groupby(col))
            for val, gg in groups:
                for st in stages:
                    v = as_float(gg[st])
                    ok = v.fillna(0) > 0
                    km = gg.trace_km if "trace_km" in gg else pd.Series(1.0, index=gg.index)
                    rows.append({"task": task, "modality": mod, "stratum": sname, "value": str(val), "stage": st,
                                 "n_systems": int(len(gg)), "n_evaluated": int(v.notna().sum()),
                                 "system_recall": float(ok.mean()) if len(gg) else np.nan,
                                 "length_recall": float(km[ok].sum() / max(km.sum(), 1e-9))})
    by = pd.DataFrame(rows)
    missed = pd.DataFrame()
    if "class" in X:
        n = X[(X["class"] == "neither")]
        missed = pd.DataFrame([{"task": t, "modality": m, "stage": st, "n_missed_by_both": int(len(g)),
                                "recovered_at_stage": int((as_float(g[st]).fillna(0) > 0).sum())}
                               for (t, m), g in n.groupby(["task", "modality"]) for st in stages])
    dsum = dens.groupby(["task", "modality", "stage"]).km_per_100km2.mean().reset_index() if len(dens) else dens
    return {"by_stratum": by, "missed_by_both": missed, "density": dsum}


# =============================================================================== M8 stages (after section 06)
def m8_stage_rows(m8, tasks, pools8: dict) -> pd.DataFrame:
    """Coverage of every system by persistent worms and by the M8 pool (linked / density-capped), per modality."""
    from ..m7.protocol import system_ids

    src = m8.cfg["signal_recovery"]
    wc = m8.cfg["worms"]
    tol, cf = float(src["tolerance_px"]), float(src["covered_frac"])
    rows = []
    for task in tasks:
        for u in m8.units_of(task):
            gr, gc = u.scorer.g_rc
            if not len(gr):
                continue
            sid = system_ids(m8.ctx, u)
            k = sid >= 0
            gr, gc, sid = gr[k], gc[k], sid[k]
            for mod in list(src["modalities"]) + ["any"]:
                masks = {}
                fams = wc["families"] if mod == "any" else [mod]
                wm = np.zeros(u.ev.shape, bool)
                for f in fams:
                    if f in wc["families"] and m8.layers.has(f"worms:{f}:persistence"):
                        p = np.asarray(m8.layers.get(f"worms:{f}:persistence")[u.sl], np.float32)
                        nl = max(len(m8.layers.scales(f, wc["levels_m"])), 1)
                        wm |= np.isfinite(p) & (p * nl >= int(wc["min_levels"]) - 1e-6)
                masks["m8_worm"] = wm if (mod == "any" or mod in wc["families"]) else None
                masks["m8_linked"], masks["m8_ranked"] = _pool_masks(
                    m8, u, mod, pools8.get(u.key), float(m8.cfg["oracle"]["gate_density_km_per_100km2"]))
                for st, mk in masks.items():
                    cov, dist = _cover(mk, gr, gc, tol)
                    for s in np.unique(sid):
                        i = sid == s
                        hit = float(np.nanmean(np.asarray(cov, float)[i])) if mk is not None else np.nan
                        rows.append({"task": task, "zone": u.zone, "system_id": int(s), "modality": mod, "stage": st,
                                     "present": bool(np.isfinite(hit) and hit >= cf) if mk is not None else np.nan,
                                     "share": hit, "localization_m": float(np.nanmedian(np.asarray(dist)[i]))})
    return pd.DataFrame(rows)


def gate1(W: pd.DataFrame, M8: pd.DataFrame, g1: dict) -> pd.DataFrame:
    """Gate 1: systems covered by M8 physics (worms or M8 linked pool) but not by M7's linked candidates."""
    rows = []
    a7 = W[W.modality == "any"].set_index(["task", "zone", "system_id"])
    a8 = M8[M8.modality == "any"].pivot_table(index=["task", "zone", "system_id"], columns="stage", values="present",
                                              aggfunc="first") if len(M8) else pd.DataFrame()
    for task in sorted(set(a7.index.get_level_values(0))):
        s7 = a7.loc[task]
        lk7 = s7["linked"].fillna(False).astype(bool) if "linked" in s7 else pd.Series(False, index=s7.index)
        raw = s7["raw"].fillna(False).astype(bool) if "raw" in s7 else pd.Series(False, index=s7.index)
        if len(a8) and task in a8.index.get_level_values(0):
            s8 = a8.loc[task].reindex(s7.index)
            m8_any = (s8.get("m8_worm", pd.Series(False, index=s7.index)).fillna(False).astype(bool) |
                      s8.get("m8_linked", pd.Series(False, index=s7.index)).fillna(False).astype(bool))
        else:
            m8_any = pd.Series(False, index=s7.index)
        add = m8_any & ~lk7
        n = int(len(s7))
        ok = int(add.sum()) >= int(g1["min_additional_systems"]) and add.mean() >= float(g1["min_additional_share"])
        lost_info = int((~raw & ~m8_any & ~lk7).sum())
        lost_extr = int((raw & ~m8_any & ~lk7).sum())
        rows.append({"task": task, "n_systems": n, "m7_linked_systems": int(lk7.sum()), "m8_physics_systems": int(m8_any.sum()),
                     "additional_systems": int(add.sum()), "additional_share": float(add.mean()) if n else np.nan,
                     "missed_raw_absent": lost_info, "missed_raw_present": lost_extr,
                     "decision": "PASS" if ok else "FAIL",
                     "diagnosis": ("" if ok else ("mostly information-limited at 100 m (raw response absent)"
                                                  if lost_info >= lost_extr else
                                                  "mostly extraction-limited (raw response present, not extracted)")),
                     "rule": f"additional >= {g1['min_additional_systems']} and share >= {g1['min_additional_share']}"})
    return pd.DataFrame(rows)


# =============================================================================== case studies
CASE_RULES = {
    "high_raw_magnetic_missed_candidate": lambda r: r.get("raw@magnetic") is True and not r.get("linked@any"),
    "high_raw_gravity_missed_candidate": lambda r: r.get("raw@gravity") is True and not r.get("linked@any"),
    "crossmodal_response_missed_candidate": lambda r: r.get("n_raw_modalities", 0) >= 2 and not r.get("linked@any"),
    "cnn_detects_structural_misses": lambda r: r.get("class") in ("both", "b0_only", "b1_only") and not r.get("linked@any"),
    "structural_detects_cnn_misses": lambda r: r.get("class") == "neither" and bool(r.get("linked@any")),
    "both_miss": lambda r: r.get("class") == "neither" and not r.get("linked@any"),
    "low_signal_short_fault": lambda r: (r.get("trace_km", 99) < 2.0) and r.get("raw@any") is False,
    "potential_regional_artifact": lambda r: bool(r.get("regional_only")),
}


def case_table(W: pd.DataFrame, systab: pd.DataFrame, attrs: pd.DataFrame, regional: pd.DataFrame) -> pd.DataFrame:
    p = W.pivot_table(index=["task", "zone", "system_id"], columns="modality",
                      values=[c for c in ("raw", "linked") if c in W], aggfunc="first")
    p.columns = [f"{a}@{b}" for a, b in p.columns]
    p = p.reset_index()
    raws = [c for c in p.columns if c.startswith("raw@") and c != "raw@any"]
    p["n_raw_modalities"] = p[raws].fillna(False).astype(bool).sum(axis=1) if raws else 0
    if len(systab) and "class" in systab:
        p = p.merge(systab[["task", "zone", "system_id", "class"]], on=["task", "zone", "system_id"], how="left")
    if len(attrs):
        p = p.merge(attrs[[c for c in ("system_id", "trace_km", "region_name", "strike_deg") if c in attrs]],
                    on="system_id", how="left")
    if len(regional):
        p = p.merge(regional, on=["task", "zone", "system_id"], how="left")
    for c in [c for c in p.columns if "@" in c]:
        p[c] = p[c].map(lambda x: None if x is None or (isinstance(x, float) and np.isnan(x)) else bool(x))
    out = []
    for name, rule in CASE_RULES.items():
        for r in p.to_dict("records"):
            try:
                if rule(r):
                    out.append({**r, "case_class": name})
            except Exception:
                continue
    return pd.DataFrame(out)


def regional_flags(m8, W: pd.DataFrame) -> pd.DataFrame:
    """Heuristic 'regional artifact' flag: the system's gradient is present only at the large scale (not at the stage
    scale) on its best modality - candidates there may follow a regional gradient, not a fault."""
    src = m8.cfg["signal_recovery"]
    L = m8.layers
    rows = []
    q = float(src["present_percentile"])
    from ..m7.protocol import system_ids

    for task in sorted(W.task.unique()):
        for u in m8.units_of(task):
            gr, gc = u.scorer.g_rc
            if not len(gr):
                continue
            sid = system_ids(m8.ctx, u)
            flags = {}
            for mod in ("magnetic", "gravity"):
                if not L.allowed(mod, "thdr")[0]:
                    continue
                small = L.get(L.name(mod, "thdr", max(float(src["stage_scale_m"]), L.scales(mod)[0])))
                large = L.get(L.name(mod, "thdr", max(float(src["large_scale_m"]), L.scales(mod)[0])))
                v = np.asarray(m8.ctx.data.valid, bool)
                rs, rl = _pct_ref(small, v), _pct_ref(large, v)
                ps = _pct(rs, np.asarray(small[u.sl], np.float32)[gr, gc])
                pl = _pct(rl, np.asarray(large[u.sl], np.float32)[gr, gc])
                for s in np.unique(sid[sid >= 0]):
                    i = sid == s
                    reg = (np.nanmean(pl[i] >= q) >= 0.5) and (np.nanmean(ps[i] >= q) < 0.5)
                    flags[int(s)] = flags.get(int(s), False) or bool(reg)
            rows += [{"task": task, "zone": u.zone, "system_id": s, "regional_only": f} for s, f in flags.items()]
    return pd.DataFrame(rows)


def run(m8) -> dict:
    src = m8.cfg["signal_recovery"]
    rng = np.random.default_rng(int(m8.cfg["seed"]))
    sm = StageMasks(m8)
    pools = {"m7": None}
    longs, dens = [], []
    tasks = list(m8.profile["signal_recovery_tasks"])
    for task in tasks:
        for u in m8.units_of(task):
            pools["m7"] = m8.m7_pool(u.key)
            lg, dn = unit_matrix(m8, u, sm, pools, rng)
            longs.append(lg)
            dens.append(dn)
            m8.say(f"s02 {u.key}: {lg.system_id.nunique() if len(lg) else 0} systems")
            m8.layers.release()
    long = pd.concat(longs, ignore_index=True) if longs else pd.DataFrame()
    dn = pd.concat(dens, ignore_index=True) if dens else pd.DataFrame()
    if not len(long):
        raise RuntimeError("no withheld system found in the evaluation units")
    W = wide(long)
    R = reasons(W)
    systab, attrs = m8.system_table(), m8.attributes()
    agg = aggregates(W, systab, attrs, dn)
    reg = regional_flags(m8, W)
    cases = case_table(W, systab, attrs, reg)
    summ = (R.groupby(["task", "failure_category"]).size().rename("n_systems").reset_index()
            if len(R) else pd.DataFrame())
    return {"long": long, "wide": W, "reasons": R, "by_stratum": agg["by_stratum"], "missed_by_both": agg["missed_by_both"],
            "density": agg["density"], "cases": cases, "summary": summ, "regional": reg}
