"""Section 06: high-recall candidate generation (unions, never hard intersections) + the mandatory oracle gate.

Families (``generation``)::

    label-free, full grid (cached): magnetic THDR / tilt-edge / TDX crests, gravity THDR / tilt crests, terrain edges /
        lines / slope ridges, conductivity edges, tolerant magnetic-gravity agreement ridges, magnetic / gravity worms
    per unit (honest maps only): B0 crests, B1 crests, B0/B1 DISAGREEMENT crests (max rank x |rank difference|),
        known-fault tip extensions (disc units: retained faults), network-completion hypotheses (section 08),
        DEM crests inside the DEM footprint (section 09 mosaic)

Physics families run on operator layers at each family scale; band families reuse M7's generator. Differences to M7:
LOCAL percentile thresholds (``local_threshold_block_km``), short fragments kept (``min_len_km``), persistence kept as a
FEATURE (``min_persistence: 1``), several linking distances / angles (``link_variants``; duplicates absorbed with
provenance), paths split where they bend more than ``max_curvature_deg_per_km``. Union -> spatially aware
deduplication (M7 ``fuse``: longest first, provenance in ``sources`` / ``member_uids``, per-modality agreement).
No candidate is dropped for low confidence before the oracle measures whether it uniquely covers withheld geometry.

Oracle gate (EVALUATION: reads the truth; never tunes the generator): coverage at label-free density caps, regional
density budget, oracle kd subsets, recall by family / provenance, coverage of B0+B1-missed systems, unique coverage of
the lowest-confidence tercile, and the M7 pool at the SAME densities. Gate 2 (``oracle.go_if``).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy import ndimage

from ..gems3.candidates import CandidateSet
from ..postprocess.topology import _disk
from . import potential_fields as PF


# =============================================================================== geometry helpers
def split_curvature(p: np.ndarray, max_deg_per_km: float, min_px: int, step: int = 5, pixel_km: float = 0.1) -> list:
    """Split a path where the heading changes faster than ``max_deg_per_km`` (between ``step``-px chords)."""
    if max_deg_per_km <= 0 or len(p) < 3 * step:
        return [p] if len(p) >= min_px else []
    idx = np.arange(0, len(p), step)
    seg = np.diff(p[idx].astype(float), axis=0)
    h = np.degrees(np.arctan2(seg[:, 0], seg[:, 1]))
    turn = np.abs((np.diff(h) + 180.0) % 360.0 - 180.0)
    km = step * pixel_km
    cut = idx[1:-1][turn / km > max_deg_per_km]
    out, s = [], 0
    for c in list(cut) + [len(p)]:
        if c - s >= min_px:
            out.append(p[s:c])
        s = c
    return out


def gated_local(strength: np.ndarray, nms: np.ndarray, coh: np.ndarray, valid: np.ndarray, gc: dict,
                block_px: int, top_frac: float | None = None, min_coh: float | None = None) -> dict | None:
    on = nms & valid & (strength > 0)
    v = strength[on]
    if not v.size:
        return None
    tf = float(top_frac if top_frac is not None else gc["top_frac_per_scale"])
    mc = float(min_coh if min_coh is not None else gc["min_coherence"])
    if block_px > 0:
        thr = PF.block_quantile_map(strength, on, block_px, 1.0 - tf)
    else:
        thr = np.full(strength.shape, float(np.quantile(v, 1.0 - tf)), np.float32)
    strong = float(np.quantile(v, 1.0 - float(gc["strong_frac"])))
    ref = float(np.quantile(v, 0.99)) or 1.0
    L = on & (strength >= thr) & (coh >= mc)
    return {"mask": L, "strong": L & (strength >= strong), "s_norm": np.clip(strength / ref, 0, 1).astype(np.float32),
            "coherence": coh, "threshold_median": float(np.nanmedian(thr[on]))}


def persistence(per: list, sigmas: list) -> np.ndarray:
    dil = []
    for lv, sig in zip(per, sigmas):
        if lv is None:
            continue
        dil.append(ndimage.binary_dilation(lv["mask"], structure=_disk(max(1, int(round(sig / 2.0))))))
    return np.sum(np.stack(dil), 0).astype(np.uint8) if dil else None


def trace_family(per: list, scales: list, P: np.ndarray, gc: dict, src: dict, *, offset=(0, 0), prefix: str = "",
                 min_persistence: int | None = None) -> tuple[list, list]:
    """Per-scale gated crests -> (link variant) skeleton paths -> curvature split -> rows with provenance."""
    from ..m7.generator import mask_to_paths, stable_id

    mp = int(min_persistence if min_persistence is not None else gc["min_persistence"])
    min_px = max(int(np.ceil(float(gc["min_len_km"]) * 1000.0 / (100.0 * 1.12))), 2)
    paths, rows = [], []
    n_sc = max(len([x for x in per if x is not None]), 1)
    for lv, s_m in zip(per, scales):
        if lv is None:
            continue
        keep = lv["mask"] & ((P >= min(mp, n_sc)) | lv["strong"])
        for gap, ang in gc["link_variants"]:
            ps, meta = mask_to_paths(keep, {**gc, "link_gap_px": int(gap), "link_angle_deg": float(ang)})
            cap = int(gc.get("max_candidates_per_source") or 0)
            if cap and len(ps) > cap:
                o = np.argsort([-len(p) for p in ps], kind="stable")[:cap]
                ps, meta = [ps[i] for i in o], [meta[i] for i in o]
            for p0, m in zip(ps, meta):
                for p in split_curvature(p0, float(gc["max_curvature_deg_per_km"]), min_px):
                    pf = p + np.asarray(offset)
                    sn, co, pe = lv["s_norm"][p[:, 0], p[:, 1]], lv["coherence"][p[:, 0], p[:, 1]], P[p[:, 0], p[:, 1]]
                    rows.append({"uid": stable_id(f"{prefix}{src['name']}|g{gap}", s_m, pf), "source": src["name"],
                                 "modality": src.get("modality", "other"), "mode": src.get("mode", "ridge"),
                                 "kind": src.get("kind", "physics"), "scale_m": float(s_m), "link_gap_px": int(gap),
                                 "network_id": f"{prefix}{src['name']}@{s_m:g}/g{gap}:{m['network']}",
                                 "junction_ends": m["junction_ends"], "linked_px": m["linked_px"],
                                 "strength_mean": float(np.nanmean(sn)), "coherence_mean": float(np.nanmean(co)),
                                 "persistence_mean": float(pe.mean()) / n_sc,
                                 "conf_lf": float(np.nanmean(sn) * np.nanmean(co) * pe.mean() / n_sc)})
                    paths.append(p)
    return paths, rows


# =============================================================================== family generators
def physics_family(m8, src: dict) -> tuple[list, list, list]:
    """One label-free layer family on the full grid."""
    from ..m7.generator import response

    from .layers import parse

    gc = m8.cfg["generation"]
    L = m8.layers
    block = int(round(float(gc["local_threshold_block_km"]) * 1000.0 / L.pixel_m))
    valid = np.asarray(m8.ctx.data.valid, bool)
    lay = src["layer"]
    per, scales, sig, log = [], [], [], []
    if "@" not in lay and lay.count(":") == 1 and lay.split(":")[0] in L.families:
        fam, op = lay.split(":")
        ok, why = L.allowed(fam, op)
        if not ok:
            return [], [], [{"source": src["name"], "status": f"refused: {why}"}]
        for s in L.scales(fam, src.get("scales_m") or gc["scales_m"]):
            a = np.asarray(L.get(L.name(fam, op, s)), np.float32)
            v = valid & np.isfinite(a)
            r = response(a, v, 0.7, "ridge")
            g = gated_local(r["strength"], r["nms"], r["coherence"], v, gc, block)
            per.append(g)
            scales.append(float(s))
            sig.append(PF.sigma_px_of(s, L.pixel_m))
            log.append({"source": src["name"], "layer": L.name(fam, op, s), "status": "ok",
                        "crest_px": int(g["mask"].sum()) if g else 0})
            del a, r
    else:
        if not L.has(lay):
            return [], [], [{"source": src["name"], "status": f"layer {lay} not produced (run its section)"}]
        a = np.asarray(L.get(lay), np.float32)
        v = valid & np.isfinite(a)
        for s in src.get("scales_m") or gc["scales_m"]:
            sg = PF.sigma_px_of(s, L.pixel_m)
            r = response(a, v, sg, "ridge")
            g = gated_local(r["strength"], r["nms"], r["coherence"], v, gc, block)
            per.append(g)
            scales.append(float(s))
            sig.append(sg)
            log.append({"source": src["name"], "layer": lay, "smoothing_m": float(s), "status": "ok",
                        "crest_px": int(g["mask"].sum()) if g else 0})
            del r
    P = persistence(per, sig)
    if P is None:
        return [], [], log
    paths, rows = trace_family(per, scales, P, gc, {**src, "kind": "physics"})
    return paths, rows, log


def band_family(m8, src: dict) -> tuple[list, list, list]:
    """Raw-band family through M7's generator (global thresholds) with M8's link / curvature / length settings."""
    from ..m7.generator import gated_crests, read_source

    gc = m8.cfg["generation"]
    a = read_source(m8.ctx, src)
    if a is None:
        return [], [], [{"source": src["name"], "status": "band missing"}]
    scales = list(src.get("scales_m") or gc["scales_m"])
    per, P = gated_crests(a, m8.ctx.data.valid & np.isfinite(a), scales, src["mode"], gc)
    paths, rows = trace_family(per, scales, P, gc, {**src, "kind": "feature"})
    return paths, rows, [{"source": src["name"], "band": src["band"], "status": "ok", "n": len(paths)}]


def label_free(m8) -> tuple[CandidateSet, pd.DataFrame]:
    gc = m8.cfg["generation"]
    allp, allr, logs = [], [], []
    for src in gc["families"]:
        t0 = time.perf_counter()
        ps, rows, lg = (band_family if src.get("band") else physics_family)(m8, src)
        allp += ps
        allr += rows
        logs += [{**x, "seconds": round(time.perf_counter() - t0, 1), "n_paths": len(ps)} for x in lg]
        m8.say(f"s06 {src['name']}: {len(ps)} paths")
        m8.layers.release()
    return CandidateSet(pd.DataFrame(allr), allp), pd.DataFrame(logs)


def label_free_key(m8) -> str:
    gc = m8.cfg["generation"]
    lays = sorted({s["layer"] for s in gc["families"] if s.get("layer")})
    fams = m8.layers.families

    def k(x):     # scaled physics families are defined by the physics config; produced layers by their stage key
        return x if ("@" not in x and x.count(":") == 1 and x.split(":")[0] in fams) else m8.layers._key(x)
    return m8.fp("s06_label_free", gc, [k(x) for x in lays], m8.cfg["physics"], m8.cfg["bands"]["families"],
                 m8.layers.compat.get("status"))


def cached_label_free(m8) -> tuple[CandidateSet, pd.DataFrame, str]:
    from ..artifacts.atomic import atomic_save

    key = label_free_key(m8)
    npz, tab, logp = (m8.cache / f"m8_label_free_{key}{e}" for e in (".npz", ".parquet", "_log.csv"))
    if npz.exists() and tab.exists() and logp.exists() and not m8.cfg.get("overwrite_existing"):
        m8.say(f"s06 label-free pool reused from {npz.name}")
        return CandidateSet.from_npz(pd.read_parquet(tab), np.load(npz)), pd.read_csv(logp), key
    lf, log = label_free(m8)
    atomic_save(npz, lambda t: np.savez_compressed(t, **lf.to_npz()))
    atomic_save(tab, lambda t: lf.table.to_parquet(t, index=False))
    atomic_save(logp, lambda t: log.to_csv(t, index=False))
    return lf, log, key


def _rank01(p, ev):
    from ..gems3.representation import rank01

    return rank01(np.nan_to_num(p, nan=0.0), ev)


def unit_pool(m8, u, lf: CandidateSet, scores: dict, extra: list | None = None) -> tuple[CandidateSet, pd.DataFrame]:
    """Label-free pool clipped to the unit + model / disagreement / tip / completion / DEM families, fused."""
    from ..m7.generator import clip_to_unit, fuse, gated_crests, geometry, stable_id, structural_score, tip_paths

    gc = m8.cfg["generation"]
    min_px = max(int(np.ceil(float(gc["min_len_km"]) / 0.112)), 2)
    off = (u.sl[0].start, u.sl[1].start)
    parts = [clip_to_unit(lf, u, min_px)]
    log = []
    stems = list(m8.sh["models"])
    for src in gc["model_families"]:
        if src.get("kind") == "disagreement":
            s0, s1 = scores.get(stems[0]), scores.get(stems[1])
            if s0 is None or s1 is None:
                continue
            r0, r1 = _rank01(s0, u.ev), _rank01(s1, u.ev)
            a = (np.maximum(r0, r1) * np.abs(r0 - r1)).astype(np.float32)
            spec = {**src, "mode": "ridge", "modality": "neural", "kind": "model_disagreement"}
        else:
            s = scores.get(src["model"])
            if s is None:
                continue
            a = np.nan_to_num(s, nan=0.0).astype(np.float32)
            spec = {**src, "mode": "ridge", "modality": "neural", "kind": "model"}
        per, P = gated_crests(a, u.ev, src["scales_m"], "ridge", gc)
        ps, rows = trace_family(per, list(src["scales_m"]), P, gc, spec, offset=off, prefix=f"{u.key}/")
        parts.append(CandidateSet(pd.DataFrame(rows), ps))
        log.append({"unit": u.key, "family": src["name"], "n": len(ps)})
    if gc.get("include_tips") and u.known is not None and u.task != "geo":
        rows, ps = [], []
        for j, (p, Lk) in enumerate(tip_paths(u.known, u.ev, gc["tip_extension_km"], gc["tip_tail_px"])):
            pf = p + np.asarray(off)
            rows.append({"uid": stable_id(f"tip{Lk:g}km", 0, pf), "source": f"tip{Lk:g}km", "modality": "network",
                         "mode": "extension", "kind": "tip", "scale_m": np.nan, "network_id": f"{u.key}/tip:{j}",
                         "junction_ends": 0, "linked_px": 0, "strength_mean": np.nan, "coherence_mean": np.nan,
                         "persistence_mean": np.nan, "conf_lf": np.nan})
            ps.append(p)
        parts.append(CandidateSet(pd.DataFrame(rows), ps))
        log.append({"unit": u.key, "family": "tips", "n": len(ps)})
    for cs in extra or []:
        if cs is not None and len(cs):
            parts.append(cs)
            log.append({"unit": u.key, "family": str(cs.table.source.iloc[0]) if "source" in cs.table else "extra",
                        "n": len(cs)})
    tabs = [p.table for p in parts if len(p)]
    raw = CandidateSet(pd.concat(tabs, ignore_index=True) if tabs else pd.DataFrame(), [x for p in parts for x in p.paths])
    for c in ("scale_m", "uid", "modality", "source"):
        if c not in raw.table and len(raw):
            raw.table[c] = np.nan if c == "scale_m" else ""
    cs = fuse(raw, u.ev.shape, gc)
    if len(cs):
        gt = geometry(cs)
        tab = pd.concat([cs.table.reset_index(drop=True), gt], axis=1)
        tab["structural_score"] = structural_score(tab)
        tab["families"] = [";".join(sorted({x.split("|")[0].split("@")[0] for x in str(s).split(";") if x}))
                           for s in tab.sources]
        tab["n_families"] = tab.families.str.count(";") + 1
        cs.table = tab
    return cs, pd.DataFrame(log)


def completion_extra(m8, u) -> CandidateSet | None:
    """Section-08 hypotheses of a disc unit as a candidate family (crop coordinates)."""
    cs = m8.completion_set(u.key)
    if cs is None or not len(cs):
        return None
    t = cs.table.copy()
    keep = {"uid", "source", "modality", "kind", "scale_m", "network_id", "conf_lf"}
    t = t[[c for c in t.columns if c in keep]].copy()
    t["modality"], t["kind"] = "network", "completion"
    t["source"] = "completion_" + t.source.astype(str)
    if "scale_m" not in t:
        t["scale_m"] = np.nan
    return CandidateSet(t.reset_index(drop=True), cs.paths)


def dem_extra(m8, u) -> CandidateSet | None:
    full = m8.dem_candidates_full()
    if full is None or not len(full):
        return None
    from ..m7.generator import clip_to_unit

    gc = m8.cfg["generation"]
    cs = clip_to_unit(full, u, max(int(np.ceil(float(gc["min_len_km"]) / 0.112)), 2))
    if len(cs):
        cs.table["modality"], cs.table["kind"] = "dem", "dem"
    return cs


# =============================================================================== oracle (EVALUATION)
def regional_subset(cs: CandidateSet, u, cap: float, block_km: float) -> CandidateSet:
    """Label-free regional density budget: per block, candidates in structural-score order until the block's
    candidate length reaches cap km per 100 km^2 of its evaluation area."""
    from ..m7.coverage import density_order

    if not len(cs):
        return cs
    b = max(int(round(block_km / 0.1)), 10)
    mid = np.array([p[len(p) // 2] for p in cs.paths])
    bid = (mid[:, 0] // b) * 100000 + mid[:, 1] // b
    ev_blocks = {}
    rr, cc = np.nonzero(u.ev)
    eb = (rr // b) * 100000 + cc // b
    ids, cnt = np.unique(eb, return_counts=True)
    ev_blocks = dict(zip(ids, cnt * 0.01))
    used = {}
    keep = []
    for i in density_order(cs):
        k = bid[i]
        km = len(cs.paths[i]) * 0.112
        lim = cap * ev_blocks.get(k, 0.0) / 100.0
        if used.get(k, 0.0) + km <= lim:
            used[k] = used.get(k, 0.0) + km
            keep.append(i)
    return cs.subset(sorted(keep))


def coverage_area_share(cs: CandidateSet, u, radius_px: int = 3) -> float:
    if not len(cs):
        return 0.0
    m = ndimage.binary_dilation(cs.raster(u.ev.shape) > 0, structure=_disk(radius_px))
    return float((m & u.ev).sum() / max(u.ev.sum(), 1))


def unit_oracle(m8, u, cs: CandidateSet, cs7: CandidateSet | None) -> dict:
    from ..gems3.oracle import candidate_truth_stats, coverage
    from ..m7.coverage import _ocfg, density_subset
    from ..m7.protocol import system_hits

    oc = m8.cfg["oracle"]
    oo = _ocfg(m8.sh)
    base = {"task": u.task, "zone": u.zone}
    summary, curve, by, systems = [], [], [], []
    pools = {"m8": cs, "m7": cs7}
    for name, pool in pools.items():
        if pool is None or not len(pool):
            continue
        summary.append({**base, "pool": name, "subset": "all", **coverage(m8.ctx, u, pool, oo),
                        "coverage_area_share": coverage_area_share(pool, u)})
        for cap in oc["density_caps_km_per_100km2"]:
            sub = density_subset(pool, u, float(cap))
            curve.append({**base, "pool": name, "density_cap": float(cap), **coverage(m8.ctx, u, sub, oo),
                          "coverage_area_share": coverage_area_share(sub, u)})
        gate = density_subset(pool, u, float(oc["gate_density_km_per_100km2"]))
        systems.append(system_hits(m8.ctx, u, gate.raster(u.ev.shape), m8.sh).assign(pool=name, subset="at_gate_density"))
        systems.append(system_hits(m8.ctx, u, pool.raster(u.ev.shape), m8.sh).assign(pool=name, subset="all"))
        reg = regional_subset(pool, u, float(oc["regional_cap_km_per_100km2"]), float(oc["regional_block_km"]))
        summary.append({**base, "pool": name, "subset": "regional_budget", **coverage(m8.ctx, u, reg, oo)})
    st = candidate_truth_stats(cs, u) if len(cs) else pd.DataFrame()
    orc = []
    for t in oc["kd_thresholds"]:
        sub = cs.subset(np.flatnonzero(st.kd_mean.to_numpy() >= t).tolist()) if len(st) else cs
        orc.append({**base, "kd_threshold": float(t), **coverage(m8.ctx, u, sub, oo)})
    fams = sorted({f for x in cs.table.get("families", pd.Series(dtype=str)) for f in str(x).split(";") if f})
    for f in fams:
        idx = [i for i, x in enumerate(cs.table.families) if f in str(x).split(";")]
        by.append({**base, "group": "family", "value": f, **coverage(m8.ctx, u, cs.subset(idx), oo)})
    for col in ("modality", "kind"):
        if col in cs.table:
            for v in sorted(cs.table[col].dropna().astype(str).unique()):
                idx = np.flatnonzero(cs.table[col].astype(str).to_numpy() == v).tolist()
                by.append({**base, "group": col, "value": v, **coverage(m8.ctx, u, cs.subset(idx), oo)})
    # unique coverage of the lowest-confidence tercile (would a confidence filter have lost systems?)
    uniq = {}
    if len(cs) >= 3 and "structural_score" in cs.table:
        s = cs.table.structural_score.to_numpy()
        lo = np.flatnonzero(s <= np.quantile(s, 1 / 3))
        hi = np.setdiff1d(np.arange(len(cs)), lo)
        hl = system_hits(m8.ctx, u, cs.subset(lo.tolist()).raster(u.ev.shape), m8.sh)
        hh = system_hits(m8.ctx, u, cs.subset(hi.tolist()).raster(u.ev.shape), m8.sh)
        if len(hl):
            m = hl.merge(hh, on="system_id", suffixes=("_lo", "_hi"))
            uniq = {**base, "n_systems": int(len(m)), "low_conf_only_systems": int((m.recovered_lo & ~m.recovered_hi).sum()),
                    "low_conf_candidates": int(len(lo))}
    return {"summary": pd.DataFrame(summary), "curve": pd.DataFrame(curve), "oracle": pd.DataFrame(orc),
            "by_group": pd.DataFrame(by), "systems": pd.concat(systems, ignore_index=True).assign(**base) if systems else
            pd.DataFrame(), "low_conf": pd.DataFrame([uniq]) if uniq else pd.DataFrame()}


def pooled(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    from ..m7.coverage import pooled as P

    if not len(df):
        return df
    out = P(df, keys)
    if "coverage_area_share" in df:
        out = out.merge(df.groupby(keys).coverage_area_share.mean().reset_index(), on=keys, how="left")
    return out


def gate2(m8, pc: pd.DataFrame, inc: pd.DataFrame) -> pd.DataFrame:
    oc, gi = m8.cfg["oracle"], m8.cfg["oracle"]["go_if"]
    cap = float(oc["gate_density_km_per_100km2"])
    rows = []
    for task in sorted(set(pc.task)) if len(pc) else []:
        at = pc[(pc.task == task) & np.isclose(pc.density_cap, cap)]
        s8 = at[at.pool == "m8"].system_recall
        s7 = at[at.pool == "m7"].system_recall
        sr8 = float(s8.iloc[0]) if len(s8) else np.nan
        sr7 = float(s7.iloc[0]) if len(s7) else np.nan
        gain = sr8 - sr7 if np.isfinite(sr8) and np.isfinite(sr7) else np.nan
        ic = inc[(inc.task == task) & (inc.pool == "m8") & (inc.subset == "at_gate_density") & (inc["class"] == "neither")] \
            if len(inc) else inc
        miss = float(ic.coverage.iloc[0]) if len(ic) else np.nan
        ok_sr = np.isfinite(sr8) and sr8 >= float(gi["system_recall"])
        ok_gain = np.isfinite(gain) and gain >= float(gi["gain_over_m7_at_matched_density"])
        ok_miss = np.isfinite(miss) and miss >= float(gi["missed_by_both_coverage"])
        go = ok_sr and (ok_gain or ok_miss)
        rows.append({"task": task, "gate": "G2_candidate_coverage", "density_km_per_100km2": cap,
                     "system_recall_m8": sr8, "system_recall_m7_same_density": sr7, "gain_over_m7": gain,
                     "missed_by_both_coverage": miss, "pass_system_recall": ok_sr, "pass_gain": ok_gain,
                     "pass_missed": ok_miss, "decision": "GO" if go else "NO_GO",
                     "rule": (f"recall >= {gi['system_recall']} AND (gain >= {gi['gain_over_m7_at_matched_density']} "
                              f"OR missed-by-both coverage >= {gi['missed_by_both_coverage']})"),
                     "profile": m8.profile["name"], "provisional": bool(m8.profile["provisional"]),
                     "note": "oracle numbers read the truth: diagnostic upper bounds, never deployable scores"})
    return pd.DataFrame(rows)


def incremental(systems: pd.DataFrame, systab: pd.DataFrame) -> pd.DataFrame:
    if not len(systems) or not len(systab) or "class" not in systab:
        return pd.DataFrame()
    m = systems.merge(systab[["task", "zone", "system_id", "class"]], on=["task", "zone", "system_id"], how="left")
    m["class"] = m["class"].fillna("unclassified")
    g = m.groupby(["task", "pool", "subset", "class"]).agg(n_systems=("recovered", "size"),
                                                          covered=("recovered", "sum")).reset_index()
    g["coverage"] = g.covered / g.n_systems.clip(lower=1)
    return g


def oracle(m8, sets: dict) -> dict:
    parts = {k: [] for k in ("summary", "curve", "oracle", "by_group", "systems", "low_conf")}
    for task in ("geo", m8.prim):
        for u in m8.units_of(task):
            cs = sets.get(u.key)
            if cs is None or not len(cs):
                continue
            r = unit_oracle(m8, u, cs, m8.m7_pool(u.key))
            for k in parts:
                parts[k].append(r[k])
            m8.say(f"s06 oracle {u.key}: {len(cs)} candidates")
    res = {k: pd.concat(v, ignore_index=True) if v else pd.DataFrame() for k, v in parts.items()}
    res["pooled_curve"] = pooled(res["curve"], ["task", "pool", "density_cap"])
    res["pooled_summary"] = pooled(res["summary"], ["task", "pool", "subset"])
    res["pooled_oracle"] = pooled(res["oracle"], ["task", "kd_threshold"])
    res["pooled_groups"] = pooled(res["by_group"], ["task", "group", "value"])
    res["incremental"] = incremental(res["systems"], m8.system_table())
    res["gate"] = gate2(m8, res["pooled_curve"], res["incremental"])
    return res
