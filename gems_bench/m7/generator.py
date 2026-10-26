"""Section 03: multiscale geological candidate generator (label-free).

A candidate is a traced polyline, not a pixel. For every source (raw band, engineered feature or honest model map)
and every scale sigma of ``candidates.scales_m``:

1. NaN-aware Gaussian smoothing at sigma;
2. response: ``edge`` = gradient magnitude (potential-field contacts / steps), ``ridge`` / ``valley`` /
   ``ridge_valley`` = Hessian line response of bright / dark / both lines (dykes, scarps, model crests);
3. non-maximum suppression across the response normal (1-px crests);
4. gates: response in the strongest ``top_frac_per_scale`` of the valid area, structure-tensor coherence >=
   ``min_coherence`` (a lineament, not a blob or a texture);
5. cross-scale persistence: the number of scales whose gated crests lie within a scale-dependent tolerance;
   a crest is kept if it persists at >= ``min_persistence`` scales or is very strong at its own scale;
6. skeleton -> endpoint linking of fragments with consistent tangents (``postprocess.topology.link_endpoints``) ->
   network labels (8-connected linked skeleton) -> junction removal -> branches -> ordered paths.

Every candidate keeps provenance: source, modality, mode, scale, network id, junction contacts at its ends, mean
normalised strength, coherence, persistence, and a stable id (hash of source, scale and its full-grid geometry).

Structural vs fault-specific evidence: a single-modality potential-field edge is ``structural`` evidence (lithologic
contacts, intrusions and basin margins produce edges too). A candidate becomes ``multi_evidence`` only after fusion,
when other modalities (or the neural maps) agree along it. Nothing here reads labels; parameters are fixed in the
configuration before any coverage is measured (section 04 never tunes them).
"""
from __future__ import annotations

import hashlib
import time

import numpy as np
import pandas as pd
from scipy import ndimage

from ..gems3.candidates import CandidateSet, tip_paths
from ..postprocess.topology import _disk, link_endpoints

PIXEL_M = 100.0


# =============================================================================== responses
def _smooth(a: np.ndarray, valid: np.ndarray, sigma_px: float) -> np.ndarray:
    from gems_eda.enhance import nan_gaussian

    return nan_gaussian(np.where(valid, a, np.nan).astype(np.float32), sigma_px)


def response(a: np.ndarray, valid: np.ndarray, sigma_px: float, mode: str) -> dict:
    """{strength, nms, coherence, strike_deg} of one source at one scale (float32 / bool arrays)."""
    from ..p3.lineaments import hessian_ridge

    ps = _smooth(a, valid, sigma_px)
    fin = np.isfinite(ps) & valid
    med = float(np.nanmedian(ps[fin])) if fin.any() else 0.0
    f = np.where(fin, ps, med).astype(np.float32)
    if mode == "edge":
        gr, gc = np.gradient(f)
        strength = np.hypot(gr, gc).astype(np.float32)
        norm = np.maximum(strength, 1e-12)
        nr, nc = (gr / norm).astype(np.float32), (gc / norm).astype(np.float32)
        field = strength
    else:
        outs = []
        if mode in ("ridge", "ridge_valley"):
            outs.append(hessian_ridge(f, sigma_px) + (f,))
        if mode in ("valley", "ridge_valley"):
            outs.append(hessian_ridge(-f, sigma_px) + (-f,))
        if len(outs) == 1:
            strength, nr, nc, field = outs[0]
        else:
            (s1, r1, c1, f1), (s2, r2, c2, f2) = outs
            take = s2 > s1
            strength = np.where(take, s2, s1).astype(np.float32)
            nr, nc = np.where(take, r2, r1).astype(np.float32), np.where(take, c2, c1).astype(np.float32)
            field = None
            nms = _nms(f1, r1, c1) & ~take | _nms(f2, r2, c2) & take
        strength = np.clip(strength, 0, None)
    if mode == "edge" or field is not None:
        nms = _nms(field, nr, nc)
    nms &= fin & (strength > 0)
    # structure-tensor coherence at an integration scale of 2 sigma
    gy, gx = np.gradient(f)
    si = max(2.0 * sigma_px, 1.0)
    jxx = ndimage.gaussian_filter(gx * gx, si)
    jyy = ndimage.gaussian_filter(gy * gy, si)
    jxy = ndimage.gaussian_filter(gx * gy, si)
    lam = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2)
    coh = np.where(jxx + jyy > 1e-20, lam / (jxx + jyy + 1e-20), 0.0).astype(np.float32)
    strike = (np.degrees(np.arctan2(nr, nc)) % 180.0).astype(np.float32)       # along-structure azimuth
    return {"strength": strength.astype(np.float32), "nms": nms, "coherence": coh, "strike_deg": strike}


def _nms(field: np.ndarray, nr: np.ndarray, nc: np.ndarray) -> np.ndarray:
    rr, cc = np.indices(field.shape, dtype=np.float32)
    fwd = ndimage.map_coordinates(field, [rr + nr, cc + nc], order=1, mode="nearest")
    bwd = ndimage.map_coordinates(field, [rr - nr, cc - nc], order=1, mode="nearest")
    del rr, cc
    return (field >= fwd) & (field >= bwd)


def gated_crests(a: np.ndarray, valid: np.ndarray, scales_m, mode: str, cc: dict) -> tuple[list[dict], np.ndarray]:
    """Per scale: gated crest mask + normalised strength / coherence; and the cross-scale persistence count."""
    per, dil = [], []
    for s_m in scales_m:
        sig = max(float(s_m) / PIXEL_M, 0.7)
        r = response(a, valid, sig, mode)
        v = r["strength"][r["nms"] & valid]
        if not v.size:
            per.append(None)
            continue
        thr = float(np.quantile(v, 1.0 - float(cc["top_frac_per_scale"])))
        strong = float(np.quantile(v, 1.0 - float(cc["strong_frac"])))
        ref = float(np.quantile(v, 0.99)) or 1.0
        L = r["nms"] & valid & (r["strength"] >= thr) & (r["coherence"] >= float(cc["min_coherence"]))
        per.append({"scale_m": float(s_m), "sigma_px": sig, "mask": L, "strong": L & (r["strength"] >= strong),
                    "s_norm": np.clip(r["strength"] / ref, 0, 1).astype(np.float32), "coherence": r["coherence"],
                    "threshold": thr})
        tol = max(1, int(round(sig / 2.0)))
        dil.append(ndimage.binary_dilation(L, structure=_disk(tol)))
        del r
    P = np.sum(np.stack(dil), 0).astype(np.uint8) if dil else np.zeros(valid.shape, np.uint8)
    return per, P


# =============================================================================== tracing
def mask_to_paths(mask: np.ndarray, cc: dict) -> tuple[list[np.ndarray], list[dict]]:
    """Linked skeleton -> branches -> ordered paths, with network id and junction contacts at the branch ends."""
    from skimage.morphology import remove_small_objects

    from ..geology.skeleton import branch_points, skeletonize
    from ..p3.lineaments import split_branches, trace

    min_px = max(int(np.ceil(float(cc["min_len_km"]) * 1000.0 / (PIXEL_M * 1.12))), 2)
    sk = skeletonize(mask)
    sk = remove_small_objects(sk, min_size=2, connectivity=2)
    if int(cc.get("link_gap_px", 0)) > 0 and sk.any():
        linked = link_endpoints(sk, int(cc["link_gap_px"]), float(cc["link_angle_deg"]))
        added = linked & ~sk
        sk = skeletonize(linked)
    else:
        added = np.zeros(sk.shape, bool)
    net, _ = ndimage.label(sk, structure=np.ones((3, 3), bool))
    jun = ndimage.binary_dilation(branch_points(sk), structure=np.ones((3, 3), bool), iterations=2)
    lab, _, _ = split_branches(sk, min_px)
    paths, meta = [], []
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None:
            continue
        loc = np.argwhere(lab[sl] == i) + np.array([sl[0].start, sl[1].start])
        if len(loc) < min_px:
            continue
        p = trace(loc)
        ends = (p[0], p[-1])
        meta.append({"network": int(net[p[0, 0], p[0, 1]]), "junction_ends": int(sum(bool(jun[e[0], e[1]]) for e in ends)),
                     "linked_px": int(added[p[:, 0], p[:, 1]].sum())})
        paths.append(p)
    return paths, meta


def stable_id(source: str, scale, p_full: np.ndarray) -> str:
    key = f"{source}|{scale}|{int(p_full[0, 0])},{int(p_full[0, 1])}|{int(p_full[-1, 0])},{int(p_full[-1, 1])}|{len(p_full)}"
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def source_candidates(a: np.ndarray, valid: np.ndarray, src: dict, cc: dict, *, offset=(0, 0),
                      net_prefix: str = "") -> tuple[list[np.ndarray], list[dict], list[dict]]:
    """All candidates of one source (paths in the array's own coordinates), their rows and a per-scale log."""
    scales = src.get("scales_m") or cc["scales_m"]
    t0 = time.perf_counter()
    per, P = gated_crests(a, valid, scales, src["mode"], cc)
    mp = int(cc["min_persistence"])
    paths, rows, log = [], [], []
    for lv in per:
        if lv is None:
            continue
        keep = lv["mask"] & ((P >= min(mp, len(scales))) | lv["strong"])
        ps, meta = mask_to_paths(keep, cc)
        cap = int(cc.get("max_candidates_per_source") or 0)
        if cap and len(ps) > cap:
            order = np.argsort([-len(p) for p in ps], kind="stable")[:cap]
            ps, meta = [ps[i] for i in order], [meta[i] for i in order]
        for p, m in zip(ps, meta):
            pf = p + np.asarray(offset)
            s_n, coh, pers = lv["s_norm"][p[:, 0], p[:, 1]], lv["coherence"][p[:, 0], p[:, 1]], P[p[:, 0], p[:, 1]]
            rows.append({"uid": stable_id(src["name"], lv["scale_m"], pf), "source": src["name"],
                         "modality": src.get("modality", "other"), "mode": src["mode"], "kind": src.get("kind", "feature"),
                         "scale_m": lv["scale_m"], "network_id": f"{net_prefix}{src['name']}@{lv['scale_m']:g}:{m['network']}",
                         "junction_ends": m["junction_ends"], "linked_px": m["linked_px"],
                         "strength_mean": float(s_n.mean()), "coherence_mean": float(coh.mean()),
                         "persistence_mean": float(pers.mean()) / max(len(scales), 1),
                         "conf_lf": float(s_n.mean() * coh.mean() * (pers.mean() / max(len(scales), 1)))})
            paths.append(p)
        log.append({"source": src["name"], "scale_m": lv["scale_m"], "threshold": lv["threshold"],
                    "crest_px": int(lv["mask"].sum()), "kept_px": int(keep.sum()), "n_paths": len(ps)})
    for r in log:
        r["seconds_source"] = round(time.perf_counter() - t0, 1)
    return paths, rows, log


def read_source(ctx, src: dict) -> np.ndarray | None:
    if src.get("band"):
        return ctx.data.read_band(src["band"]) if src["band"] in ctx.data.band_names else None
    if src.get("feature") and ctx.store.has(src["feature"]):
        return np.asarray(ctx.store.get(src["feature"]), np.float32)
    return None


def label_free(ctx, cfg: dict) -> tuple[CandidateSet, pd.DataFrame]:
    """Every label-free source on the full grid (computed once; the caller caches it)."""
    cc = cfg["candidates"]
    allp, allr, logs = [], [], []
    for src in cc["sources"]:
        a = read_source(ctx, src)
        if a is None:
            logs.append({"source": src["name"], "status": "missing (band / feature not available)"})
            print(f"[m7] s03 {src['name']}: missing, skipped")
            continue
        ps, rows, log = source_candidates(a, ctx.data.valid & np.isfinite(a), {**src, "kind": "feature"}, cc)
        allp += ps
        allr += rows
        logs += [{**x, "status": "ok"} for x in log]
        print(f"[m7] s03 {src['name']}: {len(ps)} candidates")
        del a
    return CandidateSet(pd.DataFrame(allr), allp), pd.DataFrame(logs)


# =============================================================================== per unit
def clip_to_unit(lf: CandidateSet, u, min_px: int) -> CandidateSet:
    off = np.array([u.sl[0].start, u.sl[1].start])
    h, w = u.ev.shape
    keep, paths = [], []
    for i, p in enumerate(lf.paths):
        q = p - off
        ok = (q[:, 0] >= 0) & (q[:, 0] < h) & (q[:, 1] >= 0) & (q[:, 1] < w)
        if ok.sum() < min_px:
            continue
        q = q[ok]
        q = q[u.ev[q[:, 0], q[:, 1]]]
        if len(q) >= min_px:
            keep.append(i)
            paths.append(q)
    return CandidateSet(lf.table.iloc[keep].reset_index(drop=True), paths)


def unit_candidates(u, lf: CandidateSet, model_scores: dict, cfg: dict) -> tuple[CandidateSet, pd.DataFrame]:
    """Label-free candidates clipped to the unit + honest model crests + known-tip extensions, then fused."""
    cc = cfg["candidates"]
    min_px = max(int(np.ceil(float(cc["min_len_km"]) / 0.112)), 2)
    off = (u.sl[0].start, u.sl[1].start)
    parts = [clip_to_unit(lf, u, min_px)]
    log = []
    for src in cc.get("model_sources") or []:
        s = model_scores.get(src["model"])
        if s is None:
            continue
        ps, rows, lg = source_candidates(np.nan_to_num(s, nan=0.0), u.ev, {**src, "mode": "ridge", "modality": "neural",
                                                                         "kind": "model"}, cc, offset=off,
                                         net_prefix=f"{u.key}/")
        parts.append(CandidateSet(pd.DataFrame(rows), ps))
        log += [{**x, "unit": u.key} for x in lg]
    if u.known is not None and u.task != "geo":
        rows, ps = [], []
        for j, (p, L) in enumerate(tip_paths(u.known, u.ev, cc["tip_extension_km"], cc["tip_tail_px"])):
            pf = p + np.asarray(off)
            rows.append({"uid": stable_id(f"tip{L:g}km", 0, pf), "source": f"tip{L:g}km", "modality": "network",
                         "mode": "extension", "kind": "tip", "scale_m": np.nan, "network_id": f"{u.key}/tip:{j}",
                         "junction_ends": 0, "linked_px": 0, "strength_mean": np.nan, "coherence_mean": np.nan,
                         "persistence_mean": np.nan, "conf_lf": np.nan})
            ps.append(p)
        parts.append(CandidateSet(pd.DataFrame(rows), ps))
    tabs = [p.table for p in parts if len(p)]
    raw = CandidateSet(pd.concat(tabs, ignore_index=True) if tabs else pd.DataFrame(),
                       [x for p in parts for x in p.paths])
    return fuse(raw, u.ev.shape, cc), pd.DataFrame(log)


def fuse(raw: CandidateSet, shape, cc: dict) -> CandidateSet:
    """Duplicate removal (longest first) keeping provenance, then cross-modal agreement per modality.

    ``agree_<modality>`` = share of the candidate within ``agreement_radius_px`` of that modality's raw crests (any
    source, any scale); ``n_agree_modalities`` counts other modalities agreeing on >= half of it; ``evidence_class``
    is ``multi_evidence`` when >= 1 other modality agrees, else ``structural_only``."""
    cols = ["cand_id", "uid", "source", "modality", "kind", "sources", "n_sources", "member_uids"]
    if not len(raw):
        return CandidateSet(pd.DataFrame(columns=cols), [])
    rad = int(cc["dedup_radius_px"])
    disk = np.argwhere(_disk(rad)) - rad
    h, w = shape
    order = np.argsort([-len(p) for p in raw.paths], kind="stable")
    claimed = np.zeros(shape, np.int32)
    keep, prov, mem, scales = [], {}, {}, {}
    src = raw.table.source.to_numpy()
    uid = raw.table.uid.to_numpy()
    sc = raw.table.scale_m.to_numpy()
    for i in order:
        p = raw.paths[i]
        own = claimed[p[:, 0], p[:, 1]]
        if (own > 0).mean() >= float(cc["dedup_overlap"]):
            o = int(np.bincount(own[own > 0]).argmax()) - 1
            prov[o].add(str(src[i]))
            mem[o].append(str(uid[i]))
            if np.isfinite(sc[i]):
                scales[o].add(float(sc[i]))
            continue
        k = len(keep)
        keep.append(i)
        prov[k], mem[k] = {str(src[i])}, [str(uid[i])]
        scales[k] = {float(sc[i])} if np.isfinite(sc[i]) else set()
        pts = (p[:, None, :] + disk[None, :, :]).reshape(-1, 2)
        pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] < h) & (pts[:, 1] >= 0) & (pts[:, 1] < w)]
        free = claimed[pts[:, 0], pts[:, 1]] == 0
        claimed[pts[free, 0], pts[free, 1]] = k + 1
    out = raw.subset(keep)
    t = out.table
    t["cand_id"] = np.arange(len(out))
    t["sources"] = [";".join(sorted(prov[k])) for k in range(len(out))]
    t["n_sources"] = [len(prov[k]) for k in range(len(out))]
    t["member_uids"] = [";".join(mem[k]) for k in range(len(out))]
    t["n_scales"] = [len(scales[k]) for k in range(len(out))]
    arad = _disk(int(cc["agreement_radius_px"]))
    mods = sorted(raw.table.modality.dropna().unique())
    for mo in mods:
        m = np.zeros(shape, bool)
        for j in np.flatnonzero(raw.table.modality.to_numpy() == mo):
            m[raw.paths[j][:, 0], raw.paths[j][:, 1]] = True
        m = ndimage.binary_dilation(m, structure=arad)
        t[f"agree_{mo}"] = [float(m[p[:, 0], p[:, 1]].mean()) for p in out.paths]
    own = t.modality.to_numpy()
    t["n_agree_modalities"] = [int(sum(t.at[i, f"agree_{mo}"] >= 0.5 for mo in mods if mo != own[i]))
                               for i in range(len(out))]
    geo_mods = [mo for mo in mods if mo not in ("neural", "network")]
    t["n_agree_geophysical"] = [int(sum(t.at[i, f"agree_{mo}"] >= 0.5 for mo in geo_mods if mo != own[i]))
                                for i in range(len(out))]
    t["evidence_class"] = np.where(t.n_agree_modalities >= 1, "multi_evidence", "structural_only")
    return out


def structural_score(t: pd.DataFrame) -> np.ndarray:
    """Label-free confidence used to order candidates (density curves, the label-free ranking control):
    mean percentile of generator confidence, cross-modal agreement, scale persistence and length. Tips (no
    generator confidence) rank on agreement and length only."""
    cols = {}
    if "conf_lf" in t:
        cols["conf"] = t.conf_lf
    for c in ("n_agree_geophysical", "n_scales", "n_sources"):
        if c in t:
            cols[c] = t[c]
    if "length_km" in t:
        cols["len"] = t.length_km
    d = pd.DataFrame(cols)
    if not len(d.columns):
        return np.zeros(len(t))
    return d.rank(pct=True).mean(axis=1, skipna=True).fillna(0).to_numpy()


def geometry(cs: CandidateSet) -> pd.DataFrame:
    from ..gems3.candidates import geometry_table

    return geometry_table(cs)
