"""Experiment 1: geological evidence atlas (what the models see, what they miss) + review persistence.

Candidates come from eight predeclared categories (config ``atlas.categories``) built on HONEST maps only: each geo
fold zone shows the B0 / B1 out-of-fold maps and the W2 held-out map of that zone (zone-stitched, the factory
construction), and the frozen submitted W2 policy map. Selection is seeded, round-robin over a 3 x 3 region grid,
with a minimum separation between any two candidates (no near-duplicates, geographic diversity); categories are not
filtered by attractiveness.

Two panel modes per candidate: EVIDENCE_ONLY (physical layers only; a cross marks the window centre, no fault label,
no model output, no candidate outline) and DIAGNOSTIC (adds mapped faults, B0 / B1 / W2 and the candidate trace).
Every panel uses the same UTM extent, an equal aspect, a 5 km scale bar; each layer's display stretch (percentiles)
is recorded in ``panels_data/<id>.json`` with the raw window arrays (``.npz``).

Reviews are APPEND-ONLY (``reviewed_annotations.csv``, one row per review event; the latest row per candidate is the
current interpretation). A review counts as independent of the original fault map only when it was made in
EVIDENCE_ONLY mode and lists no mapped-fault / model source. Unreviewed candidates are NOT negatives.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from scipy import ndimage

INTERPRETATIONS = ("LIKELY_FAULT", "POSSIBLE_FAULT", "LIKELY_CONTACT", "POSSIBLE_INTRUSION_OR_DYKE",
                   "TERRAIN_OR_DRAINAGE_ARTIFACT", "SURVEY_OR_PROCESSING_ARTIFACT", "INSUFFICIENT_EVIDENCE",
                   "NOT_REVIEWED")
REVIEW_MODES = ("EVIDENCE_ONLY", "DIAGNOSTIC")
DEPENDENT_SOURCES = ("mapped_faults", "w2", "b0", "b1", "model")
REVIEW_FIELDS = ("review_id", "candidate_id", "interpretation", "confidence", "supporting_evidence",
                 "alternative_explanation", "reviewer_notes", "reviewer", "review_mode", "evidence_sources",
                 "review_date", "data_license", "geometry_wkt", "independent_of_original_fault_map", "label_class",
                 "review_split")
CATALOG_FIELDS = ("candidate_id", "category", "region_id", "review_split", "row", "col", "x_utm", "y_utm",
                  "window_extent", "length_km", "n_px", "source_models", "source_layers", "distance_to_known_fault_km",
                  "review_status", "interpretation", "confidence", "supporting_evidence", "alternative_explanation",
                  "reviewer_notes")
S8 = np.ones((3, 3), bool)
PX_KM = 0.1


# =============================================================================== pure: selection
def region_ids(rows: np.ndarray, cols: np.ndarray, bbox, grid) -> np.ndarray:
    """Region index (row-major) of each pixel in a ``grid`` = (n_rows, n_cols) partition of ``bbox`` (r0, r1, c0, c1)."""
    r0, r1, c0, c1 = bbox
    gr, gc = grid
    i = np.clip(((np.asarray(rows) - r0) * gr // max(r1 - r0, 1)).astype(int), 0, gr - 1)
    j = np.clip(((np.asarray(cols) - c0) * gc // max(c1 - c0, 1)).astype(int), 0, gc - 1)
    return i * gc + j


def components(mask: np.ndarray, min_px: int = 1) -> list[tuple[np.ndarray, np.ndarray]]:
    """Pixel sets (rows, cols) of the 8-connected components of ``mask`` with >= ``min_px`` pixels."""
    lab, n = ndimage.label(np.asarray(mask, bool), structure=S8)
    if not n:
        return []
    objs = ndimage.find_objects(lab)
    out = []
    for i, sl in enumerate(objs, start=1):
        rr, cc = np.nonzero(lab[sl] == i)
        if len(rr) >= min_px:
            out.append((rr + sl[0].start, cc + sl[1].start))
    return out


def quotas(categories: dict, n: int) -> dict:
    w = {k: float(v) for k, v in categories.items() if float(v) > 0}
    tot = sum(w.values())
    q = {k: int(np.floor(n * v / tot)) for k, v in w.items()}
    for k in sorted(w, key=lambda k: -w[k])[: n - sum(q.values())]:
        q[k] += 1
    return q


def select(pools: dict, quota: dict, region_of, min_sep_px: float, rng) -> list[dict]:
    """Seeded selection: per category, shuffle the pool, visit regions round-robin, keep a candidate only when it is
    at least ``min_sep_px`` from every candidate already kept (any category). Returns kept items with category and
    region; a category may end below its quota (recorded, never filled with weaker candidates of another one)."""
    kept, pts = [], []
    for cat in quota:
        items = list(pools.get(cat) or [])
        if not items:
            continue
        order = rng.permutation(len(items))
        by_reg: dict[int, list] = {}
        for i in order:
            it = items[i]
            r, c = float(np.mean(it[0])), float(np.mean(it[1]))
            by_reg.setdefault(int(region_of(r, c)), []).append((r, c, it))
        regs = sorted(by_reg)
        n = 0
        while n < quota[cat] and any(by_reg.values()):
            for g in regs:
                if n >= quota[cat] or not by_reg[g]:
                    continue
                r, c, it = by_reg[g].pop(0)
                if pts and min(np.hypot(r - a, c - b) for a, b in pts) < min_sep_px:
                    continue
                pts.append((r, c))
                kept.append({"category": cat, "region_id": g, "row": r, "col": c, "rr": it[0], "cc": it[1]})
                n += 1
    return kept


def stretch(a: np.ndarray, pct=(2, 98)) -> tuple[float, float]:
    v = a[np.isfinite(a)]
    if not v.size:
        return (np.nan, np.nan)
    lo, hi = np.percentile(v, pct)
    return float(lo), float(hi if hi > lo else lo + 1e-6)


# =============================================================================== pure: reviews
def review_split(region_id: int, eval_regions) -> str:
    return "EVAL" if int(region_id) in {int(x) for x in eval_regions} else "TRAIN"


def label_class(interp: str, mode: str, sources: str, confidence, hard_negative=()) -> tuple[bool, str]:
    """(independent_of_original_fault_map, label class). INDEPENDENT_REVIEWED only for evidence-only reviews that
    cite no mapped-fault / model source; DIAGNOSTIC_REVIEWED otherwise; HARD_NEGATIVE = a confident non-fault
    interpretation; UNVERIFIED = not reviewed / insufficient evidence."""
    src = {s.strip().lower() for s in str(sources or "").replace(",", ";").split(";") if s.strip()}
    indep = mode == "EVIDENCE_ONLY" and not (src & set(DEPENDENT_SOURCES))
    if interp in ("NOT_REVIEWED", "INSUFFICIENT_EVIDENCE"):
        return indep, "UNVERIFIED"
    conf = _conf(confidence)
    if interp in hard_negative:
        return indep, "HARD_NEGATIVE" if conf >= 0.66 else "UNVERIFIED_NONFAULT"
    return indep, "INDEPENDENT_REVIEWED" if indep else "DIAGNOSTIC_REVIEWED"


def _conf(x) -> float:
    m = {"LOW": 0.33, "MEDIUM": 0.66, "MED": 0.66, "HIGH": 1.0}
    if isinstance(x, str) and x.strip().upper() in m:
        return m[x.strip().upper()]
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def validate_review(r: dict, known_ids) -> list[str]:
    bad = []
    if r.get("candidate_id") not in known_ids:
        bad.append(f"unknown candidate_id {r.get('candidate_id')!r}")
    if r.get("interpretation") not in INTERPRETATIONS:
        bad.append(f"interpretation {r.get('interpretation')!r} not in {INTERPRETATIONS}")
    if r.get("review_mode") not in REVIEW_MODES:
        bad.append(f"review_mode {r.get('review_mode')!r} not in {REVIEW_MODES}")
    c = _conf(r.get("confidence"))
    if not 0.0 <= c <= 1.0:
        bad.append("confidence must be in [0, 1] or LOW / MEDIUM / HIGH")
    if not str(r.get("reviewer") or "").strip():
        bad.append("reviewer is required (provenance)")
    return bad


def append_reviews(existing: pd.DataFrame, new: list[dict], catalog: pd.DataFrame, hard_negative=()) -> tuple[
        pd.DataFrame, list[str]]:
    """Append validated review events (never edits or drops an earlier row); identical repeats are skipped."""
    ids = set(catalog.candidate_id)
    cat = catalog.set_index("candidate_id")
    rows, errs = [], []
    cols = [k for k in REVIEW_FIELDS if k not in ("review_id", "review_date")]
    seen = set(existing.reindex(columns=cols).fillna("").astype(str).apply(tuple, axis=1)) if len(existing) else set()
    for r in new:
        r = {k: ("" if isinstance(v, float) and np.isnan(v) else v) for k, v in r.items()}   # template NaNs
        r = {k: r.get(k, "") for k in REVIEW_FIELDS if k not in ("review_id", "review_date", "label_class",
                                                                  "independent_of_original_fault_map",
                                                                  "review_split", "geometry_wkt")} | r
        e = validate_review(r, ids)
        if e:
            errs += [f"{r.get('candidate_id')}: {x}" for x in e]
            continue
        indep, cls = label_class(r["interpretation"], r["review_mode"], r.get("evidence_sources"), r.get("confidence"),
                                 hard_negative)
        c = cat.loc[r["candidate_id"]]
        r.update(independent_of_original_fault_map=indep, label_class=cls, review_split=c.review_split,
                 geometry_wkt=r.get("geometry_wkt") or c.get("geometry_wkt", ""))
        key = tuple("" if r.get(k) is None else str(r.get(k, "")) for k in cols)
        if key in seen:
            continue
        r["review_id"] = f"r{len(existing) + len(rows) + 1:05d}"
        r["review_date"] = r.get("review_date") or time.strftime("%Y-%m-%dT%H:%M:%S")
        rows.append({k: r.get(k, "") for k in REVIEW_FIELDS})
        seen.add(key)
    out = pd.concat([existing, pd.DataFrame(rows, columns=list(REVIEW_FIELDS))], ignore_index=True) if rows else \
        existing
    return out, errs


def latest_reviews(R: pd.DataFrame) -> pd.DataFrame:
    if not len(R):
        return R
    return R.sort_values(["candidate_id", "review_id"]).groupby("candidate_id").tail(1).reset_index(drop=True)


# =============================================================================== Colab: honest maps + pools
def honest_maps(R) -> dict:
    """Full-grid zone-stitched honest maps: B0 / B1 / W2 ranks of each geo zone from that zone's held-out maps and the
    frozen submitted W2 policy map (``atlas/cache/honest_maps.npz``)."""
    from ..gems3 import representation as REP
    from ..geomoe.evaluate import policy_q
    from ..p6.final_submission import zone_partition

    rel = "atlas/cache/honest_maps.npz"
    p = R.ws.path(rel)
    if p.exists():
        with np.load(p) as z:
            out = {k: z[k] for k in z.files}
        out["w2_frozen"] = np.unpackbits(out.pop("w2_frozen_bits"))[:int(np.prod(out["shape"]))].reshape(
            tuple(out["shape"])).astype(bool)
        return out
    d = R.ctx.data
    shape = d.shape
    v3 = [su for su in R.units if su.scenario == "V3"]
    zid, _ = zone_partition(R.ctx.folds.fold_map, d.valid, [su.zone for su in v3])
    out = {k: np.full(shape, np.nan, np.float32) for k in ("w2", "b0", "b1")}
    fro = np.zeros(shape, bool)
    fz = R.gm.frozen_policy()["cell"]
    for su in v3:
        u = su.unit
        m = (zid[u.sl] == su.zone) & u.ev
        b0, b1 = R.bst.cnn(su.base)
        w = R.gm.w2_smap(su)
        s = R.gm.frozen_smap(su)
        for k, a in (("b0", REP.rank01(b0, u.ev)), ("b1", REP.rank01(b1, u.ev)), ("w2", w)):
            if a is not None:
                v = out[k][u.sl]
                v[m] = np.asarray(a, np.float32)[m]
        if s is not None:
            q = policy_q(R.gm, u, s, str(fz["rep"]), float(fz["frac"])) > 0
            fv = fro[u.sl]
            fv[m] = q[m]
        R.gm.release(su.base)
    R.save_npz(rel, **{k: v.astype(np.float16) for k, v in out.items()}, zone_id=zid.astype(np.int16),
               w2_frozen_bits=np.packbits(fro), shape=np.asarray(shape))
    out.update(w2_frozen=fro, zone_id=zid, shape=np.asarray(shape))
    return out


def category_pools(R, L: dict, M: dict) -> tuple[dict, dict]:
    """{category: [(rows, cols), ...]} and the label-derived full-grid helpers (distance to mapped faults)."""
    from ..geology.skeleton import skeletonize
    from .physics import rank_valid

    th = R.rc["atlas"]["thresholds"]
    d = R.ctx.data
    valid = np.asarray(d.valid, bool) & np.asarray(d.template_valid, bool) & (M["zone_id"] >= 0)
    lab = np.asarray(d.labels, bool)
    dist = (ndimage.distance_transform_edt(~lab) * PX_KM).astype(np.float32)
    sk = skeletonize(lab)
    known = components(sk, 3)
    klen = [len(c[0]) * 0.112 for c in known]
    lin = np.nan_to_num(L["lineament"], nan=0.0)
    kmean = [float(lin[c].mean()) for c in known]
    q67 = np.quantile(kmean, 0.67) if kmean else 1.0
    fro = M["w2_frozen"]
    w2 = np.nan_to_num(M["w2"].astype(np.float32), nan=0.0)
    far_from_w2 = ndimage.distance_transform_edt(~fro) * PX_KM > float(th["missed_km"]) if fro.any() else \
        np.ones(lab.shape, bool)
    thr_lin = np.quantile(lin[valid], 1 - float(th["lineament_top"])) if valid.any() else 1.0
    crest = skeletonize((lin >= thr_lin) & valid)
    b0, b1 = (np.nan_to_num(M[k].astype(np.float32), nan=0.0) for k in ("b0", "b1"))
    t = float(th["disagreement"])
    dis = valid & ((np.abs(b0 - b1) > t) | (np.abs(w2 - 0.5 * (b0 + b1)) > t)) & (np.fmax(np.fmax(b0, b1), w2) >= 0.9)
    pools = {"A_known_supported": [c for c, m in zip(known, kmean) if m >= q67 and valid[c].all()],
             "B_missed_mapped": components(sk & far_from_w2 & valid, 10),
             "C_w2_unmapped": [c for c in components(fro & valid, 3) if dist[c].min() > float(th["unmapped_km"])],
             "D_lineament_low_w2": [c for c in components(crest, 10) if w2[c].mean() < 0.5 and
                                    dist[c].min() > float(th["unmapped_km"])],
             "E_cnn_disagreement": components(dis, 20),
             "G_short_fragmented": [c for c, n in zip(known, klen) if n < float(th["short_km"]) and valid[c].all()] +
             [c for c in components(fro & valid, 2) if len(c[0]) * 0.112 < 1.0]}
    f_mask = np.zeros(lab.shape, bool)
    if "mag_thdr_1" in L and "grav_thdr_1" in L:
        rm, rg = rank_valid(L["mag_thdr_1"], valid), rank_valid(L["grav_thdr_1"], valid)
        f_mask |= (np.nan_to_num(rm) >= float(th["single_edge_q"])) & (np.nan_to_num(rg, nan=1.0) <
                                                                       float(th["weak_other_q"]))
    if "slope" in L:
        rs = rank_valid(L["slope"], valid)
        f_mask |= (np.nan_to_num(rs) >= 0.97) & (lin < np.quantile(lin[valid], 0.5))
    pools["F_suspected_nonfault"] = components(skeletonize(f_mask & valid), 10)
    rr, cc = np.nonzero(valid)
    rng = np.random.default_rng(int(R.rc["atlas"]["seed"]) + 7)
    pick = rng.choice(len(rr), min(2000, len(rr)), replace=False) if len(rr) else []
    pools["H_regional_random"] = [(np.array([rr[i]]), np.array([cc[i]])) for i in pick]
    return pools, {"dist_known_km": dist, "valid": valid}


def build_catalog(R, L: dict, M: dict) -> tuple[pd.DataFrame, dict]:
    """Candidate catalog (CATALOG_FIELDS + geometry) and {candidate_id: (rows, cols)}."""
    ac = R.rc["atlas"]
    pools, aux = category_pools(R, L, M)
    valid = aux["valid"]
    rr, cc = np.nonzero(valid)
    bbox = (int(rr.min()), int(rr.max()) + 1, int(cc.min()), int(cc.max()) + 1)
    grid = tuple(int(x) for x in ac["region_grid"])
    reg = (lambda r, c: int(region_ids(np.array([r]), np.array([c]), bbox, grid)[0]))
    kept = select(pools, quotas(ac["categories"], int(ac["n_examples"])), reg,
                  float(ac["min_separation_km"]) / PX_KM, np.random.default_rng(int(ac["seed"])))
    T = R.ctx.data.template_profile["transform"]
    half = int(ac["patch_size_px"]) // 2
    src = {"A_known_supported": ("mapped_faults", "lineament"), "B_missed_mapped": ("mapped_faults", "w2_frozen"),
           "C_w2_unmapped": ("w2_frozen",), "D_lineament_low_w2": ("lineament", "w2"),
           "E_cnn_disagreement": ("b0", "b1", "w2"), "F_suspected_nonfault": ("mag_thdr", "grav_thdr", "slope"),
           "G_short_fragmented": ("mapped_faults", "w2_frozen"), "H_regional_random": ()}
    rows, geom = [], {}
    for i, k in enumerate(kept):
        cid = f"FDR{i:03d}_{k['category'][0]}"
        r, c = int(round(k["row"])), int(round(k["col"]))
        x, y = T * (c + 0.5, r + 0.5)
        x0, y0 = T * (c - half, r - half)
        x1, y1 = T * (c + half, r + half)
        geom[cid] = (np.asarray(k["rr"]), np.asarray(k["cc"]))
        s = src[k["category"]]
        rows.append({"candidate_id": cid, "category": k["category"], "region_id": k["region_id"],
                     "review_split": review_split(k["region_id"], ac["eval_regions"]), "row": r, "col": c,
                     "x_utm": float(x), "y_utm": float(y),
                     "window_extent": json.dumps([float(min(x0, x1)), float(min(y0, y1)), float(max(x0, x1)),
                                                  float(max(y0, y1))]),
                     "length_km": float(len(k["rr"]) * 0.112) if len(k["rr"]) > 1 else 0.0, "n_px": int(len(k["rr"])),
                     "source_models": ";".join(x_ for x_ in s if x_ in ("w2", "w2_frozen", "b0", "b1")),
                     "source_layers": ";".join(x_ for x_ in s if x_ not in ("w2", "w2_frozen", "b0", "b1")),
                     "distance_to_known_fault_km": float(aux["dist_known_km"][r, c]),
                     "review_status": "NOT_REVIEWED", "interpretation": "NOT_REVIEWED", "confidence": "",
                     "supporting_evidence": "", "alternative_explanation": "", "reviewer_notes": "",
                     "geometry_wkt": wkt(geom[cid], T)})
    C = pd.DataFrame(rows)
    short = {c: int((C.category == c).sum()) for c in pools}
    return C, {"geometry": geom, "category_counts": short, "pool_sizes": {c: len(v) for c, v in pools.items()},
               "bbox": bbox}


def wkt(g, T) -> str:
    rr, cc = g
    if len(rr) < 2:
        x, y = T * (float(cc[0]) + 0.5, float(rr[0]) + 0.5)
        return f"POINT ({x:.1f} {y:.1f})"
    th = np.mod(0.5 * np.arctan2(2 * np.mean((rr - rr.mean()) * (cc - cc.mean())), np.var(cc) - np.var(rr)), np.pi)
    p = (cc - cc.mean()) * np.cos(th) + (rr - rr.mean()) * np.sin(th)
    a, b = int(np.argmin(p)), int(np.argmax(p))
    xa, ya = T * (float(cc[a]) + 0.5, float(rr[a]) + 0.5)
    xb, yb = T * (float(cc[b]) + 0.5, float(rr[b]) + 0.5)
    return f"LINESTRING ({xa:.1f} {ya:.1f}, {xb:.1f} {yb:.1f})"


# =============================================================================== Colab: panels / html
EVIDENCE_LAYERS = (("mag", "Magnetic (RTP, native units)", "RdBu_r"), ("mag_tilt", "Magnetic tilt (rad)", "twilight"),
                   ("grav", "Gravity (native units)", "RdBu_r"), ("grav_thdr_1", "Gravity |grad| (100 m)", "magma"),
                   ("hillshade", "Terrain multi-hillshade", "gray"),
                   ("struct", "Structural coherence proxy (mag coherence x strike agreement)", "viridis"),
                   ("lineament", "Label-free lineament score", "inferno"), ("fused", "Fused: R mag, G grav, B slope", None))
DIAG_LAYERS = (("w2", "W2 held-out rank", "viridis"), ("b0b1", "B0 (R) / B1 (G) ranks", None))


def window_layers(L: dict, M: dict, sl) -> dict:
    from .physics import multi_hillshade

    w = {k: L[k][sl] for k in ("mag", "mag_tilt", "grav", "grav_thdr_1", "lineament") if k in L}
    if "elev" in L:
        w["hillshade"] = multi_hillshade(L["elev"][sl])
    if "mag_coh" in L:
        w["struct"] = L["mag_coh"][sl] * (L["mg_agree"][sl] if "mg_agree" in L else 1.0)
    for k in ("w2", "b0", "b1"):
        w[k] = M[k][sl].astype(np.float32)
    w["w2_frozen"] = M["w2_frozen"][sl]

    def rk(a):
        v = np.asarray(a, np.float32)
        lo, hi = stretch(v)
        return np.clip((v - lo) / (hi - lo), 0, 1) if np.isfinite(lo) else np.zeros_like(v)

    w["fused"] = np.dstack([rk(L["mag_thdr_1"][sl]) if "mag_thdr_1" in L else np.zeros(w["w2"].shape),
                            rk(L["grav_thdr_1"][sl]) if "grav_thdr_1" in L else np.zeros(w["w2"].shape),
                            rk(L["slope"][sl]) if "slope" in L else np.zeros(w["w2"].shape)])
    w["b0b1"] = np.dstack([np.nan_to_num(w["b0"]), np.nan_to_num(w["b1"]), np.zeros(w["b0"].shape)])
    return w


def render(R, row, L: dict, M: dict, geom, mode: str, labels: np.ndarray):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    half = int(R.rc["atlas"]["patch_size_px"]) // 2
    H, W = labels.shape
    r, c = int(row.row), int(row.col)
    sl = (slice(max(r - half, 0), min(r + half, H)), slice(max(c - half, 0), min(c + half, W)))
    w = window_layers(L, M, sl)
    T = R.ctx.data.template_profile["transform"]
    xa, ya = T * (sl[1].start, sl[0].start)
    xb, yb = T * (sl[1].stop, sl[0].stop)
    ext = [xa / 1e3, xb / 1e3, yb / 1e3, ya / 1e3]
    layers = list(EVIDENCE_LAYERS) + (list(DIAG_LAYERS) if mode == "DIAGNOSTIC" else [])
    n = len(layers)
    nc = 5
    fig, axs = plt.subplots(int(np.ceil(n / nc)), nc, figsize=(4.2 * nc, 4.2 * int(np.ceil(n / nc))), squeeze=False)
    norm = {}
    kn = labels[sl]
    for ax, (k, title, cmap) in zip(axs.ravel(), layers):
        a = w.get(k)
        if a is None:
            ax.text(0.5, 0.5, f"{title}\nUNAVAILABLE", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        if a.ndim == 3:
            ax.imshow(a, extent=ext, interpolation="nearest")
            norm[k] = "per-channel 2-98 % stretch" if k == "fused" else "ranks [0, 1]"
        else:
            lo, hi = (-np.pi / 2, np.pi / 2) if k == "mag_tilt" else (0.0, 1.0) if k in ("w2", "lineament") else \
                stretch(a, tuple(R.rc["atlas"]["stretch_percentiles"]))
            ax.imshow(a, extent=ext, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
            norm[k] = {"vmin": lo, "vmax": hi}
        if mode == "DIAGNOSTIC":
            kr, kc = np.nonzero(kn)                              # mapped-fault pixels at their UTM centres
            if len(kr):
                kx, ky = T * (kc + sl[1].start + 0.5, kr + sl[0].start + 0.5)
                ax.plot(np.asarray(kx) / 1e3, np.asarray(ky) / 1e3, ",", color="k")
            gr, gc_ = geom
            x, y = T * (gc_ + 0.5, gr + 0.5)
            ax.plot(np.asarray(x) / 1e3, np.asarray(y) / 1e3, ".", ms=1.5, color="cyan")
        x0, y0 = T * (c + 0.5, r + 0.5)
        ax.plot(x0 / 1e3, y0 / 1e3, "+", color="lime", ms=10, mew=1.5)
        ax.plot([ext[0] + 1, ext[0] + 6], [ext[2] + 1, ext[2] + 1], "k-", lw=3)
        ax.text(ext[0] + 3.5, ext[2] + 1.6, "5 km", ha="center", fontsize=7)
        ax.set_title(title, fontsize=8)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=6)
    for ax in axs.ravel()[n:]:
        ax.set_axis_off()
    fig.suptitle(f"{row.candidate_id} | {row.category} | region {row.region_id} | {mode} | UTM 11N "
                 f"({row.x_utm:.0f}, {row.y_utm:.0f}) | km grid", fontsize=10)
    return fig, w, norm


def write_panels(R, C: pd.DataFrame, L: dict, M: dict, geom: dict) -> pd.DataFrame:
    from ..artifacts.atomic import atomic_save

    labels = np.asarray(R.ctx.data.labels, bool)
    rows = []
    for row in C.itertuples():
        out = {"candidate_id": row.candidate_id}
        for mode in REVIEW_MODES:
            fig, w, norm = render(R, row, L, M, geom[row.candidate_id], mode, labels)
            rel = f"atlas/panels/{row.candidate_id}_{mode.lower()}.png"
            p = R.ws.path(rel)
            atomic_save(p, lambda t, f=fig: f.savefig(t, dpi=110, bbox_inches="tight", format="png"))
            import matplotlib.pyplot as plt

            plt.close(fig)
            out[f"panel_{mode.lower()}"] = rel
        R.save_npz(f"atlas/panels_data/{row.candidate_id}.npz",
                   **{k: np.asarray(v, np.float16) for k, v in w.items() if np.asarray(v).dtype != bool},
                   w2_frozen=np.asarray(w["w2_frozen"], np.uint8))
        R.ws.save_json({"candidate_id": row.candidate_id, "normalization": norm, "units": "native band units for "
                        "raw fields; ranks in [0, 1] for model maps and the lineament score"},
                       f"atlas/panels_data/{row.candidate_id}.json")
        rows.append(out)
    return pd.DataFrame(rows)


def overview_html(C: pd.DataFrame, P: pd.DataFrame, reviews: pd.DataFrame) -> str:
    lr = latest_reviews(reviews).set_index("candidate_id") if len(reviews) else pd.DataFrame()
    head = ("<html><head><meta charset='utf-8'><title>GEMS fault-discovery atlas</title><style>body{font-family:"
            "sans-serif}td{vertical-align:top;padding:4px;border-bottom:1px solid #ccc}img{width:420px}</style></head>"
            "<body><h1>GEMS fault-discovery evidence atlas</h1><p>EVIDENCE_ONLY panels hide mapped faults and model "
            "outputs (review these first); DIAGNOSTIC panels reveal them. Interpretations are review categories, not "
            "truth labels. Unreviewed candidates are not negatives.</p><table>")
    rows = []
    P = P.set_index("candidate_id")
    for r in C.itertuples():
        st = lr.loc[r.candidate_id].interpretation if len(lr) and r.candidate_id in lr.index else "NOT_REVIEWED"
        e, d = (P.loc[r.candidate_id, f"panel_{m}"].replace("atlas/", "") for m in ("evidence_only", "diagnostic"))
        rows.append(f"<tr><td><b>{r.candidate_id}</b><br>{r.category}<br>region {r.region_id} ({r.review_split})<br>"
                    f"UTM {r.x_utm:.0f}, {r.y_utm:.0f}<br>{r.distance_to_known_fault_km:.2f} km to mapped fault<br>"
                    f"review: <b>{st}</b></td><td><a href='{e}'><img src='{e}'></a></td>"
                    f"<td><details><summary>diagnostic overlay</summary><a href='{d}'><img src='{d}'></a></details>"
                    "</td></tr>")
    return head + "".join(rows) + "</table></body></html>"


def review_widget(R):
    """ipywidgets form (Colab): pick a candidate, view its EVIDENCE_ONLY panel, record a review (append-only)."""
    try:
        import ipywidgets as wd
        from IPython.display import Image, clear_output, display
    except ImportError:
        print("ipywidgets unavailable: fill atlas/annotation_template.csv and run R.import_reviews(path)")
        return None
    C = R.ws.load_table("atlas/atlas_index.csv")
    pick = wd.Dropdown(options=list(C.candidate_id), description="candidate")
    mode = wd.ToggleButtons(options=list(REVIEW_MODES), description="mode")
    interp = wd.Dropdown(options=list(INTERPRETATIONS), value="NOT_REVIEWED", description="interpretation")
    conf = wd.Dropdown(options=["LOW", "MEDIUM", "HIGH"], value="LOW", description="confidence")
    fields = {k: wd.Text(description=k[:14]) for k in ("supporting_evidence", "alternative_explanation",
                                                       "reviewer_notes", "reviewer", "evidence_sources",
                                                       "data_license")}
    out, btn = wd.Output(), wd.Button(description="save review", button_style="success")

    def show(*_):
        with out:
            clear_output()
            p = R.ws.path(f"atlas/panels/{pick.value}_{mode.value.lower()}.png")
            if p.exists():
                display(Image(str(p)))

    def save(*_):
        res = R.record_review({"candidate_id": pick.value, "interpretation": interp.value, "confidence": conf.value,
                               "review_mode": mode.value, **{k: f.value for k, f in fields.items()}})
        with out:
            print(res)

    pick.observe(show, "value")
    mode.observe(show, "value")
    btn.on_click(save)
    show()
    return wd.VBox([wd.HBox([pick, mode]), wd.HBox([interp, conf]), *fields.values(), btn, out])
