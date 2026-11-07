"""Section 11 - label-free Atlas evidence at correct detections, false positives and missed faults.

Evidence layers come only from the Geological Data Atlas registry (no fault labels). Continuous layers are
converted to percentile ranks over the eligible footprint; Atlas layers that already are ranks
(``*_boundary_strength``, ``xm_*``) are used as delivered. A missing layer is reported as missing, never as
weak evidence; DEM evidence is used only where DEM coverage exists.

Regions per discovery fold (full-area prediction):
* FP regions - 8-connected pixels with p >= the fold's ``fp_high_quantile`` of scored pixels, > 300 m from
  held truth (k = 0, the whole p counts as FP), at least ``min_component_px``;
* missed systems - held components with weighted recall < ``recall_missed``;
* correct systems - held components with weighted recall >= ``recall_correct``.
Each region gets descriptive flags ("consistent with ..."); none of them confirms or rejects a fault.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

EVIDENCE = {   # name -> (atlas feature ids in order of preference, already_rank, meaning)
    "magnetic_edge": (["mag_boundary_strength", "mag_thd"], True, "magnetic boundary strength (rank)"),
    "gravity_edge": (["grav_boundary_strength", "grav_thd"], True, "gravity boundary strength (rank)"),
    "terrain_slope_break": (["terrain_slope_break_1000m"], False, "terrain slope break 1 km (rank)"),
    "mag_persistence": (["mag_edge_persistence"], True, "magnetic edge persistence across continuation (fraction)"),
    "grav_persistence": (["grav_edge_persistence"], True, "gravity edge persistence (fraction)"),
    "mag_thd_uc500": (["mag_thd_uc500"], False, "magnetic THD after 500 m continuation (rank)"),
    "mag_thd_uc1000": (["mag_thd_uc1000"], False, "magnetic THD after 1 km continuation (rank)"),
    "mag_thd_uc2000": (["mag_thd_uc2000"], False, "magnetic THD after 2 km continuation (rank)"),
    "mag_thd_uc5000": (["mag_thd_uc5000"], False, "magnetic THD after 5 km continuation (rank)"),
    "grav_thd_uc5000": (["grav_thd_uc5000"], False, "gravity THD after 5 km continuation (rank)"),
    "grav_long_wavelength": (["grav_uc5000", "grav_residual1000"], False, "gravity long-wavelength field (rank)"),
    "xm_evidence_mean": (["xm_evidence_mean"], True, "combined cross-modal edge evidence (rank)"),
    "xm_multiscale_persistence": (["xm_multiscale_persistence"], True, "multiscale boundary persistence"),
    "xm_continuity": (["xm_continuity"], False, "along-strike continuity (rank)"),
    "xm_orient_mag_grav": (["xm_orient_magn_grav"], True, "magnetic-gravity strike agreement (cos 2dtheta)"),
    "xm_evidence_count": (["xm_evidence_count"], True, "independent strong-edge count (0-3)"),
    "xm_available": (["xm_available_modalities"], True, "modalities with data (0-3)"),
}


class AtlasEvidence:
    def __init__(self, atlas_root, valid: np.ndarray):
        from gems_bench.geological_atlas.atlas import GeologicalAtlas

        self.valid = valid
        self.atlas = None
        self.missing, self.source = {}, {}
        self.cache: dict = {}
        try:
            self.atlas = GeologicalAtlas.open(atlas_root)
        except Exception as e:                                         # noqa: BLE001
            self.error = f"{type(e).__name__}: {e}"
        else:
            self.error = None
        self.dem = None
        if self.atlas is not None:
            cov = self.atlas.load_coverage()
            for k in ("dem_coverage_DETAILED_2M", "dem_coverage_FAST_10M"):
                if k in cov:
                    self.dem = np.nan_to_num(cov[k], nan=0.0)
                    self.source["dem_coverage"] = k
                    break

    @property
    def available(self) -> bool:
        return self.atlas is not None and len(self.atlas.registry) > 0

    def get(self, name: str):
        if name in self.cache:
            return self.cache[name]
        from gems_bench.geological_atlas.feature_registry import pct_rank

        ids, is_rank, _ = EVIDENCE[name]
        arr = None
        if self.available:
            for fid in ids:
                try:
                    a = self.atlas.load_feature(fid).astype(np.float32)
                except (KeyError, FileNotFoundError) as e:
                    self.missing[f"{name}:{fid}"] = str(e)
                    continue
                if a.shape != self.valid.shape:
                    self.missing[f"{name}:{fid}"] = f"shape {a.shape} != grid {self.valid.shape}"
                    continue
                a[~self.valid] = np.nan
                arr = a if is_rank or fid.startswith("xm_evidence_count") else pct_rank(a, self.valid)
                self.source[name] = fid
                break
        self.cache[name] = arr
        return arr

    def table(self) -> pd.DataFrame:
        rows = []
        for n, (ids, is_rank, meaning) in EVIDENCE.items():
            a = self.get(n)
            rows.append({"evidence": n, "atlas_feature": self.source.get(n), "meaning": meaning,
                         "available": a is not None,
                         "valid_fraction": float(np.isfinite(a[self.valid]).mean()) if a is not None else 0.0})
        rows.append({"evidence": "dem_coverage", "atlas_feature": self.source.get("dem_coverage"),
                     "meaning": "DEM support fraction", "available": self.dem is not None,
                     "valid_fraction": float((self.dem[self.valid] > 0).mean()) if self.dem is not None else 0.0})
        return pd.DataFrame(rows)


def _mean(a, m):
    if a is None:
        return np.nan
    v = a[m]
    v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else np.nan


def flags(ev: dict, kind: str, d_visible_m: float, d_truth_m: float, d_edge_m: float, dem_cov: float,
          near_p_ok: bool = False) -> list[str]:
    f = []
    mag, grav, topo = ev.get("magnetic_edge"), ev.get("gravity_edge"), ev.get("terrain_slope_break")

    def hi(x):
        return x is not None and np.isfinite(x) and x >= 0.8

    def lo(x):
        return x is not None and np.isfinite(x) and x < 0.5

    if hi(mag) and lo(grav):
        f.append("strong magnetic / weak gravity edge")
    if hi(grav) and lo(mag):
        f.append("strong gravity / weak magnetic edge")
    if hi(topo) and lo(mag) and lo(grav):
        f.append("terrain structure without potential-field support")
    pers, cont = ev.get("xm_multiscale_persistence"), ev.get("xm_continuity")
    if pers is not None and np.isfinite(pers) and pers >= 0.5 and cont is not None and np.isfinite(cont) and cont >= 0.75:
        f.append("multiscale continuity")
    o = ev.get("xm_orient_mag_grav")
    if o is not None and np.isfinite(o) and o >= 0.5:
        f.append("magnetic-gravity orientation agreement")
    if kind == "false_positive" and d_visible_m <= 1000:
        f.append("near a training-visible mapped trace (possible extension / branch)")
    if kind == "false_positive" and 300 < d_truth_m <= 1000:
        f.append("close to held truth but outside the 300 m kernel (poor alignment)")
    if kind == "missed" and near_p_ok:
        f.append("confidence present within 1 km but not on the trace (poor alignment)")
    if d_edge_m <= 2000:
        f.append("within 2 km of the survey / footprint edge (possible source artifact)")
    if dem_cov is None or not np.isfinite(dem_cov) or dem_cov < 0.5:
        f.append("little or no DEM coverage")
    av = ev.get("xm_available")
    if av is not None and np.isfinite(av) and av < 2:
        f.append("fewer than two modalities available")
    if not f:
        f.append("no distinctive label-free evidence pattern")
    return f


def regions_for_unit(unit, U, prob, maps, evidence: AtlasEvidence, valid, cfg, transform) -> list[dict]:
    """Region rows (FP / missed / correct) for one discovery fold."""
    D = cfg["diagnostics"]
    G = maps["truth"]
    excl = U.exclusion(cfg["scoring"]["primary_exclusion"])
    scored = valid & ~excl
    p = np.where(scored, np.nan_to_num(prob, nan=0.0), 0.0)
    d_truth = ndimage.distance_transform_edt(~G) * 100.0 if G.any() else np.full(G.shape, np.inf)
    d_vis = ndimage.distance_transform_edt(~U.training_visible) * 100.0
    d_edge = ndimage.distance_transform_edt(valid) * 100.0
    names = list(EVIDENCE)
    arrs = {n: evidence.get(n) for n in names}
    a, c, e, f = transform.a, transform.c, transform.e, transform.f
    rows = []

    def describe(kind, rid, sl, m, extra):
        """``m`` is the region mask inside the bounding-box slice ``sl``."""
        ev = {n: _mean(arrs[n][sl] if arrs[n] is not None else None, m) for n in names}
        dem_cov = _mean(evidence.dem[sl], m) if evidence.dem is not None else np.nan
        rr, cc = np.nonzero(m)
        rr, cc = rr + sl[0].start, cc + sl[1].start
        dv = float(d_vis[sl][m].min())
        dt = float(d_truth[sl][m].min()) if kind == "false_positive" else 0.0
        de = float(d_edge[sl][m].min())
        fl = flags(ev, kind, dv, dt, de, dem_cov, extra.pop("near_p_ok", False))
        rows.append({"unit": unit, "region_id": rid, "kind": kind, "n_px": int(m.sum()),
                     "mean_p": float(p[sl][m].mean()), "max_p": float(p[sl][m].max()),
                     "centroid_e": float(c + a * (cc.mean() + 0.5)), "centroid_n": float(f + e * (rr.mean() + 0.5)),
                     "row": int(rr.mean()), "col": int(cc.mean()), "dist_training_visible_m": dv,
                     "dist_held_truth_m": dt, "dist_footprint_edge_m": de, "dem_coverage": dem_cov,
                     **{f"ev_{n}": ev[n] for n in names}, **extra, "flags": "; ".join(fl)})

    thr = float(np.quantile(p[scored], D["fp_high_quantile"]))
    seed = scored & (p >= thr) & (d_truth > 300)
    lab, n = ndimage.label(seed, structure=np.ones((3, 3), bool))
    if n:
        fpm = ndimage.sum(maps["fp_mass"], lab, index=np.arange(1, n + 1))
        size = ndimage.sum(np.ones_like(p), lab, index=np.arange(1, n + 1))
        keep = [i for i in np.argsort(-fpm) if size[i] >= D["min_component_px"]][: 4 * D["n_regions_report"]]
        objs = ndimage.find_objects(lab)
        for i in keep:
            sl = objs[i]
            describe("false_positive", f"{unit}_FP{i + 1}", sl, lab[sl] == i + 1,
                     {"fp_mass": float(fpm[i]), "recall_w": np.nan})
    hl, hn = ndimage.label(G, structure=np.ones((3, 3), bool))
    if hn:
        tp = ndimage.sum(maps["tp_support"], hl, index=np.arange(1, hn + 1))
        npx = ndimage.sum(G.astype(float), hl, index=np.arange(1, hn + 1))
        rec = tp / np.maximum(npx, 1)
        near_max = ndimage.maximum_filter(p, size=21)                    # max p within ~1 km
        objs = ndimage.find_objects(hl)
        for i in range(hn):
            if npx[i] < D["min_component_px"]:
                continue
            kind = "missed" if rec[i] < D["recall_missed"] else ("correct" if rec[i] >= D["recall_correct"] else None)
            if kind is None:
                continue
            sl = objs[i]
            m = hl[sl] == i + 1
            near_ok = bool(kind == "missed" and near_max[sl][m].max() >= thr)
            describe(kind, f"{unit}_{'MS' if kind == 'missed' else 'OK'}{i + 1}", sl, m,
                     {"fp_mass": np.nan, "recall_w": float(rec[i]), "near_p_ok": near_ok})
    return rows


def class_profiles(unit, prob, maps, evidence: AtlasEvidence, valid, excl, sample, cfg) -> list[dict]:
    """Evidence distributions at block-sampled pixels of four classes (TP-supported truth, missed truth,
    high FP pixels, background)."""
    D = cfg["diagnostics"]
    G = maps["truth"]
    scored = valid & ~excl
    p = np.where(scored, np.nan_to_num(prob, nan=0.0), 0.0)
    thr = float(np.quantile(p[scored], D["fp_high_quantile"]))
    cls = np.full(p.shape, -1, np.int8)
    cls[scored] = 3                                                              # background
    cls[scored & (p >= thr) & (maps["fp_mass"] >= 0.5 * p)] = 2                  # high-FP pixel
    cls[G & (maps["tp_support"] < D["recall_missed"])] = 1                       # missed truth
    cls[G & (maps["tp_support"] >= D["recall_correct"])] = 0                     # supported truth
    names = {0: "supported truth", 1: "missed truth", 2: "high-confidence FP", 3: "background"}
    rows = []
    rr, cc = sample
    for k, nm in names.items():
        if k in (0, 1, 2):                      # rare classes: all pixels, thinned to <= 20000
            idx = np.flatnonzero(cls.ravel() == k)
            if idx.size > 20000:
                idx = idx[:: idx.size // 20000]
            r_, c_ = np.unravel_index(idx, p.shape)
        else:
            sel = cls[rr, cc] == 3
            r_, c_ = rr[sel], cc[sel]
        for n in EVIDENCE:
            a = evidence.get(n)
            if a is None or not len(r_):
                continue
            v = a[r_, c_]
            v = v[np.isfinite(v)]
            if not v.size:
                continue
            q = np.quantile(v, [0.1, 0.25, 0.5, 0.75, 0.9])
            rows.append({"unit": unit, "class": nm, "evidence": n, "n": int(v.size), "mean": float(v.mean()),
                         "q10": q[0], "q25": q[1], "median": q[2], "q75": q[3], "q90": q[4]})
    return rows


def response_curve(prob, evidence_arr, mask, sample, nbins: int = 10, n_boot: int = 200, seed: int = 0) -> pd.DataFrame:
    """Mean model confidence per evidence decile on block-sampled pixels, with a block-bootstrap 90% CI."""
    rr, cc = sample
    x, y = evidence_arr[rr, cc], prob[rr, cc]
    ok = np.isfinite(x) & np.isfinite(y) & mask[rr, cc]
    x, y = x[ok], y[ok]
    if x.size < 50:
        return pd.DataFrame()
    edges = np.unique(np.quantile(x, np.linspace(0, 1, nbins + 1)))
    idx = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(len(edges) - 1):
        yy = y[idx == b]
        if not yy.size:
            continue
        boots = [rng.choice(yy, yy.size).mean() for _ in range(n_boot)]
        rows.append({"bin": b, "x_lo": edges[b], "x_hi": edges[b + 1], "x_mid": float(np.median(x[idx == b])),
                     "mean_p": float(yy.mean()), "ci_lo": float(np.quantile(boots, 0.05)),
                     "ci_hi": float(np.quantile(boots, 0.95)), "n": int(yy.size)})
    return pd.DataFrame(rows)
