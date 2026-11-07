"""Section 04 - where the confidence mass sits relative to mapped faults (DIAGNOSTIC ONLY).

The distance to mapped faults is never a model input, router input or postprocessing feature in this
milestone; nothing is suppressed near mapped faults. Population: eligible pixels outside mapped faults.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage


def skeleton_endpoints(labels: np.ndarray):
    from skimage.morphology import skeletonize

    sk = skeletonize(labels)
    nb = ndimage.convolve(sk.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant") - sk
    return sk, sk & (nb == 1)


def band_label(lo, hi) -> str:
    return f"{lo / 1000:g}-{hi / 1000:g} km" if hi is not None else f">{lo / 1000:g} km"


def compute(prob, valid, labels, transform, cfg, px: float = 100.0) -> dict:
    D = cfg["diagnostics"]
    elig = valid & ~labels & np.isfinite(prob)
    p = np.where(elig, np.clip(prob, 0, 1), 0.0)
    dist = (ndimage.distance_transform_edt(~labels) * px).astype(np.float32)
    total, area = float(p.sum()), float(elig.sum())
    thr = float(np.quantile(p[elig], D["high_conf_quantile"]))
    high = elig & (p >= thr)
    rows = []
    b = D["distance_bands_m"]
    for lo, hi in zip(b[:-1], b[1:]):
        m = elig & (dist > lo if lo > 0 else dist >= 0) & ((dist <= hi) if hi is not None else True)
        a_s, m_s = m.sum() / area, p[m].sum() / total
        rows.append({"band": band_label(lo, hi), "d_min_m": lo, "d_max_m": hi if hi is not None else np.inf,
                     "n_px": int(m.sum()), "area_share": float(a_s), "mass_share": float(m_s),
                     "mean_confidence": float(p[m].mean()) if m.any() else np.nan,
                     "enrichment": float(m_s / a_s) if a_s > 0 else np.nan,
                     "high_conf_share": float((high & m).sum() / max(high.sum(), 1))})
    bands = pd.DataFrame(rows)
    # cumulative mass / area vs distance (100 m steps)
    dmax = float(D["cumulative_max_m"])
    edges = np.arange(0, dmax + px, px)
    dv, pv = dist[elig], p[elig]
    order = np.argsort(dv)
    dvs, cm = dv[order], np.cumsum(pv[order])
    k = np.searchsorted(dvs, edges, side="right")
    cum = pd.DataFrame({"distance_m": edges,
                        "cumulative_mass_share": np.where(k > 0, cm[np.maximum(k - 1, 0)], 0) / total,
                        "cumulative_area_share": k / area})
    # mapped-trace endpoints vs mid-trace at the same distance range
    sk, ends = skeleton_endpoints(labels)
    d_end = ndimage.distance_transform_edt(~ends) * px
    r = float(D["endpoint_radius_m"])
    ring = elig & (dist > 300) & (dist <= r)
    near_end, mid = ring & (d_end <= r), ring & (d_end > 2 * r)
    endpoint = pd.DataFrame([
        {"context": f"300 m - {r:g} m from a mapped trace, within {r:g} m of a trace endpoint",
         "n_px": int(near_end.sum()), "mean_confidence": float(p[near_end].mean()) if near_end.any() else np.nan},
        {"context": f"300 m - {r:g} m from a mapped trace, > {2 * r:g} m from any endpoint",
         "n_px": int(mid.sum()), "mean_confidence": float(p[mid].mean()) if mid.any() else np.nan}])
    # high-confidence structures far from every mapped trace
    lab, n = ndimage.label(high, structure=np.ones((3, 3), bool))
    comps = []
    if n:
        sizes = np.bincount(lab.ravel())
        mind = ndimage.minimum(dist, lab, index=np.arange(1, n + 1))
        objs = ndimage.find_objects(lab)
        a, c, e, f = transform.a, transform.c, transform.e, transform.f
        for i in range(1, n + 1):
            if sizes[i] < D["min_component_px"] or mind[i - 1] < D["far_from_mapped_m"]:
                continue
            sl = objs[i - 1]
            rr, cc = np.nonzero(lab[sl] == i)
            rr, cc = rr + sl[0].start, cc + sl[1].start
            E, N = c + a * (cc + 0.5), f + e * (rr + 0.5)
            X = np.stack([E - E.mean(), N - N.mean()], 1)
            w, v = np.linalg.eigh(np.cov(X.T)) if len(rr) > 2 else (np.array([0, 1.0]), np.eye(2))
            d = v[:, 1]
            proj = X @ d
            comps.append({"component_id": int(i), "n_px": int(sizes[i]), "length_km": float((np.ptp(proj) + px) / 1e3),
                          "linearity": float(w[1] / (w.sum() + 1e-12)),
                          "strike_deg": float(np.degrees(np.arctan2(d[0], d[1])) % 180),
                          "mean_p": float(p[rr, cc].mean()), "max_p": float(p[rr, cc].max()),
                          "min_distance_to_mapped_m": float(mind[i - 1]),
                          "centroid_e": float(E.mean()), "centroid_n": float(N.mean())})
    far = pd.DataFrame(comps, columns=["component_id", "n_px", "length_km", "linearity", "strike_deg", "mean_p",
                                       "max_p", "min_distance_to_mapped_m", "centroid_e", "centroid_n"])
    far = far.sort_values("n_px", ascending=False).reset_index(drop=True)
    # geographic variation: share of mass within 1 km per 10 km block vs area share within 1 km
    B = int(D["region_block_px"])
    h, w_ = prob.shape
    nbr, nbc = (h + B - 1) // B, (w_ + B - 1) // B
    ri, ci = np.indices(prob.shape, dtype=np.int32)
    bid = (ri // B) * nbc + (ci // B)
    del ri, ci
    near = elig & (dist <= 1000)
    m_all = np.bincount(bid.ravel(), weights=p.ravel(), minlength=nbr * nbc)
    m_near = np.bincount(bid.ravel(), weights=(p * near).ravel(), minlength=nbr * nbc)
    a_all = np.bincount(bid.ravel(), weights=elig.ravel().astype(float), minlength=nbr * nbc)
    a_near = np.bincount(bid.ravel(), weights=near.ravel().astype(float), minlength=nbr * nbc)
    with np.errstate(invalid="ignore", divide="ignore"):
        conc = (m_near / m_all) / (a_near / a_all)
    conc[(a_all < 0.25 * B * B) | (a_near == 0) | (m_all <= 0)] = np.nan
    blocks = pd.DataFrame({"block": np.arange(nbr * nbc), "row0": (np.arange(nbr * nbc) // nbc) * B,
                           "col0": (np.arange(nbr * nbc) % nbc) * B, "mass": m_all, "mass_within_1km": m_near,
                           "area_px": a_all, "area_within_1km_px": a_near, "concentration_ratio": conc})
    s1 = bands[bands.d_max_m <= 1000]
    stats = {"total_mass": total, "eligible_px": int(area), "high_conf_threshold": thr,
             "mass_within_1km": float(s1.mass_share.sum()), "area_within_1km": float(s1.area_share.sum()),
             "mass_beyond_3km": float(bands[bands.d_min_m >= 3000].mass_share.sum()),
             "enrichment_0_300": float(bands.enrichment.iloc[0]),
             "n_far_high_conf_components": int(len(far)),
             "endpoint_ratio": float(endpoint.mean_confidence.iloc[0] / endpoint.mean_confidence.iloc[1])
             if endpoint.mean_confidence.iloc[1] > 0 else np.nan,
             "median_block_concentration": float(np.nanmedian(conc)) if np.isfinite(conc).any() else np.nan}
    return {"table": bands, "cumulative": cum, "endpoints": endpoint, "far_components": far, "blocks": blocks,
            "stats": stats, "_dist": dist, "_high": high, "_conc_grid": conc.reshape(nbr, nbc)}
