"""Mapped-fault context: network statistics, strike, density, feature response vs distance, fault-normal profiles.

All of it is DESCRIPTIVE. Mapped faults are an incomplete reference; nothing here selects features or tunes the
lineament rules. Pixels are not independent: every distance-band statistic uses one random pixel per
``proximity.block_px`` x ``block_px`` block (1 km at 100 m) and reports the number of blocks, not of pixels.

Trace length: the label raster is skeletonised; length = sum over 8-connected skeleton links of their length
(100 m orthogonal, 141.4 m diagonal), each link counted once.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

from .circular_statistics import axial_mean, circular_std_deg, rayleigh_axial
from .magnetic import block_sample


def skeleton(labels: np.ndarray) -> np.ndarray:
    from skimage.morphology import skeletonize

    return skeletonize(labels)


def link_length_m(sk: np.ndarray, px: float) -> np.ndarray:
    """Per skeleton pixel: half the summed length of its links (so the sum over a trace is its length)."""
    s = sk.astype(np.float32)
    orth = sum(np.roll(s, sh, ax) for ax in (0, 1) for sh in (1, -1))
    diag = sum(np.roll(np.roll(s, a, 0), b, 1) for a in (1, -1) for b in (1, -1))
    return np.where(sk, 0.5 * (orth * px + diag * px * np.sqrt(2)), 0.0).astype(np.float32)


def network(labels: np.ndarray, grid, regions=(3, 3)) -> dict:
    px = grid.pixel_m
    sk = skeleton(labels)
    ll = link_length_m(sk, px)
    lab, n = ndimage.label(sk, structure=np.ones((3, 3), bool))
    idx = np.arange(1, n + 1)
    seg_len = ndimage.sum(ll, lab, idx) / 1e3 if n else np.array([])
    rows = []
    a, _, c, _, e, f = grid.transform[:6]
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None:
            continue
        rr, cc = np.nonzero(lab[sl] == i)
        E_ = c + a * (cc + sl[1].start + 0.5)
        N_ = f + e * (rr + sl[0].start + 0.5)
        if len(rr) >= 3:
            X = np.stack([E_ - E_.mean(), N_ - N_.mean()], 1)
            w, v = np.linalg.eigh(np.cov(X.T))
            strike = float(np.degrees(np.arctan2(v[0, 1], v[1, 1])) % 180)
            lin = float(w[1] / (w.sum() + 1e-12))
        else:
            strike, lin = np.nan, np.nan
        rows.append({"trace_id": i, "length_km": float(seg_len[i - 1]), "n_px": int(len(rr)), "strike_deg": strike,
                     "linearity": lin, "centroid_e": float(E_.mean()), "centroid_n": float(N_.mean()),
                     "row": float(rr.mean() + sl[0].start), "col": float(cc.mean() + sl[1].start)})
    traces = pd.DataFrame(rows)
    H, W = labels.shape
    reg = []
    for i in range(regions[0]):
        for j in range(regions[1]):
            r0, r1 = i * H // regions[0], (i + 1) * H // regions[0]
            c0, c1 = j * W // regions[1], (j + 1) * W // regions[1]
            km2 = (r1 - r0) * (c1 - c0) * px * px / 1e6
            L = float(ll[r0:r1, c0:c1].sum() / 1e3)
            reg.append({"region": f"r{i}c{j}", "row_from": r0, "row_to": r1, "col_from": c0, "col_to": c1,
                        "trace_km": L, "km_per_1000km2": L / km2 * 1000})
    tot = float(ll.sum() / 1e3)
    mean_strike, R = axial_mean(traces.strike_deg, traces.length_km) if len(traces) else (np.nan, np.nan)
    summary = {"label_pixels": int(labels.sum()), "skeleton_pixels": int(sk.sum()), "total_trace_km": tot,
               "n_traces": int(n), "median_trace_km": float(np.median(seg_len)) if n else np.nan,
               "p90_trace_km": float(np.percentile(seg_len, 90)) if n else np.nan,
               "traces_per_100km": float(n / max(tot, 1e-9) * 100),
               "length_weighted_mean_strike_deg": mean_strike, "strike_resultant_R": R,
               "strike_circular_std_deg": circular_std_deg(R),
               "rayleigh_trace_p": rayleigh_axial(traces.strike_deg)["p"] if len(traces) else np.nan,
               "length_method": "skeleton 8-connected link length (100 m / 141.4 m)"}
    return {"skeleton": sk, "link_length": ll, "traces": traces, "regions": pd.DataFrame(reg), "summary": summary}


def distance_m(labels: np.ndarray, px: float) -> np.ndarray:
    return (ndimage.distance_transform_edt(~labels) * px).astype(np.float32)


def band_table(feature_ids: list[str], loader, dist: np.ndarray, footprint: np.ndarray, cfg, rng) -> pd.DataFrame:
    """``loader(feature_id) -> array``; one feature in memory at a time."""
    pc = cfg["proximity"]
    edges = list(pc["bands_m"]) + [np.inf]
    rows = []
    samples = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = footprint & (dist >= lo) & (dist < hi)
        samples.append((lo, hi, *block_sample(m, int(pc["block_px"]), rng, int(pc["max_blocks_per_band"]))))
    for fid in feature_ids:
        a = loader(fid)
        for lo, hi, rr, cc in samples:
            v = a[rr, cc]
            v = v[np.isfinite(v)]
            if not v.size:
                continue
            q = np.percentile(v, [10, 25, 50, 75, 90])
            rows.append({"feature_id": fid, "band": f"{lo:g}-{hi:g} m", "band_lo_m": lo, "n_blocks": int(v.size),
                         "p10": q[0], "p25": q[1], "median": q[2], "p75": q[3], "p90": q[4], "mean": float(v.mean())})
    df = pd.DataFrame(rows)
    if len(df):
        near = df[df.band_lo_m == 0].set_index("feature_id")
        far = df[df.band_lo_m == edges[-2]].set_index("feature_id")
        iqr = (far.p75 - far.p25).replace(0, np.nan)
        contrast = ((near["median"] - far["median"]) / iqr).rename("near_far_median_contrast_iqr")
        df = df.merge(contrast, left_on="feature_id", right_index=True, how="left")
    return df


def response_curve(a: np.ndarray, dist: np.ndarray, footprint: np.ndarray, rng, step_m=250.0, max_m=5000.0,
                   block=10) -> pd.DataFrame:
    rr, cc = block_sample(footprint & np.isfinite(a) & (dist <= max_m + step_m), block, rng, 200000)
    d, v = dist[rr, cc], a[rr, cc]
    b = np.floor(d / step_m) * step_m
    df = pd.DataFrame({"dist_m": b, "v": v}).groupby("dist_m").v.agg(
        median="median", q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75), n="size").reset_index()
    return df[df.dist_m <= max_m]


def pick_profile_traces(traces: pd.DataFrame, cfg, regions=(3, 3), shape=None) -> pd.DataFrame:
    """Long, linear training-visible traces, at most one per region (deterministic). Descriptive examples only."""
    pc = cfg["proximity"]
    t = traces[(traces.length_km >= float(pc["profile_min_trace_km"])) & (traces.linearity >= 0.9)].copy()
    if shape is not None and len(t):
        H, W = shape
        t["region"] = (t.row * regions[0] // H).astype(int) * regions[1] + (t.col * regions[1] // W).astype(int)
        t = t.sort_values(["region", "length_km"], ascending=[True, False]).drop_duplicates("region")
    return t.sort_values("length_km", ascending=False).head(int(pc["profile_examples"]))


def profiles(traces: pd.DataFrame, labels_sk_lab: tuple, features: dict, grid, cfg, n_along: int = 9) -> pd.DataFrame:
    """Fault-normal profiles: features sampled every 100 m along the normal (+- half length) at ``n_along``
    stations along each selected trace, averaged per offset."""
    lab, _ = labels_sk_lab
    px = grid.pixel_m
    L = float(cfg["proximity"]["profile_half_length_m"])
    offs = np.arange(-L, L + px, px)
    rows = []
    for t in traces.itertuples():
        rr, cc = np.nonzero(lab == t.trace_id)
        if len(rr) < 5:
            continue
        az = np.radians(t.strike_deg)
        dr_s, dc_s = -np.cos(az), np.sin(az)                 # along strike (row, col)
        dr_n, dc_n = dc_s, -dr_s                              # normal
        proj = (rr - rr.mean()) * dr_s + (cc - cc.mean()) * dc_s
        q = np.quantile(proj, np.linspace(0.1, 0.9, n_along))
        st_r = rr.mean() + q * dr_s
        st_c = cc.mean() + q * dc_s
        R = st_r[:, None] + (offs / px)[None, :] * dr_n
        C = st_c[:, None] + (offs / px)[None, :] * dc_n
        for fid, a in features.items():
            v = ndimage.map_coordinates(np.nan_to_num(a, nan=np.nan), [R.ravel(), C.ravel()], order=1, cval=np.nan,
                                        mode="constant").reshape(R.shape)
            with np.errstate(all="ignore"):
                prof = np.nanmedian(v, 0)
            for o, val in zip(offs, prof):
                rows.append({"trace_id": t.trace_id, "trace_strike_deg": t.strike_deg, "trace_km": t.length_km,
                             "feature_id": fid, "offset_m": float(o), "value": float(val)})
    return pd.DataFrame(rows)
