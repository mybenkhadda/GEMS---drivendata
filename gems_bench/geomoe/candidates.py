"""Experiment B: one unified candidate inventory per scenario unit + the oracle coverage ceiling.

Construction (label-free): every available source layer (``experts.POOL_SOURCES``) gives low-threshold 1-px lines
(``pool_rep`` at ``pool_frac`` of the unit's eligible domain, ``gems3.representation.apply``); straight continuations of
the PERMITTED known traces give ``tips`` lines (``gems3.candidates.tip_paths``). Lines are split into branches, long
branches cut into pieces of at most ``max_segment_km``, and deduplicated across sources in priority order with a
spatial + orientation test (``p6.segment_recovery.dedup_inventory``: a crossing line is never a duplicate, a short line
is never dropped for being short). Every kept segment records which sources have a line on most of it
(``proposed_by``), the means of every layer, geometry and network descriptors.

Truth enters only two clearly named places, both evaluation-side: ``y_near_truth`` (share of the segment within 300 m
of the unit's scored truth; read as a training target only for TRAINING zones, ``gating.training_rows``) and the
coverage / oracle tables (diagnostics; ``role = oracle`` rows are refused by every selection).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

from .experts import POOL_SOURCES, SOURCE_FAMILY

S8 = np.ones((3, 3), bool)
STEP_KM = 0.112
LENGTH_BINS_KM = (1.0, 2.0, 5.0)


# =============================================================================== pure: segments
def split_long(lab: np.ndarray, max_px: int) -> np.ndarray:
    """Cut every label longer than ``max_px`` into consecutive pieces along its major axis (relabelled 1..n)."""
    from ..p6.segment_recovery import label_geometry

    lab = np.asarray(lab, np.int32)
    n = int(lab.max())
    if not n:
        return lab.copy()
    G = label_geometry(lab)
    rr, cc = np.nonzero(lab)
    lb = lab[rr, cc]
    th, mr, mc = G.theta.to_numpy(), G.row.to_numpy(), G.col.to_numpy()
    proj = (cc - mc[lb - 1]) * np.cos(th[lb - 1]) + (rr - mr[lb - 1]) * np.sin(th[lb - 1])
    lo = np.asarray(ndimage.minimum(proj, lb, np.arange(1, n + 1)))
    piece = np.floor((proj - lo[lb - 1]) / max(int(max_px), 1)).astype(np.int64)
    key = lb.astype(np.int64) * 100_000 + piece
    _, new = np.unique(key, return_inverse=True)
    out = np.zeros(lab.shape, np.int32)
    out[rr, cc] = new.astype(np.int32) + 1
    return out


def connectivity(lab: np.ndarray) -> np.ndarray:
    """Number of distinct other segments touching each segment (8-neighbourhood), index = label - 1."""
    lab = np.asarray(lab, np.int32)
    n = int(lab.max())
    deg = np.zeros(n, int)
    if not n:
        return deg
    h, w = lab.shape
    pairs = []
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):              # every 8-neighbour pair once (dr >= 0)
        a = lab[0:h - dr, max(0, -dc):w - max(0, dc)]
        b = lab[dr:h, max(0, dc):w + min(0, dc)]
        m = (a > 0) & (b > 0) & (a != b)
        if m.any():
            pairs.append(np.c_[a[m], b[m]])
    if pairs:
        P = np.unique(np.sort(np.vstack(pairs), 1), axis=0)
        deg += np.bincount(P[:, 0] - 1, minlength=n) + np.bincount(P[:, 1] - 1, minlength=n)
    return deg


def near_share(lab: np.ndarray, mask: np.ndarray, tol_px: float) -> np.ndarray:
    """Share of every segment's pixels within ``tol_px`` of ``mask`` (index = label - 1)."""
    n = int(lab.max())
    if not n:
        return np.zeros(0)
    m = np.asarray(mask, bool)
    if not m.any():
        return np.zeros(n)
    d = ndimage.distance_transform_edt(~m)
    rr, cc = np.nonzero(lab)
    lb = lab[rr, cc]
    cnt = np.bincount(lb, minlength=n + 1)[1:]
    return np.bincount(lb[d[rr, cc] <= tol_px], minlength=n + 1)[1:] / np.maximum(cnt, 1)


def layer_means(lab: np.ndarray, layers: dict) -> pd.DataFrame:
    """``mean_<layer>`` over the finite pixels of each segment and ``fin_<layer>`` (finite share)."""
    n = int(lab.max())
    rr, cc = np.nonzero(lab)
    lb = lab[rr, cc]
    cnt = np.bincount(lb, minlength=n + 1)[1:].astype(float)
    out = {"label": np.arange(1, n + 1)}
    for k, a in layers.items():
        v = np.asarray(a, np.float64)[rr, cc]
        ok = np.isfinite(v)
        s = np.bincount(lb[ok], weights=v[ok], minlength=n + 1)[1:]
        m = np.bincount(lb[ok], minlength=n + 1)[1:]
        out[f"mean_{k}"] = np.where(m > 0, s / np.maximum(m, 1), np.nan)
        out[f"fin_{k}"] = m / np.maximum(cnt, 1)
    return pd.DataFrame(out)


def segment_table(lab: np.ndarray, layers: dict, pools: dict, tol_px: float, eligible: np.ndarray,
                  truth: np.ndarray | None, offset=(0, 0), dedup_tol_px: float = 2) -> pd.DataFrame:
    """One row per candidate segment: geometry, provenance, layer means, network descriptors and (evaluation side)
    ``y_near_truth``."""
    from ..p6.segment_recovery import label_geometry

    n = int(lab.max())
    if not n:
        return pd.DataFrame(columns=["label"])
    T = label_geometry(lab).rename(columns={"row": "row_crop", "col": "col_crop"})
    T["row_full"], T["col_full"] = T.row_crop + offset[0], T.col_crop + offset[1]
    T["length_km"] = T.n_px * STEP_KM
    T["connectivity"] = connectivity(lab)
    T = T.merge(layer_means(lab, layers), on="label")
    if "tip_dist_km" in layers:
        rr, cc = np.nonzero(lab)
        v = np.nan_to_num(np.asarray(layers["tip_dist_km"], float)[rr, cc], nan=np.inf)
        T["min_tip_dist_km"] = np.asarray(ndimage.minimum(v, lab[rr, cc], T.label.to_numpy()))
    prop = []
    for s, m in pools.items():
        T[f"on_{s}"] = near_share(lab, m, dedup_tol_px)
        prop.append(s)
    by = T[[f"on_{s}" for s in prop]].to_numpy() >= 0.5 if prop else np.zeros((n, 0), bool)
    T["proposed_by"] = [";".join(p for p, b in zip(prop, row) if b) for row in by]
    T["n_proposers"] = by.sum(axis=1)
    fams = {f: [i for i, p in enumerate(prop) if SOURCE_FAMILY.get(p) == f] for f in sorted(set(SOURCE_FAMILY.values()))}
    for f, ix in fams.items():
        T[f"prop_{f}"] = by[:, ix].any(axis=1) if ix else False
    T["eligible_share"] = near_share(lab, eligible, 0)
    if truth is not None:
        T["y_near_truth"] = near_share(lab, truth, tol_px)
    return T


def tips_raster(known: np.ndarray, ev: np.ndarray, lengths_km, tail_px: int) -> np.ndarray:
    """Straight continuations of the permitted known traces (label-free), clipped to ``ev``."""
    from ..gems3.candidates import tip_paths

    out = np.zeros(ev.shape, bool)
    for p, _ in tip_paths(np.asarray(known, bool), np.asarray(ev, bool), lengths_km, int(tail_px)):
        out[p[:, 0], p[:, 1]] = True
    return out & ev


def inventory(pools: dict, cc: dict) -> tuple[np.ndarray, pd.DataFrame]:
    """(segment labels, dedup table) of the pools in POOL_SOURCES priority order (+ tips last)."""
    from ..p6.segment_recovery import dedup_inventory

    order = [s for s in (*POOL_SOURCES, "tips") if s in pools and np.asarray(pools[s]).any()]
    if not order:
        shape = next(iter(pools.values())).shape if pools else (1, 1)
        return np.zeros(shape, np.int32), pd.DataFrame()
    d = cc["dedup"]
    lab, D = dedup_inventory({s: pools[s] for s in order}, order, float(d["tol_px"]), float(d["min_share"]),
                             float(d["max_angle_deg"]))
    lab = split_long(lab, int(round(float(cc["max_segment_km"]) / STEP_KM)))
    sizes = np.bincount(lab.ravel(), minlength=int(lab.max()) + 1)
    small = sizes < int(cc["min_segment_px"])
    small[0] = False
    if small.any():
        lab = np.where(small[lab], 0, lab)
        _, inv = np.unique(lab, return_inverse=True)
        lab = inv.reshape(lab.shape).astype(np.int32)
    return lab, D


# =============================================================================== pure: coverage / oracle
def truth_pixels(u, sysmap: np.ndarray, known_dist_km: np.ndarray | None) -> pd.DataFrame:
    """Scored truth pixels (crop coordinates) with system id and distance to the permitted known faults."""
    if not u.scorer.n:
        return pd.DataFrame(columns=["r", "c", "system_id", "known_dist_km"])
    gr, gc = u.scorer.g_rc
    d = pd.DataFrame({"r": np.asarray(gr), "c": np.asarray(gc), "system_id": np.asarray(sysmap)[gr, gc]})
    d["known_dist_km"] = np.asarray(known_dist_km)[gr, gc] if known_dist_km is not None else np.nan
    return d[d.system_id >= 0].reset_index(drop=True)


def system_coverage(px: pd.DataFrame, masks: dict, tol_px: float, share: float) -> pd.DataFrame:
    """Per (system, pool): share of the system's scored pixels within ``tol_px`` of the pool, covered flag."""
    rows = []
    if not len(px):
        return pd.DataFrame(columns=["system_id", "pool", "near_share", "covered", "n_px"])
    for name, m in masks.items():
        near = near_pixels(px, m, tol_px)
        g = pd.DataFrame({"s": px.system_id.to_numpy(), "n": near}).groupby("s").n.agg(["mean", "size"])
        rows.append(pd.DataFrame({"system_id": g.index.to_numpy(), "pool": name, "near_share": g["mean"].to_numpy(),
                                  "covered": g["mean"].to_numpy() >= float(share), "n_px": g["size"].to_numpy()}))
    return pd.concat(rows, ignore_index=True)


def near_pixels(px: pd.DataFrame, mask: np.ndarray, tol_px: float) -> np.ndarray:
    m = np.asarray(mask, bool)
    if not m.any() or not len(px):
        return np.zeros(len(px), bool)
    return ndimage.distance_transform_edt(~m)[px.r.to_numpy(), px.c.to_numpy()] <= tol_px


def band_coverage(px: pd.DataFrame, masks: dict, tol_px: float, edges) -> pd.DataFrame:
    """GEOMETRIC coverage of truth pixels (length) per pool and distance band to the permitted known faults. Being
    within 300 m is NOT the official TP support (linear kernel); see the kernel-weighted oracle for that."""
    from .scenarios import BANDS, band_of

    if not len(px):
        return pd.DataFrame(columns=["pool", "band", "truth_px", "covered_px", "covered_share"])
    b = band_of(px.known_dist_km.to_numpy(float), edges)
    rows = []
    for name, m in masks.items():
        near = near_pixels(px, m, tol_px)
        for i, nm in enumerate(BANDS):
            k = b == i
            rows.append({"pool": name, "band": nm, "truth_px": int(k.sum()), "covered_px": int((near & k).sum()),
                         "covered_share": float((near & k).sum() / max(int(k.sum()), 1))})
    return pd.DataFrame(rows)


def system_attributes(px: pd.DataFrame, events: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per system: scored pixels, length, median distance to the permitted known faults, length bin, subtype."""
    if not len(px):
        return pd.DataFrame(columns=["system_id", "n_px", "length_km", "known_dist_km_median", "length_bin", "subtype"])
    g = px.groupby("system_id").agg(n_px=("r", "size"), known_dist_km_median=("known_dist_km", "median")).reset_index()
    g["length_km"] = g.n_px * STEP_KM
    edges = (0.0, *LENGTH_BINS_KM, np.inf)
    g["length_bin"] = pd.cut(g.length_km, edges, labels=[f"<{LENGTH_BINS_KM[0]:g}km"] + [
        f"{a:g}-{b:g}km" for a, b in zip(LENGTH_BINS_KM[:-1], LENGTH_BINS_KM[1:])] + [f">{LENGTH_BINS_KM[-1]:g}km"],
        right=False).astype(str)
    if events is not None and len(events):
        g = g.merge(events[["event_id", "class"]].rename(columns={"event_id": "system_id", "class": "subtype"}),
                    on="system_id", how="left")
    else:
        g["subtype"] = "mapped_system"
    return g


def unique_coverage(C: pd.DataFrame, families: list[str]) -> pd.DataFrame:
    """Per system: covered by which families, uniquely covered by one family (pools named ``fam:<family>``)."""
    cols = [f"fam:{f}" for f in families]
    h = C[C.pool.astype(str).str.startswith("fam:")]
    if not len(h):
        return pd.DataFrame(columns=["system_id", "n_families"] + [f"{k}_{f}" for f in families
                                                                    for k in ("covered", "unique")])
    F = (h.assign(covered=h.covered.astype(bool).astype(int))
         .pivot_table(index="system_id", columns="pool", values="covered", aggfunc="max", fill_value=0))
    F = F.reindex(columns=cols, fill_value=0).astype(bool)
    out = pd.DataFrame({"system_id": F.index.to_numpy(), "n_families": F[cols].sum(axis=1).to_numpy()})
    for f in families:
        own = F[f"fam:{f}"].to_numpy(bool)
        others = [c for c in cols if c != f"fam:{f}"]
        other = F[others].to_numpy(bool).any(axis=1) if others else np.zeros(len(F), bool)
        out[f"covered_{f}"] = own
        out[f"unique_{f}"] = own & ~other
    return out


def oracle_order(lab: np.ndarray, truth: np.ndarray, tol_px: float) -> np.ndarray:
    """DIAGNOSTIC ONLY (held-out labels): labels with any truth support, by near-truth precision then size."""
    n = int(lab.max())
    if not n:
        return np.zeros(0, int)
    prec = near_share(lab, truth, tol_px)
    size = np.bincount(lab.ravel(), minlength=n + 1)[1:]
    ids = np.arange(1, n + 1)
    keep = ids[prec > 0]
    return keep[np.lexsort((-size[keep - 1], -prec[keep - 1]))]


def score_order(lab: np.ndarray, score: np.ndarray) -> np.ndarray:
    """Labels by a label-free segment score (NaN last), ties by size."""
    n = int(lab.max())
    if not n:
        return np.zeros(0, int)
    s = np.nan_to_num(np.asarray(score, float), nan=-np.inf)
    size = np.bincount(lab.ravel(), minlength=n + 1)[1:]
    return np.lexsort((-size, -s)) + 1


# =============================================================================== Colab side
INVENTORY_ALIASES = {                  # brief vocabulary -> repository columns (NaN when the source is absent)
    "length_km": "length_km", "straightness": "straightness", "continuity": "connectivity",
    "source_membership": "proposed_by", "cnn_b0_score": "mean_cnn_b0", "cnn_b1_score": "mean_cnn_b1",
    "w2_score": "mean_w2", "magnetic_score": "mean_mag_hgrad", "gravity_score": "mean_grav_hgrad",
    "mag_grav_agreement": "mean_mg_orient_agree", "multiscale_persistence": "mean_pf_persistence",
    "g02_continuity": "mean_g02c", "g02_displacement": "mean_g02a", "g02_truncation": "mean_g02_truncation",
    "known_fault_distance": "mean_known_dist_km", "known_tip_distance": "min_tip_dist_km",
    "tip_alignment": "mean_tip_along_cos", "terrain_availability": "mean_dem_observed", "terrain_evidence": "mean_dem"}


def inventory_view(T: pd.DataFrame) -> pd.DataFrame:
    """The persisted candidate inventory: ids, geometry reference, provenance and the brief's named columns
    (``g02_truncation`` has no raster key in the G02 run: recorded as NaN = UNAVAILABLE, never zero)."""
    out = pd.DataFrame({"candidate_id": T.unit.astype(str) + "#" + T.label.astype(str), "scenario": T.scenario,
                        "zone": T.zone, "unit": T.unit, "label": T.label,
                        "orientation_deg": np.degrees(T.theta.to_numpy(float)) % 180.0,
                        "row_full": T.row_full, "col_full": T.col_full, "n_px": T.n_px,
                        "geometry": "candidates/candidate_geometries.gpkg (candidate_id) | label raster "
                                    "candidates/units/<fp>/<unit>.npz"})
    for k, c in INVENTORY_ALIASES.items():
        out[k] = T[c] if c in T else np.nan
    for c in [c for c in T.columns if c.startswith(("on_", "prop_"))] + ["n_proposers", "eligible_share"]:
        if c in T:
            out[c] = T[c]
    return out


def unpack_pools(z) -> dict:
    shape = tuple(int(x) for x in z["shape"])
    n = int(np.prod(shape))
    return {k[5:]: np.unpackbits(z[k])[:n].reshape(shape).astype(bool) for k in z.files if k.startswith("pool_")}


def pack_pools(pools: dict) -> dict:
    return {f"pool_{k}": np.packbits(np.asarray(v, bool)) for k, v in pools.items()}


def build_unit(G, su, L: dict) -> dict:
    """Pools, inventory and segment table of one scenario unit (``L`` = experts.unit_layers). Label-free except the
    evaluation-side ``y_near_truth`` column. Coverage is a separate step (``unit_coverage``) so a coverage failure
    never discards the expensive inventory."""
    from ..gems3 import representation as R

    cc = G.cfg["candidates"]
    u, b = su.unit, su.base
    el = G.elig(u)
    sh = G.sh
    rep = sh["baseline"]["representations"][cc["pool_rep"]]
    sigma = float(sh["evaluation"]["ridge_sigma_px"])
    pools, prov = {}, []
    for s in POOL_SOURCES:
        if s not in L or not np.isfinite(L[s][b.ev]).any():
            prov.append({"unit": su.key, "source": s, "family": SOURCE_FAMILY[s], "status": "UNAVAILABLE"})
            continue
        sc = np.where(np.isfinite(L[s]), L[s], np.nan).astype(np.float32)
        q = R.apply(rep, sc, u.ev, float(cc["pool_frac"]), known=u.known, sigma_px=sigma, eligible=el)
        pools[s] = (q > 0) & el
        prov.append({"unit": su.key, "source": s, "family": SOURCE_FAMILY[s], "status": "OK",
                     "pool_px": int(pools[s].sum())})
    t = tips_raster(su.known, el, cc["tip_lengths_km"], int(cc["tip_tail_px"])) if "network" in L else \
        np.zeros(el.shape, bool)
    if t.any():
        pools["tips"] = t
    prov.append({"unit": su.key, "source": "tips", "family": SOURCE_FAMILY["tips"],
                 "status": "OK" if t.any() else ("DISABLED (experts.network)" if "network" not in L else
                                                 "EMPTY (no permitted known tip in reach)"), "pool_px": int(t.sum())})
    lab, D = inventory(pools, cc)
    truth = u.eval_truth()
    layers = {k: v for k, v in L.items() if k in (*POOL_SOURCES, *G.descriptor_layers())}
    T = segment_table(lab, layers, pools, float(cc["tol_px"]), el, truth, (u.sl[0].start, u.sl[1].start),
                      float(cc["dedup"]["tol_px"]))
    T = T.assign(scenario=su.scenario, zone=su.zone, unit=su.key)
    return {"lab": lab, "T": T, "dedup": D.assign(unit=su.key) if len(D) else D, "prov": pd.DataFrame(prov),
            "pools": pools, "pool_px": {s: int(m.sum()) for s, m in pools.items()}, "eligible_px": int(el.sum())}


def coverage_masks(pools: dict, lab: np.ndarray) -> dict:
    from .experts import FAMILIES

    masks = {f"src:{s}": m for s, m in pools.items()}
    for f in FAMILIES:
        fm = [m for s, m in pools.items() if SOURCE_FAMILY.get(s) == f]
        masks[f"fam:{f}"] = np.logical_or.reduce(fm) if fm else np.zeros(lab.shape, bool)
    masks["inventory"] = lab > 0
    masks["label_free"] = masks["fam:E2_potential"] | masks["fam:E3_structural"]
    return masks


def unit_coverage(G, su, lab: np.ndarray, T: pd.DataFrame, pools: dict, known_dist_km: np.ndarray) -> dict:
    """DIAGNOSTIC (held-out truth): system / length / distance-band coverage, attributes, unique families and the
    kernel-weighted oracle of one unit, from the saved inventory and pools (no layer is recomputed)."""
    from .experts import FAMILIES

    cc = G.cfg["candidates"]
    tag = {"scenario": su.scenario, "zone": su.zone, "unit": su.key}
    px = truth_pixels(su.unit, su.sysmap, known_dist_km)
    masks = coverage_masks(pools, lab)
    C = system_coverage(px, masks, float(cc["tol_px"]), float(cc["covered_share"])).assign(**tag)
    A = system_attributes(px, su.events if su.scenario == "V2" else None).assign(**tag)
    U = unique_coverage(C, list(FAMILIES)).assign(**tag)
    Bd = band_coverage(px, {k: v for k, v in masks.items() if not k.startswith("src:")}, float(cc["tol_px"]),
                       G.cfg["distance_bands_km"]).assign(**tag)
    O = oracle_curve(G, su, lab, T)
    return {"coverage": C, "attrs": A, "unique": U, "bands": Bd, "oracle": O}


def oracle_curve(G, su, lab: np.ndarray, T: pd.DataFrame) -> pd.DataFrame:
    """OFFICIAL-KERNEL-WEIGHTED curves inside the inventory, scored with the official components (linear 300 m
    kernel, pixel-exact known mask) under the evaluation area budgets: ``oracle`` (segments ordered by held-out truth
    proximity: an upper-bound DIAGNOSTIC, never a selection input) and ``w2_order`` (by mean W2 rank: label-free)."""
    from ..p6.boost import fill_budget

    cc = G.cfg["candidates"]
    u = su.unit
    el = G.elig(u)
    truth = u.eval_truth()
    orders = {"oracle": oracle_order(lab, truth, float(cc["tol_px"]))}
    if "mean_w2" in T:
        s = np.full(int(lab.max()), np.nan)
        s[T.label.to_numpy() - 1] = T.mean_w2.to_numpy()
        orders["w2_order"] = score_order(lab, s)
    rows = []
    n_el = max(int(el.sum()), 1)
    for name, o in orders.items():
        for f in G.cfg["evaluation"]["area_fracs"]:                  # the SAME final area constraints
            q = fill_budget(lab, o, int(np.floor(float(f) * n_el))) & el
            r = G.score(u, q.astype(np.float32))
            rows.append({**r, "scenario": su.scenario, "unit": su.key, "order": name, "frac": float(f),
                         "role": "oracle" if name == "oracle" else "label_free_diagnostic"})
    return pd.DataFrame(rows)


def to_geodataframe(lab: np.ndarray, T: pd.DataFrame, sl, transform, crs):
    """Segments as LineStrings in map coordinates (pixel centres; ``p3.lineaments.trace`` order)."""
    import geopandas as gpd
    from shapely.geometry import LineString, Point

    from ..p3.lineaments import trace

    objs = ndimage.find_objects(lab)
    geoms = []
    for lb in T.label.to_numpy():
        o = objs[int(lb) - 1]
        if o is None:
            geoms.append(None)
            continue
        rr, cc = np.nonzero(lab[o] == lb)
        p = trace(np.c_[rr + o[0].start + sl[0].start, cc + o[1].start + sl[1].start])
        xy = [transform * (c + 0.5, r + 0.5) for r, c in p]
        geoms.append(LineString(xy) if len(xy) > 1 else Point(xy[0]))
    keep = [c for c in T.columns if not c.startswith("y_")]          # labels never leave in the geometry file
    return gpd.GeoDataFrame(T[keep].reset_index(drop=True), geometry=geoms, crs=crs)
