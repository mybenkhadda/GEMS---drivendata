"""Model-ready datasets (features and labels stored separately, joined by ids; strict allowlists at fit time).

D1 pixel        datasets/pixels/pool_meta.parquet       row_id, row, col, x, y, stratum flags, block, incl_prob
                datasets/pixels/pool_features.parquet   row_id + every store feature at the pool pixels
                datasets/pixels/labels/<fit>.parquet    row_id, fit_id, cls, target, weight, label_conf, system_id,
                                                        system_px, dist_km, target_dist_km, inner_fold, inner_buffer
D2 patch        datasets/patches/patch_index.parquet    patch_id, r0, c0, size, footprint share, fold, eval-zone shares
                (multichannel arrays: ``models_patch.create_patch_dataset`` writes the channel memmap once)
D3 segment      datasets/segments/units/<unit>.parquet  one row per candidate lineament: ids, geometry, geology,
                                                        orientation, permitted known context, honest model evidence
                datasets/segments/units/<unit>_labels.parquet  near_share, cls, target, label_conf, system (eval side)
                datasets/segments/units/<unit>_lab.npz  candidate label raster (crop) + crop window
                datasets/segments/nested_known/<scenario>_<outer>__<unit zone>.parquet   known context recomputed
                                                        without the outer unit's truth (strict cross-fitting)
D4 connection   datasets/connections/units/<unit>.parquet (+ _labels): candidate fragment pairs
Every manifest (datasets/dataset_manifests/<name>.json) records schema version, feature-schema hash, provenance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import DatasetSpec, LeakageError, ProvenanceError
from .supervision import POSITIVE, build_pixel_supervision, class_confidence

SCHEMA_VERSION = "g06.datasets.1"
PIXEL_META = ["row_id", "row", "col", "x", "y", "stratum", "in_label", "in_evidence", "in_background", "block",
              "incl_prob"]
PIXEL_LABELS = ["row_id", "fit_id", "stratum", "cls", "target", "weight", "label_conf", "system_id", "system_px", "dist_km",
                "target_dist_km", "inner_fold", "inner_buffer"]
SEG_META = ["unit", "scenario", "zone", "label", "row", "col", "r_full", "c_full", "seg_id"]
SEG_LABELS = ["unit", "label", "near_share", "cls", "target", "label_conf", "system_id"]
STEP_KM = 0.112


def unit_file(key: str) -> str:
    return key.replace(":", "_").replace("|", "__")


def pixel_xy(grid, rows, cols) -> tuple[np.ndarray, np.ndarray]:
    a, b, c, d, e, f = grid.transform
    cc, rr = np.asarray(cols, np.float64) + 0.5, np.asarray(rows, np.float64) + 0.5
    return a * cc + b * rr + c, d * cc + e * rr + f


# =============================================================================== D1 pixel
def build_pixel_pool(store, labels: np.ndarray, system_map: np.ndarray, cfg: dict, chunk: int = 250000
                     ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(pool meta, pool features). ``cfg`` = benchmarks.sampling; features = every store channel except the
    physical QC copies (models choose their allowlist at fit time)."""
    from .sampling import build_pool

    fp = store.footprint()
    ev_name = cfg.get("evidence_feature")
    evidence = store.load([ev_name])[0] if ev_name and ev_name in store.names else None
    pool = build_pool(labels, system_map, fp, evidence, cfg)
    del evidence
    x, y = pixel_xy(store.grid, pool.row, pool.col)
    pool["x"], pool["y"] = x, y
    names = [n for n in store.names if not n.startswith("phys_")]
    parts = []
    for i in range(0, len(pool), chunk):
        p = pool.iloc[i:i + chunk]
        parts.append(store.sample(names, p.row.to_numpy(), p.col.to_numpy()))
    X = pd.DataFrame(np.concatenate(parts) if parts else np.zeros((0, len(names)), np.float32), columns=names)
    X.insert(0, "row_id", pool.row_id.to_numpy())
    return pool[PIXEL_META], X


def label_pool_for_fit(pool: pd.DataFrame, vis, valid: np.ndarray, system_map: np.ndarray, sup_cfg: dict,
                       samp_cfg: dict, inner_cfg: dict, seed: int) -> pd.DataFrame:
    """Labels / weights / inner folds of the pool rows usable by one training visibility."""
    from .sampling import compute_sample_weights
    from .splits import inner_buffer_mask, make_inner_folds

    sup = build_pixel_supervision(vis.visible, vis.region, valid, sup_cfg)
    r, c = pool.row.to_numpy(), pool.col.to_numpy()
    cls = sup["classes"][r, c]
    visible_px = vis.visible[r, c]
    keep = (pool.in_background.to_numpy() | pool.in_evidence.to_numpy()) | (pool.in_label.to_numpy() & visible_px)
    keep &= cls != -1
    # a hidden-fault pixel kept through the background / evidence strata counts as THAT stratum, never 'label'
    stratum = pool.stratum.to_numpy().astype(object)
    relabel = pool.in_label.to_numpy() & ~visible_px
    stratum[relabel] = np.where(pool.in_evidence.to_numpy()[relabel], "evidence", "background")
    T = pd.DataFrame({"row_id": pool.row_id.to_numpy()[keep], "row": r[keep], "col": c[keep],
                      "stratum": stratum[keep], "cls": cls[keep],
                      "dist_km": sup["dist_km"][r, c][keep], "target_dist_km": sup["target_dist_km"][r, c][keep]})
    sm = np.asarray(system_map)
    sid = np.where(T.cls.to_numpy() == POSITIVE, sm[T.row.to_numpy(), T.col.to_numpy()], -1)
    T["system_id"] = sid
    vis_ids = sm[vis.visible]
    n_sys = pd.Series(vis_ids[vis_ids >= 0]).value_counts()
    T["system_px"] = pd.Series(sid).map(n_sys).fillna(0).to_numpy().astype(np.int64)
    T["target"] = (T.cls.to_numpy() == POSITIVE).astype(np.int8)
    T["label_conf"] = class_confidence(T.cls.to_numpy(), sup_cfg)
    T["weight"] = compute_sample_weights(T, {**sup_cfg, **samp_cfg})
    inner = make_inner_folds(vis.region, vis.visible, float(inner_cfg.get("block_km", 20.0)),
                             int(inner_cfg.get("n_folds", 3)), int(seed))
    ib = inner_buffer_mask(inner, float(inner_cfg.get("buffer_km", 2.0)))
    T["inner_fold"] = inner[T.row.to_numpy(), T.col.to_numpy()].astype(np.int8)
    T["inner_buffer"] = ib[T.row.to_numpy(), T.col.to_numpy()]
    T["fit_id"] = vis.fit_id
    if (T.cls.to_numpy() == POSITIVE).any() and not vis.visible[T.row[T.cls == POSITIVE], T.col[T.cls == POSITIVE]
                                                                ].all():
        raise LeakageError(f"{vis.fit_id}: a positive row is not a visible label")
    if vis.hidden[T.row.to_numpy(), T.col.to_numpy()].any() and (T.cls.to_numpy()[
            vis.hidden[T.row.to_numpy(), T.col.to_numpy()]] == POSITIVE).any():
        raise LeakageError(f"{vis.fit_id}: a hidden fault pixel was labelled positive")
    return T[PIXEL_LABELS]


def build_patch_index(footprint: np.ndarray, patch_px: int, stride_px: int, fold_map: np.ndarray | None,
                      eval_zones: dict | None, min_footprint: float = 0.5) -> pd.DataFrame:
    """D2 index (label-free): patch windows over the footprint with fold / evaluation-zone shares."""
    fp = np.asarray(footprint, bool)
    h, w = fp.shape
    rows = []
    for r0 in range(0, max(h - patch_px, 0) + 1, stride_px):
        for c0 in range(0, max(w - patch_px, 0) + 1, stride_px):
            sl = (slice(r0, r0 + patch_px), slice(c0, c0 + patch_px))
            share = float(fp[sl].mean())
            if share < min_footprint:
                continue
            row = {"patch_id": f"p{r0:05d}_{c0:05d}", "r0": r0, "c0": c0, "size": patch_px, "footprint_share": share}
            if fold_map is not None:
                f = np.asarray(fold_map)[sl]
                f = f[f >= 0]
                row["fold"] = int(np.bincount(f).argmax()) if f.size else -1
            for d, m in (eval_zones or {}).items():
                row[f"eval_share_{d}"] = float(np.asarray(m)[sl].mean())
            rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================== D3 segments
def unit_pools(engine, store, vu, cfg: dict, extra_maps: dict | None = None) -> tuple[dict, dict]:
    """Label-free candidate pools of one validation unit (crop masks) + their provenance."""
    from ..gems3.representation import rank01, top_mask

    u, b = vu.unit, vu.base
    ev = u.ev
    elig = u.scored_mask() & engine.template_valid()[u.sl]
    pools, prov = {}, {}
    pc = {k: v for k, v in (cfg.get("pools") or {}).items() if v}       # a null pool is disabled
    if "w2" in pc:
        pools["w2"] = top_mask(engine.w2_rank(b), elig, float(pc["w2"]["frac"]))
        prov["w2"] = engine.provenance(b)["w2"]
    if "cnn" in pc:
        b0, b1 = engine.cnn(b)
        fused = 0.5 * (rank01(b0, ev) + rank01(b1, ev))
        pools["cnn"] = top_mask(fused, elig, float(pc["cnn"]["frac"]))
        prov["cnn"] = engine.provenance(b)["cnn_b0"]
    if "physics" in pc:
        win = (u.sl[0].start, u.sl[0].stop, u.sl[1].start, u.sl[1].stop)
        a = store.load([pc["physics"]["feature"]], win)[0]
        pools["physics"] = top_mask(rank01(np.nan_to_num(a, nan=-np.inf), ev), elig, float(pc["physics"]["frac"]))
        prov["physics"] = f"store:{pc['physics']['feature']}"
    for k, m in (extra_maps or {}).items():
        if k in pc and m is not None:
            pools[k] = top_mask(rank01(np.nan_to_num(m, nan=0.0), ev), elig, float(pc[k]["frac"]))
            prov[k] = f"honest_prediction:{k}"
    return pools, prov


def generate_candidates(pools: dict, cfg: dict) -> np.ndarray:
    """Deduplicated, length-limited candidate labels (int32 crop, 1..n) from the pools (label-free)."""
    from ..geomoe.candidates import split_long
    from ..p6.segment_recovery import dedup_inventory

    if not pools or not any(np.asarray(v).any() for v in pools.values()):
        shape = next(iter(pools.values())).shape if pools else (1, 1)
        return np.zeros(shape, np.int32)
    lab, _ = dedup_inventory(pools, list(pools), float(cfg.get("dedup_tol_px", 2)),
                             float(cfg.get("dedup_min_share", 0.6)), float(cfg.get("dedup_max_angle_deg", 20)))
    lab = split_long(lab, int(cfg.get("max_len_px", 60)))
    n = int(lab.max())
    if not n:
        return lab.astype(np.int32)
    size = np.bincount(lab.ravel(), minlength=n + 1)
    keep = np.zeros(n + 1, bool)
    keep[1:] = size[1:] >= int(cfg.get("min_px", 5))
    new = np.zeros(n + 1, np.int32)
    new[keep] = np.arange(1, int(keep.sum()) + 1)
    return new[lab]


def _means(lb: np.ndarray, vals: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    ok = np.isfinite(vals)
    cnt = np.bincount(lb[ok], minlength=n + 1)[1:].astype(float)
    s = np.bincount(lb[ok], weights=vals[ok], minlength=n + 1)[1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan), cnt


def _stds(lb, vals, n, mean):
    ok = np.isfinite(vals)
    cnt = np.bincount(lb[ok], minlength=n + 1)[1:].astype(float)
    s2 = np.bincount(lb[ok], weights=vals[ok] ** 2, minlength=n + 1)[1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.sqrt(np.maximum(np.where(cnt > 0, s2 / np.maximum(cnt, 1), np.nan) - mean ** 2, 0))


def segment_family(col: str) -> str:
    for pre, fam in (("seg_", "geometry"), ("on_", "geometry"), ("geo_", "geology"), ("ori_", "orientation"),
                     ("kn_", "context_known"), ("ev_", "model_evidence"), ("unc_", "uncertainty")):
        if col.startswith(pre):
            return fam
    return "meta"


def known_columns(lab: np.ndarray, known_ctx: dict, seg_c2: np.ndarray, seg_s2: np.ndarray) -> pd.DataFrame:
    """Per-segment permitted known-fault context (means / minima over the segment pixels)."""
    n = int(lab.max())
    rr, cc = np.nonzero(lab)
    lb = lab[rr, cc]
    out = {}
    for k in ("known_dist_km", "known_density", "tip_along_cos", "known_net_density"):
        out[f"kn_{k}_mean"], _ = _means(lb, np.asarray(known_ctx[k])[rr, cc].astype(np.float64), n)
    td = np.asarray(known_ctx["tip_dist_km"])[rr, cc].astype(np.float64)
    mn = np.full(n, np.nan)
    ok = np.isfinite(td)
    if ok.any():
        s = pd.Series(td[ok]).groupby(lb[ok]).min()
        mn[s.index.to_numpy() - 1] = s.to_numpy()
    out["kn_tip_dist_km_min"] = mn
    kc, _ = _means(lb, np.asarray(known_ctx["known_orient_cos2"])[rr, cc].astype(np.float64), n)
    ks, _ = _means(lb, np.asarray(known_ctx["known_orient_sin2"])[rr, cc].astype(np.float64), n)
    out["kn_orient_agree"] = 0.5 * (1.0 + seg_c2 * kc + seg_s2 * ks)
    return pd.DataFrame(out)


def segment_features(lab: np.ndarray, store, window: tuple, cfg: dict, known_ctx: dict | None,
                     evidence: dict | None) -> pd.DataFrame:
    """One row per candidate label: geometry, geology (store means), orientation agreement, cross-trace field step,
    permitted known context, honest model evidence, uncertainty. No truth is read here."""
    from ..geomoe.candidates import connectivity
    from ..p6.segment_recovery import label_geometry

    n = int(lab.max())
    if not n:
        return pd.DataFrame()
    G = label_geometry(lab)
    th = G.theta.to_numpy()                                   # from +column toward +row
    c2, s2 = np.cos(2 * th), -np.sin(2 * th)                  # doubled math angle (north up)
    T = pd.DataFrame({"label": G.label.to_numpy(), "row": G.row.to_numpy(), "col": G.col.to_numpy(),
                      "seg_n_px": G.n_px.to_numpy(), "seg_length_km": G.n_px.to_numpy() * STEP_KM,
                      "seg_cos2": c2, "seg_sin2": s2, "seg_linearity": G.linearity.to_numpy(),
                      "seg_straightness": G.straightness.to_numpy(), "seg_degree": connectivity(lab)})
    rr, cc = np.nonzero(lab)
    lb = lab[rr, cc]
    fin_cnt = np.zeros(n)
    feats = [f for f in cfg.get("geology_features", []) if f in store.names]
    for f in feats:
        a = store.load([f], window)[0]
        v = a[rr, cc].astype(np.float64)
        m, cnt = _means(lb, v, n)
        T[f"geo_mean_{f}"] = m
        fin_cnt += cnt
        if f in cfg.get("std_features", []):
            T[f"geo_std_{f}"] = _stds(lb, v, n, m)
    T["unc_feature_finite_share"] = fin_cnt / np.maximum(len(feats) * T.seg_n_px.to_numpy(), 1)
    for fld, (fc, fs) in (cfg.get("orientation_pairs") or {}).items():
        if fc in store.names and fs in store.names:
            mc, _ = _means(lb, store.load([fc], window)[0][rr, cc].astype(np.float64), n)
            ms, _ = _means(lb, store.load([fs], window)[0][rr, cc].astype(np.float64), n)
            T[f"ori_{fld}_agree"] = 0.5 * (1.0 - (c2 * mc + s2 * ms))      # vs strike = gradient axis + 90 deg
    d = float(cfg.get("jump_offset_px", 3))
    for f in cfg.get("jump_fields", []):
        if f not in store.names:
            continue
        a = store.load([f], window)[0].astype(np.float64)
        nr, nc = np.cos(th[lb - 1]), -np.sin(th[lb - 1])     # normal to the segment axis (row, col components)
        from scipy import ndimage

        p = ndimage.map_coordinates(a, [rr + d * nr, cc + d * nc], order=1, mode="nearest")
        q = ndimage.map_coordinates(a, [rr - d * nr, cc - d * nc], order=1, mode="nearest")
        diff = p - q
        md, _ = _means(lb, diff, n)
        mabs, _ = _means(lb, np.abs(diff), n)
        T[f"geo_jump_{f}"] = np.abs(md)
        with np.errstate(invalid="ignore", divide="ignore"):
            T[f"geo_jumpcons_{f}"] = np.abs(md) / np.maximum(mabs, 1e-12)
    if known_ctx is not None:
        K = known_columns(lab, known_ctx, c2, s2)
        for k in K:
            T[k] = K[k].to_numpy()
    for k, a in (evidence or {}).items():
        v = np.asarray(a, np.float64)[rr, cc]
        m, _ = _means(lb, v, n)
        T[f"ev_{k}_mean"] = m
        mx = pd.Series(v).groupby(lb).max()
        arr = np.full(n, np.nan)
        arr[mx.index.to_numpy() - 1] = mx.to_numpy()
        T[f"ev_{k}_max"] = arr
        if k == "w2_rank":
            T["unc_w2_rank_std"] = _stds(lb, v, n, m)
    return T


def build_segment_dataset(engine, store, vu, cfg: dict, sup_cfg: dict, known_cfg: dict,
                          extra_maps: dict | None = None) -> dict:
    """D3 rows + labels + label raster of one unit. Candidates and features are label-free; the label table is
    computed from the unit's own truth (evaluation side)."""
    from ..gems3.representation import rank01
    from .feature_context import known_context_provenance, permitted_fault_context
    from .supervision import build_segment_supervision, dominant_system

    u = vu.unit
    pools, prov = unit_pools(engine, store, vu, cfg, extra_maps)
    lab = generate_candidates(pools, cfg)
    window = (u.sl[0].start, u.sl[0].stop, u.sl[1].start, u.sl[1].stop)
    valid_crop = engine.valid()[u.sl]
    kctx = permitted_fault_context(vu.known, valid_crop, known_cfg)
    ev = {}
    if cfg.get("track_b_evidence", True):
        ev["w2_rank"] = engine.w2_rank(vu.base)
        b0, b1 = engine.cnn(vu.base)
        r0, r1 = rank01(b0, u.ev), rank01(b1, u.ev)
        ev.update(cnn_b0_rank=r0, cnn_b1_rank=r1, cnn_disagree=np.abs(r0 - r1))
    T = segment_features(lab, store, window, cfg, kctx, ev)
    if len(T):
        for k, m in pools.items():
            rr, cc = np.nonzero(lab)
            share, _ = _means(lab[rr, cc], m[rr, cc].astype(np.float64), int(lab.max()))
            T[f"on_{k}"] = share
        T.insert(0, "unit", vu.key)
        T.insert(1, "scenario", vu.scenario)
        T.insert(2, "zone", vu.zone)
        T["r_full"] = T.row + u.sl[0].start
        T["c_full"] = T.col + u.sl[1].start
        T["seg_id"] = vu.key + "#" + T.label.astype(str)
    L = build_segment_supervision(lab, vu.eval_truth(), sup_cfg)
    if len(L):
        L["system_id"] = dominant_system(lab, vu.sysmap, vu.eval_truth(), float(sup_cfg["tol_px"]))
        L.insert(0, "unit", vu.key)
    provenance = {"unit": vu.key, "pools": prov, "known": known_context_provenance(vu.key, vu.known),
                  "evidence": engine.provenance(vu.base) if ev else {}, "window": window}
    return {"T": T, "L": L, "lab": lab, "pools": pools, "provenance": provenance}


def outer_hidden(engine, scenario: str, outer: int) -> np.ndarray:
    """Full-grid labels carrying the truth of outer unit ``scenario:outer`` (removed from nested known context)."""
    lab = engine.labels()
    if scenario == "V1":
        return engine.ctx.splits.withheld(outer) & lab
    return lab & (np.asarray(engine.ctx.folds.fold_map) == int(outer))


def nested_known_table(engine, vu, outer: int, lab: np.ndarray, known_cfg: dict) -> pd.DataFrame:
    """Known-context columns of unit ``vu`` recomputed without the truth of outer unit ``outer`` (same scenario)."""
    from .feature_context import permitted_fault_context

    hid = outer_hidden(engine, vu.scenario, outer)[vu.sl]
    known = vu.known & ~hid
    if (known & vu.eval_truth()).any():
        raise LeakageError(f"{vu.key}: nested known context contains the unit truth")
    kctx = permitted_fault_context(known, engine.valid()[vu.sl], known_cfg)
    from ..p6.segment_recovery import label_geometry

    G = label_geometry(lab)
    th = G.theta.to_numpy()
    K = known_columns(lab, kctx, np.cos(2 * th), -np.sin(2 * th))
    K.insert(0, "label", G.label.to_numpy())
    K.insert(0, "unit", vu.key)
    K["outer"] = int(outer)
    return K


# =============================================================================== D4 connections
def segment_endpoints(lab: np.ndarray) -> pd.DataFrame:
    """Two extreme pixels of every label along its major axis (crop coordinates)."""
    from ..p6.segment_recovery import label_geometry

    G = label_geometry(lab).set_index("label")
    rr, cc = np.nonzero(lab)
    lb = lab[rr, cc]
    th = G.theta.to_numpy()[lb - 1]
    proj = (cc - G.col.to_numpy()[lb - 1]) * np.cos(th) + (rr - G.row.to_numpy()[lb - 1]) * np.sin(th)
    df = pd.DataFrame({"label": lb, "r": rr, "c": cc, "p": proj})
    lo = df.loc[df.groupby("label").p.idxmin()].assign(end=0)
    hi = df.loc[df.groupby("label").p.idxmax()].assign(end=1)
    return pd.concat([lo, hi], ignore_index=True)[["label", "end", "r", "c"]]


def line_pixels(r0, c0, r1, c1) -> tuple[np.ndarray, np.ndarray]:
    n = int(max(abs(r1 - r0), abs(c1 - c0))) + 1
    t = np.linspace(0, 1, n)
    return np.round(r0 + t * (r1 - r0)).astype(int), np.round(c0 + t * (c1 - c0)).astype(int)


def build_connection_dataset(lab: np.ndarray, T: pd.DataFrame, store, window: tuple, cfg: dict,
                             evidence: dict | None, unit_key: str) -> pd.DataFrame:
    """D4: candidate fragment pairs (nearest endpoints within ``max_gap_km``) with gap / alignment / evidence
    features (label-free). Gap pixels are returned in ``gap_r`` / ``gap_c`` (json) for the evaluation-side target."""
    from scipy.spatial import cKDTree

    if not len(T) or int(lab.max()) < 2:
        return pd.DataFrame()
    E = segment_endpoints(lab)
    tree = cKDTree(E[["r", "c"]].to_numpy())
    mg = float(cfg.get("max_gap_km", 2.0)) / 0.1
    pairs = tree.query_pairs(mg, output_type="ndarray")
    if not len(pairs):
        return pd.DataFrame()
    a, b = E.iloc[pairs[:, 0]].reset_index(drop=True), E.iloc[pairs[:, 1]].reset_index(drop=True)
    ok = a.label.to_numpy() != b.label.to_numpy()
    a, b = a[ok].reset_index(drop=True), b[ok].reset_index(drop=True)
    if not len(a):
        return pd.DataFrame()
    sw = a.label.to_numpy() > b.label.to_numpy()                 # canonical order label_a < label_b
    a2, b2 = a.copy(), b.copy()
    a2.loc[sw, :] = b.loc[sw, :].to_numpy()
    b2.loc[sw, :] = a.loc[sw, :].to_numpy()
    a, b = a2, b2
    P = pd.DataFrame({"label_a": a.label, "label_b": b.label, "ra": a.r, "ca": a.c, "rb": b.r, "cb": b.c})
    P["gap_km"] = np.hypot(P.ra - P.rb, P.ca - P.cb) * 0.1
    P = P.sort_values("gap_km").drop_duplicates(["label_a", "label_b"]).reset_index(drop=True)
    S = T.set_index("label")
    ca2, sa2 = S.loc[P.label_a, "seg_cos2"].to_numpy(), S.loc[P.label_a, "seg_sin2"].to_numpy()
    cb2, sb2 = S.loc[P.label_b, "seg_cos2"].to_numpy(), S.loc[P.label_b, "seg_sin2"].to_numpy()
    gth = np.arctan2(-(P.rb - P.ra).to_numpy(), (P.cb - P.ca).to_numpy())
    gc2, gs2 = np.cos(2 * gth), np.sin(2 * gth)
    P["con_angle_agree"] = 0.5 * (1 + ca2 * cb2 + sa2 * sb2)
    P["con_gap_align_a"] = 0.5 * (1 + ca2 * gc2 + sa2 * gs2)
    P["con_gap_align_b"] = 0.5 * (1 + cb2 * gc2 + sb2 * gs2)
    P["con_len_a_km"] = S.loc[P.label_a, "seg_length_km"].to_numpy()
    P["con_len_b_km"] = S.loc[P.label_b, "seg_length_km"].to_numpy()
    feats = [f for f in cfg.get("gap_features", []) if f in store.names]
    arrays = {f: store.load([f], window)[0] for f in feats}
    arrays.update({f"ev_{k}": v for k, v in (evidence or {}).items()})
    gr, gcs = [], []
    vals = {k: [] for k in arrays}
    for row in P.itertuples():
        r, c = line_pixels(row.ra, row.ca, row.rb, row.cb)
        gr.append(r)
        gcs.append(c)
        for k, a in arrays.items():
            v = np.asarray(a)[r, c]
            vals[k].append(float(np.nanmean(v)) if np.isfinite(v).any() else np.nan)
    for k, v in vals.items():
        P[f"con_gap_{k}" if not k.startswith("ev_") else f"ev_gap_{k[3:]}"] = v
    P["gap_r"] = [",".join(map(str, x)) for x in gr]
    P["gap_c"] = [",".join(map(str, x)) for x in gcs]
    P.insert(0, "unit", unit_key)
    return P


def gap_near_share(P: pd.DataFrame, eval_truth: np.ndarray, tol_px: float) -> np.ndarray:
    """EVALUATION SIDE: share of each gap line within ``tol_px`` of the unit truth."""
    from scipy import ndimage

    t = np.asarray(eval_truth, bool)
    if not t.any() or not len(P):
        return np.zeros(len(P))
    d = ndimage.distance_transform_edt(~t)
    out = []
    for gr, gc in zip(P.gap_r, P.gap_c):
        r = np.array([int(x) for x in str(gr).split(",") if x != ""], dtype=int)
        c = np.array([int(x) for x in str(gc).split(",") if x != ""], dtype=int)
        out.append(float((d[r, c] <= tol_px).mean()) if r.size else 0.0)
    return np.asarray(out)


# =============================================================================== manifests / loading
def dataset_manifest(name: str, kind: str, feature_columns, target_columns, meta_columns, n_rows: int,
                     fingerprint: str, schema_hash: str, path: str, provenance: dict | None = None) -> dict:
    spec = DatasetSpec(name, kind, SCHEMA_VERSION, tuple(feature_columns), tuple(target_columns),
                       tuple(meta_columns), int(n_rows), fingerprint, schema_hash, path)
    from dataclasses import asdict

    return {**asdict(spec), "provenance": provenance or {}}


def load_dataset(root, kind: str, fit_id: str | None = None, columns=None, expected_schema_hash: str | None = None
                 ) -> dict:
    """Reload a persisted dataset: {"meta", "X", "labels"} (pixel), {"T", "L"} (segment / connection: all units).
    The feature-schema hash of the manifest must match ``expected_schema_hash`` when given."""
    from pathlib import Path

    from .persistence import read_json

    root = Path(root)
    man = read_json(root / "datasets" / "dataset_manifests" / f"{kind}.json") or {}
    if expected_schema_hash and man.get("feature_schema_hash") not in (None, expected_schema_hash):
        raise ProvenanceError(f"{kind} dataset was built on feature schema {man.get('feature_schema_hash')}, the "
                              f"store is {expected_schema_hash}: rebuild the dataset")
    if kind == "pixel":
        d = root / "datasets" / "pixels"
        out = {"meta": pd.read_parquet(d / "pool_meta.parquet"),
               "X": pd.read_parquet(d / "pool_features.parquet", columns=(["row_id"] + list(columns))
                                    if columns else None)}
        if fit_id is not None:
            out["labels"] = pd.read_parquet(d / "labels" / f"{unit_file(fit_id)}.parquet")
        return out
    sub = {"segment": "segments", "connection": "connections"}[kind]
    d = root / "datasets" / sub / "units"
    T = [pd.read_parquet(p) for p in sorted(d.glob("*.parquet")) if not p.stem.endswith("_labels")]
    L = [pd.read_parquet(p) for p in sorted(d.glob("*_labels.parquet"))]
    return {"T": pd.concat(T, ignore_index=True) if T else pd.DataFrame(),
            "L": pd.concat(L, ignore_index=True) if L else pd.DataFrame(), "manifest": man}
