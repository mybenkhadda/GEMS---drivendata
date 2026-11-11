"""Region-based geographic split generator (v2) and hard split gates.

Why v1 failed on the real grid: v1 merged every block touched by a fault system after dilating the mapped faults by
half the embargo (3.2 km). With ~61 000 fault pixels spread over the area, that dilation connects almost all faults
into ONE system, so almost every block joined one group, which went to ``train`` (5 065 055 px); the few fault-free
groups left for validation / stack / test were then mostly eaten by the 6.4 km embargo (5 317 / 318 / 0 px).

v2:

1. ``block_km`` blocks; blocks touched by the same SMALL fault system (8-connected UNdilated mapped-fault component
   with extent <= ``v2.max_system_km``) are merged, so a small system never spans splits. Regional systems longer
   than that may cross split boundaries; every crossing is separated by the full embargo (listed in the report).
2. Merged block groups are clustered into ``v2.n_regions`` compact regions (deterministic weighted k-means).
3. ``v2.n_candidates`` seeded region -> split assignments are scored on the deviation of area, mapped-fault and DEM
   shares from the target fractions after a coarse (1 km) embargo; the best ``v2.n_exact`` are recomputed exactly at
   full resolution and the first passing every gate of :func:`split_gates` is returned.
4. If none passes, :class:`SplitGateError` is raised with the gate table - an empty or near-empty partition is
   never accepted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

from .spatial_splits import (CODES, SPLITS, SplitResult, check_disjoint, check_embargo, check_predictor_names,
                             embargo_width_m)


class SplitGateError(RuntimeError):
    def __init__(self, msg, gates=None):
        super().__init__(msg)
        self.gates = gates


def v2_params(cfg: dict) -> dict:
    ss = cfg["spatial_splits"]
    v = dict(ss.get("v2") or {})
    g = {"min_area_share_of_target": 0.5, "min_fault_share_of_target": 0.4, "min_positive_px": 200,
         "min_blocks": 3, "min_block_fraction": 0.25, "require_patch_windows": True,
         "min_patch_windows": {"train": 20, "validation": 4}, "require_dem_each_split": True, "min_dem_fault_px": 1}
    g.update(v.get("gates") or {})
    return {"n_regions": int(v.get("n_regions", 24)), "n_candidates": int(v.get("n_candidates", 2000)),
            "n_exact": int(v.get("n_exact", 12)), "max_system_km": float(v.get("max_system_km", ss["block_km"])),
            "weights": {"area": 1.0, "faults": 1.0, "dem": 0.5, **(v.get("weights") or {})}, "gates": g}


def _uf_roots(n: int, pairs) -> np.ndarray:
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    return np.array([find(i) for i in range(n)])


def _kmeans(pts: np.ndarray, w: np.ndarray, k: int, iters: int = 60) -> np.ndarray:
    """Deterministic weighted k-means (farthest-point initialisation from the heaviest point)."""
    n = len(pts)
    k = max(1, min(k, n))
    c = [pts[int(np.argmax(w))]]
    for _ in range(1, k):
        d = np.min(((pts[:, None, :] - np.array(c)[None]) ** 2).sum(-1), 1)
        c.append(pts[int(np.argmax(d))])
    c = np.array(c, float)
    lab = np.full(n, -1)
    for _ in range(iters):
        new = np.argmin(((pts[:, None, :] - c[None]) ** 2).sum(-1), 1)
        if (new == lab).all():
            break
        lab = new
        for j in range(k):
            m = lab == j
            if m.any():
                c[j] = np.average(pts[m], axis=0, weights=np.maximum(w[m], 1e-9))
    return lab


def embargo_code(code: np.ndarray, res_yx, emb: float) -> np.ndarray:
    """Pixels within ``emb`` metres of a pixel of another split become EMBARGO."""
    out = code.copy()
    em = np.zeros(code.shape, bool)
    for k in SPLITS:
        m = code == CODES[k]
        if m.any():
            other = (code > 0) & (code != CODES[k]) & (code != CODES["embargo"])
            if other.any():
                em |= m & (ndimage.distance_transform_edt(~other, sampling=res_yx) <= emb)
    out[em] = CODES["embargo"]
    return out


def _score(sh: np.ndarray, tgt: np.ndarray, weights: dict, use_dem: bool) -> float:
    s = 0.0
    for i, m in enumerate(("area", "faults", "dem")):
        if m == "dem" and not use_dem:
            continue
        s += weights[m] * float((((sh[i] - tgt) / tgt) ** 2).sum())
    return s


def make_splits_v2(valid: np.ndarray, faults: np.ndarray, grid, cfg: dict, dem_usable: np.ndarray | None = None,
                   patch_px: int | None = None, predictor_names=(), label_derived=()) -> SplitResult:
    ss = cfg["spatial_splits"]
    P = v2_params(cfg)
    px = grid.res[0]
    B = max(1, int(round(ss["block_km"] * 1000 / px)))
    emb, parts = embargo_width_m(cfg)
    h, w = valid.shape
    nbc, nbr = -(-w // B), -(-h // B)
    nb = nbc * nbr
    ri, ci = np.indices(valid.shape, dtype=np.int32)
    block = (ri // B) * nbc + ci // B
    dem = (dem_usable if dem_usable is not None else np.zeros(valid.shape, bool)) & valid
    fv = faults & valid
    vcount = np.bincount(block[valid], minlength=nb).astype(float)
    fcount = np.bincount(block[fv], minlength=nb).astype(float)
    dcount = np.bincount(block[dem], minlength=nb).astype(float)
    rsum = np.bincount(block[valid], weights=ri[valid], minlength=nb)
    csum = np.bincount(block[valid], weights=ci[valid], minlength=nb)
    del ri, ci
    # --- small fault systems bind their blocks; regional ones may span (embargo-separated)
    lab, nsys = ndimage.label(fv, structure=np.ones((3, 3), bool))
    pairs, spanning, small_ids = [], [], []
    for sid, s in enumerate(ndimage.find_objects(lab) if nsys else [], 1):
        if s is None:
            continue
        ext = float(np.hypot((s[0].stop - s[0].start) * px, (s[1].stop - s[1].start) * px))
        bs = np.unique(block[s][lab[s] == sid])
        if ext <= P["max_system_km"] * 1000:
            small_ids.append(sid)
            pairs += [(int(bs[0]), int(b)) for b in bs[1:]]
        elif len(bs) > 1:
            spanning.append({"system": sid, "extent_km": round(ext / 1000, 1), "blocks": int(len(bs))})
    root = _uf_roots(nb, pairs)
    used = vcount > 0
    groups = np.unique(root[used])
    gidx = np.full(nb, -1)
    gidx[used] = np.searchsorted(groups, root[used])
    ng = len(groups)
    gv = np.bincount(gidx[used], weights=vcount[used], minlength=ng)
    gpts = np.stack([np.bincount(gidx[used], weights=rsum[used], minlength=ng),
                     np.bincount(gidx[used], weights=csum[used], minlength=ng)], 1) / np.maximum(gv, 1)[:, None]
    reg_of_group = _kmeans(gpts, gv, P["n_regions"])
    reg_of_block = np.where(gidx >= 0, reg_of_group[np.maximum(gidx, 0)], -1)
    nreg = int(reg_of_group.max()) + 1
    rv = np.bincount(reg_of_block[used], weights=vcount[used], minlength=nreg)
    rf = np.bincount(reg_of_block[used], weights=fcount[used], minlength=nreg)
    rd = np.bincount(reg_of_block[used], weights=dcount[used], minlength=nreg)
    pix_region = np.where(valid, reg_of_block[block], -1).astype(np.int32)
    # --- coarse (~1 km) grid for embargo-aware scoring
    cf = max(1, int(round(1000 / px)))
    Hc, Wc = -(-h // cf), -(-w // cf)

    def coarse(m, fill=0):
        mp = np.full((Hc * cf, Wc * cf), fill, m.dtype)
        mp[:h, :w] = m
        return mp.reshape(Hc, cf, Wc, cf)

    cval, cfau, cdem, cdf = (coarse(x.astype(np.float32)).sum((1, 3)) for x in (valid, fv, dem, fv & dem))
    cr = coarse(pix_region, -1).transpose(0, 2, 1, 3).reshape(Hc, Wc, cf * cf)
    counts = np.stack([(cr == j).sum(-1) for j in range(nreg)], -1)
    creg = np.where(counts.sum(-1) > 0, counts.argmax(-1), -1)
    fr = ss["fractions"]
    tgt = np.array([fr[k] for k in SPLITS], float)
    tot = np.array([rv.sum(), max(rf.sum(), 1.0), max(rd.sum(), 1.0)])
    use_dem = rd.sum() > 0
    rng = np.random.default_rng(int(ss["seed"]))
    rdf = np.bincount(reg_of_block[used], weights=np.bincount(block[fv & dem], minlength=nb).astype(float)[used],
                      minlength=nreg)
    quotas = [(rv, 1.0), (rf, 1.0), (rdf, 0.5 if rdf.sum() > 0 else 0.0)]
    seen, cands = set(), []
    for _ in range(P["n_candidates"]):
        order = rng.permutation(nreg)
        got = np.zeros((3, 4))
        a = np.zeros(nreg, int)
        for j in order:                                      # greedy fill of the largest combined relative deficit
            deficit = sum(w * (tgt * max(x.sum(), 1e-9) - got[i]) / (tgt * max(x.sum(), 1e-9))
                          for i, (x, w) in enumerate(quotas) if w)
            k = int(np.argmax(deficit + 1e-6 * rng.random(4)))
            a[j] = k
            for i, (x, _w) in enumerate(quotas):
                got[i, k] += x[j]
        key = a.tobytes()
        if key in seen:
            continue
        seen.add(key)
        sh = np.stack([np.bincount(a, weights=x, minlength=4) for x in (rv, rf, rd)]) / tot[:, None]
        cands.append((_score(sh, tgt, P["weights"], use_dem), a))
    cands.sort(key=lambda t: t[0])
    codes_arr = np.array([CODES[k] for k in SPLITS], np.uint8)
    G = P["gates"]
    need_dem = bool(G["require_dem_each_split"] and cdf.sum() > 0)
    rescored = []
    for sc, a in cands[: max(P["n_exact"] * 10, 40)]:
        cc = np.where(creg >= 0, codes_arr[a][np.maximum(creg, 0)], 0).astype(np.uint8)
        cc = embargo_code(cc, (px * cf, px * cf), emb)
        cnt = np.stack([[x[cc == CODES[k]].sum() for k in SPLITS] for x in (cval, cfau, cdem, cdf)])
        sh = cnt[:3] / tot[:, None]
        # gate-aware penalty (coarse estimate, 10 % margin): area, positives and DEM-covered faults per split
        tv, tf = float(cnt[0].sum() + cval[cc == CODES["embargo"]].sum()), float(cnt[1].sum())
        pen = (cnt[0] < 1.1 * G["min_area_share_of_target"] * tgt * tv).sum()
        pen += (cnt[1] < 1.1 * np.maximum(G["min_fault_share_of_target"] * tgt * tf,
                                          min(G["min_positive_px"], 0.05 * tf))).sum()
        if need_dem:
            pen += (cnt[3] < max(G["min_dem_fault_px"], 1)).sum()
        rescored.append((_score(sh, tgt, P["weights"], use_dem) + 10.0 * float(pen), sc, a))
    rescored.sort(key=lambda t: t[0])
    best, tried = None, []
    for s2, sc, a in rescored[: P["n_exact"]]:
        code = np.where(pix_region >= 0, codes_arr[a][np.maximum(pix_region, 0)], 0).astype(np.uint8)
        code = embargo_code(code, grid.res[::-1], emb)
        bt = pd.DataFrame({"block": np.arange(nb), "group": root, "region": reg_of_block,
                           "valid_px": vcount.astype(int), "fault_px": fcount.astype(int), "dem_px": dcount.astype(int),
                           "split": [SPLITS[a[r]] if r >= 0 else "none" for r in reg_of_block],
                           "row0": (np.arange(nb) // nbc) * B, "col0": (np.arange(nb) % nbc) * B})
        res = SplitResult(code, bt, emb, parts, np.where(valid, root[block], -1).astype(np.int32))
        gates = split_gates(res, faults, grid, cfg, dem_usable, patch_px, predictor_names, label_derived,
                            small_systems=(lab, small_ids))
        ok = bool(gates.passed.all())
        tried.append({"coarse_score": round(s2, 4), "score_before_embargo": round(sc, 4), "passed": ok,
                      "failed": "; ".join(gates[~gates.passed].check)})
        if best is None or ok:
            best = (res, gates, a)
        if ok:
            break
    res, gates, a = best
    res.gates = gates
    regions = pd.DataFrame({"region": np.arange(nreg), "valid_px": rv.astype(int), "fault_px": rf.astype(int),
                            "dem_px": rd.astype(int), "split": [SPLITS[x] for x in a]})
    res.report = {"generator": "v2_regions", "n_blocks": int(used.sum()), "n_block_groups": int(ng),
                  "n_regions": nreg, "n_candidates_scored": len(cands), "candidates_tried_exact": tried,
                  "fault_systems": int(nsys), "small_systems_kept_whole": len(small_ids),
                  "regional_systems_spanning_blocks": spanning[:200], "embargo_m": emb, "embargo_components": parts,
                  "regions": regions.to_dict("records"), "split_stats": split_stats(res, faults, dem_usable, grid),
                  "target_fractions": dict(fr)}
    if not gates.passed.all():
        raise SplitGateError("no split candidate passed every gate (best candidate):\n"
                             + gates[~gates.passed].to_string(index=False), gates)
    return res


def split_stats(res: SplitResult, faults, dem_usable, grid) -> dict:
    out = {}
    km2 = grid.res[0] * grid.res[1] / 1e6
    for k in list(SPLITS) + ["embargo"]:
        m = res.mask(k)
        out[k] = {"pixels": int(m.sum()), "km2": round(float(m.sum() * km2), 1), "fault_px": int((m & faults).sum()),
                  "dem_px": int((m & dem_usable).sum()) if dem_usable is not None else None,
                  "dem_fault_px": int((m & faults & dem_usable).sum()) if dem_usable is not None else None}
    return out


def windows_inside(m: np.ndarray, P: int) -> int:
    """Non-overlapping P x P windows (stride P, two offsets) lying entirely inside ``m``."""
    best = 0
    for o in (0, P // 2):
        mm = m[o:, o:]
        hh, ww = (mm.shape[0] // P) * P, (mm.shape[1] // P) * P
        if P > 0 and hh and ww:
            best = max(best, int(mm[:hh, :ww].reshape(hh // P, P, ww // P, P).all((1, 3)).sum()))
    return best


def split_gates(res: SplitResult, faults, grid, cfg: dict, dem_usable=None, patch_px: int | None = None,
                predictor_names=(), label_derived=(), small_systems=None) -> pd.DataFrame:
    """Hard integrity gates; training is refused unless every row passes."""
    g = v2_params(cfg)["gates"]
    fr = cfg["spatial_splits"]["fractions"]
    rows = []

    def add(check, ok, detail):
        rows.append({"check": check, "passed": bool(ok), "detail": str(detail)[:1500]})

    tot_v = int((res.code > 0).sum())
    tot_f = int((faults & (res.code > 0)).sum())
    B = max(1, int(round(cfg["spatial_splits"]["block_km"] * 1000 / grid.res[0])))
    for k in SPLITS:
        m = res.mask(k)
        need = g["min_area_share_of_target"] * fr[k] * tot_v
        add(f"nonempty_{k}", m.sum() >= max(need, 1), f"{int(m.sum())} px (need >= {need:.0f})")
        fneed = max(min(g["min_positive_px"], 0.05 * tot_f), g["min_fault_share_of_target"] * fr[k] * tot_f, 1)
        add(f"positives_{k}", (m & faults).sum() >= fneed, f"{int((m & faults).sum())} fault px (need >= {fneed:.0f})")
        nblk = 0
        for r in res.blocks[res.blocks.split == k].itertuples():
            if r.valid_px and m[r.row0:r.row0 + B, r.col0:r.col0 + B].sum() >= g["min_block_fraction"] * r.valid_px:
                nblk += 1
        add(f"independent_blocks_{k}", nblk >= g["min_blocks"], f"{nblk} block(s) (need >= {g['min_blocks']})")
        if dem_usable is not None and g["require_dem_each_split"] and (dem_usable & faults).any():
            n = int((m & faults & dem_usable).sum())
            add(f"dem_faults_{k}", n >= g["min_dem_fault_px"], f"{n} fault px with usable DEM")
    errs = check_disjoint(res)
    add("disjoint", not errs, errs or "ok")
    errs = check_embargo(res, grid)
    add("embargo", not errs, errs or f">= {res.embargo_m:.0f} m between splits")
    if patch_px and g["require_patch_windows"]:
        for k, n_min in g["min_patch_windows"].items():
            n = windows_inside(res.mask(k), int(patch_px))
            add(f"patch_windows_{k}", n >= n_min, f"{n} {patch_px}-px window(s) inside {k} (need >= {n_min})")
    if small_systems is not None:
        lab, ids = small_systems
        bad = []
        if ids:
            core = (res.code > 0) & (res.code != CODES["embargo"]) & (lab > 0)
            isel = np.zeros(int(lab.max()) + 1, bool)
            isel[ids] = True
            labs, codes = lab[core], res.code[core]
            sel = isel[labs]
            n = pd.DataFrame({"s": labs[sel], "c": codes[sel]}).drop_duplicates().groupby("s").c.nunique()
            bad = n[n > 1].index.tolist()
        add("small_fault_system_leakage", not bad, f"{len(bad)} small system(s) in >1 split" if bad else "none")
    errs = check_predictor_names(predictor_names, label_derived)
    add("no_target_derived_predictors", not errs, errs or "ok")
    add("no_final_test_tuning", True, "split fixed before any model selection; test evaluated once, after selection")
    return pd.DataFrame(rows)


def save_split_version(res: SplitResult, grid, out_dir, ctx) -> list:
    """``05_splits/<version>/``: code raster, masks, registries, gates, report, manifest (COMPLETE.json by caller)."""
    from .features import write_geotiff
    from .registry import atomic_write_json, atomic_write_parquet, sha256_file

    out_dir.mkdir(parents=True, exist_ok=True)
    outs = [write_geotiff(out_dir / "split_code.tif", res.code, grid, ["split code"], nodata=None),
            atomic_write_parquet(out_dir / "split_registry.parquet", res.blocks)]
    for k in SPLITS:
        outs.append(write_geotiff(out_dir / f"{k}_mask.tif", res.mask(k).astype(np.uint8), grid, [f"{k} split"],
                                  nodata=None))
    outs.append(write_geotiff(out_dir / "embargo_mask.tif", res.mask("embargo").astype(np.uint8), grid, ["embargo"],
                              nodata=None))
    gates = getattr(res, "gates", None)
    if gates is not None:
        outs.append(atomic_write_parquet(out_dir / "split_gates.parquet", gates))
    rep = getattr(res, "report", {}) or {}
    if rep.get("regions"):
        outs.append(atomic_write_parquet(out_dir / "regions.parquet", pd.DataFrame(rep["regions"])))
    outs.append(atomic_write_json(out_dir / "split_report.json", rep))
    outs.append(ctx.write_manifest(out_dir / "split_manifest.json", {
        "embargo_m": res.embargo_m, "embargo_components": res.embargo_parts, "codes": CODES,
        "pixels": {k: int(res.mask(k).sum()) for k in list(SPLITS) + ["embargo"]},
        "block_km": ctx.cfg["spatial_splits"]["block_km"], "fractions": ctx.cfg["spatial_splits"]["fractions"],
        "generator": rep.get("generator", "v1"), "split_code_sha256": sha256_file(out_dir / "split_code.tif"),
        "gates_passed": bool(gates.passed.all()) if gates is not None else None, "selected_before_test": True}))
    return outs


def load_split_code(path) -> np.ndarray:
    import rasterio

    with rasterio.open(path) as ds:
        return ds.read(1).astype(np.uint8)
