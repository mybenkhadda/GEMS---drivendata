"""Section 09: geographic and discovery validation of the dense channel (CV-A ... CV-F), Gates 2 and 3.

Held-out predictions (``heldout``): every trained unit predicts ONLY its own held-out evaluation mask
    geo_f<k>   -> unit geo:<k>            no map context (the whole mapped network of the fold is the truth)
    disc_s<d>  -> unit disc_official:<d>  context none (independent discovery) AND known (retained faults, the
                                          test-time situation: the known map is provided with the data)
Raw heads are stored as float16 crops (``predictions/<variant>/<task>_<zone>__<mode>.npz``) and as GeoTIFFs.

Methods (held-out maps saved under ``validation/maps/<method>/`` with protocol keys; the M7 comparison re-scores
every one of them with the official FastScorer on the same units, eligible budget and masks)::

    m9_dense_<variant>          geo: no context; disc: ``validation.production_context`` (default known)
    m9_dense_<variant>_noctx    geo: identical map; disc: no context (independent-discovery control, Gate 3)

Representation (binary top / crest 1 px) and area fraction are chosen per zone on the OTHER zones
(``nested_select``). With a single trained zone (PILOT) nothing can be nested: the champion's all-zone (rep, frac)
choice for the same task is borrowed and the result is labelled ``selection = borrowed_from_champion``.

Diagnostics: CV-A per-fold / worst-fold DTI, CV-B system recall, CV-C reconstruction of synthetically damaged
geometry (context = damaged map vs none), CV-D recall by system length, CV-E DTI by label-free geological domain,
CV-F recall / FP mass by distance to known faults, block-bootstrap uncertainty (spatially grouped, never pixels).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MODES = ("none", "known")


# =============================================================================== held-out predictions
def _mask_full(m9, u) -> np.ndarray:
    m = np.zeros(m9.ctx.data.shape, bool)
    m[u.sl] = u.ev
    return m


def _ctx_known(m9, d: int) -> np.ndarray:
    lab = np.asarray(m9.ctx.data.labels, bool)
    return (lab & ~m9.ctx.splits.withheld(d)).astype(np.float32)


def pred_path(m9, vname: str, u, mode: str):
    return m9.ws.root / "predictions" / vname / f"{u.task}_{u.zone}__{mode}.npz"


def load_pred(m9, vname: str, u, mode: str) -> dict | None:
    p = pred_path(m9, vname, u, mode)
    if not p.exists():
        return None
    z = np.load(p)
    return {h: z[h].astype(np.float32) for h in ("seg", "ocos", "osin", "ep", "jn") if h in z}


def heldout(m9, vname: str) -> pd.DataFrame:
    """Predict every trained unit's held-out mask (both context modes on disc); returns an index table."""
    from .infer import predict_mask
    from .train import load_unit

    vcfg = m9.variant_cfg(vname)
    meta, cube, pooled = m9.cube_for(vcfg)
    dc, tc = vcfg["data"], vcfg["training"]
    P = int(dc["patch_px"])
    kw = dict(P=P, f=int(dc["context_factor"]), stride=int(P * float(tc["stride_frac"])), amp=bool(tc["amp"]),
              batch=int(tc["infer_batch"]))
    rows = []
    jobs = [("geo", k, f"geo_f{k}") for k in m9.train_folds()] + \
           [(m9.prim, d, f"disc_s{d}") for d in m9.train_splits()]
    for task, z, unit in jobs:
        u = m9.unit(f"{task}:{z}")
        if u is None:
            continue
        modes = ("none",) if task == "geo" else MODES
        if all(pred_path(m9, vname, u, md).exists() for md in modes) and not m9.cfg.get("overwrite_existing"):
            rows += [{"variant": vname, "task": task, "zone": z, "mode": md, "status": "reused"} for md in modes]
            continue
        lu = load_unit(m9, vname, unit)
        if lu is None:
            rows += [{"variant": vname, "task": task, "zone": z, "mode": md, "status": "model missing"} for md in modes]
            continue
        model, sc, psc, obj = lu
        dev = next(model.parameters()).device
        full = _mask_full(m9, u)
        for md in modes:
            cm = _ctx_known(m9, z) if md == "known" else None
            out = predict_mask([(model, sc, psc)], cube, pooled, full, ctx_map=cm, flag=1.0 if md == "known" else 0.0,
                               device=dev, **kw)
            arr = {h: np.nan_to_num(a[u.sl]).astype(np.float16) for h, a in out.items()}
            m9.ws.save_npz({**arr, "protocol_key": np.array(m9.pkey(u)), "checkpoint_sha": np.array(str(obj.get("spec_hash", "")))},
                           f"predictions/{vname}/{task}_{z}__{md}.npz")
            rows.append({"variant": vname, "task": task, "zone": z, "mode": md, "status": "predicted",
                         "seg_mean_ev": float(np.nanmean(out["seg"][full]))})
        del model
    return pd.DataFrame(rows)


def stitch_rasters(m9, vname: str) -> None:
    """GeoTIFFs of the held-out seg maps: OOF (geo folds) and one band per (split, context mode)."""
    H, W = m9.ctx.data.shape
    oof = np.full((H, W), np.nan, np.float32)
    disc = {}
    for u in m9.units_of("geo"):
        p = load_pred(m9, vname, u, "none")
        if p is not None:
            sub = oof[u.sl]
            sub[u.ev] = p["seg"][u.ev]
    for u in m9.units_of(m9.prim):
        for md in MODES:
            p = load_pred(m9, vname, u, md)
            if p is not None:
                a = np.full((H, W), np.nan, np.float32)
                a[u.sl] = np.where(u.ev, p["seg"], np.nan)
                disc[f"disc{u.zone}_{md}"] = a
    if np.isfinite(oof).any():
        m9.ws.save_raster({"oof_fault_evidence": oof}, f"predictions/oof/{vname}_seg.tif", m9.ctx.data.grid)
    if disc:
        m9.ws.save_raster(disc, f"predictions/discovery/{vname}_seg.tif", m9.ctx.data.grid)


# =============================================================================== nested selection + held-out maps
def grid_rows(m9, vname: str, mode_disc: str, method: str) -> pd.DataFrame:
    from ..gems3 import representation as R
    from ..m7.protocol import score_row

    reps = m9.cfg["validation"]["representations"]
    sigma = float(m9.sh["evaluation"]["ridge_sigma_px"])
    rows = []
    for task in ("geo", m9.prim):
        for u in m9.units_of(task):
            p = load_pred(m9, vname, u, "none" if task == "geo" else mode_disc)
            if p is None:
                continue
            s = R.rank01(p["seg"], u.ev)
            el = m9.elig(u, 0)
            crest = R.crest_map(s, u.ev, sigma) if any(r.get("select") == "crest" for r in reps.values()) else None
            for name, rep in reps.items():
                for f in m9.cfg["validation"]["area_fracs"]:
                    q = R.apply(rep, s, u.ev, float(f), known=u.known, crest=crest, sigma_px=sigma, eligible=el)
                    rows.append(score_row(u, q, el, m9.pkey(u), method=method, rep=name, frac=float(f)))
    return pd.DataFrame(rows)


def borrowed_choice(m9, task: str) -> dict | None:
    """Champion's all-zone (rep, frac) of the same task (PILOT fallback when nothing can be nested)."""
    sel = m9.baseline_selection() or {}
    ch = (sel.get(task) or {}).get(m9.cfg["baseline"]["champion"])
    if not ch:
        return None
    pa = ch["params_all"]
    rep = "crest_1px" if pa.get("rep") in ("crest_1px", "skeleton_1px") else "binary"
    return {"rep": rep, "frac": float(pa["frac"])}


def select_and_save(m9, grid: pd.DataFrame, method: str, score_of) -> tuple[pd.DataFrame, dict]:
    """Nested (rep, frac) per zone; the held-out map of every zone is rebuilt with its own choice and saved.
    ``score_of(u)`` returns the unit's rank score (or None)."""
    from ..gems3 import representation as R
    from ..gems3.metric_opt import nested_select
    from ..m7.rasterize import save_heldout

    reps = m9.cfg["validation"]["representations"]
    sigma = float(m9.sh["evaluation"]["ridge_sigma_px"])
    per_all, summ = [], {}
    for task in ("geo", m9.prim):
        g = grid[(grid.task == task) & (grid.method == method)] if len(grid) else grid
        if not len(g):
            continue
        per, s = nested_select(g, ["rep", "frac"])
        how = "nested (other zones)"
        if not len(per):
            ch = borrowed_choice(m9, task)
            if ch is None:
                continue
            fr = np.asarray(sorted(g.frac.unique()), float)
            ch["frac"] = float(fr[np.argmin(np.abs(fr - ch["frac"]))])         # nearest grid fraction
            per = g[(g.rep == ch["rep"]) & (np.isclose(g.frac, ch["frac"]))][["zone", "rep", "frac", "tp_w", "fp_w",
                                                                             "fn_w", "n_truth", "dti"]].copy()
            per = per.rename(columns={"dti": "dti_heldout"})
            how = "borrowed_from_champion (single zone: nothing to nest)"
            s = {"dti_crossfit": float(per.dti_heldout.mean()) if len(per) else np.nan, "params_all": ch}
        for r in per.itertuples():
            u = m9.unit(f"{task}:{int(r.zone)}")
            sc = score_of(u)
            if sc is None:
                continue
            q = R.apply(reps[r.rep], sc, u.ev, float(r.frac), known=u.known, sigma_px=sigma, eligible=m9.elig(u, 0))
            save_heldout(m9.ws, method, u, q, m9.pkey(u))
        per_all.append(per.assign(task=task, method=method, selection=how))
        summ[task] = {**{k: v for k, v in s.items() if k != "params_all"}, "params_all": s.get("params_all"),
                      "selection": how}
    return (pd.concat(per_all, ignore_index=True) if per_all else pd.DataFrame()), summ


def dense_methods(m9, vname: str) -> dict:
    """Grid + nested held-out maps of m9_dense_<v> (production context) and m9_dense_<v>_noctx."""
    from ..gems3 import representation as R

    prod = m9.cfg["validation"]["production_context"]
    out = {}
    for method, mode in ((f"m9_dense_{vname}", prod), (f"m9_dense_{vname}_noctx", "none")):
        g = grid_rows(m9, vname, mode, method)

        def score_of(u, mode=mode):
            p = load_pred(m9, vname, u, "none" if u.task == "geo" else mode)
            return None if p is None else R.rank01(p["seg"], u.ev)
        per, summ = select_and_save(m9, g, method, score_of)
        out[method] = {"grid": g, "per_zone": per, "summary": summ}
    return out


# =============================================================================== CV-C reconstruction benchmark
def reconstruction(m9, vname: str) -> pd.DataFrame:
    """Recovery of synthetically removed geometry on each trained fold's validation zone: context = damaged map
    (the model never saw these systems) vs no context, at several area fractions of the class zone."""
    from ..gems3.representation import top_mask
    from .corruption import benchmark_units, load_sets
    from .infer import predict_mask
    from .train import load_unit

    sets = load_sets(m9.ws)
    if not sets:
        return pd.DataFrame()
    vcfg = m9.variant_cfg(vname)
    meta, cube, pooled = m9.cube_for(vcfg)
    dc, tc = vcfg["data"], vcfg["training"]
    P = int(dc["patch_px"])
    kw = dict(P=P, f=int(dc["context_factor"]), stride=int(P * float(tc["stride_frac"])), amp=bool(tc["amp"]),
              batch=int(tc["infer_batch"]))
    rows = []
    for k in sorted(sets):
        lu = load_unit(m9, vname, f"geo_f{k}")
        if lu is None:
            continue
        model, sc, psc, _ = lu
        dev = next(model.parameters()).device
        units = benchmark_units(m9, k, sets)
        if not units:
            continue
        zone = np.zeros(m9.ctx.data.shape, bool)
        for u in units:
            zone[u.sl] |= u.ev
        preds = {md: predict_mask([(model, sc, psc)], cube, pooled, zone,
                                  ctx_map=sets[k]["damaged"].astype(np.float32) if md == "damaged" else None,
                                  flag=1.0 if md == "damaged" else 0.0, device=dev, **kw)["seg"]
                 for md in ("damaged", "none")}
        for u in units:
            for md, s in preds.items():
                sc_u = np.nan_to_num(s[u.sl])
                dom = u.scored_mask()
                for f in (0.01, 0.02, 0.05):
                    q = top_mask(sc_u, dom, f).astype(np.float32)
                    c = u.components(q)
                    rows.append({"variant": vname, "fold": k, "class": u.task.replace("cvc_", ""), "context": md,
                                 "frac": f, "recall_w": c.tp / max(c.n_truth, 1), "dti": c.dti, "truth_px": c.n_truth})
            u.release()
        del model
    return pd.DataFrame(rows)


# =============================================================================== CV-D / CV-E
def length_strata(m9, systems: pd.DataFrame) -> pd.DataFrame:
    if not len(systems):
        return pd.DataFrame()
    km = m9.ctx.splits.systems.set_index("system_id").trace_km
    bins = [0.0] + list(m9.cfg["data"]["length_bins_km"]) + [np.inf]
    d = systems.copy()
    d["trace_km"] = km.reindex(d.system_id).to_numpy()
    d["length_bin_km"] = pd.cut(d.trace_km, bins, right=False).astype(str)
    return (d.groupby(["task", "method", "length_bin_km"])
             .agg(n_systems=("system_id", "size"), system_recall=("recovered", "mean"),
                  support_mean=("support_mean", "mean")).reset_index())


def domain_dti(m9, methods: list[str]) -> pd.DataFrame:
    """CV-E: DTI of each method's geo held-out maps inside each label-free geological domain."""
    from ..gems3.official import FastScorer
    from ..m7.rasterize import load_heldout

    dom = m9.domains()
    if dom is None:
        return pd.DataFrame()
    rows = []
    for mth in methods:
        for u in m9.units_of("geo"):
            q = load_heldout(m9.ws, mth, u)
            if q is None:
                continue
            dm = dom[u.sl]
            for d in np.unique(dm[u.ev & (dm >= 0)]):
                ev = u.ev & (dm == d)
                G = u.eval_truth() & ev
                if not G.any():
                    continue
                c = FastScorer(G, ev).components(np.asarray(q, np.float64))
                rows.append({"method": mth, "zone": u.zone, "domain": int(d), "tp_w": c.tp, "fp_w": c.fp, "fn_w": c.fn,
                             "n_truth": c.n_truth, "dti": c.dti})
    D = pd.DataFrame(rows)
    if not len(D):
        return D
    g = D.groupby(["method", "domain"])[["tp_w", "fp_w", "fn_w", "n_truth"]].sum().reset_index()
    g["dti"] = g.tp_w / (g.tp_w + 0.2 * g.fp_w + 0.8 * g.fn_w + 1e-6)
    return g


# =============================================================================== gates 2 / 3
def unique_systems(systems: pd.DataFrame, a: str, b: str, task: str) -> int:
    if not len(systems):
        return 0
    s = systems[systems.task == task]
    piv = s.pivot_table(index=["zone", "system_id"], columns="method", values="recovered", aggfunc="max")
    if a not in piv or b not in piv:
        return 0
    return int((piv[a].fillna(False).astype(bool) & ~piv[b].fillna(False).astype(bool)).sum())


def gate2(m9, verdicts: pd.DataFrame, systems: pd.DataFrame, fusion_gain: float | None = None) -> pd.DataFrame:
    g = m9.cfg["gates"]["g2_dense"]
    ref = f"s01_{m9.cfg['baseline']['champion']}"
    m = "m9_dense_core"
    v = verdicts[verdicts.method == m] if len(verdicts) else verdicts
    verdict = str(v.verdict.iloc[0]) if len(v) else "NOT_EVALUATED"
    uq = unique_systems(systems, m, ref, m9.prim)
    if verdict == "NOT_EVALUATED":
        dec = "NOT_EVALUATED"
    elif verdict == "PROMOTE":
        dec = "PROMOTE"
    elif uq >= int(g["min_unique_systems"]) or (fusion_gain is not None and fusion_gain >= float(g["min_fusion_gain"])):
        dec = "COMPLEMENT"
    else:
        dec = "FAIL"
    return pd.DataFrame([{"gate": "G2_dense", "task": m9.prim, "decision": dec, "method": m, "verdict_vs_champion": verdict,
                          "unique_systems_vs_champion": uq, "fusion_gain": fusion_gain, "profile": m9.profile["name"],
                          "provisional": bool(m9.profile["provisional"]),
                          "rule": f"PROMOTE if the dense verdict is PROMOTE; COMPLEMENT if >= {g['min_unique_systems']} "
                                  f"unique systems or a fusion gain >= {g['min_fusion_gain']}"}])


def gate3(m9, pooled: pd.DataFrame, recon: pd.DataFrame) -> pd.DataFrame:
    g = m9.cfg["gates"]["g3_reconstruction"]
    base = {"gate": "G3_reconstruction", "task": m9.prim, "profile": m9.profile["name"],
            "provisional": bool(m9.profile["provisional"])}
    if "recon_off" not in m9.variants() or not len(pooled):
        return pd.DataFrame([{**base, "decision": "NOT_EVALUATED",
                              "reason": "variant recon_off not trained in this profile (add it to profiles.<P>.variants)"}])

    def dti(task, method):
        r = pooled[(pooled.task == task) & (pooled.method == method)]
        return float(r.dti_pooled.iloc[0]) if len(r) else np.nan

    disc_drop = dti(m9.prim, "m9_dense_recon_off_noctx") - dti(m9.prim, "m9_dense_core_noctx")
    geo_drop = dti("geo", "m9_dense_recon_off") - dti("geo", "m9_dense_core")
    rg = np.nan
    if len(recon):
        r = recon[(recon.context == "damaged") & (recon.frac == 0.02)].groupby("variant").recall_w.mean()
        if "core" in r and "recon_off" in r:
            rg = float(r["core"] - r["recon_off"])
    ok = (np.isfinite(rg) and rg >= float(g["min_recall_gain"]) and (not np.isfinite(disc_drop) or
          disc_drop <= float(g["max_discovery_drop"])) and (not np.isfinite(geo_drop) or geo_drop <= float(g["max_geo_drop"])))
    return pd.DataFrame([{**base, "decision": "RETAIN_RECONSTRUCTION" if ok else "DROP_RECONSTRUCTION",
                          "reconstruction_recall_gain": rg, "independent_discovery_drop": disc_drop, "geo_drop": geo_drop,
                          "rule": f"recall gain >= {g['min_recall_gain']} AND discovery drop <= {g['max_discovery_drop']} "
                                  f"AND geo drop <= {g['max_geo_drop']}"}])


def variant_deltas(pooled: pd.DataFrame) -> pd.DataFrame:
    """Every dense variant against core (same protocol, same zones): Stage C / spectral / topology ablations."""
    if not len(pooled):
        return pd.DataFrame()
    d = pooled[pooled.method.str.startswith("m9_dense_")].copy()
    rows = []
    for task, g in d.groupby("task"):
        ref = g[g.method == "m9_dense_core"]
        if not len(ref):
            continue
        for r in g.itertuples():
            if r.method == "m9_dense_core":
                continue
            rows.append({"task": task, "method": r.method, "dti_pooled": r.dti_pooled,
                         "delta_vs_core": r.dti_pooled - float(ref.dti_pooled.iloc[0]),
                         "worst_zone_delta": r.worst_zone_dti - float(ref.worst_zone_dti.iloc[0])})
    return pd.DataFrame(rows)
