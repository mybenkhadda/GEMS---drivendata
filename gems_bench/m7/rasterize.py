"""Section 12: metric-aware rasterisation of ranked geological candidates (exact competition grid).

Budget: candidate pixels are ranked by the candidate score (max where lines cross) and the top ``frac`` x
|eligible| pixels are kept (pixel-level budget; a long weak line can be cut). The eligible domain excludes template
nodata, pixel-exact known faults and the optional ring, so masked pixels never consume budget. A representation
then shapes the kept core; its area after shaping is recorded (``area_frac_eligible``), and nested selection chooses
(representation, confidence blend, frac, ring) on the other zones.

Representations (``rasterization.representations``)::

    centerline        1-px lines (value 1)
    corridor          core 1, then max(0, 1 - d / (w + 1)) up to ``width_px`` (narrow, distance-decaying)
    confidence_width  the top ``hi_quantile`` of kept pixels get ``extra_px`` more width (binary)
    antialiased       sub-pixel lines: each kept candidate is smoothed (3-point running mean) and drawn with
                      anti-aliased segments (``skimage.draw.line_aa``); value = coverage of the pixel by the line

The official kernel tolerance (300 m) is NOT a reason to draw 300 m wide faults: width is an experimental
parameter, and thin lines have won in every earlier experiment. Confidence blend: w1 rank(ranker) + w2
rank(label-free structural score) + w3 rank(neural support), per unit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

from ..gems3.representation import top_mask
from ..gems3.submission import ring_zone


def rank_pct(x) -> np.ndarray:
    s = pd.Series(np.asarray(x, float))
    return s.rank(pct=True).fillna(0.0).to_numpy()


def blend(scores: dict, w) -> np.ndarray:
    """w = (ranker, label_free, neural) weights over per-unit percentile ranks; missing parts get their weight
    redistributed to the others."""
    parts = [("ranker", w[0]), ("label_free", w[1]), ("neural", w[2])]
    have = [(k, float(x)) for k, x in parts if x > 0 and scores.get(k) is not None]
    if not have:
        return rank_pct(scores.get("ranker", scores.get("label_free")))
    tot = sum(x for _, x in have)
    return sum(x / tot * rank_pct(scores[k]) for k, x in have)


def score_raster(cs, scores: np.ndarray, shape) -> np.ndarray:
    s = np.full(shape, -np.inf, np.float32)
    for p, v in zip(cs.paths, np.asarray(scores, np.float32)):
        np.maximum.at(s, (p[:, 0], p[:, 1]), v)
    return s


def select_core(sr: np.ndarray, elig: np.ndarray, frac: float) -> np.ndarray:
    on = np.isfinite(sr) & elig
    if not on.any():
        return np.zeros(sr.shape, bool)
    k = int(np.ceil(frac * int(elig.sum())))
    return top_mask(np.where(on, sr, -np.inf), on, min(1.0, k / max(int(on.sum()), 1)))


def _antialiased(cs, scores_px_mask: np.ndarray, shape) -> np.ndarray:
    from skimage.draw import line_aa

    out = np.zeros(shape, np.float32)
    for p in cs.paths:
        hit = scores_px_mask[p[:, 0], p[:, 1]]
        if not hit.any():
            continue
        q = p[hit].astype(float)
        if len(q) >= 3:
            q = np.vstack([q[0], (q[:-2] + q[1:-1] + q[2:]) / 3.0, q[-1]])
        for a, b in zip(q[:-1], q[1:]):
            rr, cc, val = line_aa(int(round(a[0])), int(round(a[1])), int(round(b[0])), int(round(b[1])))
            ok = (rr >= 0) & (rr < shape[0]) & (cc >= 0) & (cc < shape[1])
            np.maximum.at(out, (rr[ok], cc[ok]), val[ok].astype(np.float32))
    return out


def shape_core(core: np.ndarray, sr: np.ndarray, rep: dict, cs=None) -> np.ndarray:
    from ..postprocess.topology import _disk

    kind = rep.get("kind", "centerline")
    if kind == "centerline" or not core.any():
        return core.astype(np.float32)
    if kind == "corridor":
        w = int(rep.get("width_px", 1))
        d = ndimage.distance_transform_edt(~core)
        q = np.where(d <= w, 1.0 - d / (w + 1.0), 0.0)
        return np.where(core, 1.0, q).astype(np.float32)
    if kind == "confidence_width":
        thr = np.quantile(sr[core], float(rep.get("hi_quantile", 0.7)))
        hi = core & (sr >= thr)
        return (core | ndimage.binary_dilation(hi, structure=_disk(int(rep.get("extra_px", 1))))).astype(np.float32)
    if kind == "antialiased":
        return core.astype(np.float32) if cs is None else _antialiased(cs, core, core.shape)
    raise KeyError(kind)


def predict(u, cs, scores: np.ndarray, rep: dict, frac: float, ring_px: int, elig_of) -> tuple[np.ndarray, list]:
    """(prediction on the unit crop, area ledger) of ranked candidates."""
    from .protocol import unit_ledger

    el = elig_of(u, ring_px)
    sr = score_raster(cs, scores, u.ev.shape)
    core = select_core(sr, el, frac)
    led = [unit_ledger(core, u, el, "core selection (top-frac of eligible candidate pixels)", ring_px)]
    q = shape_core(core, sr, rep, cs)
    led.append(unit_ledger(q, u, el, f"representation {rep.get('kind')}", ring_px))
    if ring_px and u.known is not None:
        q = np.where(ring_zone(u.known, ring_px), 0.0, q).astype(np.float32)
        led.append(unit_ledger(q, u, el, f"ring {ring_px} px", ring_px))
    q = np.where(u.ev, q, 0.0).astype(np.float32)
    return q, led


def hybrid(u, base_q: np.ndarray, cs, scores: np.ndarray, frac: float, share: float, ring_px: int, elig_of) -> np.ndarray:
    """Union of the pixel-fusion map (budget (1 - share) frac, already selected) and candidate centerlines with the
    remaining share, chosen outside the fusion pixels so the total budget stays exact."""
    el = elig_of(u, ring_px) & ~(base_q > 0)
    sr = score_raster(cs, scores, u.ev.shape)
    k_total = frac * int(elig_of(u, ring_px).sum())
    f_c = share * k_total / max(int(el.sum()), 1)
    core = select_core(sr, el, f_c)
    q = np.maximum(base_q, core.astype(np.float32))
    if ring_px and u.known is not None:
        q = np.where(ring_zone(u.known, ring_px), 0.0, q)
    return np.where(u.ev, q, 0.0).astype(np.float32)


# =============================================================================== held-out map persistence
def save_heldout(ws, method: str, u, q: np.ndarray, pkey: str = "") -> None:
    """Sparse (or dense float16 when soft and dense) crop of a held-out prediction: validation/maps/<method>/,
    stamped with the protocol key it was built under (section 13 refuses stale or foreign maps)."""
    q = np.asarray(q, np.float32)
    pos = q > 0
    rel = f"validation/maps/{method}/{u.key.replace(':', '_')}.npz"
    key = np.array(str(pkey))
    if pos.mean() > 0.05:
        ws.save_npz({"dense": q.astype(np.float16), "shape": np.array(q.shape), "protocol_key": key}, rel)
    else:
        rr, cc = np.nonzero(pos)
        ws.save_npz({"r": rr.astype(np.int32), "c": cc.astype(np.int32), "v": q[rr, cc].astype(np.float16),
                     "shape": np.array(q.shape), "protocol_key": key}, rel)


def load_heldout(ws, method: str, u, with_key: bool = False):
    p = ws.root / "validation" / "maps" / method / f"{u.key.replace(':', '_')}.npz"
    if not p.exists():
        return (None, "") if with_key else None
    z = np.load(p)
    key = str(z["protocol_key"]) if "protocol_key" in z else ""
    if "dense" in z:
        q = z["dense"].astype(np.float32)
    else:
        q = np.zeros(tuple(z["shape"]), np.float32)
        q[z["r"], z["c"]] = z["v"].astype(np.float32)
    return (q, key) if with_key else q


def saved_methods(ws) -> list[str]:
    d = ws.root / "validation" / "maps"
    return sorted(p.name for p in d.iterdir() if p.is_dir()) if d.exists() else []


# =============================================================================== section 12 experiment
CONTROLS = ("label_free", "neural_support")


def best_ranker(summary: pd.DataFrame, task: str) -> str | None:
    s = summary[(summary.task == task) & ~summary.method.isin(CONTROLS)] if len(summary) else summary
    if not len(s):
        return None
    return str(s.sort_values("dti_crossfit", ascending=False).method.iloc[0])


def experiment(m7, rk: dict, sets: dict) -> dict:
    """Representation x confidence blend x budget x ring of the best ranker (and the label-free control), plus the
    hybrid with the section-01 fusion; every choice nested over zones."""
    from ..gems3.metric_opt import nested_select
    from .baseline import MemberMaps
    from .baseline import predict as fusion_predict
    from .protocol import score_row

    cfg = m7.cfg
    rz, ev = cfg["rasterization"], cfg["evaluation"]
    prim = ev["primary_task"]
    T = rk["table"]
    best = best_ranker(rk["summary"], prim) or "label_free"
    rankers = list(dict.fromkeys([best, "label_free"]))
    reps = rz["representations"]
    sel01 = m7.baseline_selection() or {}
    mm = MemberMaps(m7.ctx, m7.models)
    rows = []
    for task in ("geo", prim):
        for u in m7.units_of(task):
            cs = sets.get(u.key)
            if cs is None or not len(cs):
                continue
            sub = T[(T.task == u.task) & (T.zone == u.zone)].sort_values("cand_id")
            rings = [int(r) for r in rz["ring_px"]] if u.is_disc else [0]
            for rk_name in rankers:
                sc = {"ranker": sub[f"score_{rk_name}"].to_numpy(), "label_free": sub.score_label_free.to_numpy(),
                      "neural": sub.score_neural_support.to_numpy()}
                for fw in rz["fusion_weights"]:
                    s = blend(sc, fw)
                    for name, rep in reps.items():
                        for f in rz["area_fracs"]:
                            for ring in rings:
                                q, _ = predict(u, cs, s, rep, float(f), ring, m7.elig)
                                rows.append(score_row(u, q, m7.elig(u, ring), m7.pkey(u), method=f"cand_{rk_name}",
                                                      rep=name, fw=json_fw(fw), frac=float(f), ring=int(ring),
                                                      share=np.nan))
            z = next((r for r in (sel01.get(task, {}).get("fusion_thin", {}).get("per_zone") or [])
                      if int(r["zone"]) == u.zone), None)
            maps = mm.get(u)
            if z is None or maps is None:
                continue
            s_best = blend({"ranker": sub[f"score_{best}"].to_numpy(), "label_free": sub.score_label_free.to_numpy(),
                            "neural": sub.score_neural_support.to_numpy()}, rz["fusion_weights"][0])
            for share in rz["hybrid_shares"]:
                for f in rz["area_fracs"]:
                    bq = fusion_predict(u, maps, {**z, "frac": (1.0 - float(share)) * float(f)},
                                        cfg["baseline"]["representations"], m7.elig, float(ev["ridge_sigma_px"]))
                    q = hybrid(u, bq, cs, s_best, float(f), float(share), int(z.get("ring", 0) or 0), m7.elig)
                    rows.append(score_row(u, q, m7.elig(u, 0), m7.pkey(u), method="hybrid_fusion_candidates",
                                          rep="fusion_thin+centerline", fw=json_fw(rz["fusion_weights"][0]),
                                          frac=float(f), ring=int(z.get("ring", 0) or 0), share=float(share)))
            mm.release(u)
            print(f"[m7] s12 {u.key}: done")
    grid = pd.DataFrame(rows)
    summ, per_all, led = [], [], []
    for (task, mth), g in grid.groupby(["task", "method"]) if len(grid) else []:
        params = ["share", "frac"] if mth.startswith("hybrid") else ["rep", "fw", "frac", "ring"]
        pz, s = nested_select(g, params)
        summ.append({"task": task, "method": mth, "dti_crossfit": s["dti_crossfit"], "dti_insample": s["dti_insample"],
                     "optimism": s["optimism"], "fold_min": s["fold_min"], "params_all": json_fw(s["params_all"]),
                     **{f"dti_zone{r.zone}": r.dti_heldout for r in pz.itertuples()}})
        per_all.append(pz.assign(task=task, method=mth))
        for r in pz.itertuples():
            u = m7.unit(f"{task}:{r.zone}")
            hit = g[(g.zone == r.zone)]
            for p in params:
                hit = hit[hit[p].astype(str) == str(getattr(r, p))]
            if len(hit):
                led.append({"task": task, "zone": r.zone, "method": mth, **{p: getattr(r, p) for p in params},
                            "area_frac_eligible": float(hit.area_frac_eligible.iloc[0]),
                            "masked_area_px": int(hit.masked_area_px.iloc[0])})
    per = pd.concat(per_all, ignore_index=True) if per_all else pd.DataFrame()
    _save_selected_maps(m7, per, T, sets, sel01, mm, best)
    reps_ab = rep_ablation(grid)
    return {"grid": grid, "summary": pd.DataFrame(summ), "per_zone": per, "area": pd.DataFrame(led),
            "rep_ablation": reps_ab, "best_ranker": best}


def json_fw(x) -> str:
    import json

    return json.dumps(x, default=float)


def rep_ablation(grid: pd.DataFrame) -> pd.DataFrame:
    """Cross-fitted DTI of each representation alone (blend, frac, ring chosen on the other zones)."""
    from ..gems3.metric_opt import nested_select

    rows = []
    g0 = grid[~grid.method.str.startswith("hybrid")] if len(grid) else grid
    for (task, mth, rep), g in g0.groupby(["task", "method", "rep"]) if len(g0) else []:
        _, s = nested_select(g, ["fw", "frac", "ring"])
        rows.append({"task": task, "method": mth, "rep": rep, "dti_crossfit": s["dti_crossfit"], "fold_min": s["fold_min"],
                     "area_frac_mean": float(g.area_frac_eligible.mean())})
    for (task, mth, ring), g in g0.groupby(["task", "method", "ring"]) if len(g0) else []:
        _, s = nested_select(g, ["rep", "fw", "frac"])
        rows.append({"task": task, "method": mth, "rep": f"ring={int(ring)} (best rep)", "dti_crossfit": s["dti_crossfit"],
                     "fold_min": s["fold_min"]})
    return pd.DataFrame(rows)


def _save_selected_maps(m7, per: pd.DataFrame, T: pd.DataFrame, sets: dict, sel01: dict, mm, best: str) -> None:
    import json

    from .baseline import predict as fusion_predict

    cfg = m7.cfg
    rz, ev = cfg["rasterization"], cfg["evaluation"]
    for r in per.itertuples() if len(per) else []:
        u = m7.unit(f"{r.task}:{r.zone}")
        cs = sets.get(u.key)
        if u is None or cs is None:
            continue
        sub = T[(T.task == u.task) & (T.zone == u.zone)].sort_values("cand_id")
        if r.method.startswith("hybrid"):
            z = next((x for x in sel01.get(r.task, {}).get("fusion_thin", {}).get("per_zone", []) if int(x["zone"]) == u.zone),
                     None)
            maps = mm.get(u)
            if z is None or maps is None:
                continue
            s_best = blend({"ranker": sub[f"score_{best}"].to_numpy(), "label_free": sub.score_label_free.to_numpy(),
                            "neural": sub.score_neural_support.to_numpy()}, rz["fusion_weights"][0])
            bq = fusion_predict(u, maps, {**z, "frac": (1.0 - float(r.share)) * float(r.frac)},
                                cfg["baseline"]["representations"], m7.elig, float(ev["ridge_sigma_px"]))
            q = hybrid(u, bq, cs, s_best, float(r.frac), float(r.share), int(z.get("ring", 0) or 0), m7.elig)
        else:
            rk_name = r.method.replace("cand_", "")
            fw = json.loads(r.fw)
            s = blend({"ranker": sub[f"score_{rk_name}"].to_numpy(), "label_free": sub.score_label_free.to_numpy(),
                       "neural": sub.score_neural_support.to_numpy()}, fw)
            q, _ = predict(u, cs, s, rz["representations"][r.rep], float(r.frac), int(r.ring), m7.elig)
        save_heldout(m7.ws, f"s12_{r.method}", u, q, m7.pkey(u))
