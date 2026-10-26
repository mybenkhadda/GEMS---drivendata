"""Section 01: the strongest existing baseline, rebuilt under ONE protocol (eligible-pixel budget, official scorer).

Methods (B0 = models[0], B1 = models[1]; per-zone rank fusion s = w R(B0) + (1 - w) R(B1))::

    b0_raw / b1_raw          the sigmoid maps as they are (soft, no budget)
    b0_binary / b1_binary    top-a of the member alone (w = 1 / 0)                        nested: frac
    fusion_binary            top-a of the fusion                                          nested: w, frac
    fusion_<thin rep>        1-px crest lines / skeleton of the fusion                    nested: w, frac
    fusion_thin              the better 1-px representation                               nested: rep, w, frac
    fusion_thin_ring300      + 300 m (3 px) suppression ring around known faults (disc)   nested: rep, w, frac
    fusion_thin_ring_sel     + ring width selected among evaluation.ring_px (0 allowed)   nested: rep, w, frac, ring

Nested selection (``gems3.metric_opt.nested_select``): every label-dependent choice for held-out zone z is made on
the other zones of the same task; the grid is fixed a priori (no refinement around any zone's optimum). The
budget fraction counts eligible pixels (``protocol.eligible``) - known faults (masked by the scorer) and the ring do
not consume it. ``disc_retained_fp`` re-scores the maps selected on ``disc_official`` (same zones, same choices).

The ring is a hypothesis about the hidden test, not a scoring rule: pseudo-discovery withholds whole systems, so it
under-represents new faults next to known traces and may reward the ring for the wrong reason.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from ..gems3 import representation as R
from ..gems3.metric_opt import nested_select
from ..gems3.official import unit_score
from .protocol import COMP, dti_of, eligible, score_row, unit_ledger

RAW = ("b0_raw", "b1_raw")


def method_specs(cfg: dict, task: str) -> dict[str, dict]:
    """{method: {"filter": {...}, "params": [...]}} over the evaluation grid of one task."""
    bc, ev = cfg["baseline"], cfg["evaluation"]
    thin = list(bc["thinning_reps"])
    rings = [int(r) for r in ev["ring_px"]]
    out = {"b0_binary": {"filter": {"rep": ["binary"], "w": [1.0], "ring": [0]}, "params": ["frac"]},
           "b1_binary": {"filter": {"rep": ["binary"], "w": [0.0], "ring": [0]}, "params": ["frac"]},
           "fusion_binary": {"filter": {"rep": ["binary"], "ring": [0]}, "params": ["w", "frac"]}}
    for t in thin:
        out[f"fusion_{t}"] = {"filter": {"rep": [t], "ring": [0]}, "params": ["w", "frac"]}
    out["fusion_thin"] = {"filter": {"rep": thin, "ring": [0]}, "params": ["rep", "w", "frac"]}
    if task.startswith("disc"):
        ring_rep = thin if bc.get("ring_rep", "best_thin") == "best_thin" else ["binary"]
        if 3 in rings:
            out["fusion_thin_ring300"] = {"filter": {"rep": ring_rep, "ring": [3]}, "params": ["rep", "w", "frac"]}
        out["fusion_thin_ring_sel"] = {"filter": {"rep": ring_rep, "ring": rings}, "params": ["rep", "w", "frac", "ring"]}
    return out


class MemberMaps:
    """Per-unit honest member scores and their per-zone ranks (cached; read-only)."""

    def __init__(self, ctx, models: list[str]):
        self.ctx, self.models = ctx, models
        self._cache: dict = {}

    def get(self, u):
        if u.key not in self._cache:
            p0, p1 = unit_score(self.ctx, self.models[0], u), unit_score(self.ctx, self.models[1], u)
            if p0 is None or p1 is None:
                self._cache[u.key] = None
            else:
                self._cache[u.key] = {"p0": p0, "p1": p1, "r0": R.rank01(p0, u.ev), "r1": R.rank01(p1, u.ev)}
        return self._cache[u.key]

    def release(self, u) -> None:
        self._cache.pop(u.key, None)


def predict(u, mm: dict, params: dict, reps: dict, elig_of, sigma_px: float, crest=None) -> np.ndarray:
    """Prediction of one (rep, w, frac, ring) on one unit with the eligible budget domain."""
    if params.get("rep") == "raw":
        p = mm["p0"] if float(params["w"]) == 1.0 else mm["p1"]
        return np.where(u.ev, np.clip(np.nan_to_num(p, nan=0.0), 0, 1), 0.0).astype(np.float32)
    s = R.fuse(mm["r0"], mm["r1"], float(params["w"]))
    ring = int(params.get("ring", 0) or 0)
    rep = R.with_ring(reps[params["rep"]], ring)
    return R.apply(rep, s, u.ev, float(params["frac"]), known=u.known, crest=crest, sigma_px=sigma_px,
                   eligible=elig_of(u, ring))


def predict_with_ledger(u, mm: dict, params: dict, reps: dict, elig_of, sigma_px: float) -> tuple[np.ndarray, list]:
    """Final prediction + the area after selection, after post operations and after the ring (each step)."""
    ring = int(params.get("ring", 0) or 0)
    el = elig_of(u, ring)
    if params.get("rep") == "raw":
        q = predict(u, mm, params, reps, elig_of, sigma_px)
        return q, [unit_ledger(q, u, el, "raw map (no budget)", ring)]
    s = R.fuse(mm["r0"], mm["r1"], float(params["w"]))
    rep = reps[params["rep"]]
    sel_only = {"select": rep.get("select", "top")}
    led = []
    q1 = R.apply(sel_only, s, u.ev, float(params["frac"]), known=u.known, sigma_px=sigma_px, eligible=el)
    led.append(unit_ledger(q1, u, el, "selection (top-frac of eligible)", ring))
    q2 = R.apply(rep, s, u.ev, float(params["frac"]), known=u.known, sigma_px=sigma_px, eligible=el)
    led.append(unit_ledger(q2, u, el, "post operations", ring))
    q3 = R.apply(R.with_ring(rep, ring), s, u.ev, float(params["frac"]), known=u.known, sigma_px=sigma_px, eligible=el)
    led.append(unit_ledger(q3, u, el, f"known-fault ring {ring} px", ring))
    led.append({**unit_ledger(q3, u, el, "scorer mask (excluded pixels carry no score)", ring),
                "note": "on_excluded_px are ignored by the official scorer; they consumed no budget"})
    return q3, led


def evaluate_grid(m7, units, mm: MemberMaps) -> pd.DataFrame:
    """Components of every (rep, w, frac, ring) on every selection unit (geo + primary task)."""
    cfg = m7.cfg
    ev, reps = cfg["evaluation"], cfg["baseline"]["representations"]
    sigma = float(ev.get("ridge_sigma_px", 1.0))
    rows = []
    for u in units:
        maps = mm.get(u)
        if maps is None:
            print(f"[m7] s01 {u.key}: member map missing, skipped")
            continue
        t0 = time.perf_counter()
        pk = m7.pkey(u)
        rings = [int(r) for r in ev["ring_px"]] if u.is_disc else [0]
        for m, w in (("b0_raw", 1.0), ("b1_raw", 0.0)):
            q = predict(u, maps, {"rep": "raw", "w": w}, reps, m7.elig, sigma)
            rows.append(score_row(u, q, m7.elig(u, 0), pk, method=m, rep="raw", w=w, frac=np.nan, ring=0))
        for w in ev["fusion_weights"]:
            s = R.fuse(maps["r0"], maps["r1"], float(w))
            crest = R.crest_map(s, u.ev, sigma) if any(r.get("select") == "crest" for r in reps.values()) else None
            for ring in rings:
                el = m7.elig(u, ring)
                for name, rep in reps.items():
                    rr = R.with_ring(rep, ring)
                    for f in ev["area_fracs"]:
                        q = R.apply(rr, s, u.ev, float(f), known=u.known, crest=crest, sigma_px=sigma, eligible=el)
                        rows.append(score_row(u, q, el, pk, method="grid", rep=name, w=float(w), frac=float(f),
                                              ring=int(ring)))
        print(f"[m7] s01 grid {u.key}: {time.perf_counter() - t0:.0f}s")
    return pd.DataFrame(rows)


def select_methods(grid: pd.DataFrame, cfg: dict, tasks: list[str]) -> tuple[pd.DataFrame, dict]:
    """Nested selection of every method on every task -> (per-zone held-out rows, {task: {method: summary}})."""
    per_rows, sel = [], {}
    for task in tasks:
        g = grid[grid.task == task]
        if not len(g):
            continue
        sel[task] = {}
        raw = g[g.method.isin(RAW)]
        for m, gm in raw.groupby("method"):
            per = gm.assign(dti_heldout=gm.dti)
            sel[task][m] = {"dti_crossfit": dti_of(gm), "dti_insample": dti_of(gm), "optimism": 0.0,
                            "fold_min": float(gm.dti.min()), "params_all": {"rep": "raw", "w": float(gm.w.iloc[0])},
                            "per_zone": [{"zone": int(r.zone), "rep": "raw", "w": float(r.w), "frac": None, "ring": 0,
                                          "dti_heldout": float(r.dti), **{c: float(getattr(r, c)) for c in COMP}}
                                         for r in gm.itertuples()]}
            per_rows.append(per.assign(stage=m)[["task", "zone", "method", "rep", "w", "ring", "dti_heldout"] + COMP]
                            .assign(method=m))
        gg = g[g.method == "grid"]
        for m, spec in method_specs(cfg, task).items():
            sub = gg
            for col, vals in spec["filter"].items():
                sub = sub[sub[col].isin(vals)]
            if not len(sub):
                continue
            per, summ = nested_select(sub, spec["params"])
            if not len(per):
                continue
            full = per.copy()
            for col in ("rep", "w", "frac", "ring"):
                if col not in full:
                    full[col] = sub[col].iloc[0] if sub[col].nunique() == 1 else np.nan
            area = []
            for r in full.itertuples():
                hit = sub[(sub.zone == r.zone) & (sub.rep == r.rep) & np.isclose(sub.w, float(r.w))
                          & np.isclose(sub.frac, float(r.frac)) & (sub.ring == int(r.ring))]
                area.append(float(hit.area_frac_eligible.iloc[0]) if len(hit) else np.nan)
            full["area_frac_eligible"] = area
            pz = [{"zone": int(r.zone), "rep": str(r.rep), "w": float(r.w), "frac": float(r.frac), "ring": int(r.ring),
                   "dti_dev": float(r.dti_dev), "dti_heldout": float(r.dti_heldout),
                   **{c: float(getattr(r, c)) for c in COMP}} for r in full.itertuples()]
            pa = dict(summ["params_all"])
            for col, vals in spec["filter"].items():
                if col not in pa and len(vals) == 1:
                    pa[col] = vals[0]
            sel[task][m] = {"dti_crossfit": summ["dti_crossfit"], "dti_insample": summ["dti_insample"],
                            "optimism": summ["optimism"], "fold_min": summ["fold_min"], "params_all": pa,
                            "params": spec["params"], "per_zone": pz,
                            "area_frac_eligible_mean": float(np.nanmean(area)) if area else np.nan}
            per_rows.append(full.assign(task=task, method=m)[["task", "zone", "method", "rep", "w", "frac", "ring",
                                                              "dti_dev", "dti_heldout", "area_frac_eligible"] + COMP])
    per = pd.concat(per_rows, ignore_index=True) if per_rows else pd.DataFrame()
    return per, sel


def retained_fp_rescore(m7, sel: dict, mm: MemberMaps) -> pd.DataFrame:
    """Score the maps selected on the primary task again on ``disc_retained_fp`` (predictions on retained faults = FP)."""
    prim = m7.cfg["evaluation"]["primary_task"]
    alt = [u for u in m7.units_of("disc_retained_fp")]
    if not alt or prim not in sel:
        return pd.DataFrame()
    reps, sigma = m7.cfg["baseline"]["representations"], float(m7.cfg["evaluation"]["ridge_sigma_px"])
    rows = []
    for ua in alt:
        up = m7.unit(f"{prim}:{ua.zone}")
        maps = mm.get(up)
        if up is None or maps is None:
            continue
        for m, s in sel[prim].items():
            z = next((r for r in s["per_zone"] if r["zone"] == ua.zone), None)
            if z is None:
                continue
            q = predict(up, maps, z, reps, m7.elig, sigma)
            q = m7.transfer(q, up, ua)
            rows.append(score_row(ua, q, m7.elig(ua, 0), m7.pkey(ua), method=m, rep=z["rep"], w=z["w"],
                                  frac=z["frac"] if z["frac"] is not None else np.nan, ring=z["ring"],
                                  selected_on=prim))
    return pd.DataFrame(rows)


def baseline_table(sel: dict, per: pd.DataFrame, alt: pd.DataFrame, pkeys: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task, ms in sel.items():
        for m, s in ms.items():
            z = {f"dti_zone{r['zone']}": r["dti_heldout"] for r in s["per_zone"]}
            rows.append({"task": task, "method": m, "dti_crossfit": s["dti_crossfit"], "fold_min": s["fold_min"],
                         "dti_insample": s["dti_insample"], "optimism": s["optimism"],
                         "params_all": json.dumps(s["params_all"], default=float),
                         "area_frac_eligible_mean": s.get("area_frac_eligible_mean", np.nan), **z})
    if len(alt):
        for m, g in alt.groupby("method"):
            rows.append({"task": "disc_retained_fp", "method": m, "dti_crossfit": dti_of(g), "fold_min": float(g.dti.min()),
                         "selected_on": g.selected_on.iloc[0],
                         **{f"dti_zone{int(r.zone)}": float(r.dti) for r in g.itertuples()}})
    t = pd.DataFrame(rows)
    if len(pkeys):
        k = pkeys.groupby("task").protocol_key.apply(lambda s: "|".join(sorted(s)))
        t["protocol_keys"] = t.task.map(k)
    t["validation_status"] = ("nested cross-fit over zones; these zones were inspected in earlier sessions "
                              "(GEMS3 / M6): not an untouched test set")
    return t


RECORDED_MAP = {"geo": {"b0_binary": "b0", "b1_binary": "b1", "fusion_binary": "fusion", "fusion_thin": "fusion_thinning"},
                "disc_official": {"b0_raw": "b0_raw", "b1_raw": "b1_raw", "fusion_binary": "rank_fusion",
                                  "fusion_skeleton_1px": "fusion_skeleton_1px", "fusion_thin_ring300": "fusion_ring_300m"}}


def recorded_comparison(table: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Recomputed vs recorded offline values: REPORTED, never reconciled (protocols differ: earlier numbers used
    the historical budget rule and other selection grids)."""
    rec = cfg.get("recorded_reference") or {}
    rows = []
    for task, mp in RECORDED_MAP.items():
        src = rec.get("geo_dti" if task == "geo" else "disc_official_dti") or {}
        for m, k in mp.items():
            if k not in src:
                continue
            hit = table[(table.task == task) & (table.method == m)]
            new = float(hit.dti_crossfit.iloc[0]) if len(hit) else np.nan
            rows.append({"task": task, "method": m, "recorded_key": k, "recorded_offline": float(src[k]),
                         "recomputed_crossfit": new, "difference": new - float(src[k]) if np.isfinite(new) else np.nan,
                         "note": "recorded = earlier session, earlier protocol; not a leaderboard score"})
    for k in ("candidate_pool_oracle_100m",):
        if k in (rec.get("disc_official_dti") or {}):
            rows.append({"task": "disc_official", "method": "candidate_oracle (section 04)", "recorded_key": k,
                         "recorded_offline": float(rec["disc_official_dti"][k]), "recomputed_crossfit": np.nan,
                         "note": "oracle = upper bound conditional on that candidate pool; see section 04"})
    return pd.DataFrame(rows)


def run(m7) -> dict:
    cfg = m7.cfg
    tasks = ["geo", cfg["evaluation"]["primary_task"]]
    units = [u for t in tasks for u in m7.units_of(t)]
    mm = MemberMaps(m7.ctx, m7.models)
    grid = evaluate_grid(m7, units, mm)
    per, sel = select_methods(grid, cfg, tasks)
    alt = retained_fp_rescore(m7, sel, mm)
    pk = m7.protocol_keys()
    table = baseline_table(sel, per, alt, pk)
    ledgers = []
    for task in tasks:
        for m, s in sel.get(task, {}).items():
            for z in s["per_zone"]:
                u = m7.unit(f"{task}:{z['zone']}")
                maps = mm.get(u) if u is not None else None
                if maps is None:
                    continue
                _, led = predict_with_ledger(u, maps, z, cfg["baseline"]["representations"], m7.elig,
                                             float(cfg["evaluation"]["ridge_sigma_px"]))
                ledgers += [{**r, "method": m} for r in led]
    return {"grid": grid, "per_zone": per, "selection": sel, "retained_fp": alt, "table": table,
            "recorded": recorded_comparison(table, cfg), "ledger": pd.DataFrame(ledgers), "protocol": pk}


def heldout_map(m7, task: str, method: str, zone: int, mm: MemberMaps | None = None) -> np.ndarray | None:
    """Cropped held-out prediction of one method on one zone, rebuilt from the persisted per-zone choice."""
    sel = m7.baseline_selection()
    s = (sel or {}).get(task, {}).get(method)
    u = m7.unit(f"{task}:{zone}")
    if s is None or u is None:
        return None
    z = next((r for r in s["per_zone"] if int(r["zone"]) == int(zone)), None)
    if z is None:
        return None
    mm = mm or MemberMaps(m7.ctx, m7.models)
    maps = mm.get(u)
    if maps is None:
        return None
    return predict(u, maps, z, m7.cfg["baseline"]["representations"], m7.elig,
                   float(m7.cfg["evaluation"]["ridge_sigma_px"]))


def figures(m7, out: dict) -> None:
    from .figures import bar_panels16, ledger_figure

    t = out["table"]
    if len(t):
        d = t[["task", "method", "dti_crossfit", "fold_min"]].copy()
        m7.ws.save_fig(bar_panels16(d, "method", "dti_crossfit", "task", ref_col="fold_min",
                                    title="Section 01 - recovered baselines (cross-fitted DTI; tick = worst zone)"),
                       "baseline_dti", d, "recovered_baselines")
    if len(out["ledger"]):
        m7.ws.save_fig(ledger_figure(out["ledger"]), "baseline_area_ledger", out["ledger"], "recovered_baselines")
