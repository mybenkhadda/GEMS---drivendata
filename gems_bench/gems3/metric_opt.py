"""Milestone A: how much DTI does the output representation leave on the table? (no retraining)

Inputs: the FULL OOF + discovery maps of two members (B0, B1), loaded through M6 (memory-mapped).

Stage 0  raw maps (A0 / A1) and absolute-threshold sweeps (A2 / A3), exact all-threshold curves.
Stage 1  per-zone rank fusion w R(B0) + (1 - w) R(B1) on the weight grid x binary top-a area grid (A4 / A5 / A10);
         w = 1 and w = 0 are B0 and B1 alone. Exact (p3.curves.BinaryCurve).
Stage 2  representations (A6 - A9, soft vs binary) at refined area fractions around each zone's own dev choice.
Stage 3  disc tasks: the zone's stage-2 choice + a suppression ring around known faults (official FP rule).

Nested selection, everywhere: for held-out zone z every label-dependent choice (weight, fraction, representation,
ring) is made on the other zones of the same task, *and* only among candidates generated from zone z's own dev
choices (``nested_select``; the M6 rule that refinements centred on another zone's choice used z's labels).
Pooled held-out components give the honest ``dti_crossfit``. The all-zone choice (in-sample) is what a submission
uses; its optimism is reported.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..p3.curves import dti_from, pooled
from . import representation as R
from .official import Unit, unit_score

COMP = ["tp_w", "fp_w", "fn_w", "n_truth"]


# =============================================================================== nested selection
def _keys(df: pd.DataFrame, params: list[str]) -> pd.Series:
    return df[params].astype(str).agg("|".join, axis=1)


def nested_select(table: pd.DataFrame, params: list[str], allowed: dict | None = None,
                  zone_col: str = "zone") -> tuple[pd.DataFrame, dict]:
    """Per held-out zone: best ``params`` on the other zones (pooled DTI), restricted to ``allowed[z]`` (set of
    param-key strings) when given; scored on the zone. Returns (per-zone rows, summary)."""
    if not len(table):
        return pd.DataFrame(), {"dti_crossfit": np.nan}
    t = table.copy()
    t["_key"] = _keys(t, params)
    rows = []
    for z in sorted(t[zone_col].unique()):
        sub = t if allowed is None else t[t._key.isin(allowed.get(z, set()))]
        dev = sub[sub[zone_col] != z]
        if not len(dev):
            continue
        g = dev.groupby("_key")[COMP].sum()
        g["dti"] = g.tp_w / (g.tp_w + 0.2 * g.fp_w + 0.8 * g.fn_w + 1e-6)
        best = g.dti.idxmax()
        held = sub[(sub[zone_col] == z) & (sub._key == best)]
        if not len(held):
            continue
        r0 = held.iloc[0]
        rows.append({zone_col: z, **{p: r0[p] for p in params}, "dti_dev": float(g.dti.max()),
                     **{c: float(held[c].sum()) for c in COMP}, "dti_heldout": dti_from(held)})
    per = pd.DataFrame(rows)
    allz = t.groupby("_key")[COMP].sum()
    allz["dti"] = allz.tp_w / (allz.tp_w + 0.2 * allz.fp_w + 0.8 * allz.fn_w + 1e-6)
    top = t[t._key == allz.dti.idxmax()].iloc[0]
    summ = {"dti_crossfit": dti_from(per) if len(per) else np.nan, "dti_insample": float(allz.dti.max()),
            "params_all": {p: (top[p].item() if hasattr(top[p], "item") else top[p]) for p in params},
            "fold_min": float(per.dti_heldout.min()) if len(per) else np.nan}
    summ["optimism"] = summ["dti_insample"] - summ["dti_crossfit"]
    return per, summ


# =============================================================================== stage 0 / 1
def stage1(ctx, units: list[Unit], models: list[str], mo: dict) -> dict[str, pd.DataFrame]:
    m0, m1 = models
    raw, thr, area, missing = [], [], [], []
    for u in units:
        p0, p1 = unit_score(ctx, m0, u), unit_score(ctx, m1, u)
        if p0 is None or p1 is None:
            missing.append({"task": u.task, "zone": u.zone, "missing": m0 if p0 is None else m1})
            continue
        for m, p, w in ((m0, p0, 1.0), (m1, p1, 0.0)):
            if mo.get("raw_models", True):
                c = u.components(p)
                raw.append({"task": u.task, "zone": u.zone, "model": m, "w": w, "rep": "raw", **c.as_dict()})
            df = u.curve(p).at(mo["thresholds"])
            df.insert(0, "model", m)
            df.insert(0, "zone", u.zone)
            df.insert(0, "task", u.task)
            thr.append(df)
        r0, r1 = R.rank01(p0, u.ev), R.rank01(p1, u.ev)
        for w in mo["fusion_weights"]:
            df = u.curve(R.fuse(r0, r1, float(w))).at_area(mo["area_fracs"])
            df.insert(0, "w", float(w))
            df.insert(0, "zone", u.zone)
            df.insert(0, "task", u.task)
            area.append(df)
        print(f"[gems3] stage 1 {u.key}: done")
    cat = (lambda x: pd.concat(x, ignore_index=True) if x else pd.DataFrame())
    out = {"raw": pd.DataFrame(raw), "thresholds": cat(thr), "area": cat(area), "missing": pd.DataFrame(missing)}
    if len(out["area"]):
        out["area"] = out["area"].rename(columns={"target_frac": "frac"})
    return out


def stage1_selection(area: pd.DataFrame, thresholds: pd.DataFrame, tasks: list[str]) -> dict[str, pd.DataFrame]:
    """Per task: joint (w, frac) choice, per-w area choice (ensemble ablation), absolute-threshold transfer."""
    joint, per_w, thr_sel, per_rows = [], [], [], []
    for task in tasks:
        a = area[area.task == task]
        if not len(a):
            continue
        per, s = nested_select(a, ["w", "frac"])
        joint.append({"task": task, "stage": "stage1_fusion_area", **s, "params_all": json.dumps(s["params_all"])})
        per_rows.append(per.assign(task=task, stage="stage1_fusion_area"))
        for w, g in a.groupby("w"):
            pw, sw = nested_select(g, ["frac"])
            per_w.append({"task": task, "w": w, "dti_crossfit": sw["dti_crossfit"], "dti_insample": sw["dti_insample"],
                          "optimism": sw["optimism"], "fold_min": sw["fold_min"],
                          "frac_all_zones": sw["params_all"]["frac"],
                          **{f"frac_zone{r.zone}": r.frac for r in pw.itertuples()},
                          **{f"dti_zone{r.zone}": r.dti_heldout for r in pw.itertuples()}})
            per_rows.append(pw.assign(task=task, w=w, stage=f"stage1_area_w{w:g}"))
        t = thresholds[thresholds.task == task]
        for m, g in t.groupby("model"):
            pt, st = nested_select(g, ["threshold"])
            thr_sel.append({"task": task, "model": m, "selection": "absolute_threshold", **st,
                            "params_all": json.dumps(st["params_all"])})
            per_rows.append(pt.assign(task=task, model=m, stage="stage0_abs_threshold"))
    cat = (lambda x: pd.concat(x, ignore_index=True) if x else pd.DataFrame())
    return {"joint": pd.DataFrame(joint), "per_w": pd.DataFrame(per_w), "thresholds": pd.DataFrame(thr_sel),
            "per_zone": cat(per_rows)}


def stage2_candidates(area: pd.DataFrame, task: str, mo: dict) -> dict:
    """{zone: {"weights": [...], "fracs": {w: [...]}}} from each zone's own dev (other-zone) choices."""
    a = area[area.task == task]
    out = {}
    for z in sorted(a.zone.unique()):
        dev = a[a.zone != z].groupby(["w", "frac"])[COMP].sum()
        dev["dti"] = dev.tp_w / (dev.tp_w + 0.2 * dev.fp_w + 0.8 * dev.fn_w + 1e-6)
        best_w = dev.dti.groupby(level="w").max().sort_values(ascending=False)
        ws = list(best_w.index[: int(mo.get("n_weights_stage2", 2))])
        fr = {}
        for w in ws:
            f = float(dev.loc[w].dti.idxmax())
            fr[w] = sorted({round(float(np.clip(f * k, 1e-4, 0.5)), 6) for k in mo["refine_factors"]})
        out[int(z)] = {"weights": ws, "fracs": fr}
    return out


# =============================================================================== stage 2 / 3 evaluation
def _row(u: Unit, q: np.ndarray, **tags) -> dict:
    c = u.components(q)
    area, n_ev = int(((q > 0) & u.ev).sum()), int(u.ev.sum())
    return {"task": u.task, "zone": u.zone, **tags, **c.as_dict(), "eval_px": n_ev, "area_px": area,
            "area_frac": area / max(n_ev, 1)}


def evaluate_reps(ctx, units: list[Unit], models: list[str], reps: dict, grid: dict, *, sigma_px: float = 1.0,
                  rings=(0,)) -> pd.DataFrame:
    """Score every (rep, w, frac, ring) of ``grid`` = {w: [fracs]} on every unit. ``rings`` > 0 only on disc."""
    m0, m1 = models
    rows = []
    for u in units:
        p0, p1 = unit_score(ctx, m0, u), unit_score(ctx, m1, u)
        if p0 is None or p1 is None:
            continue
        r0, r1 = R.rank01(p0, u.ev), R.rank01(p1, u.ev)
        for w, fracs in grid.items():
            s = R.fuse(r0, r1, float(w))
            need_crest = any(r.get("select") == "crest" for r in reps.values())
            crest = R.crest_map(s, u.ev, sigma_px) if need_crest else None
            for f in fracs:
                for name, rep in reps.items():
                    for ring in rings if u.is_disc else (0,):
                        q = R.apply(R.with_ring(rep, ring), s, u.ev, f, known=u.known, crest=crest, sigma_px=sigma_px)
                        rows.append(_row(u, q, rep=name, family=R.family(name, rep), w=float(w), frac=float(f),
                                         ring=int(ring)))
        print(f"[gems3] representations {u.key}: {sum(len(v) for v in grid.values())} (w, frac) x {len(reps)} reps")
    return pd.DataFrame(rows)


def _union_grid(cands: dict) -> dict:
    g: dict = {}
    for c in cands.values():
        for w in c["weights"]:
            g.setdefault(float(w), set()).update(c["fracs"][w])
    return {w: sorted(v) for w, v in g.items()}


def _allowed(cands: dict, rep_names, ring_of=None) -> dict:
    out = {}
    for z, c in cands.items():
        keys = set()
        for w in c["weights"]:
            for f in c["fracs"][w]:
                for r in rep_names:
                    keys.add("|".join(map(str, [r, float(w), float(f)] + ([0] if ring_of else []))))
        out[z] = keys
    return out


def run(ctx, units: list[Unit], models: list[str], cfg: dict) -> dict:
    """All stages; returns every table (written by the caller) and the selections."""
    mo, tasks = cfg["metric_opt"], list(dict.fromkeys(u.task for u in units))
    reps = mo["representations"]
    s1 = stage1(ctx, units, models, mo)
    sel1 = stage1_selection(s1["area"], s1["thresholds"], tasks) if len(s1["area"]) else {}
    stage2_tasks = [t for t in tasks if t in ("geo", cfg["primary_task"])]
    s2_tabs, s3_tabs, sel_rows, per_rows, cand_log = [], [], [], [], []
    choice = {}
    for task in stage2_tasks:
        tu = [u for u in units if u.task == task]
        cands = stage2_candidates(s1["area"], task, mo)
        cand_log += [{"task": task, "zone": z, "weights": json.dumps(c["weights"]),
                      "fracs": json.dumps({str(k): v for k, v in c["fracs"].items()})} for z, c in cands.items()]
        t2 = evaluate_reps(ctx, tu, models, reps, _union_grid(cands), sigma_px=mo.get("ridge_sigma_px", 1.0))
        s2_tabs.append(t2)
        per2, sm2 = nested_select(t2, ["rep", "w", "frac"], _allowed(cands, reps))
        sel_rows.append({"task": task, "stage": "stage2_representation", **sm2, "params_all": json.dumps(sm2["params_all"])})
        per_rows.append(per2.assign(task=task, stage="stage2_representation"))
        best, per_best = sm2, per2
        rings = [int(r) for r in mo.get("known_ring_px") or []]
        if task.startswith("disc") and rings and len(per2):
            grid3, allowed3 = {}, {}
            for r in per2.itertuples():
                grid3.setdefault((r.rep, float(r.w)), set()).add(float(r.frac))
            t3 = []
            for (rep, w), fr in grid3.items():
                t3.append(evaluate_reps(ctx, tu, models, {rep: reps[rep]}, {w: sorted(fr)},
                                        sigma_px=mo.get("ridge_sigma_px", 1.0), rings=[0] + rings))
            t3 = pd.concat(t3, ignore_index=True)
            s3_tabs.append(t3)
            for r in per2.itertuples():
                allowed3[int(r.zone)] = {"|".join(map(str, [r.rep, float(r.w), float(r.frac), k])) for k in [0] + rings}
            per3, sm3 = nested_select(t3, ["rep", "w", "frac", "ring"], allowed3)
            sel_rows.append({"task": task, "stage": "stage3_known_ring", **sm3, "params_all": json.dumps(sm3["params_all"])})
            per_rows.append(per3.assign(task=task, stage="stage3_known_ring"))
            best, per_best = sm3, per3
        choice[task] = {"params_all": best["params_all"], "dti_crossfit": best["dti_crossfit"],
                        "dti_insample": best["dti_insample"], "per_zone": per_best.to_dict("records")}
    cat = (lambda x: pd.concat(x, ignore_index=True) if x else pd.DataFrame())
    t2, t3 = cat(s2_tabs), cat(s3_tabs)
    rep_ablation = representation_ablation(t2, reps)
    out = {**s1, "sel1": sel1, "stage2": t2, "stage3": t3, "candidates": pd.DataFrame(cand_log),
           "selection": cat([pd.DataFrame(sel_rows)] + ([sel1["joint"], sel1["thresholds"]] if sel1 else [])),
           "per_zone": cat(per_rows + ([sel1["per_zone"]] if sel1 else [])), "rep_ablation": rep_ablation,
           "choice": choice}
    out["paired"] = paired_table(out, models, tasks)
    return out


def representation_ablation(t2: pd.DataFrame, reps: dict) -> pd.DataFrame:
    """Cross-fitted DTI of each representation alone (its (w, frac) chosen on the other zones)."""
    rows = []
    for (task, rep), g in t2.groupby(["task", "rep"]) if len(t2) else []:
        per, s = nested_select(g, ["w", "frac"])
        rows.append({"task": task, "rep": rep, "family": R.family(rep, reps.get(rep, {})),
                     "dti_crossfit": s["dti_crossfit"], "dti_insample": s["dti_insample"], "optimism": s["optimism"],
                     "fold_min": s["fold_min"], "params_all": json.dumps(s["params_all"]),
                     "area_frac_mean": float(g.area_frac.mean()),
                     **{f"dti_zone{r.zone}": r.dti_heldout for r in per.itertuples()}})
    return pd.DataFrame(rows)


def paired_table(out: dict, models: list[str], tasks: list[str]) -> pd.DataFrame:
    """Held-out DTI per zone of every stage, with deltas vs each member's own cross-fitted binary map."""
    pz = out["per_zone"]
    if not len(pz):
        return pd.DataFrame()
    rows = []
    m0, m1 = models
    for task in tasks:
        d = pz[pz.task == task]
        ref = {}
        for w, m in ((1.0, m0), (0.0, m1)):
            s = d[d.stage == f"stage1_area_w{w:g}"]
            ref[m] = s.set_index("zone").dti_heldout
        for stage, g in d.groupby("stage"):
            if stage.startswith("stage1_area_w") and stage not in ("stage1_area_w1", "stage1_area_w0"):
                continue
            label = {"stage1_area_w1": f"{m0}_binary", "stage1_area_w0": f"{m1}_binary"}.get(stage, stage)
            if stage == "stage0_abs_threshold":
                for m, gm in g.groupby("model"):
                    for r in gm.itertuples():
                        rows.append({"task": task, "zone": r.zone, "method": f"{m}_abs_threshold",
                                     "dti_heldout": r.dti_heldout})
                continue
            for r in g.itertuples():
                rows.append({"task": task, "zone": r.zone, "method": label, "dti_heldout": r.dti_heldout})
        raw = out["raw"]
        if len(raw):
            for r in raw[raw.task == task].itertuples():
                rows.append({"task": task, "zone": r.zone, "method": f"{r.model}_raw", "dti_heldout": r.dti})
        for m in (m0, m1):
            if m in ref:
                for row in rows:
                    if row["task"] == task:
                        row[f"delta_vs_{m}_binary"] = row["dti_heldout"] - float(ref[m].get(row["zone"], np.nan))
    return pd.DataFrame(rows)


# =============================================================================== reporting helpers
def pooled_curves(out: dict) -> dict[str, pd.DataFrame]:
    th = out["thresholds"]
    area = out["area"]
    t2 = out["stage2"]
    res = {}
    if len(th):
        res["threshold_sweep"] = pd.concat([pooled(th, ["task", "model", "threshold"]).assign(zone="pooled"), th],
                                           ignore_index=True)
    if len(area):
        res["area_budget_curves"] = pd.concat([pooled(area, ["task", "w", "frac"]).assign(zone="pooled", rep="binary"),
                                               area.assign(rep="binary")], ignore_index=True)
    if len(t2):
        p2 = pooled(t2, ["task", "rep", "family", "w", "frac"]).assign(zone="pooled")
        res["width_ablation"] = p2[p2.family.isin(["width", "soft", "area"])]
        res["dilation_ablation"] = p2[p2.family.isin(["dilation", "area"])]
    if len(out["stage3"]):
        res["known_ring_ablation"] = pooled(out["stage3"], ["task", "rep", "w", "frac", "ring"]).assign(zone="pooled")
    return res


def figures(ws, out: dict, curves: dict, models: list[str]) -> None:
    from .figures import bar_panels, line_panels

    th = curves.get("threshold_sweep")
    if th is not None and len(th):
        d = th[th.zone == "pooled"]
        ws.save_fig(line_panels(d, "threshold", "dti", "model", "task", xlabel="absolute threshold on the raw map"),
                    "threshold_sweep", d, sub="metric_optimization/figures")
    ab = curves.get("area_budget_curves")
    if ab is not None and len(ab):
        d = ab[(ab.zone == "pooled")].copy()
        d["member"] = d.w.map(lambda w: models[0] if w == 1 else models[1] if w == 0 else f"fusion w={w:g}")
        ws.save_fig(line_panels(d, "area_frac", "dti", "member", "task", xlabel="predicted area (share of zone)",
                                logx=True), "area_budget_curves", d, sub="metric_optimization/figures")
    pw = out["sel1"].get("per_w") if out.get("sel1") else None
    if pw is not None and len(pw):
        pw = pw.assign(series="cross-fitted")
        ws.save_fig(line_panels(pw, "w", "dti_crossfit", "series", "task",
                                xlabel=f"weight of {models[0]} in the rank fusion"), "ensemble_weight_curve", pw,
                    sub="metric_optimization/figures")
    ra = out["rep_ablation"]
    if len(ra):
        ref = {}
        if pw is not None and len(pw):
            ref = pw.groupby("task").dti_crossfit.max().to_dict()
        ws.save_fig(bar_panels(ra, "rep", "dti_crossfit", "task", ref=ref), "representation_ablation", ra,
                    sub="metric_optimization/figures")
    pt = out["paired"]
    if len(pt):
        piv = pt.groupby(["task", "method"]).dti_heldout.mean().reset_index()
        ws.save_fig(bar_panels(piv, "method", "dti_heldout", "task", ylabel="mean held-out DTI over zones"),
                    "paired_methods", pt, sub="ensembles")


def heldout_maps(ctx, units: list[Unit], models: list[str], cfg: dict, task: str, per_zone: list[dict]) -> dict:
    """Full-grid held-out prediction of ``task`` with each zone's own dev-selected parameters (for reuse)."""
    mo, reps = cfg["metric_opt"], cfg["metric_opt"]["representations"]
    H, W = ctx.data.shape
    layers = {}
    by_zone = {int(r["zone"]): r for r in per_zone}
    full = np.full((H, W), np.nan, np.float32) if task == "geo" else None
    for u in units:
        if u.task != task or u.zone not in by_zone:
            continue
        r = by_zone[u.zone]
        p0, p1 = unit_score(ctx, models[0], u), unit_score(ctx, models[1], u)
        s = R.fuse(R.rank01(p0, u.ev), R.rank01(p1, u.ev), float(r["w"]))
        q = R.apply(R.with_ring(reps[r["rep"]], int(r.get("ring", 0) or 0)), s, u.ev, float(r["frac"]),
                    known=u.known, sigma_px=mo.get("ridge_sigma_px", 1.0))
        tgt = full if full is not None else np.full((H, W), np.nan, np.float32)
        view = tgt[u.sl]
        view[u.ev] = q[u.ev]
        if full is None:
            layers[f"{task}_split{u.zone}"] = tgt
    if full is not None:
        layers[f"{task}_heldout"] = full
    return layers
