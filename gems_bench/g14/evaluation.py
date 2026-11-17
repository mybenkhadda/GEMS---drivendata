"""One evaluation interface for every G14 experiment and every reference model (Phase 10).

* Scoring: official components through ``gems_bench.gems3.official`` units (pixel-exact known-fault mask for
  ``disc_official``), the same code path as G13 / P6 / P7 (``gems_bench.metric`` is verified against an independent
  KD-tree reference in tests/test_gems3.py).
* Nested selection: (representation, area fraction) for each held-out zone is chosen on the OTHER zones of the same
  task (``g13.evaluation.crossfit``); thresholds are never tuned on the zone being scored nor on the locked TEST.
  The area grid reaches 30 %: the real G13 run selected the largest value of its 0.5-12 % grid for weak models.
* Diagnostics at the selected operating point: per-zone DTI, fault-length-stratified recall, centre-line
  localisation, fragmentation, false-positive mass, distance of predictions to held-out faults, time and memory.
* Locked TEST: ``evaluate_locked_test`` scores ONE (model, operating point) and appends to an audit log; a second
  look is refused unless explicitly forced and is then labelled ``SECOND_LOOK`` in every table.
* Never: a local score presented as a leaderboard score.

UNTESTED: written without execution.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

from ..g13.evaluation import area_table, block_components, bootstrap_dti, crossfit, dti_of, pooled_dti

PROVENANCE = {"scorer": "gems_bench.gems3.official.Unit.components (FastScorer)", "radius_m": 300,
              "alpha": 0.2, "beta": 0.8, "known_mask": "pixel-exact, excluded pixels cannot support TP (disc_official)",
              "selection": "cross-fitted over zones of the same task", "is_leaderboard_score": False}


# =============================================================================== map containers
def maps_from_units(maps_by_key: dict[str, np.ndarray], tasks=("geo", "disc_official")) -> dict:
    """Normalise to {(task, zone): full-grid array}. Keys like 'geo_1', 'disc_official_0'."""
    out = {}
    for k, v in maps_by_key.items():
        task, zone = k.rsplit("_", 1)
        if task in tasks:
            out[(task, int(zone))] = np.asarray(v, np.float32)
    return out


def load_g13_model(pred_dir: str | Path, system) -> dict:
    """G13 predictions/<model>/{oof,disc<d>}.npy -> {(task, zone): full grid}. Only valid with the g13 backend."""
    if system.backend != "g13":
        raise ValueError("G13 maps were produced on G13 splits; score them with the g13 backend only")
    p = Path(pred_dir)
    out = {}
    if (p / "oof.npy").exists():
        oof = np.load(p / "oof.npy")
        for k in system.cv_zones:
            out[("geo", k)] = oof
    for d in range(system.n_disc):
        f = p / f"disc{d}.npy"
        if f.exists():
            out[("disc_official", d)] = np.load(f)
    if (p / "full.npy").exists():
        out[("full", 0)] = np.load(p / "full.npy", mmap_mode="r")
    return out


def load_unit_maps(root: str | Path, system, pattern: str = "{task}_{zone}.npy") -> dict:
    """Unit maps saved per evaluation unit (e.g. P6 W2 ``stack/maps``). Accepts full-grid arrays or unit crops (then
    embedded with the unit's slice); any other shape is refused rather than guessed."""
    root = Path(root)
    out = {}
    for u in system.eval_units():
        f = root / pattern.format(task=u.task, zone=u.zone)
        if not f.exists():
            continue
        a = np.load(f)
        if a.shape == system.data.shape:
            out[(u.task, u.zone)] = a.astype(np.float32)
        elif a.shape == u.ev.shape:
            full = np.full(system.data.shape, np.nan, np.float32)
            full[u.sl] = a
            out[(u.task, u.zone)] = full
        else:
            raise ValueError(f"{f}: shape {a.shape} matches neither the grid {system.data.shape} nor unit crop "
                             f"{u.ev.shape} (different split backend?)")
    return out


# =============================================================================== scoring
def operating_tables(maps: dict, system, cfg) -> pd.DataFrame:
    """Components for every (unit, representation, area fraction)."""
    from ..g13.dti_opt import render
    from ..gems3.representation import crest_map

    rows = []
    for u in system.eval_units(tuple(cfg.split.tasks)):
        s = maps.get((u.task, u.zone))
        if s is None:
            continue
        if "top" in cfg.eval.representations:
            t = area_table(s, [u], cfg.eval.area_fracs)
            t["rep"] = "top"
            rows.append(t)
        if "crest_w1" in cfg.eval.representations:
            sc = u.crop(s)
            crest = crest_map(np.where(u.ev, np.nan_to_num(sc), 0.0), u.ev)
            for fr in cfg.eval.area_fracs:
                spec = {"rep": {"select": "crest", "post": [["dilate", {"radius_px": 1}]]}, "frac": fr, "ring_px": 0}
                q = render(spec, sc, u.ev, u.known, None, u.scored_mask(), crest)
                c = u.components(q)
                rows.append(pd.DataFrame([{"task": u.task, "zone": u.zone, "target_frac": fr, "tp_w": c.tp,
                                           "fp_w": c.fp, "fn_w": c.fn, "area_px": int((q > 0).sum()), "rep": "crest_w1"}]))
        u.release()
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def score(name: str, maps: dict, system, cfg, out_dir: str | Path | None = None) -> dict:
    """Nested (cross-fitted) DTI per task + diagnostics. Returns {'row': scorecard dict, 'tables': {...}}."""
    tab = operating_tables(maps, system, cfg)
    row = {"model": name, "backend": system.backend, **{f"prov_{k}": v for k, v in PROVENANCE.items()}}
    tables = {"operating": tab}
    chosen = {}
    for task, t in tab.groupby("task") if len(tab) else []:
        per, summ = crossfit(t.assign(rep=t.rep.astype(str)), ["rep", "target_frac"])
        tables[f"{task}_choice"] = per
        key = "disc" if task.startswith("disc") else task
        row.update({f"{key}_dti": summ["dti_crossfit"], f"{key}_zone_min": summ["zone_min"],
                    f"{key}_optimism": summ["optimism"], f"{key}_rep": summ["params_all"]["rep"],
                    f"{key}_frac": summ["params_all"]["target_frac"],
                    f"{key}_frac_on_grid_edge": float(summ["params_all"]["target_frac"]) == max(cfg.eval.area_fracs)})
        chosen[task] = per
    tables["diagnostics"] = diagnostics(maps, system, cfg, chosen)
    if out_dir is not None:
        root = Path(out_dir)
        root.mkdir(parents=True, exist_ok=True)
        for k, v in tables.items():
            if isinstance(v, pd.DataFrame):
                v.to_csv(root / f"{name}__{k}.csv", index=False)
        (root / f"{name}__scorecard.json").write_text(json.dumps(row, indent=1, default=str))
    return {"row": row, "tables": tables}


def _binary(u, s, rep: str, frac: float) -> np.ndarray:
    from ..g13.dti_opt import render

    spec = {"rep": {"select": "top"} if rep == "top" else {"select": "crest", "post": [["dilate", {"radius_px": 1}]]},
            "frac": frac, "ring_px": 0}
    return render(spec, u.crop(s), u.ev, u.known, None, u.scored_mask()) > 0


def diagnostics(maps: dict, system, cfg, chosen: dict) -> pd.DataFrame:
    """Per held-out zone at its cross-fitted operating point."""
    from ..geology.skeleton import skeletonize

    rows = []
    for u in system.eval_units(tuple(cfg.split.tasks)):
        s = maps.get((u.task, u.zone))
        per = chosen.get(u.task)
        if s is None or per is None or u.zone not in set(per.zone):
            continue
        r = per[per.zone == u.zone].iloc[0]
        q = _binary(u, s, str(r.rep), float(r.target_frac))
        truth = u.eval_truth()
        dt = ndimage.distance_transform_edt(~truth) * 0.1 if truth.any() else np.full(q.shape, np.inf)
        sk = skeletonize(q)
        lab, ncomp = ndimage.label(q, structure=np.ones((3, 3), bool))
        sk_km = sk.sum() * 0.1
        near = dt[sk] <= 1.0
        sc = u.scorer
        p = q.astype(np.float64)
        p[sc.zero] = 0
        sup = sc.support(p, cleaned=True)
        rows.append({"task": u.task, "zone": u.zone, "rep": r.rep, "frac": float(r.target_frac),
                     "dti": float(r.dti_heldout), "tp_w": float(r.tp_w), "fp_w": float(r.fp_w), "fn_w": float(r.fn_w),
                     "weighted_recall": float(sup.mean()) if len(sup) else np.nan,
                     "truth_px": int(len(sup)), "pred_px": int(q.sum()),
                     "fragmentation_per_100km": float(ncomp / max(sk_km, 1e-9) * 100) if sk_km else np.nan,
                     "centerline_mean_offset_km": float(dt[sk][near].mean()) if near.any() else np.nan,
                     "pred_within_300m_frac": float((dt[sk] <= 0.3).mean()) if sk.any() else np.nan,
                     "median_pred_to_truth_km": float(np.median(dt[q])) if q.any() and np.isfinite(dt).any() else np.nan})
        u.release()
    return pd.DataFrame(rows)


def length_stratified_recall(maps: dict, system, cfg, chosen: dict, sysmap: np.ndarray, systems_km: dict,
                             bins_km=(0, 2, 5, 10, 25, 1e9)) -> pd.DataFrame:
    """Weighted recall of held-out truth grouped by the trace length of its fault system."""
    rows = []
    for u in system.eval_units(tuple(cfg.split.tasks)):
        s, per = maps.get((u.task, u.zone)), chosen.get(u.task)
        if s is None or per is None or u.zone not in set(per.zone):
            continue
        r = per[per.zone == u.zone].iloc[0]
        q = _binary(u, s, str(r.rep), float(r.target_frac)).astype(np.float64)
        sc = u.scorer
        q[sc.zero] = 0
        sup = sc.support(q, cleaned=True)
        gr, gc = sc.g_rc
        ids = sysmap[u.sl][gr, gc]
        km = np.array([systems_km.get(int(i), np.nan) for i in ids])
        b = np.digitize(km, bins_km) - 1
        for k in np.unique(b):
            m = b == k
            rows.append({"task": u.task, "zone": u.zone, "length_bin": int(k), "n_truth": int(m.sum()),
                         "tp_w": float(sup[m].sum())})
        u.release()
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    g = df.groupby(["task", "length_bin"])[["n_truth", "tp_w"]].sum().reset_index()
    g["recall_w"] = g.tp_w / g.n_truth.clip(lower=1)
    return g


def bootstrap_task(maps: dict, system, cfg, chosen: dict, task: str, n_boot: int = 300) -> dict:
    blocks = []
    for u in system.eval_units((task,)):
        s, per = maps.get((u.task, u.zone)), chosen.get(task)
        if s is None or per is None or u.zone not in set(per.zone):
            continue
        r = per[per.zone == u.zone].iloc[0]
        if str(r.rep) == "top":
            blocks.append(block_components(s, u, float(r.target_frac)))
        u.release()
    return bootstrap_dti(pd.concat(blocks), n_boot) if blocks else {}


# =============================================================================== locked TEST
def evaluate_locked_test(name: str, full_map: np.ndarray, system, rep: str, frac: float, log_dir: str | Path,
                         force_second_look: bool = False) -> dict:
    """Once-only TEST evaluation of the operating point chosen on the CV units (never re-tuned here)."""
    if not system.has_locked_test:
        raise ValueError("this split backend has no locked TEST partition")
    log_p = Path(log_dir) / "locked_test_log.jsonl"
    log_p.parent.mkdir(parents=True, exist_ok=True)
    prior = [json.loads(x) for x in log_p.read_text().splitlines()] if log_p.exists() else []
    if prior and not force_second_look:
        raise PermissionError(f"the locked TEST was already evaluated ({prior[0]['model']}); "
                              "pass force_second_look=True to look again (labelled SECOND_LOOK)")
    u = system.test_unit()
    q = _binary(u, np.asarray(full_map, np.float32), rep, frac)
    c = u.components(q.astype(np.float64))
    rec = {"model": name, "rep": rep, "frac": frac, "test_proxy_dti": c.dti, "tp_w": c.tp, "fp_w": c.fp, "fn_w": c.fn,
           "status": "SECOND_LOOK" if prior else "FIRST_LOOK", "time_unix": time.time(),
           "map_sha": hashlib.sha256(np.ascontiguousarray(full_map, np.float32).tobytes()).hexdigest()[:16],
           "note": "geographic TEST fold, all mapped faults in it as truth (rediscovery proxy, not a leaderboard score)"}
    with open(log_p, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def compare(scorecards: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(scorecards)
    keep = [c for c in df.columns if not c.startswith("prov_")]
    return df[keep].sort_values(next((c for c in ("disc_dti", "geo_dti") if c in df), keep[0]), ascending=False)


__all__ = ["PROVENANCE", "maps_from_units", "load_g13_model", "load_unit_maps", "operating_tables", "score",
           "diagnostics", "length_stratified_recall", "bootstrap_task", "evaluate_locked_test", "compare", "dti_of",
           "pooled_dti"]
