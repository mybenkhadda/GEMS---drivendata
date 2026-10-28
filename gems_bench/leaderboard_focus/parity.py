"""Notebook 02 (part 2): diagnosis of the historical W2 export-parity blocker + the champion's deployment parity.

Three evidence sources, each labelled with where it comes from (nothing is inferred without a table behind it):
A. ``recorded``   P6.1 recovery table ``recovery/export_parity/train_inference_parity.csv`` (training vs inference
                  construction per unit and feature), classified by ``cause_of``.
B. ``recomputed`` cheap CPU checks on frozen artifacts:
                  * expected vs actual score map: W2 held-out map vs the historical full-region score crop
                    (``stack/maps/full_score.tif``), both ranked inside the unit (Spearman, top-k overlap);
                  * fold assignment: each geo unit mask == valid & fold_map == k on its crop;
                  * duplicated / missing pixels: geo zones disjoint, valid pixels outside every zone;
                  * candidate budget and representation of the historical submission vs its sidecar and the champion;
                  * OOF vs in-sample: the historical score is a refit on every primary-task row (``stack.full_region``),
                    so each zone's score saw that zone's rows (recorded as a fact of the code, not a measurement).
C. ``champion``   the zone-stitched parity rows of the dry build (``validation/zone_stitched_parity.csv``).
The stage is COMPLETE when the CHAMPION construction has no blocking parity row; the historical blocker is
documented (it concerns an export M1 does not use) and never silently cleared.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import BLOCKED, COMPLETE

OUT = ("parity/parity_report.csv", "parity/parity_report.json", "parity/historical_parity_summary.csv")
CAUSES = {
    "rank_domain": "feature transformation: ranked inside each unit (training / evaluation) vs inside the whole "
                   "footprint (historical full-region export)",
    "score_map": "score map: held-out cross-fitted W2 map vs the historical full-region refit score",
    "by_design": "by design: permitted known faults (training) vs every mapped fault (inference)",
    "representation_budget": "binary map outcome at the deployment cell (representation / budget)",
    "fusion_weight": "fusion weight: per-zone champion w (evaluation) vs all-zone w (deployment)",
    "other": "not classified (inspect the row)",
}


# =============================================================================== pure
def cause_of(row: dict, ranked: set) -> str:
    f, chk = str(row.get("feature")), str(row.get("check"))
    if chk == "w2_score":
        return "score_map"
    if chk == "map_agreement":
        return "representation_budget"
    if chk == "fusion_weight":
        return "fusion_weight"
    if str(row.get("expected_by_design")).strip().lower() == "true":      # NaN / absent (CSV) is not True
        return "by_design"
    if chk == "feature" and f in ranked:
        return "rank_domain"
    return "other"


def summarize_historical(par: pd.DataFrame, ranked: set) -> pd.DataFrame:
    if par is None or not len(par):
        return pd.DataFrame(columns=["cause", "status", "n_rows", "units", "features", "explanation"])
    P = par.copy()
    P["cause"] = [cause_of(r, ranked) for r in P.to_dict("records")]
    g = P.groupby(["cause", "status"]).agg(n_rows=("status", "size"), units=("unit", lambda x: ";".join(sorted(set(
        map(str, x))))), features=("feature", lambda x: ";".join(sorted(set(map(str, x)))))).reset_index()
    g["explanation"] = g.cause.map(CAUSES)
    return g


def fold_assignment(ev_crop: np.ndarray, valid_crop: np.ndarray, fold_crop: np.ndarray, k: int) -> dict:
    want = np.asarray(valid_crop, bool) & (np.asarray(fold_crop) == int(k))
    diff = np.asarray(ev_crop, bool) ^ want
    return {"mismatch_px": int(diff.sum()), "status": "PASS" if not diff.any() else "BLOCKED"}


def zone_overlaps(masks: dict, valid: np.ndarray) -> dict:
    """Pixels claimed by more than one zone (duplicated) and valid pixels claimed by none (missing)."""
    cnt = np.zeros(valid.shape, np.int16)
    for m in masks.values():
        cnt += np.asarray(m, bool)
    return {"duplicated_px": int((cnt > 1).sum()), "missing_valid_px": int((np.asarray(valid, bool) & (cnt == 0)).sum())}


def final_status(champion_rows: pd.DataFrame, fresh: pd.DataFrame, strict: bool) -> tuple[str, list[str]]:
    why = []
    zs = champion_rows[champion_rows.check == "zone score vs evaluation construction"] if len(champion_rows) else \
        champion_rows
    bad = zs[zs.status != "PASS"] if strict else zs[zs.status == "BLOCKED"]
    if not len(zs):
        why.append("champion zone-stitched parity rows absent (run validation first)")
    elif len(bad):
        why.append(f"champion zone score parity not PASS on {list(bad.unit)}")
    hard = fresh[(fresh.scope == "champion") & (fresh.status == "BLOCKED")] if len(fresh) else fresh
    why += [f"{r.check} {r.unit}: {r.detail}" for r in hard.itertuples()]
    return (COMPLETE if not why else BLOCKED), why


# =============================================================================== Colab
def fresh_checks(sess) -> pd.DataFrame:
    from ..p6 import export_parity as EP
    from ..p6 import stack as ST
    from ..p6.final_submission import zone_partition

    fg, bst, p6 = sess.fg, sess.bst, sess.p6
    data = p6.ctx.data
    valid = np.asarray(data.valid, bool)
    fm = np.asarray(p6.ctx.folds.fold_map)
    geo = [u for u in bst.units() if u.task == "geo"]
    rows, masks = [], {}
    fs = p6.ws.root / "stack" / "maps" / "full_score.tif"
    full = EP.read_band(fs) if fs.exists() else None
    seed = int(p6.cfg["seed"])
    for u in geo:
        fa = fold_assignment(u.ev, valid[u.sl], fm[u.sl], u.zone)
        rows.append({"scope": "champion", "check": "fold assignment (unit mask == valid & fold_map == k)",
                     "unit": u.key, "status": fa["status"], "detail": f"{fa['mismatch_px']} px differ",
                     "source": "recomputed"})
        m = np.zeros(valid.shape, bool)
        m[u.sl] = u.ev
        masks[u.zone] = m
        if full is not None:
            a = EP.agreement(ST.rank_in(bst.w2_map(u), u.ev), ST.rank_in(full[u.sl], u.ev), u.ev, (0.01, 0.05),
                             200_000, seed)
            rows.append({"scope": "historical", "check": "score map: W2 held-out vs historical full-region score",
                         "unit": u.key, "status": EP.classify_parity(a, fg.rc.cfg["export_parity"]["parity"]),
                         "detail": json.dumps({k: a.get(k) for k in ("spearman", "top_0.01", "top_0.05")}, default=str),
                         "source": "recomputed", **{k: a.get(k) for k in ("spearman",)}})
        u.release()
    if full is None:
        rows.append({"scope": "historical", "check": "score map", "unit": "*", "status": "NOT_AVAILABLE",
                     "detail": "stack/maps/full_score.tif absent", "source": "recomputed"})
    ov = zone_overlaps(masks, valid)
    _, cov = zone_partition(fm, valid, [u.zone for u in geo])
    rows.append({"scope": "champion", "check": "duplicated pixels across geo zones", "unit": "*",
                 "status": "PASS" if ov["duplicated_px"] == 0 else "BLOCKED", "detail": json.dumps(ov),
                 "source": "recomputed"})
    rows.append({"scope": "champion", "check": "missing valid pixels (outside every geo zone)", "unit": "*",
                 "status": "PASS" if cov["uncovered_frac"] <= float(sess.cfg["validation"]["max_uncovered_frac"])
                 else "BLOCKED", "detail": json.dumps(cov, default=str), "source": "recomputed"})
    rows.append({"scope": "historical", "check": "OOF vs in-sample", "unit": "*", "status": "IN_SAMPLE_BY_CONSTRUCTION",
                 "detail": "stack.full_region refits LightGBM on every primary-task row: each zone's full-region score "
                           "saw that zone's rows (held-out maps are cross-fitted)", "source": "code (p6/stack.py)"})
    hist = sorted(p6.ws.root.joinpath("submissions").glob("p6_stack_fusion*.json"))
    rec = sess.ws.read_json("selection/champion_selection.json") or {}
    if hist:
        side = json.loads(hist[-1].read_text(encoding="utf-8"))
        pa = side.get("parameters") or {}
        rows.append({"scope": "historical", "check": "historical export recipe (rep / blend source vs frac source)",
                     "unit": "*", "status": "BLOCKED" if str(side.get("frac_source")) == "geo" else "WARNING",
                     "detail": json.dumps({"parameters": pa, "frac_source": side.get("frac_source")}, default=str),
                     "source": f"recorded ({hist[-1].name})"})
    rows.append({"scope": "champion", "check": "champion recipe = one all-zone selection", "unit": "*",
                 "status": "PASS" if rec.get("recipe_id") else "BLOCKED",
                 "detail": f"recipe_id {rec.get('recipe_id')}", "source": "notebook 01"})
    return pd.DataFrame(rows)


def diagnose(sess) -> dict:
    """Notebook 02 stage ``parity``."""
    from ..p6 import stack as ST

    sess.require("validation")
    fp = sess.fp("parity", sess.manifest("validation").get("fingerprint"),
                 sess.fg.rc.read_json("export_parity", "decision"))

    def run():
        hist = sess.fg.rc.load_table("train_inference_parity", "export_parity")
        HS = summarize_historical(hist, set(ST.RANKED))
        sess.ws.save_table(HS, "parity/historical_parity_summary.csv")
        FR = fresh_checks(sess)
        CH = sess.ws.load_table("parity/zone_stitched_parity.csv")
        R = pd.concat([FR, CH.assign(scope="champion", source="dry build (final_submission.build_map)",
                                     detail=CH.get("note", ""))], ignore_index=True)
        sess.ws.save_table(R, "parity/parity_report.csv")
        st, why = final_status(CH, FR, bool(sess.cfg["validation"]["strict_parity"]))
        blocked_hist = HS[HS.status == "BLOCKED"] if len(HS) else HS
        rep = {"status": st, "reasons": why,
               "historical_blocker": {"recorded_rows_blocked": int(blocked_hist.n_rows.sum()) if len(blocked_hist) else 0,
                                      "causes": blocked_hist[["cause", "features", "units", "explanation"]].to_dict(
                                          "records") if len(blocked_hist) else [],
                                      "recorded_table_present": bool(len(hist)),
                                      "resolution": "not fixed in place: the historical export is not used. The "
                                                    "champion is built zone by zone from the evaluated held-out maps "
                                                    "(no footprint ranking, no in-sample refit, one selection)"},
               "champion_parity": CH[["unit", "check", "status"]].to_dict("records") if len(CH) else []}
        sess.ws.save_json(rep, "parity/parity_report.json")
        from .reporting import parity_figure

        parity_figure(sess, R)
        return st, why, {}

    return sess.run_stage("parity", fp, OUT, run, inputs={"validation_fp": sess.manifest("validation").get(
        "fingerprint")})
