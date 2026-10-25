"""Screening: proxy evaluation of SCREEN / VALIDATE runs, matched comparisons, shortlists, read-only reports.

Four jobs, all reading saved artifacts only (except ``evaluate_fidelity_run``, which scores a run's own rasters):

``evaluate_fidelity_run``  the evaluate stage of a SCREEN / VALIDATE cnn5 run. Per geographic fold: DTI, weighted
                           recall and recall@300 m at the matched predicted-area budget (``evaluation.match_frac``),
                           best-area DTI, predicted-area fraction, collapse indicators, on the fidelity's scored
                           mask (its sha256 recorded). Per discovery split (VALIDATE): the same at the matched budget
                           with withheld traces as truth inside ``eval_zone`` and retained known faults excluded
                           ("excl" policy). These are PROXY metrics, labelled ``proxy_*``: they are not the
                           section-17 cross-fitted scores and never enter a leaderboard or the promotion gate.
``metrics_comparison`` /   read-only consolidation of every experiment's saved metrics (FULL scoreboards and
``paired_deltas``          fidelity proxy tables), candidate - control deltas under identical protocols only
                           (same scoreboard table for FULL; same fidelity protocol hash and scored-mask digests for
                           SCREEN / VALIDATE). Missing values are ``NOT_EVALUATED``, never zero.
``shortlist``              configurable, recorded decisions (PROMOTE_TO_VALIDATE / PROMOTE_TO_FULL / REVIEW / REJECT /
                           PARITY_CONFIRMED / INCOMPARABLE / NOT_EVALUATED) with the reasons. Thresholds are triage
                           heuristics (``DEFAULT_RULES``), not scientific acceptance criteria; a decision only
                           recommends the next fidelity level. Submission promotion stays a separate manual step.
``export_screening_report`` / ``research_dashboard``  CSV / PNG / SVG / Markdown under
                           ``reports/experiment_screening`` and a compact notebook status view.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

NE = "NOT_EVALUATED"
OUT_DIR = "reports/experiment_screening"

# triage heuristics (configurable via experiments.screening); NOT acceptance criteria
DEFAULT_RULES = {
    "primary": "proxy_dti_at_match",      # geographic proxy DTI at the matched area budget
    "promote_min_gain": 0.005,            # mean candidate - control gain needed to recommend the next level
    "reject_max_loss": 0.010,             # mean loss beyond which the candidate is rejected at this level
    "max_recall300_drop": 0.020,          # recall@300 m at matched area may not fall more than this
    "min_folds_beating": 2,               # VALIDATE: folds where the candidate beats the control
    "collapse_high_frac": 0.50,           # share of scored pixels with p > 0.99 -> near-one collapse
    "collapse_low_mean": 1e-3,            # mean p below this -> near-zero collapse
    "parity_tol": 1e-4,                   # |delta| below this on every metric = numerically identical
}
DECISIONS = ("PROMOTE_TO_VALIDATE", "PROMOTE_TO_FULL", "REVIEW", "REJECT", "PARITY_CONFIRMED", "INCOMPARABLE",
             "NOT_EVALUATED")
NEXT_LEVEL = {"SCREEN": "PROMOTE_TO_VALIDATE", "VALIDATE": "PROMOTE_TO_FULL"}
PROXY_METRICS = ("proxy_dti_at_match", "proxy_recall300_at_match", "proxy_recall_w_at_match", "proxy_dti_best_area",
                 "pred_area_frac_p05", "mean_p", "frac_p_gt_099", "frac_p_lt_001")
FULL_METRICS = ("geo_dti_area", "geo_fold_min", "geo_fold_max", "geo_fold_std", "dti_fold0", "dti_fold1", "dti_fold2",
                "disc_dti_excl", "disc_fold_min_excl", "disc_recall300_at_match_excl", "disc_dti_retained_fp",
                "geo_recall300_at_match", "geo_far_share_at_match", "geo_frac", "disc_frac_excl", "short_recall_w",
                "recall_w_len_<1", "recall_w_len_1-2", "unique_systems_vs_G1_excl")
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
PROFILE_MARKER = {"SCREEN": "o", "VALIDATE": "s", "FULL": "D"}


# =============================================================================== proxy evaluation
def proxy_scores(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray, frac: float,
                 exclude: np.ndarray | None = None) -> dict:
    """Matched-area DTI / recall + distribution diagnostics of ``pred`` on ``mask`` (NaN outside = not predicted)."""
    from ..p3.curves import BinaryCurve

    mask = np.asarray(mask, bool) & np.isfinite(pred)
    p = np.asarray(pred, np.float64)[mask]
    out = {"scored_px": int(mask.sum()), "n_truth_px": int((np.asarray(truth, bool) & mask).sum())}
    if not mask.any():
        return out
    c = BinaryCurve(pred, truth, mask, exclude)
    at = c.at_area([frac]).iloc[0]
    out.update(proxy_dti_at_match=float(at["dti"]), proxy_recall_w_at_match=float(at["recall_w"]),
               proxy_recall300_at_match=float(at.get("recall@300m", np.nan)), match_frac=float(frac))
    if c.n:
        b = c.best()
        out.update(proxy_dti_best_area=float(b["dti"]), proxy_best_area_frac=float(b["target_frac"]))
    out.update(pred_area_frac_p05=float((p >= 0.5).mean()), mean_p=float(p.mean()),
               frac_p_gt_099=float((p > 0.99).mean()), frac_p_lt_001=float((p < 0.01).mean()))
    return out


def _unit_training(rd) -> pd.DataFrame:
    if not rd.has_table("training_history"):
        return pd.DataFrame(columns=["unit"])
    h = rd.load_table("training_history")
    g = h.groupby("unit")
    out = pd.DataFrame({"epochs_run": g.epoch.max(), "train_s": g.epoch_s.sum(),
                        "final_loss": g.train_loss.last(), "loss_drop": g.train_loss.first() - g.train_loss.last()})
    if "loss_unlabeled_mean_p" in h:
        out["unlabeled_p_last"] = g.loss_unlabeled_mean_p.last()
    if "collapsed" in h:
        out["collapse_guard"] = g.collapsed.max().astype(bool)
    out["nan_loss"] = g.train_loss.apply(lambda s: bool(~np.isfinite(s).all()))
    return out.reset_index()


def evaluate_fidelity_run(runner) -> dict:
    """Evaluate stage of a SCREEN / VALIDATE cnn5 run (see module docstring). Writes ``metrics/fidelity_metrics.csv``
    and ``reports/fidelity_protocol.json``; touches neither the registry nor any scoreboard."""
    from .fidelity import LABEL, mask_digest

    fid, ctx, rd, cfg = runner.fidelity, runner.ctx, runner.rd, runner.cfg
    data = ctx.data
    frac = float(cfg.get("evaluation.match_frac", 0.05))
    phash = fid.protocol_hash(cfg.get("validation.folds_dir", ""), cfg.get("validation.discovery_dir", ""))
    rows = []
    for k in fid.fold_ids(ctx.fold_ids):
        if not rd.raster_path(f"oof_fold{k}").exists():
            rows.append({"zone": "geo", "unit": f"cv_fold{k}", "status": NE})
            continue
        mask = fid.val_mask(ctx.folds.val(k) & data.valid, k)
        rows.append({"zone": "geo", "unit": f"cv_fold{k}", "status": "ok", "mask_digest": mask_digest(mask),
                     **proxy_scores(rd.read_raster(f"oof_fold{k}"), data.labels, mask, frac)})
    splits = fid.split_ids(ctx.splits.n_splits) if cfg.get("discovery", True) else []
    for d in splits:
        if not rd.raster_path(f"disc{d}").exists():
            rows.append({"zone": "discovery", "unit": f"disc{d}", "status": NE})
            continue
        zone = fid.zone_mask(ctx.splits.eval_zone(d) & data.valid, d)
        held = ctx.splits.withheld(d) & data.labels
        rows.append({"zone": "discovery", "unit": f"disc{d}", "status": "ok", "mask_digest": mask_digest(zone),
                     **proxy_scores(rd.read_raster(f"disc{d}"), held, zone, frac, exclude=data.labels & ~held)})
    df = pd.DataFrame(rows).merge(_unit_training(rd), on="unit", how="left")
    df.insert(0, "experiment_id", runner.exp.id)
    df.insert(1, "model", runner.model_name)
    df.insert(2, "profile", fid.profile)
    df.insert(3, "protocol_hash", phash)
    rd.save_table(df, "fidelity_metrics")
    proto = {"profile": fid.profile, "label": LABEL[fid.profile], "protocol_hash": phash, **fid.semantic_block(),
             "match_frac": frac, "folds_dir": cfg.get("validation.folds_dir", ""),
             "discovery_dir": cfg.get("validation.discovery_dir", ""), "metrics": "proxy (not section-17)",
             "leaderboard_eligible": False, "promotion_eligible": False,
             "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    rd.save_json(proto, "fidelity_protocol")
    runner.exp.manifest.data["protocol"] = {**proto, "kind": "fidelity_proxy"}
    runner.exp.manifest.save()
    geo = df[(df.zone == "geo") & (df.status == "ok")]
    return {"profile": fid.profile, "protocol_hash": phash,
            "proxy_dti_at_match_mean": float(geo.proxy_dti_at_match.mean()) if len(geo) else None,
            "leaderboard_eligible": False}


# =============================================================================== read-only consolidation
def _root(pc, root):
    from .inventory import inventory_root

    return inventory_root(pc, root)


def _csv(p: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(p) if p.exists() else None
    except Exception:
        return None


def _candidate_models(d: Path) -> list[str]:
    try:
        m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    names = list((m.get("maps") or {}).keys())
    fm = m.get("fidelity_maps") or {}
    if fm.get("model"):
        names.append(fm["model"])
    return names


def metrics_comparison(pc=None, root: Path | None = None, inventory: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per (experiment, model) with every available metric; ``NOT_EVALUATED`` where absent."""
    from .inventory import experiment_inventory

    base = _root(pc, root)
    inv = inventory if inventory is not None else experiment_inventory(pc, root=root)
    rows = []
    for r in inv.itertuples():
        d = base / "experiments" / r.experiment_id
        cands = _candidate_models(d)
        common = {"experiment": r.experiment, "experiment_id": r.experiment_id, "profile": r.profile,
                  "role": r.role, "complete": r.complete, "config_compatible": r.config_compatible,
                  "gpu_train_min": r.gpu_train_min}
        if r.profile == "FULL":
            sb = _csv(d / "metrics" / "scoreboard.csv")
            sc = _csv(d / "metrics" / "scorecard.csv")
            if sb is None and sc is None:
                rows.append({**common, "model": cands[0] if cands else "", "is_candidate": True,
                             "metric_kind": "section17", "status": NE})
                continue
            t = sb if sb is not None else sc
            if sb is not None and sc is not None:
                extra = [c for c in sc.columns if c not in sb.columns]
                t = sb.merge(sc[["model"] + extra], on="model", how="left")
            geom = _csv(d / "metrics" / "geometry_metrics.csv")
            if geom is not None and "model" in geom:
                gcols = [c for c in geom.columns if c not in ("model", "frac") and not c.startswith("n_px")]
                t = t.merge(geom[["model"] + gcols].rename(columns={c: f"geom_{c}" for c in gcols}), on="model",
                            how="left")
            for _, row in t.iterrows():
                rec = {**common, "model": row["model"], "is_candidate": row["model"] in cands,
                       "metric_kind": "section17", "status": "ok", "protocol_hash": row.get("protocol_hash", "")}
                for c in list(FULL_METRICS) + [c for c in t.columns if c.startswith("geom_")]:
                    v = row.get(c, np.nan)
                    rec[c] = v if pd.notna(v) else NE
                rows.append(rec)
        else:
            fm = _csv(d / "metrics" / "fidelity_metrics.csv")
            if fm is None:
                rows.append({**common, "model": cands[0] if cands else "", "is_candidate": True,
                             "metric_kind": "proxy", "status": NE})
                continue
            for zone, g in fm.groupby("zone"):
                ok = g[g.get("status", "ok") == "ok"] if "status" in g else g
                rec = {**common, "model": g.model.iloc[0], "is_candidate": True, "metric_kind": f"proxy_{zone}",
                       "status": "ok" if len(ok) else NE, "protocol_hash": g.protocol_hash.iloc[0],
                       "units": ",".join(ok.unit)}
                for c in PROXY_METRICS:
                    rec[c] = float(ok[c].mean()) if c in ok and len(ok) and ok[c].notna().any() else NE
                if zone == "geo" and "proxy_dti_at_match" in ok and len(ok) > 1:
                    rec["proxy_fold_min"] = float(ok.proxy_dti_at_match.min())
                    rec["proxy_fold_std"] = float(ok.proxy_dti_at_match.std(ddof=0))
                rows.append(rec)
    return pd.DataFrame(rows)


def _num(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def paired_deltas(pc=None, root: Path | None = None, inventory: pd.DataFrame | None = None,
                  comparison: pd.DataFrame | None = None) -> pd.DataFrame:
    """Candidate - control deltas under identical protocols only (see module docstring)."""
    from .experiments import CATALOGUE, control_stem

    base = _root(pc, root)
    mc = comparison if comparison is not None else metrics_comparison(pc, root, inventory)
    rows = []
    if not len(mc):
        return pd.DataFrame(rows)
    # FULL: control row of the same scoreboard table
    for eid, g in mc[(mc.profile == "FULL") & (mc.status == "ok")].groupby("experiment_id"):
        exp = g.experiment.iloc[0]
        ctrl = CATALOGUE[exp].control if exp in CATALOGUE else None
        cand = g[g.is_candidate]
        crow = g[g.model == ctrl]
        for _, c in cand.iterrows():
            if c["model"] == ctrl:
                continue
            base_rec = {"experiment": exp, "experiment_id": eid, "profile": "FULL", "candidate": c["model"],
                        "control": ctrl, "protocol": "same scoreboard table"}
            if not len(crow):
                rows.append({**base_rec, "metric": "*", "status": "INCOMPARABLE",
                             "reason": f"control {ctrl} not in this experiment's scoreboard"})
                continue
            for m in FULL_METRICS:
                a, b = _num(c.get(m)), _num(crow.iloc[0].get(m))
                rows.append({**base_rec, "metric": m, "candidate_value": a if a is not None else NE,
                             "control_value": b if b is not None else NE,
                             "delta": a - b if a is not None and b is not None else NE,
                             "status": "ok" if a is not None and b is not None else NE})
    # SCREEN / VALIDATE: the control experiment at the same protocol hash, unit by unit, identical masks
    fid = mc[(mc.profile != "FULL") & (mc.status == "ok")]
    for eid, g in fid.groupby("experiment_id"):
        exp, prof = g.experiment.iloc[0], g.profile.iloc[0]
        cstem = control_stem(exp)
        base_rec = {"experiment": exp, "experiment_id": eid, "profile": prof, "candidate": g.model.iloc[0]}
        if cstem is None:
            continue
        cand_t = _csv(base / "experiments" / eid / "metrics" / "fidelity_metrics.csv")
        ph = g.protocol_hash.iloc[0]
        ctrl_rows = fid[(fid.experiment == cstem) & (fid.protocol_hash == ph)]
        if not len(ctrl_rows):
            rows.append({**base_rec, "control": cstem, "metric": "*", "status": "INCOMPARABLE",
                         "reason": f"no {cstem} run at {prof} with protocol {ph}"})
            continue
        ctrl_id = ctrl_rows.experiment_id.iloc[0]
        ctrl_t = _csv(base / "experiments" / ctrl_id / "metrics" / "fidelity_metrics.csv")
        if cand_t is None or ctrl_t is None:
            rows.append({**base_rec, "control": ctrl_id, "metric": "*", "status": "INCOMPARABLE",
                         "reason": "fidelity_metrics.csv missing"})
            continue
        j = cand_t.merge(ctrl_t, on=["zone", "unit"], suffixes=("", "_ctrl"))
        if "mask_digest" in j and (j.mask_digest != j.mask_digest_ctrl).any():
            rows.append({**base_rec, "control": ctrl_id, "metric": "*", "status": "INCOMPARABLE",
                         "reason": "scored masks differ (mask_digest)"})
            continue
        for _, u in j.iterrows():
            for m in PROXY_METRICS:
                a, b = _num(u.get(m)), _num(u.get(f"{m}_ctrl"))
                rows.append({**base_rec, "control": ctrl_id, "protocol": f"{prof} {ph}", "zone": u["zone"],
                             "unit": u["unit"], "metric": m, "candidate_value": a if a is not None else NE,
                             "control_value": b if b is not None else NE,
                             "delta": a - b if a is not None and b is not None else NE,
                             "status": "ok" if a is not None and b is not None else NE})
    return pd.DataFrame(rows)


def resource_usage(pc=None, root: Path | None = None, inventory: pd.DataFrame | None = None) -> pd.DataFrame:
    """Recorded GPU training minutes, epochs, inference seconds and peak memory per experiment (read-only)."""
    from .inventory import experiment_inventory

    base = _root(pc, root)
    inv = inventory if inventory is not None else experiment_inventory(pc, root=root)
    rows = []
    for r in inv.itertuples():
        d = base / "experiments" / r.experiment_id / "metrics"
        rp, fm, dr = _csv(d / "runtime_profile.csv"), _csv(d / "fold_metrics.csv"), _csv(d / "discovery_runs.csv")
        rows.append({"experiment": r.experiment, "experiment_id": r.experiment_id, "profile": r.profile,
                     "units_trained": int(len(rp)) if rp is not None else 0,
                     "epochs_total": int(rp.epochs.sum()) if rp is not None and "epochs" in rp else NE,
                     "mean_epoch_s": round(float(rp.mean_epoch_s.mean()), 1) if rp is not None and "mean_epoch_s" in rp else NE,
                     "gpu_train_min": r.gpu_train_min if r.gpu_train_min == r.gpu_train_min else NE,
                     "oof_predict_s": round(float(fm.predict_s.sum()), 1) if fm is not None and "predict_s" in fm else NE,
                     "gpu_peak_mb": round(float(fm.gpu_peak_mb.max()), 0) if fm is not None and "gpu_peak_mb" in fm
                     and fm.gpu_peak_mb.notna().any() else NE,
                     "discovery_train_s": round(float(dr.train_s.sum()), 1) if dr is not None and "train_s" in dr else NE})
    return pd.DataFrame(rows)


# =============================================================================== shortlist
def _rules(pc=None, rules: dict | None = None) -> dict:
    r = dict(DEFAULT_RULES)
    if pc is not None:
        r.update(pc.cfg.get("experiments.screening") or {})
    r.update(rules or {})
    return r


def shortlist(pc=None, root: Path | None = None, rules: dict | None = None, deltas: pd.DataFrame | None = None,
              comparison: pd.DataFrame | None = None) -> pd.DataFrame:
    """Recorded triage decision per SCREEN / VALIDATE candidate (read-only; see DEFAULT_RULES)."""
    from .experiments import role_of

    R = _rules(pc, rules)
    mc = comparison if comparison is not None else metrics_comparison(pc, root)
    pdl = deltas if deltas is not None else paired_deltas(pc, root, comparison=mc)
    rows = []
    fid = mc[mc.profile != "FULL"] if len(mc) else mc
    for eid, g in fid.groupby("experiment_id") if len(fid) else []:
        exp, prof = g.experiment.iloc[0], g.profile.iloc[0]
        rec = {"experiment": exp, "experiment_id": eid, "profile": prof, "role": role_of(exp)}
        if rec["role"] == "control":
            continue
        d = pdl[pdl.experiment_id == eid] if len(pdl) else pdl
        geo = g[g.metric_kind == "proxy_geo"]
        if not len(geo) or geo.status.iloc[0] != "ok":
            rows.append({**rec, "decision": "NOT_EVALUATED", "reason": "no proxy metrics saved"})
            continue
        if not len(d) or (d.status == "INCOMPARABLE").all():
            why = d.reason.iloc[0] if len(d) and "reason" in d else "no matched control run"
            rows.append({**rec, "decision": "INCOMPARABLE", "reason": why})
            continue
        ok = d[(d.status == "ok") & (d.zone == "geo")]
        prim = ok[ok.metric == R["primary"]]
        gain = float(pd.to_numeric(prim.delta).mean()) if len(prim) else np.nan
        rec300 = ok[ok.metric == "proxy_recall300_at_match"]
        drop300 = -float(pd.to_numeric(rec300.delta).mean()) if len(rec300) else 0.0
        beats = int((pd.to_numeric(prim.delta) > 0).sum())
        g0 = geo.iloc[0]
        hi, mean_p = _num(g0.get("frac_p_gt_099")) or 0.0, _num(g0.get("mean_p"))
        collapsed = hi > R["collapse_high_frac"] or (mean_p is not None and mean_p < R["collapse_low_mean"])
        identical = len(ok) and (pd.to_numeric(ok.delta).abs() < R["parity_tol"]).all()
        rec.update(primary_gain=gain, recall300_drop=drop300, folds_beating=beats, folds_compared=len(prim),
                   frac_p_gt_099=hi, mean_p=mean_p)
        if rec["role"] == "parity_check":
            rec.update(decision="PARITY_CONFIRMED" if identical else "REVIEW",
                       reason="numerically identical to its control (implementation parity)" if identical else
                       "parity check differs from its control: inspect architecture routing / nondeterminism")
        elif collapsed:
            rec.update(decision="REJECT", reason=f"prediction collapse (p>0.99 share {hi:.2f}, mean p {mean_p})")
        elif identical:
            rec.update(decision="REVIEW", reason="numerically identical to the control: the change may not be active")
        elif gain != gain:
            rec.update(decision="NOT_EVALUATED", reason="primary metric missing")
        elif gain <= -R["reject_max_loss"]:
            rec.update(decision="REJECT", reason=f"{R['primary']} {gain:+.4f} vs control (<= -{R['reject_max_loss']})")
        elif gain >= R["promote_min_gain"] and drop300 <= R["max_recall300_drop"] and \
                (prof == "SCREEN" or beats >= min(R["min_folds_beating"], len(prim))):
            rec.update(decision=NEXT_LEVEL[prof], reason=f"{R['primary']} {gain:+.4f} vs control at matched area; "
                       f"recall@300 change {-drop300:+.4f}; folds beating {beats}/{len(prim)}")
        else:
            rec.update(decision="REVIEW", reason=f"{R['primary']} {gain:+.4f} (promotion needs >= "
                       f"+{R['promote_min_gain']}); recall@300 change {-drop300:+.4f}; folds beating {beats}/{len(prim)}")
        rows.append(rec)
    out = pd.DataFrame(rows, columns=["experiment", "experiment_id", "profile", "role", "decision", "reason",
                                      "primary_gain", "recall300_drop", "folds_beating", "folds_compared",
                                      "frac_p_gt_099", "mean_p"])
    out["rules"] = json.dumps(R, sort_keys=True)
    return out


def shortlisted(pc=None, from_profile: str = "SCREEN", root: Path | None = None) -> list[str]:
    """Experiment stems recommended for the next fidelity level (the user decides to run them)."""
    s = shortlist(pc, root)
    return list(s[(s.profile == from_profile) & (s.decision == NEXT_LEVEL.get(from_profile, ""))].experiment.unique())


# =============================================================================== figures + report
def _style(ax):
    ax.grid(True, color="#e6e5e0", linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _colors(names) -> dict:
    return {n: SERIES[i % len(SERIES)] for i, n in enumerate(dict.fromkeys(names))}


def _fig_score_runtime(mc: pd.DataFrame):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    t = mc[mc.is_candidate == True].copy()  # noqa: E712
    t["score"] = [(_num(r.get("geo_dti_area")) if r["profile"] == "FULL" else
                   _num(r.get("proxy_dti_at_match")) if r.get("metric_kind") == "proxy_geo" else None)
                  for _, r in t.iterrows()]
    t["gpu"] = pd.to_numeric(t.gpu_train_min, errors="coerce")
    t = t.dropna(subset=["score", "gpu"])
    col = _colors(t.experiment)
    for _, r in t.iterrows():
        ax.scatter(r.gpu, r.score, s=64, marker=PROFILE_MARKER.get(r.profile, "o"), color=col[r.experiment],
                   edgecolor="white", linewidth=1.5, label=f"{r.experiment} ({r.profile})", zorder=3)
        ax.annotate(r.experiment, (r.gpu, r.score), xytext=(5, 4), textcoords="offset points", fontsize=8,
                    color="#52514e")
    ax.set_xlabel("recorded GPU training time (min)")
    ax.set_ylabel("geographic DTI (FULL: section 17; SCREEN/VALIDATE: proxy)")
    ax.set_title("Score vs GPU cost - profiles are not comparable with each other", fontsize=10)
    _style(ax)
    if len(t):
        ax.legend(fontsize=7, frameon=False, loc="best")
    else:
        ax.text(0.5, 0.5, "no evaluated experiment with recorded runtime", ha="center", transform=ax.transAxes)
    return fig


def _fig_fold_stability(mc: pd.DataFrame):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.2))
    t = mc[(mc.profile == "FULL") & (mc.status == "ok")].drop_duplicates("model")
    folds = [c for c in ("dti_fold0", "dti_fold1", "dti_fold2") if c in t]
    col = _colors(t.model)
    n = 0
    for _, r in t.iterrows():
        y = [_num(r.get(c)) for c in folds]
        if all(v is None for v in y):
            continue
        ax.plot(range(len(folds)), [np.nan if v is None else v for v in y], marker="o", markersize=8, linewidth=2,
                color=col[r.model], label=r.model)
        n += 1
    ax.set_xticks(range(len(folds)), [f.replace("dti_", "") for f in folds])
    ax.set_ylabel("geographic DTI per held-out fold (FULL)")
    ax.set_title("Fold stability (section-17 per-fold DTI)", fontsize=10)
    _style(ax)
    if n:
        ax.legend(fontsize=8, frameon=False)
    else:
        ax.text(0.5, 0.5, "per-fold DTI not available", ha="center", transform=ax.transAxes)
    return fig


def _fig_discovery(mc: pd.DataFrame):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.2))
    t = mc[(mc.profile == "FULL") & (mc.status == "ok")].drop_duplicates("model")
    t = t.assign(v=[_num(x) for x in t.get("disc_dti_excl", pd.Series([None] * len(t)))]).dropna(subset=["v"])
    ax.bar(range(len(t)), t.v, color=SERIES[0], width=0.6, edgecolor="white", linewidth=2)
    ax.set_xticks(range(len(t)), t.model, rotation=20, ha="right", fontsize=8)
    for i, v in enumerate(t.v):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8, color="#52514e")
    ax.set_ylabel("pseudo-discovery DTI (excl policy)")
    ax.set_title("Pseudo-discovery comparison (FULL runs)", fontsize=10)
    _style(ax)
    if not len(t):
        ax.text(0.5, 0.5, "no pseudo-discovery scores available", ha="center", transform=ax.transAxes)
    return fig


def _fig_budget_curves(base: Path, inv: pd.DataFrame):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.2))
    seen, n = set(), 0
    curves = []
    for r in inv[inv.profile == "FULL"].itertuples():
        c = _csv(base / "experiments" / r.experiment_id / "metrics" / "predicted_area_curve.csv")
        if c is not None and {"model", "dti"} <= set(c.columns):
            curves.append(c)
    allc = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()
    xcol = next((x for x in ("target_frac", "area_frac") if x in allc), None)
    col = _colors(allc.model if len(allc) else [])
    for m, g in (allc.groupby("model") if xcol else []):
        if m in seen:
            continue
        seen.add(m)
        g = g.groupby(xcol).dti.mean().reset_index().sort_values(xcol)
        ax.plot(g[xcol], g.dti, linewidth=2, color=col[m], label=m)
        n += 1
    ax.set_xscale("log")
    ax.set_xlabel("predicted-area fraction of the held-out zone")
    ax.set_ylabel("geographic DTI")
    ax.set_title("DTI vs prediction budget (FULL runs)", fontsize=10)
    _style(ax)
    if n:
        ax.legend(fontsize=8, frameon=False)
    else:
        ax.text(0.5, 0.5, "no predicted-area curves saved", ha="center", transform=ax.transAxes)
    return fig


def _md_table(df: pd.DataFrame, cols: list[str], n: int = 40) -> str:
    cols = [c for c in cols if c in df]
    if not len(df) or not cols:
        return "_none_\n"
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) and v == v else str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(r[c]) for c in cols) + " |" for _, r in df.head(n).iterrows()]
    return "\n".join(lines) + "\n"


def export_screening_report(pc=None, root: Path | None = None, out_root: Path | None = None) -> dict:
    """Write ``reports/experiment_screening/`` (CSV, PNG + SVG figures, SCREENING_REPORT.md) from saved results.

    Read-only with respect to experiments; figures come only from saved tables. Returns the written paths."""
    import matplotlib.pyplot as plt

    from .inventory import experiment_inventory

    base = _root(pc, root)
    out = Path(out_root or (pc.local if pc is not None else base)) / OUT_DIR
    (out / "figures").mkdir(parents=True, exist_ok=True)
    inv = experiment_inventory(pc, root=root)
    mc = metrics_comparison(pc, root, inv)
    pdl = paired_deltas(pc, root, inv, mc)
    ru = resource_usage(pc, root, inv)
    sl = shortlist(pc, root, deltas=pdl, comparison=mc)
    tables = {"experiment_inventory": inv, "metrics_comparison": mc, "paired_deltas": pdl, "resource_usage": ru,
              "shortlist_decisions": sl}
    for k, t in tables.items():
        t.to_csv(out / f"{k}.csv", index=False)
    figs = {"score_vs_runtime": _fig_score_runtime(mc) if len(mc) else None,
            "fold_stability": _fig_fold_stability(mc) if len(mc) else None,
            "discovery_comparison": _fig_discovery(mc) if len(mc) else None,
            "prediction_budget_curves": _fig_budget_curves(base, inv)}
    for k, f in figs.items():
        if f is None:
            continue
        f.tight_layout()
        f.savefig(out / "figures" / f"{k}.png", dpi=160)
        f.savefig(out / "figures" / f"{k}.svg")
        plt.close(f)
    full = mc[(mc.profile == "FULL")] if len(mc) else mc
    deltas_ok = pdl[(pdl.get("status", pd.Series(dtype=str)) == "ok")] if len(pdl) else pdl
    key = deltas_ok[deltas_ok.metric.isin(["geo_dti_area", "disc_dti_excl", "geo_recall300_at_match",
                                           "proxy_dti_at_match"])] if len(deltas_ok) else deltas_ok
    md = [f"# Experiment screening report\n\nGenerated {time.strftime('%Y-%m-%d %H:%M')} from saved artifacts under "
          f"`{base}` (read-only). Missing values are `{NE}`; nothing is imputed.\n",
          "Profiles: **FULL** = section-17 protocol (scoreboard / promotion gate). **SCREEN** / **VALIDATE** = proxy "
          "metrics on reduced budgets: triage only, never comparable with FULL numbers, never on the leaderboard.\n",
          "## Inventory\n", _md_table(inv, ["experiment_id", "role", "profile", "folds_done", "discovery_done",
                                           "evaluation_done", "reusable", "next_action"]),
          "## FULL results (section 17)\n", _md_table(full, ["experiment", "model", "is_candidate", "geo_dti_area",
                                                            "geo_fold_min", "disc_dti_excl", "geo_recall300_at_match",
                                                            "short_recall_w", "gpu_train_min"]),
          "## Paired candidate - control deltas (identical protocols only)\n",
          _md_table(key, ["experiment", "profile", "candidate", "control", "metric", "candidate_value",
                          "control_value", "delta"]),
          "## Shortlist decisions (triage heuristics, see `rules` column)\n",
          _md_table(sl, ["experiment_id", "profile", "decision", "reason"]),
          "## Resource usage (recorded)\n", _md_table(ru, ["experiment_id", "profile", "units_trained",
                                                          "epochs_total", "mean_epoch_s", "gpu_train_min"]),
          "## Notes\n",
          "* `e1_gsn_parity` is a parity check (same modules, widths, construction order and seed as B0): identical "
          "training to B0 is the expected outcome, not an architectural improvement.\n",
          "* `b0_reference` compares against the legacy `P4_B_gated`; when that map is not on Drive the delta is "
          f"`INCOMPARABLE`/`{NE}`.\n",
          "* Final model promotion for submission is a separate, manual registry transition.\n"]
    (out / "SCREENING_REPORT.md").write_text("\n".join(md), encoding="utf-8")
    if pc is not None:
        pc.push([OUT_DIR])
    return {"dir": str(out), **{k: str(out / f"{k}.csv") for k in tables}}


# =============================================================================== dashboard
def research_dashboard(pc, save: bool = True) -> dict:
    """Compact status of data, features, experiments, profile, remaining units, GPU time, maps, decisions, Drive."""
    import matplotlib.pyplot as plt

    from .experiments import plan_experiments, session_fidelity
    from .inventory import experiment_inventory, last_checkpoint, remaining_units
    from .resource_budget import profile_cost

    root = pc.drive or pc.local
    inv = experiment_inventory(pc)
    plan = plan_experiments(pc, inventory=inv)
    rem = remaining_units(inv)
    fid = session_fidelity(pc)
    fam = {}
    try:
        from .stages import feature_catalogue

        cat = feature_catalogue(pc, root)
        cat["cached"] = [(root / r).exists() for r in cat.rel]
        fam = cat.groupby("family").agg(n=("name", "count"), cached=("cached", "sum")).reset_index()
    except Exception as e:                                     # dashboard must never fail a session
        fam = pd.DataFrame([{"family": "?", "error": f"{type(e).__name__}: {e}"}])
    drive_gib = {}
    for rel in ("data/raw", "outputs/benchmark/cache/features", "outputs/phase4/cache/features", "experiments",
                "reports"):
        p = root / rel
        if p.exists():
            drive_gib[rel] = round(sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 2**30, 2)
    to_run = plan[plan.action.isin(["RUN", "RESUME"])] if len(plan) else plan
    cost = profile_cost(fid)
    summary = pd.DataFrame([
        {"item": "experiment profile", "value": fid.profile},
        {"item": "experiments complete / total", "value": f"{int(inv.complete.sum()) if len(inv) else 0} / {len(inv)}"},
        {"item": "incomplete (compatible)", "value": int(len(rem))},
        {"item": "folds remaining", "value": int(rem.folds_remaining.sum()) if len(rem) else 0},
        {"item": "discovery splits remaining", "value": int(rem.splits_remaining.sum()) if len(rem) else 0},
        {"item": f"planned runs at {fid.profile}", "value": ", ".join(to_run.experiment) if len(to_run) else "none"},
        {"item": "estimated GPU min per planned run (typical / upper)",
         "value": f"{cost['gpu_min_typical']} / {cost['gpu_min_upper']} (estimate)"},
        {"item": "recorded GPU training min (all runs)",
         "value": round(float(pd.to_numeric(inv.gpu_train_min, errors="coerce").sum()), 1) if len(inv) else 0},
        {"item": "model maps available (FULL)", "value": int(inv.maps.sum()) if len(inv) else 0},
        {"item": "last checkpoint", "value": json.dumps(last_checkpoint(root))},
        {"item": "Drive usage GiB", "value": json.dumps(drive_gib)},
    ])
    sl = shortlist(pc)
    out = {"summary": summary, "inventory": inv, "plan": plan, "features": fam, "shortlist": sl}
    fig, ax = plt.subplots(figsize=(7, max(2.2, 0.35 * len(inv) + 1)))
    if len(inv):
        lab = [f"{r.experiment_id} [{r.profile}]" for r in inv.itertuples()]
        done = [len([x for x in str(r.folds_done).split(",") if x not in ("", "nan")]) +
                len([x for x in str(r.discovery_done).split(",") if x not in ("", "nan")]) for r in inv.itertuples()]
        exp_ = [len([x for x in str(r.folds_expected).split(",") if x not in ("", "nan")]) +
                len([x for x in str(r.discovery_expected).split(",") if x not in ("", "nan")]) for r in inv.itertuples()]
        y = np.arange(len(lab))
        ax.barh(y, exp_, color="#e6e5e0", height=0.6, label="expected units")
        ax.barh(y, done, color=SERIES[0], height=0.6, label="completed units")
        ax.set_yticks(y, lab, fontsize=7)
        ax.invert_yaxis()
        ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax.set_xlabel("training units (folds + discovery splits)")
    ax.set_title(f"Experiment progress (session profile {fid.profile})", fontsize=10)
    _style(ax)
    fig.tight_layout()
    out["figure"] = fig
    try:
        from IPython.display import display

        for k in ("summary", "plan", "inventory", "shortlist"):
            display(out[k])
        display(fig)
    except ImportError:
        print(summary.to_string(index=False))
    if save:
        d = pc.reports
        for k in ("summary", "inventory", "plan", "shortlist"):
            out[k].to_csv(d / f"research_{k}.csv", index=False)
        fig.savefig(d / "figures" / "research_progress.png", dpi=160)
        fig.savefig(d / "figures" / "research_progress.svg")
        pc.push(["reports/pipeline"])
    plt.close(fig)
    return out
