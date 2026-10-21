"""Experiments A1 / A4 and section 17: one protocol, one scoreboard, explicit promotion decisions.

``evaluate_models``         the Phase-4 scorecard (geographic cross-fitted binary-area DTI, pseudo-discovery under
                            ``excl`` and ``retained_fp``, thin-structure recall) for any set of zoo / Phase-5 models,
                            all on the same folds, splits, budgets and policies, plus unique withheld systems.
``competition_scoreboard``  the A4 alignment table: every score family in its own column, never substituted:
                            area-selected vs raw geographic DTI, per-fold dispersion, discovery per policy,
                            short-fault recall, unique recovered fault length, public leaderboard (only when a
                            submission of exactly that model was scored and logged in reports/leaderboard_log.csv).
``promotion_decision``      section-17 gate (primary discovery policy, alternative policy, geographic, unique
                            systems, false-positive footprint, fold stability, reproducibility).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import phase4_eval
from ..metric import dti_by_zone

LEADERBOARD_COLUMNS = ["submitted_at", "model", "submission_file", "known_fault_policy", "public_score", "notes"]

DEFAULT_RULES = {
    "primary_policy": "excl", "alt_policy": "retained_fp",
    "min_disc_gain": 0.005,           # 1. primary-policy pseudo-discovery gain (cross-fitted area)
    "max_alt_drop": 0.002,            # 2. alternative policy must not regress by more than this
    "max_geo_drop": 0.005,            # 3. geographic DTI may not drop by more than this
    "min_unique_systems": 1,          # 4. withheld systems recovered (>= 0.5) that the control misses (by >= 0.3)
    "max_far_share_increase": 0.05,   # 5. share of the matched-budget mask > 1 km from any fault
    "max_fold_min_drop": 0.005,       # 6. worst geographic fold and worst discovery split
    "min_folds_beating": 2,           # 6. paired splits on which the primary discovery score improves
}


def evaluate_models(ctx, names: list[str], *, match_frac: float = 0.05, refs=("G1",), policies=("excl", "retained_fp")) -> dict:
    """Scorecard for ``names`` (+ refs) under the context's folds / splits; geographic and discovery separate.

    Same composition as ``phase4_eval.scorecard`` (identical numbers), but the discovery curves are kept so the
    per-split cross-fitted scores (stability criterion) come from the same computation.
    """
    import time

    from ..p3 import discovery3
    from ..p3.curves import crossfit_select

    t0 = time.perf_counter()
    names = list(dict.fromkeys(list(names) + [r for r in refs if r not in names]))
    maps = {n: ctx.oof(n) for n in names if ctx.has_oof(n)}
    disc = [n for n in names if ctx.has_discovery(n)]
    geo, curves = phase4_eval.geo_score(ctx, maps, match_frac)
    dc, objs, ds = pd.DataFrame(), {}, pd.DataFrame()
    per_split = []
    if disc:
        dc, objs = discovery3.discovery_curves(ctx, disc, policies=policies, variants=("model",))
        ds = discovery3.discovery_summary(dc)
        for (m, pol), g in dc.groupby(["model", "policy"]):
            per, _ = crossfit_select(g, ["target_frac"])
            per_split.append(per.assign(model=m, policy=pol))
    rec, K = phase4_eval.recovery_table(ctx, maps, match_frac)
    wide = geo.set_index("model").join(_recovery_columns(rec))
    dw = _disc_wide(ds, disc, policies, match_frac)
    if len(dw):
        wide = wide.join(dw.set_index("model"), how="left")
    sc = {"score": wide.reset_index(), "geo_curves": curves, "disc_curves": dc, "disc_summary": ds, "disc_objs": objs,
          "recovery": rec, "support": K,
          "disc_per_split": pd.concat(per_split, ignore_index=True) if per_split else pd.DataFrame()}
    uniq = pd.DataFrame()
    for pol in policies:
        u, lists = phase4_eval.unique_discovery_systems(ctx, objs, disc, refs=refs, frac=match_frac, policy=pol)
        uniq = pd.concat([uniq, u], ignore_index=True)
        sc[f"unique_lists_{pol}"] = lists
    sc["unique"] = uniq
    sc["raw_geo"] = raw_geo_dti(ctx, maps)
    print(f"evaluate_models: {len(maps)} maps, {len(disc)} with discovery, {time.perf_counter() - t0:.0f}s")
    return sc


def _recovery_columns(rec: pd.DataFrame) -> list:
    short = rec[(rec.stratum == "short_fault") & (rec.value == "<2 km")].set_index("model").weighted_recall.rename("short_recall_w")
    lb = rec[rec.stratum == "length_bin"].pivot_table(index="model", columns="value", values="weighted_recall")
    lb.columns = [f"recall_w_len_{c}" for c in lb.columns]
    st = rec[rec.stratum == "structure"].pivot_table(index="model", columns="value", values="weighted_recall")
    st.columns = [f"recall_w_{c}" for c in st.columns]
    cu = rec[(rec.stratum == "curved") & (rec.value == "curved")].set_index("model").weighted_recall.rename("recall_w_curved")
    return [short, lb, st, cu]


def _disc_wide(ds: pd.DataFrame, models, policies, frac) -> pd.DataFrame:
    """One row per model, as ``phase4_eval.discovery_score``."""
    if ds is None or not len(ds):
        return pd.DataFrame()
    cf = ds[ds.row == "crossfit_area"]
    rows = []
    for m in models:
        r = {"model": m}
        for pol in policies:
            s = cf[(cf.model == m) & (cf.policy == pol)]
            if len(s):
                s = s.iloc[0]
                r.update({f"disc_dti_{pol}": s.dti, f"disc_fold_min_{pol}": s.fold_min, f"disc_frac_{pol}": s.target_frac})
            mt = ds[(ds.row == "matched_budget") & (ds.model == m) & (ds.policy == pol) & np.isclose(ds.target_frac, frac)]
            if len(mt):
                r[f"disc_recall300_at_match_{pol}"] = float(mt.iloc[0]["recall@300m"])
        rows.append(r)
    return pd.DataFrame(rows)


def raw_geo_dti(ctx, maps: dict[str, np.ndarray]) -> pd.DataFrame:
    """DTI of the raw probability maps, pooled over folds (score transformation: none)."""
    rows = []
    for m, p in maps.items():
        t = dti_by_zone(p, ctx.data.labels, ctx.folds.fold_map, valid=ctx.data.valid)
        pooled = t[t.zone.astype(str) == "pooled_components"]
        rows.append({"model": m, "geo_dti_raw": float(pooled.dti.iloc[0]) if len(pooled) else np.nan})
    return pd.DataFrame(rows)


def competition_scoreboard(sc: dict, protocol_hash: str, leaderboard: pd.DataFrame | None = None,
                           ref: str = "G1") -> pd.DataFrame:
    s = sc["score"].copy()
    fold_cols = [c for c in s.columns if c.startswith("dti_fold")]
    out = pd.DataFrame({"model": s.model})
    out["geo_dti_area"] = s.geo_dti
    out = out.merge(sc["raw_geo"], on="model", how="left")
    out["geo_fold_min"], out["geo_fold_max"] = s.geo_fold_min.values, s.geo_fold_max.values
    out["geo_fold_std"] = s[fold_cols].std(axis=1, ddof=0).values if fold_cols else np.nan
    for pol in ("excl", "retained_fp"):
        for c in (f"disc_dti_{pol}", f"disc_fold_min_{pol}", f"disc_frac_{pol}", f"disc_recall300_at_match_{pol}"):
            out[c] = s[c].values if c in s else np.nan
    for c in ("short_recall_w", "recall_w_len_<1", "recall_w_len_1-2", "geo_far_share_at_match", "geo_recall300_at_match"):
        out[c] = s[c].values if c in s else np.nan
    u = sc.get("unique", pd.DataFrame())
    if len(u):
        for pol in ("excl", "retained_fp"):
            x = u[(u.ref == ref) & (u.policy == pol)].set_index("model")
            out[f"unique_systems_vs_{ref}_{pol}"] = out.model.map(x.n_unique_systems)
            out[f"unique_km_vs_{ref}_{pol}"] = out.model.map(x.unique_trace_km)
    out["public_lb"], out["public_lb_note"] = np.nan, ""
    if leaderboard is not None and len(leaderboard):
        lb = leaderboard.dropna(subset=["public_score"]).sort_values("submitted_at").groupby("model").last()
        out["public_lb"] = out.model.map(lb.public_score)
        out["public_lb_note"] = out.model.map(lb.known_fault_policy).fillna("")
    out["protocol_hash"] = protocol_hash
    return out


def load_leaderboard(path) -> pd.DataFrame:
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=LEADERBOARD_COLUMNS)
    return pd.read_csv(p)


def _per_split(sc: dict, model: str, policy: str) -> pd.Series:
    """Held-out (cross-fitted area) discovery DTI per split."""
    ps = sc.get("disc_per_split")
    if ps is None or not len(ps):
        return pd.Series(dtype=float)
    r = ps[(ps.model == model) & (ps.policy == policy)]
    return r.set_index("zone").dti_heldout


def promotion_decision(sc: dict, cand: str, ctrl: str, rules: dict | None = None, *,
                       artifacts_ok: bool | None = None) -> dict:
    """Section-17 promotion: every criterion is reported; ``promote`` needs all required ones."""
    R = {**DEFAULT_RULES, **(rules or {})}
    s = sc["score"].set_index("model")
    if cand not in s.index or ctrl not in s.index:
        return {"candidate": cand, "control": ctrl, "promote": False, "reason": "missing scorecard rows"}
    a, b = s.loc[cand], s.loc[ctrl]
    pp, ap = R["primary_policy"], R["alt_policy"]
    crit, notes = {}, {}

    def val(row, col):
        v = row.get(col, np.nan)
        return float(v) if pd.notna(v) else np.nan

    d1 = val(a, f"disc_dti_{pp}") - val(b, f"disc_dti_{pp}")
    crit["1_primary_discovery"] = bool(d1 >= R["min_disc_gain"])
    d2 = val(a, f"disc_dti_{ap}") - val(b, f"disc_dti_{ap}")
    crit["2_alt_policy_no_regression"] = bool(d2 >= -R["max_alt_drop"])
    d3 = val(a, "geo_dti") - val(b, "geo_dti")
    crit["3_geographic_competitive"] = bool(d3 >= -R["max_geo_drop"])
    u = sc.get("unique", pd.DataFrame())
    n_u = 0
    if len(u):
        x = u[(u.model == cand) & (u.ref == ctrl) & (u.policy == pp)]
        n_u = int(x.n_unique_systems.max()) if len(x) else 0
    crit["4_unique_systems"] = bool(n_u >= R["min_unique_systems"])
    d5 = val(a, "geo_far_share_at_match") - val(b, "geo_far_share_at_match")
    crit["5_fp_footprint"] = bool(np.isnan(d5) or d5 <= R["max_far_share_increase"])
    dmin_geo = val(a, "geo_fold_min") - val(b, "geo_fold_min")
    dmin_disc = val(a, f"disc_fold_min_{pp}") - val(b, f"disc_fold_min_{pp}")
    pa, pb = _per_split(sc, cand, pp), _per_split(sc, ctrl, pp)
    beating = int(((pa - pb.reindex(pa.index)) > 0).sum()) if len(pa) and len(pb) else None
    crit["6_stability"] = bool(dmin_geo >= -R["max_fold_min_drop"] and (np.isnan(dmin_disc) or dmin_disc >= -R["max_fold_min_drop"])
                               and (beating is None or beating >= R["min_folds_beating"]))
    crit["7_reproducible"] = bool(artifacts_ok) if artifacts_ok is not None else False
    notes.update(primary=f"disc {pp} {d1:+.4f}", alt=f"disc {ap} {d2:+.4f}", geo=f"geo {d3:+.4f}",
                 unique=f"{n_u} unique withheld systems vs {ctrl}", fp=f"far-share {d5:+.3f}",
                 stability=f"geo fold-min {dmin_geo:+.4f}, disc split-min {dmin_disc:+.4f}, splits beating {beating}")
    return {"candidate": cand, "control": ctrl, "promote": all(crit.values()), "criteria": crit, "notes": notes,
            "rules": R}


def decision_table(decisions: list[dict]) -> pd.DataFrame:
    rows = []
    for d in decisions:
        rows.append({"candidate": d["candidate"], "control": d["control"], "promote": d["promote"],
                     **{k: v for k, v in d.get("criteria", {}).items()},
                     "notes": "; ".join(d.get("notes", {}).values()) or d.get("reason", "")})
    return pd.DataFrame(rows)


def champion_report(board: pd.DataFrame, decisions: list[dict], title: str = "Champion vs challengers") -> str:
    """Markdown: scoreboard (separate score families) + one block per decision."""
    cols = [c for c in ("model", "geo_dti_area", "geo_dti_raw", "geo_fold_min", "disc_dti_excl", "disc_dti_retained_fp",
                        "short_recall_w", "geo_far_share_at_match", "public_lb") if c in board]
    lines = [f"# {title}", "", "Local scores are NOT leaderboard scores; geographic and pseudo-discovery numbers are",
             "different quantities and are never averaged or substituted.", "",
             board[cols].to_markdown(index=False, floatfmt=".4f"), ""]
    for d in decisions:
        lines += [f"## {d['candidate']} vs {d['control']}: {'PROMOTE' if d['promote'] else 'do not promote'}", ""]
        for k, v in d.get("criteria", {}).items():
            lines.append(f"- [{'x' if v else ' '}] {k}")
        for k, v in d.get("notes", {}).items():
            lines.append(f"  - {k}: {v}")
        lines.append("")
    if len(board) and "protocol_hash" in board:
        lines.append(f"protocol_hash: `{board.protocol_hash.iloc[0]}`")
    return "\n".join(lines)
