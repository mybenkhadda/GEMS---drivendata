"""P6 full benchmark: every candidate through ONE evaluation interface, then a predeclared robust selection.

Inputs are saved grids only: the official components of every fixed (candidate, blend, rep, frac) cell on every unit,
written by the shared protocol (``m7.protocol.score_row``, eligibility ``m9.elig(u, 0)``): boost A-D, recovery X3 and
the W1 seed ensembles. Rows of different grids are pooled only when their protocol keys and truth mass agree on every
unit with the boost-A reference grid (same evaluation mask, truth, exclusion, budget domain and template). Each
candidate is a policy space evaluated by ``export_parity.evaluate_policy``: per outer zone ONE cell chosen on the other
zones, applied unchanged to that zone in both tasks, so every number is held-out and single-policy (or flagged).

    G, prov = load_grids(fg)                                       # Colab: grids + provenance
    T = benchmark_tables(G, specs, sel, prim, tasks, exclusion)    # pure: metrics, per zone, rankings, selection
    rc = choose(T["metrics"], sel)                                 # pure: eligibility, winner, baseline, reasons

Official quantities (``disc_*`` = disc_official DTI and components, ``geo_*`` = geographic DTI) are kept apart from
internal diagnostics (systems, line length, fragmentation, area), which never enter the selection rule.
"""
from __future__ import annotations

import json
from fnmatch import fnmatchcase

import numpy as np
import pandas as pd

from . import boost as B
from . import export_parity as EP

COMP = B.COMP
KEEP = ["task", "zone", "candidate", "blend", "rep", "frac", *COMP, "protocol_key", "area_px", "area_frac_eligible",
        "length_km", "n_segments", "short_share", "systems", "n_systems", "n_recovered", "role"]
DEPLOYABLE = ("fusion", "cnn", "stack", "seeds")         # kinds with a zone-stitched full-region builder
STATUSES = ("EVALUATED", "INCOMPLETE", "NOT_RUN", "BLOCKED", "NOT_FOUND", "INCOMPATIBLE", "FAILED")
OFFICIAL = ("disc_dti", "disc_worst", "disc_worst_zone", "disc_fp_w", "disc_precision_w", "disc_recall_w", "geo_dti",
            "geo_worst", "geo_fp_w")
# Matched-cell ablations on predeclared pairs (grid-prefixed names; None blend = the side has a single blend / NaN).
ABLATIONS = [("stacker_vs_fusion", ("boost_a:w2", 1.0), ("boost_a:fallback_fusion", None)),
             ("hybrid_vs_stacker", ("boost_a:hybrid", 0.5), ("boost_a:w2", 1.0)),
             ("fusion_vs_b0", ("boost_a:fallback_fusion", None), ("boost_a:cnn_b0", None)),
             ("fusion_vs_b1", ("boost_a:fallback_fusion", None), ("boost_a:cnn_b1", None)),
             ("b1_vs_b0", ("boost_a:cnn_b1", None), ("boost_a:cnn_b0", None)),
             ("g02c_prior", ("boost_b:S1", 1.0), ("boost_b:S0", 1.0)),
             ("g02a_prior", ("boost_b:S2", 1.0), ("boost_b:S1", 1.0)),
             ("g01_wavelet_persistence", ("boost_b:S3", 1.0), ("boost_b:S2", 1.0)),
             ("segment_h1", ("rec_x3:H1", 1.0), ("rec_x3:H0", 1.0)),
             ("segment_h2", ("rec_x3:H2", 1.0), ("rec_x3:H0", 1.0)),
             ("seed_tta", ("w1_seeds:seeds_tta|w0.5", None), ("w1_seeds:seeds|w0.5", None))]


# =============================================================================== pure: grids
def normalize_grid(G: pd.DataFrame | None, source: str, kind: str) -> pd.DataFrame:
    """One saved grid in the common schema; ``candidate`` becomes ``<source>:<grid candidate>`` (``orig_candidate``
    keeps the grid name). Seeds grids name a cell ``<method>|w<w>`` (the fusion weight is part of the candidate, not a
    blend). Oracle rows raise ``OracleLeak``; nothing else is dropped except the published-fallback duplicate."""
    cols = KEEP + ["source", "orig_candidate"]
    if G is None or not len(G):
        return pd.DataFrame(columns=cols)
    g = G.copy()
    if kind == "seeds":
        if not {"method", "w"} <= set(g):
            raise ValueError(f"{source}: a seeds grid needs 'method' and 'w' columns")
        g["candidate"] = [f"{m}|w{float(w):g}" for m, w in zip(g.method, g.w)]
        g["blend"] = np.nan
    miss = [c for c in ("task", "zone", "candidate", "rep", "frac", *COMP) if c not in g]
    if miss:
        raise ValueError(f"{source}: grid columns missing {miss}")
    EP.guard_oracle(g)
    g = g[g.candidate.astype(str) != "fallback_published"].copy()
    if "blend" not in g:
        g["blend"] = np.nan
    g["orig_candidate"] = g.candidate.astype(str)
    g["candidate"] = source + ":" + g.orig_candidate
    g["source"] = source
    g["zone"] = g.zone.astype(int)
    g["frac"] = g.frac.astype(float)
    g["blend"] = pd.to_numeric(g.blend, errors="coerce")
    return g[[c for c in cols if c in g]].reset_index(drop=True)


def match_cells(names, patterns) -> list[str]:
    """Grid candidate names matching any pattern (``fnmatch``: ``seeds|*`` = every seeds fusion weight)."""
    return sorted({str(n) for n in names for p in (patterns or []) if fnmatchcase(str(n), str(p))})


def protocol_check(G: pd.DataFrame, ref_source: str = "boost_a") -> tuple[pd.DataFrame, dict]:
    """Per (source, unit): protocol keys and truth mass vs the reference source. A source is consistent when every
    unit it shares with the reference has identical protocol keys (when recorded) and identical ``n_truth``."""
    if not len(G):
        return pd.DataFrame(), {}
    has_pk = "protocol_key" in G

    def facts(d):
        pk = set(d.protocol_key.dropna().astype(str)) if has_pk else set()
        return pk, set(np.round(d.n_truth.astype(float), 6))

    ref = {k: facts(d) for k, d in G[G.source == ref_source].groupby(["task", "zone"])}
    rows = []
    for (src, task, zone), d in G.groupby(["source", "task", "zone"]):
        pk, nt = facts(d)
        rk, rt = ref.get((task, zone), (set(), set()))
        pk_ok = (not pk or not rk) or pk == rk
        rows.append({"source": src, "task": task, "zone": int(zone), "protocol_keys": ";".join(sorted(pk)),
                     "n_truth": ";".join(f"{x:g}" for x in sorted(nt)), "in_reference": (task, zone) in ref,
                     "protocol_key_ok": bool(pk_ok), "truth_ok": bool(not rt or nt == rt), "unique_key": len(pk) <= 1})
    T = pd.DataFrame(rows)
    T["ok"] = T.protocol_key_ok & T.truth_ok & T.unique_key
    return T, {s: bool(d.ok.all()) for s, d in T.groupby("source")}


def candidate_space(spec: dict, G: pd.DataFrame) -> dict:
    """``export_parity.policy_rows`` spec of one configured candidate on the pooled grid."""
    src = G[G.source == spec["grid"]] if len(G) else G
    names = match_cells(src.orig_candidate.unique() if len(src) else [], spec.get("cells"))
    sp = {"candidates": [f"{spec['grid']}:{n}" for n in names]}
    for k in ("reps", "fracs", "blends", "fixed"):
        if spec.get(k) is not None:
            sp[k] = spec[k]
    return sp


# =============================================================================== pure: evaluation
def evaluate_candidates(G: pd.DataFrame, specs: dict, objective: str, prim: str, tasks: list[str],
                        exclusion: dict | None = None, geo_floor: float = 0.0, consistent: dict | None = None,
                        zones=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Nested single-policy numbers of every candidate on the SAME zones (the reference grid's): a candidate missing
    a zone or a task is INCOMPLETE, never pooled on fewer zones. Specs carrying ``status`` (absent grid, unresolved
    recovery choice) are reported as such."""
    zones = sorted(zones) if zones is not None else (sorted(G[G.source == "boost_a"].zone.unique()) if len(G) else [])
    rows, pers = [], []
    for name, spec in specs.items():
        base = {"name": name, "role": spec.get("role", "candidate"), "family": spec.get("family"),
                "grid": spec.get("grid"), "deploy": spec.get("deploy", "none"), "label": spec.get("label", ""),
                "objective": objective, "members": json.dumps(list(spec.get("members") or []))}
        if spec.get("status"):
            rows.append({**base, "status": spec["status"], "status_reason": spec.get("reason", "")})
            continue
        sp = candidate_space(spec, G)
        if not sp["candidates"]:
            rows.append({**base, "status": "NOT_FOUND", "status_reason": f"no rows for cells {spec.get('cells')} in "
                                                                          f"grid {spec.get('grid')}"})
            continue
        if consistent is not None and not consistent.get(spec["grid"], False):
            rows.append({**base, "status": "INCOMPATIBLE", "status_reason": f"grid {spec['grid']}: protocol keys or "
                                                                             "truth mass differ from boost_a"})
            continue
        s, P = EP.evaluate_policy(G, name, sp, objective, prim, float(geo_floor), tasks, zones or None, exclusion)
        done = all(np.isfinite(s.get(f"{'disc' if t == prim else 'geo'}_dti", np.nan)) for t in tasks)
        n_cells = int(EP.policy_rows(G, sp).groupby(EP.CELL, dropna=False).ngroups)
        rows.append({**base, **{k: v for k, v in s.items() if k != "policy"}, "status": "EVALUATED" if done else
                     "INCOMPLETE", "status_reason": "" if done else "a task or zone has no held-out row",
                     "space": json.dumps(sp, default=str), "n_cells_space": n_cells})
        pers.append(P.rename(columns={"candidate": "cell_candidate", "policy": "name"}))
    M = pd.DataFrame(rows)
    PZ = pd.concat(pers, ignore_index=True) if pers else pd.DataFrame()
    return M, PZ


def zone_metrics(PZ: pd.DataFrame, prim: str) -> pd.DataFrame:
    """Per candidate, task and zone: the held-out cell and its official components; weakest zone flagged."""
    if not len(PZ):
        return PZ
    Z = PZ.copy()
    Z["metric"] = np.where(Z.task == prim, "disc_official_dti", "geo_dti")
    Z["is_weakest"] = False
    ok = Z.dti.notna()
    idx = Z[ok].groupby(["name", "task"]).dti.idxmin()
    Z.loc[idx.to_numpy(), "is_weakest"] = True
    return Z


def matched_ablation(G: pd.DataFrame, name: str, a: tuple, b: tuple, prim: str) -> list[dict]:
    """In-sample matched-cell comparison of two grid candidates (same rep and frac; blends fixed per side): mean and
    best delta of pooled DTI, share of cells where ``a`` wins, worst-zone delta. Diagnostic: unselected cells."""
    (ca, ba), (cb, bb) = a, b
    D = G[G.candidate.isin([ca, cb])] if len(G) else G
    if not len(D) or D.candidate.nunique() < 2:
        return [{"ablation": name, "a": ca, "b": cb, "status": "absent", "note": "one side has no grid rows"}]
    M = B.matched_comparison(D)
    out = []
    for task, g in M.groupby("task"):
        sa, sb = g[g.candidate == ca], g[g.candidate == cb]
        if ba is not None:
            sa = sa[np.isclose(sa.blend.fillna(-1.0), float(ba))]
        if bb is not None:
            sb = sb[np.isclose(sb.blend.fillna(-1.0), float(bb))]
        X = sa.merge(sb, on=["rep", "frac"], suffixes=("_a", "_b"))
        if not len(X) or X.duplicated(["rep", "frac"]).any():
            out.append({"ablation": name, "task": task, "a": ca, "b": cb, "status": "unmatched",
                        "note": "no unique (rep, frac) pairing"})
            continue
        d = X.dti_a - X.dti_b
        out.append({"ablation": name, "task": task, "metric": "disc_official_dti" if task == prim else "geo_dti",
                    "a": ca, "b": cb, "blend_a": ba, "blend_b": bb, "status": "ok", "n_cells": int(len(X)),
                    "mean_delta": float(d.mean()), "best_cell_delta": float((X.dti_a.max() - X.dti_b.max())),
                    "win_share": float((d > 0).mean()), "worst_zone_delta_mean": float((X.worst_zone_dti_a -
                                                                                        X.worst_zone_dti_b).mean()),
                    "note": "fixed cells, unselected (in-sample diagnostic; not a nested estimate)"})
    return out


def nested_deltas(M: pd.DataFrame, baseline: str | None) -> pd.DataFrame:
    """Nested deltas of every evaluated candidate against the baseline (disc, worst, geo; per zone)."""
    if baseline is None or not len(M) or baseline not in set(M.name):
        return pd.DataFrame()
    b = M[M.name == baseline].iloc[0]
    rows = []
    for r in M[M.status == "EVALUATED"].to_dict("records"):
        pz, bz = _zones(r.get("disc_per_zone")), _zones(b.get("disc_per_zone"))
        rows.append({"ablation": f"{r['name']}_vs_{baseline}", "a": r["name"], "b": baseline, "status": "ok",
                     "metric": "nested single-policy", "delta_disc": _num(r.get("disc_dti")) - _num(b.get("disc_dti")),
                     "delta_worst": _num(r.get("disc_worst")) - _num(b.get("disc_worst")),
                     "delta_geo": _num(r.get("geo_dti")) - _num(b.get("geo_dti")),
                     "zone_deltas": json.dumps({z: pz[z] - bz[z] for z in pz if z in bz})})
    return pd.DataFrame(rows)


# =============================================================================== pure: selection
def _num(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return np.nan
    return v


def _zones(d) -> dict:
    if isinstance(d, str):
        d = json.loads(d) if d else {}
    return {int(k): float(v) for k, v in (d or {}).items()}


def _members(r) -> list[str]:
    m = r.get("members")
    if isinstance(m, str):
        m = json.loads(m) if m else []
    return list(m or [])


def ensemble_check(r: dict, by_name: dict, min_gain: float) -> tuple[bool, str]:
    """An ensemble (``members`` set) must beat its best evaluated member by ``min_gain`` (nested disc DTI) without a
    lower worst zone. Out-of-fold evidence only."""
    mem = _members(r)
    if not mem:
        return True, ""
    ev = [by_name.get(m) for m in mem]
    miss = [m for m, x in zip(mem, ev) if x is None or x.get("status") != "EVALUATED"]
    if miss:
        return False, f"ensemble members not evaluated: {miss}"
    best = max(ev, key=lambda x: _num(x.get("disc_dti")))
    gain = _num(r.get("disc_dti")) - _num(best.get("disc_dti"))
    if not gain >= float(min_gain) - 1e-12:
        return False, f"ensemble gain {gain:+.4f} over best member {best['name']} < {float(min_gain):g}"
    if _num(r.get("disc_worst")) < _num(best.get("disc_worst")) - 1e-12:
        return False, f"ensemble worst zone below member {best['name']}"
    return True, f"ensemble gain {gain:+.4f} over {best['name']}"


def geo_check(r: dict, rule: dict, ref_geo: float | None) -> tuple[bool, str]:
    mode = rule.get("mode", "relative")
    g = _num(r.get("geo_dti"))
    if mode == "report":
        return True, ""
    if not np.isfinite(g):
        return False, "no geographic DTI"
    if mode == "gate":
        return (g >= float(rule["floor"]), "" if g >= float(rule["floor"]) else f"geo {g:.4f} < floor {rule['floor']}")
    if ref_geo is None or not np.isfinite(ref_geo):
        return True, "no reference geo DTI (relative rule not applied)"
    lim = float(ref_geo) - float(rule.get("max_drop", 0.0))
    return (g >= lim - 1e-12, "" if g >= lim - 1e-12 else f"geo {g:.4f} < reference {ref_geo:.4f} - "
                                                         f"{float(rule.get('max_drop', 0.0)):g}")


def choose(M: pd.DataFrame, sel: dict) -> dict:
    """Predeclared rule. Eligible = evaluated, one deployable policy, a validated full-region builder, protocol-
    consistent, worst zone >= floor, geographic rule, ensemble justified, role allowed. Baseline = the strongest
    evaluated deployable reference. Winner = the best eligible candidate (ties within ``tie_tolerance``: higher worst
    zone, then fewer members, then name), kept only if it beats the baseline by ``min_gain_over_reference`` AND is not
    worse on at least ``min_zones_not_worse`` zones; otherwise the baseline stays."""
    rows = M.to_dict("records") if len(M) else []
    by = {r["name"]: r for r in rows}
    refs = [by[n] for n in sel.get("references", []) if n in by and by[n].get("status") == "EVALUATED" and
            by[n].get("deployable") and by[n].get("deploy") in DEPLOYABLE]
    baseline = max(refs, key=lambda r: (_num(r.get("disc_dti")), r["name"])) if refs else None
    ref_geo = _num(baseline.get("geo_dti")) if baseline else None
    out_rows = []
    for r in rows:
        why = []
        if r.get("status") != "EVALUATED":
            why.append(f"status {r.get('status')}: {r.get('status_reason', '')}".strip())
        else:
            if not r.get("deployable"):
                why.append("not one deployable policy")
            if r.get("deploy") not in DEPLOYABLE:
                why.append(f"no validated full-region builder for deploy kind {r.get('deploy')!r}")
            w = _num(r.get("disc_worst"))
            if not (np.isfinite(w) and w >= float(sel["worst_floor"]) - 1e-12):
                why.append(f"worst zone {w:.4f} < {float(sel['worst_floor']):g}")
            ok, msg = geo_check(r, sel["geo_rule"], ref_geo)
            if not ok:
                why.append(msg)
            ok, msg = ensemble_check(r, by, float(sel["ensemble_min_gain"]))
            if not ok:
                why.append(msg)
            if r.get("role") in set(sel.get("never_winner_roles", [])) or str(r.get("role")).startswith(
                    ("oracle", "historical")):
                why.append(f"role {r.get('role')} is never selected")
        out_rows.append({**r, "eligible": not why, "eligibility_reasons": "; ".join(why)})
    E = sorted([r for r in out_rows if r["eligible"]], key=lambda r: -_num(r.get("disc_dti")))
    top = None
    if E:
        tol = float(sel["tie_tolerance"])
        tied = [r for r in E if _num(r.get("disc_dti")) >= _num(E[0].get("disc_dti")) - tol]
        top = sorted(tied, key=lambda r: (-_num(r.get("disc_worst")), len(_members(r)), r["name"]))[0]
    reasons, decision, gain = [], None, None
    if top is None:
        winner, decision = baseline, "NO_ELIGIBLE_CANDIDATE"
        reasons.append("no candidate passes eligibility: the strongest reference is exported")
    elif baseline is None or top["name"] == baseline["name"]:
        winner, decision = top, ("REFERENCE_RETAINED" if baseline else "NO_REFERENCE")
    else:
        gain = _num(top.get("disc_dti")) - _num(baseline.get("disc_dti"))
        pz, bz = _zones(top.get("disc_per_zone")), _zones(baseline.get("disc_per_zone"))
        nw = sum(pz[z] >= bz[z] - float(sel["tie_tolerance"]) for z in pz if z in bz)
        if gain >= float(sel["min_gain_over_reference"]) - 1e-12 and nw >= int(sel["min_zones_not_worse"]):
            winner, decision = top, "REPLACE"
        else:
            winner, decision = baseline, "REFERENCE_RETAINED"
            reasons.append(f"best eligible {top['name']}: gain {gain:+.4f} (min {sel['min_gain_over_reference']}), "
                           f"zones not worse {nw} (min {sel['min_zones_not_worse']})")
    slot = bool(decision == "REPLACE" and gain is not None and gain >= float(sel["slot_min_gain"]) - 1e-12)
    for r in out_rows:
        r["is_winner"] = bool(winner and r["name"] == winner["name"])
        r["is_baseline"] = bool(baseline and r["name"] == baseline["name"])
        r["delta_disc_vs_baseline"] = (_num(r.get("disc_dti")) - _num(baseline.get("disc_dti"))) if baseline else None
    return {"rows": out_rows, "winner": winner["name"] if winner else None, "baseline": baseline["name"] if baseline
            else None, "best_eligible": top["name"] if top else None, "decision": decision, "gain_vs_baseline": gain,
            "slot_worthy": slot, "reasons": reasons}


def rankings(rows: list[dict]) -> pd.DataFrame:
    """Eligible first, then nested disc DTI, worst zone, name."""
    R = pd.DataFrame(rows)
    if not len(R):
        return R
    num = (lambda c: pd.to_numeric(R[c], errors="coerce").fillna(-np.inf) if c in R else -np.inf)
    R = R.assign(_d=num("disc_dti"), _w=num("disc_worst"))
    R = R.sort_values(["eligible", "_d", "_w", "name"], ascending=[False, False, False, True]).drop(columns=["_d", "_w"])
    R.insert(0, "rank", np.arange(1, len(R) + 1))
    keep = ["rank", "name", "label", "role", "family", "status", "eligible", "eligibility_reasons", "disc_dti",
            "disc_worst", "disc_worst_zone", "geo_dti", "delta_disc_vs_baseline", "deploy", "deploy_cell",
            "is_winner", "is_baseline"]
    return R[[c for c in keep if c in R]]


def benchmark_tables(G: pd.DataFrame, specs: dict, sel: dict, prim: str, tasks: list[str],
                     exclusion: dict | None = None, consistent: dict | None = None,
                     exclusion_feasible: bool = True) -> dict:
    """Every benchmark table from the pooled grid (pure). The deployment objective falls back to ``disc`` when the
    overlap exclusion is infeasible (geo labels of the held-out geography would otherwise enter the selection)."""
    obj = sel["objective"] if exclusion_feasible or sel["objective"] == "disc" else "disc"
    floor = float(sel["geo_rule"].get("floor", 0.0))
    M, PZ = evaluate_candidates(G, specs, obj, prim, tasks, exclusion, floor, consistent)
    sens = []
    for o in [x for x in sel.get("sensitivity_objectives", []) if x != obj]:
        Ms, _ = evaluate_candidates(G, specs, o, prim, tasks, exclusion, floor, consistent)
        sens.append(Ms.assign(sensitivity=True))
    ch = choose(M, sel)
    abl = [x for n, a, b in ABLATIONS for x in matched_ablation(G, n, a, b, prim)]
    A = pd.concat([pd.DataFrame(abl), nested_deltas(M, ch["baseline"])], ignore_index=True)
    return {"metrics": M, "per_zone": zone_metrics(PZ, prim), "sensitivity": pd.concat(sens, ignore_index=True)
            if sens else pd.DataFrame(), "selection": ch, "rankings": rankings(ch["rows"]), "ablations": A,
            "objective": obj, "objective_predeclared": sel["objective"]}


def meta_estimate(G: pd.DataFrame, M: pd.DataFrame, specs: dict, ch: dict, sel: dict, prim: str, tasks: list[str],
                  exclusion: dict | None, objective: str) -> dict:
    """Honest estimate of 'choose the candidate, then deploy it' (``export_parity.meta_nested``) over the eligible
    candidates: the choice itself is nested (noisy with three zones)."""
    names = [r["name"] for r in ch["rows"] if r["eligible"]]
    if len(names) < 1:
        return {"status": "NOT_RUN", "reason": "no eligible candidate"}
    sp = {n: candidate_space(specs[n], G) for n in names}
    gates = {"geo_floor": float(sel["geo_rule"]["floor"]) if sel["geo_rule"].get("mode") == "gate" else -np.inf,
             "worst_floor": float(sel["worst_floor"])}
    meta, MP = EP.meta_nested(G, sp, objective, prim, gates, tasks, {n: "deploy_candidate" for n in names}, exclusion)
    return {**meta, "status": "RUN", "candidates": names, "per_zone": MP.to_dict("records") if len(MP) else []}


# =============================================================================== Colab side
def _read_json(p):
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    except (OSError, json.JSONDecodeError):
        return None


def load_grids(fg) -> tuple[pd.DataFrame, dict]:
    """Every configured grid present on disk, normalised, with provenance (sha256, producing-stage fingerprint, code
    commit, status). Absent grids are recorded with the producing stage's own status / decision."""
    from .boost_audit import file_hash

    root = fg.p6.ws.root
    parts, prov = [], {}
    for src, gc in fg.cfg["grids"].items():
        p = root / gc["path"]
        man = _read_json(root / gc["stage"] / "manifest.json") or {}
        st = _read_json(root / gc["stage"] / "status.json") or {}
        info = {"path": gc["path"], "exists": p.exists(), "stage_status": st.get("status"),
                "stage_fingerprint": man.get("fingerprint"), "stage_commit": (man.get("code") or {}).get("git_commit"),
                "stage_seconds": man.get("seconds")}
        if p.exists():
            G = pd.read_csv(p, dtype={"systems": str})
            g = normalize_grid(G, src, gc["kind"])
            parts.append(g)
            info.update(sha256=file_hash(p), rows=int(len(g)), candidates=sorted(g.orig_candidate.unique()))
        else:
            notes = {}
            for rel in fg.cfg.get("not_run_sources", {}).get(src, []):
                j = _read_json(root / rel)
                if j is not None:
                    notes[rel] = {k: j.get(k) for k in ("status", "stage_status", "reason", "repair_status", "decision",
                                                        "s1_decision", "verdict") if k in j}
            info["not_run_evidence"] = notes
        prov[src] = info
    G = pd.concat([g for g in parts if len(g)], ignore_index=True) if parts else pd.DataFrame(columns=KEEP)
    return G, prov


def resolve_specs(fg, prov: dict) -> dict:
    """Configured candidates with absent grids marked NOT_RUN / BLOCKED, and the recovery choice resolved from
    ``recovery/export_parity/decision.json`` (its policy space in recovery.yaml)."""
    specs = {}
    for name, sp in fg.cfg["candidates"].items():
        sp = dict(sp)
        if sp.get("from_recovery"):
            dec = fg.rc.read_json("export_parity", "decision") or {}
            pol = dec.get("chosen_policy")
            pspec = (fg.rc.cfg["export_parity"].get("policies") or {}).get(pol) if pol else None
            if not pspec:
                sp.update(status="NOT_RUN", reason="recovery/export_parity/decision.json has no chosen policy")
            else:
                sp.update({k: pspec[k] for k in ("candidates", "reps", "fracs", "blends", "fixed") if k in pspec})
                sp["cells"] = sp.pop("candidates", [])
                sp["label"] = f"{sp.get('label', '')}: {pol} ({dec.get('objective')})"
        g = prov.get(sp.get("grid"), {})
        if not sp.get("status") and not g.get("exists"):
            ev = g.get("not_run_evidence") or {}
            blocked = any("BLOCKED" in json.dumps(v) or "READY_FOR_BACKFILL" in json.dumps(v) for v in ev.values())
            sp.update(status="BLOCKED" if blocked else "NOT_RUN",
                      reason=f"grid {g.get('path')} absent; evidence: {json.dumps(ev, default=str)[:400]}")
        specs[name] = sp
    return specs


def provenance_rows(fg, M: pd.DataFrame, prov: dict, specs: dict) -> list[dict]:
    """experiment_manifest entries: source grid / checkpoint / configuration / data fingerprint / code commit."""
    from .._version import provenance

    code = provenance()
    w2 = fg.bst.w2_inputs()
    try:
        members = fg.p6.m9.m7.g3.members()
    except Exception as e:                                   # provenance never blocks the benchmark
        members = [{"error": f"{type(e).__name__}: {e}"}]
    out = []
    for r in M.to_dict("records"):
        sp, g = specs.get(r["name"], {}), prov.get(r.get("grid"), {})
        ck = {"cnn_members": [{"model": m.get("model"), "experiment_id": m.get("experiment_id"),
                               "checkpoints": m.get("checkpoints")} for m in members]}
        if r.get("deploy") == "stack":
            ck["w2"] = {"candidate_json": (w2.get("candidate") or {}).get("features"), "maps_sha256": w2["maps_sha256"],
                        "rows_sha256": w2["rows_sha256"]}
        out.append({"name": r["name"], "status": r.get("status"), "role": r.get("role"), "grid": r.get("grid"),
                    "grid_sha256": g.get("sha256"), "producing_stage_fingerprint": g.get("stage_fingerprint"),
                    "producing_stage_commit": g.get("stage_commit"), "producing_stage_status": g.get("stage_status"),
                    "spec": sp, "checkpoints": ck, "benchmark_commit": code.get("git_commit")})
    return out


def run(fg) -> dict:
    """Benchmark stage: grids -> protocol check -> nested evaluation -> rankings / ablations -> selection files."""
    stage = "benchmark"
    sel, prim, tasks = fg.cfg["selection"], fg.prim, fg.cfg["tasks"]
    G, prov = load_grids(fg)
    specs = resolve_specs(fg, prov)
    fp = fg.fp(stage, fg.cfg["grids"], fg.cfg["candidates"], sel, {k: v.get("sha256") for k, v in prov.items()},
               fg.rc.read_json("export_parity", "decision"))
    if fg.done(stage, fp):
        return fg.load(stage)
    t0 = fg.begin(stage, fp)
    PC, consistent = protocol_check(G)
    ov = fg.rc.load_table("unit_overlap", "export_parity")
    zones = sorted(G[G.source == "boost_a"].zone.unique()) if len(G) else []
    if not len(ov):
        ov = EP.unit_overlap(fg.bst)
    ex, feasible = EP.overlap_exclusion(ov, zones, prim) if len(ov) and zones else (None, False)
    T = benchmark_tables(G, specs, sel, prim, tasks, ex, consistent, feasible)
    ch = T["selection"]
    meta = meta_estimate(G, T["metrics"], specs, ch, sel, prim, tasks, ex, T["objective"])
    rows = pd.DataFrame([B._flat(r) for r in ch["rows"]])
    off = [c for c in rows if c in OFFICIAL or c.endswith(("_per_zone",))]
    fg.save_table(rows, "candidate_metrics", stage)
    fg.save_parquet(rows, "candidate_metrics", stage)
    fg.save_table(T["per_zone"].drop(columns=["systems"], errors="ignore"), "zone_metrics", stage)
    fg.save_table(T["rankings"], "candidate_rankings", stage)
    fg.save_table(T["ablations"], "candidate_ablations", stage)
    fg.save_table(T["sensitivity"].pipe(lambda d: pd.DataFrame([B._flat(r) for r in d.to_dict("records")])),
                  "objective_sensitivity", stage)
    fg.save_table(PC, "protocol_consistency", stage)
    man = {"stage": stage, "objective": T["objective"], "objective_predeclared": T["objective_predeclared"],
           "overlap_exclusion_feasible": feasible, "grids": prov, "protocol_keys": fg.bst.protocol_keys(),
           "input_data_fingerprint": fg.data_fingerprint(), "config_fp": fg.fp(fg.cfg),
           "candidates": provenance_rows(fg, T["metrics"], prov, specs),
           "official_columns": off, "diagnostic_columns": ["systems_*", "length_km", "n_segments", "short_share",
                                                            "area_*"],
           "note": "nested held-out numbers from saved grids; NOT leaderboard scores; unexecuted candidates are listed "
                   "with their status, never with a number"}
    fg.save_json(man, stage, "experiment_manifest")
    win = next((r for r in ch["rows"] if r["is_winner"]), None)
    msel = {"rule": {k: sel[k] for k in sel}, "objective": T["objective"], "decision": ch["decision"],
            "winner": ch["winner"], "baseline": ch["baseline"], "best_eligible": ch["best_eligible"],
            "gain_vs_baseline": ch["gain_vs_baseline"], "slot_worthy": ch["slot_worthy"], "reasons": ch["reasons"],
            "winner_summary": B._flat(win) if win else None, "meta_nested": meta,
            "eligible": [r["name"] for r in ch["rows"] if r["eligible"]],
            "ineligible": {r["name"]: r["eligibility_reasons"] for r in ch["rows"] if not r["eligible"]},
            "note": "selection uses held-out zones only; leaderboard feedback is never an input"}
    fg.save_json(msel, stage, "model_selection")
    fg.finish(stage, fp, t0, {"grids": {k: v.get("sha256") for k, v in prov.items()}},
              {"winner": ch["winner"], "decision": ch["decision"]})
    return fg.load(stage)
