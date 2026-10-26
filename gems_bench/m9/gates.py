"""Research gates 0-6 (rebuilt from the saved tables only; thresholds live in the configuration).

    G0 valid baseline submission   the B0/B1 fallback GeoTIFF is written and passes validation      (section 01)
    G1 geological representation   spectral / new cross-modal layers add held-out value             (section 03)
    G2 dense model                 M9 dense improves on (PROMOTE) or complements (COMPLEMENT) B0/B1 (sections 09, 16)
    G3 conditional reconstruction  reconstruction recovers removed geometry without hurting discovery (section 09)
    G4 structured tracing          traced candidates cover enough held-out systems to justify ranking (section 11)
    G5 Monte Carlo / RL            value beyond beam search and basic ensembling                     (sections 12, 13)
    G6 final fusion                leakage-safe, reproducible, every promotion rule passed           (section 16)

A gate decided under a provisional profile (PILOT) is labelled PROVISIONAL. Oracle-based gates (G4) decide whether
work is worth doing; they never promote a model. Only G6 promotes, and the B0/B1 champion is retained otherwise.
"""
from __future__ import annotations

import json

import pandas as pd

SOURCES = [("G0_baseline_submission", "gate0", "baseline"), ("G1_representation", "gate1", "spectral"),
           ("G2_dense", "gate2", "validation"), ("G3_reconstruction", "gate3", "validation"),
           ("G4_tracing", "gate4", "tracing/oracle"), ("G5_extensions", "gate5", "reports"),
           ("G6_final_fusion", "gate6", "fusion")]
MEANING = {
    "PASS": "satisfied", "FAIL": "not satisfied",
    "INCLUDE": "keep the component in production inputs", "EXCLUDE": "drop the component from production inputs",
    "PROMOTE": "candidate for the leaderboard (still one upload at most)", "COMPLEMENT": "keep for fusion only",
    "RETAIN_RECONSTRUCTION": "keep the corruption mixture", "DROP_RECONSTRUCTION": "train with mixture none = 1 (recon_off)",
    "CONTINUE": "train / use candidate rankers", "STOP": "do not train rankers; widen tracing first",
    "UNCERTAINTY_REPORTING_ONLY": "keep Monte Carlo for uncertainty maps only", "RETAIN_BEAM": "keep beam search",
    "PROMOTE_CHALLENGER": "the RL / learned tracer may replace beam search", "RETAIN_CHAMPION": "submit the B0/B1 fallback",
    "NOT_EVALUATED": "section not run / gated / not in this profile"}
NEXT = {
    "G0_baseline_submission": {"FAIL": "fix the submission writer before anything else (template / CRS / NaN policy)"},
    "G1_representation": {"EXCLUDE": "keep spectral layers for diagnostics only (no_spectral is production)"},
    "G2_dense": {"FAIL": "inspect validation/metrics/paired_zone_deltas and the loss logs; do not tune on these zones",
                 "COMPLEMENT": "run section 16 (fusion with the legacy channel)"},
    "G3_reconstruction": {"DROP_RECONSTRUCTION": "set variants core.corruption.mixture none = 1 or promote recon_off",
                          "NOT_EVALUATED": "add recon_off to the profile variants (FULL)"},
    "G4_tracing": {"STOP": "widen tracing (lower seed_quantile, longer max_gap_px) before ranking",
                   "CONTINUE": "run sections 14-16"},
    "G6_final_fusion": {"RETAIN_CHAMPION": "upload nothing new; the fallback stays the reference",
                        "PROMOTE": "one controlled upload of the champion file, logged in submissions/"}}


def collect(m9) -> pd.DataFrame:
    rows = []
    for gate, name, sub in SOURCES:
        t = m9.ws.load_table(name, sub)
        if not len(t):
            rows.append({"gate": gate, "component": "", "decision": "NOT_EVALUATED",
                         "meaning": MEANING["NOT_EVALUATED"], "next_action": ""})
            continue
        for _, r in t.iterrows():
            r = r.to_dict()
            dec = str(r.get("decision"))
            prov = bool(r.get("provisional", m9.profile["provisional"]))
            rows.append({"gate": gate, "component": r.get("component", ""),
                         "decision": dec + (" (PROVISIONAL)" if prov and dec != "NOT_EVALUATED" else ""),
                         "meaning": MEANING.get(dec, ""), "next_action": NEXT.get(gate, {}).get(dec, ""),
                         "profile": r.get("profile", m9.profile["name"]), "rule": r.get("rule", ""),
                         "details": json.dumps({k: v for k, v in r.items() if k not in ("gate", "decision", "rule",
                                                                                       "profile", "component")},
                                               default=str)})
    return pd.DataFrame(rows)


def gate0(m9, info: dict | None, err: str | None = None) -> pd.DataFrame:
    base = {"gate": "G0_baseline_submission", "profile": m9.profile["name"], "provisional": False}
    if info is None:
        return pd.DataFrame([{**base, "decision": "FAIL", "reason": err or "fallback not written"}])
    fails = list(info["validation"].get("checks_failed") or [])
    return pd.DataFrame([{**base, "decision": "PASS" if not fails else "FAIL", "file": info["submission"],
                          "frac_of_eligible": info.get("final_frac_of_eligible"), "checks_failed": ";".join(fails),
                          "rule": "single float32 band, EPSG:32611, template grid, values in [0, 1], NaN only where the "
                                  "template is NaN (gems3.submission.validate)"}])


def gate1(m9, pooled: pd.DataFrame, uniq: pd.DataFrame, nested: pd.DataFrame) -> pd.DataFrame:
    """One row per component (spectral, crossmodal_new) from the nested pixel screen."""
    g = m9.cfg["gates"]["g1_representation"]
    rows = []
    for comp, a, b in (("spectral", "base_spectral", "base"), ("crossmodal_new", "base", "base_no_new")):
        p = pooled[pooled.task == m9.prim].set_index("feature_set") if len(pooled) else pd.DataFrame()
        if a not in getattr(p, "index", []) or b not in getattr(p, "index", []):
            rows.append({"gate": "G1_representation", "component": comp, "decision": "NOT_EVALUATED",
                         "reason": "screen rows missing"})
            continue
        gain = float(p.loc[a, "dti_pooled"] - p.loc[b, "dti_pooled"])
        u = uniq[(uniq.set_a == a) & (uniq.set_b == b) & (uniq.task == m9.prim)] if len(uniq) else uniq
        nu = int(u.unique_systems.iloc[0]) if len(u) else 0
        nz = nested[(nested.component == comp)] if len(nested) else nested
        ok = gain >= float(g["min_conditional_gain"]) or nu >= int(g["min_unique_systems"])
        rows.append({"gate": "G1_representation", "component": comp, "task": m9.prim,
                     "decision": "INCLUDE" if ok else "EXCLUDE", "conditional_dti_gain": gain, "unique_systems": nu,
                     "zones_agreeing": int(nz.agrees.fillna(False).astype(bool).sum()) if len(nz) else None,
                     "n_zones": int(len(nz)) if len(nz) else None,
                     "profile": m9.profile["name"], "provisional": bool(m9.profile["provisional"]),
                     "rule": f"pooled DTI gain >= {g['min_conditional_gain']} OR >= {g['min_unique_systems']} systems "
                             "recovered only with the component (pixel LightGBM screen, training geography only)"})
    return pd.DataFrame(rows)


def nested_component(t: pd.DataFrame, comp: str, a: str, b: str, min_gain: float) -> pd.DataFrame:
    """Per held-out zone: the include decision made on the OTHER zones vs the zone's own gain (consistency)."""
    if not len(t):
        return pd.DataFrame()
    rows = []
    for task, g in t.groupby("task"):
        zones = sorted(g.zone.unique())
        for z in zones:
            dev = g[g.zone != z]
            own = g[g.zone == z].set_index("feature_set").dti
            if a not in own or b not in own:
                continue
            d_own = float(own[a] - own[b])
            if len(dev):
                pa = dev[dev.feature_set == a]
                pb = dev[dev.feature_set == b]
                dt = lambda x: x.tp_w.sum() / (x.tp_w.sum() + 0.2 * x.fp_w.sum() + 0.8 * x.fn_w.sum() + 1e-6)
                dec = (dt(pa) - dt(pb)) >= min_gain
                rows.append({"component": comp, "task": task, "zone": z, "decided_on_other_zones": bool(dec),
                             "own_gain": d_own, "agrees": bool(dec) == (d_own >= min_gain)})
            else:
                rows.append({"component": comp, "task": task, "zone": z, "decided_on_other_zones": None,
                             "own_gain": d_own, "agrees": None})
    return pd.DataFrame(rows)
