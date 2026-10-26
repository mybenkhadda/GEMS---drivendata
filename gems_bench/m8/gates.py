"""Research gates 1-6 (decisions are rebuilt from the saved tables only; thresholds live in the configuration).

    G1 signal recovery      do M8 physics stages cover withheld systems that the M7 chain lost?   (section 02 + 06)
    G2 candidate coverage   system recall at controlled density vs the M7 pool (oracle; diagnostic) (section 06)
    G3 Monte Carlo          unique coverage at matched density OR ranking gain                     (sections 07 + 11)
    G4 DEM                  systems recovered only with DEM at matched density                     (section 09)
    G5 ranking              best learned ranker beats max(label-free, CNN-only)                    (section 11)
    G6 deployable gain      cross-fitted, geographically robust DTI gain over the CNN champion     (section 14)

A gate decided under a provisional profile (PILOT) is labelled PROVISIONAL. G2 uses truth-assisted coverage to decide
whether ranking is worth running; it never promotes anything. Only G6 (held-out maps, nested choices, paired zones)
can promote a model, and the B0/B1 champion is retained otherwise.
"""
from __future__ import annotations

import json

import pandas as pd

SOURCES = [("G1_signal_recovery", "gate1", "signal_recovery"), ("G2_candidate_coverage", "gate2", "candidate_generation/oracle"),
           ("G3_monte_carlo", "gate3", "monte_carlo/stability"), ("G4_dem", "gate4", "dem/candidates"),
           ("G5_ranking", "gate5", "ranking"), ("G6_deployable", "gate6", "ensemble")]
MEANING = {"G1_signal_recovery": {"PASS": "physics recovers systems the M7 chain lost", "FAIL": "see diagnosis"},
           "G2_candidate_coverage": {"GO": "rank the candidates", "NO_GO": "stop ranking; widen generation"},
           "G3_monte_carlo": {"USE_IN_MODEL": "keep MC features / structures",
                              "UNCERTAINTY_REPORTING_ONLY": "keep MC for uncertainty maps only"},
           "G4_dem": {"EXPAND_TARGETED": "expand DEM to targeted tiles", "STOP": "no large-scale DEM expansion",
                      "INCONCLUSIVE": "pilot too small to decide"},
           "G5_ranking": {"PASS": "learned ranking adds value", "FAIL": "do not increase model complexity"},
           "G6_deployable": {"PROMOTE": "new champion candidate for a leaderboard probe",
                             "RETAIN_CHAMPION": "keep the B0/B1 champion"}}


def collect(m8) -> pd.DataFrame:
    rows = []
    for gate, name, sub in SOURCES:
        t = m8.ws.load_table(name, sub)
        if not len(t):
            rows.append({"gate": gate, "decision": "NOT_EVALUATED", "meaning": "section not run / gated"})
            continue
        t = t[t.task == m8.prim] if "task" in t and (t.task == m8.prim).any() else t
        r = t.iloc[0].to_dict()
        dec = str(r.get("decision"))
        prov = bool(r.get("provisional", m8.profile["provisional"]))
        rows.append({"gate": gate, "decision": dec + (" (PROVISIONAL)" if prov and dec not in ("NOT_EVALUATED",) else ""),
                     "meaning": MEANING.get(gate, {}).get(dec, ""), "profile": r.get("profile", m8.profile["name"]),
                     "rule": r.get("rule", ""),
                     "details": json.dumps({k: v for k, v in r.items() if k not in ("gate", "decision", "rule", "profile")},
                                           default=str)})
    return pd.DataFrame(rows)
