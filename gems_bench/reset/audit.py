"""Section 1: what we actually have, and how much of what we believe is established.

``prior_findings`` classifies every important historical claim as VERIFIED_EXECUTED_OUTPUT,
REPRODUCIBLE_NOT_VERIFIED, METHODOLOGICALLY_QUESTIONABLE, HYPOTHESIS_ONLY or NOT_AVAILABLE, with the reason and the
check that runs in Colab (``verify``). The checks read existing files only (EDA report / manifest, labels, frozen
tables); none of them re-trains or re-runs the EDA.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

V, R_, Q, H, NA = ("VERIFIED_EXECUTED_OUTPUT", "REPRODUCIBLE_NOT_VERIFIED", "METHODOLOGICALLY_QUESTIONABLE",
                   "HYPOTHESIS_ONLY", "NOT_AVAILABLE")
EDA = "outputs/eda/EDA_FINDINGS.md"

# (id, claim, source, classification, reason, colab check)
PRIOR = [
    ("eda_grid", "EDA ran on a 200 m working grid (native 100 m)", EDA, V,
     "stated in the executed report header; every EDA scale / width number is at 200 m", "header_grid"),
    ("eda_segment_median", "median mapped segment 1.5 km (p90 4.9 km), 2318 segments", EDA, R_,
     "computed on the 200 m grid; re-measured here at 100 m", "segment_lengths"),
    ("eda_transform_labels", "best pixel transform per modality (table column)", EDA, Q,
     "the column prints a pandas `<bound method Series.transform ...>` repr: the code read `row.transform` (a "
     "method) instead of the 'transform' field; the table's transform names are not trustworthy as printed",
     "bound_method_repr"),
    ("eda_evidence_saturation", "median max-evidence 1.00 for every geometry; 0 % weak segments", EDA, Q,
     "a calibrated evidence that saturates at 1.00 everywhere (background 0.5) carries no ranking information; "
     "'0 % weak' is a consequence of the saturation, not a property of the faults", "saturated_evidence"),
    ("eda_modality_ranking", "terrain > gravity > magnetic > basement > conductivity", EDA, R_,
     "AUC near vs far from MAPPED faults at 200 m (0.55-0.63 pixel AUC): a weak, label-defined contrast; it ranks "
     "association with mapped faults, not discovery power", None),
    ("eda_enrichment", "pixels with >= 3 modality edges are 135.7x enriched near mapped faults (best case)", EDA, Q,
     "maximum over configurations (selection bias) and enrichment of a very small set; no held-out estimate", None),
    ("eda_response_width", "fault response FWHM ~ 9.4 km (top-8 signals, p75)", EDA, R_,
     "measured on aligned profiles at 200 m; implies broad responses ~30x wider than the 300 m tolerance "
     "(re-measured here as localization error at 100 m)", None),
    ("eda_lineament_recovery", "fused lineament recovers 14 % of mapped trace within 2 km", EDA, Q,
     "2 km tolerance is ~7x the official 300 m; says nothing about official TP support", None),
    ("eda_unmapped_lineaments", "10 coherent lineaments >= 6 km with no mapped fault within 6 km", EDA, H,
     "candidates only (contacts / dykes possible); they argue for a PU setting, not for new faults", None),
    ("eda_top_candidates", "top unmapped candidates table", EDA, H, "exploratory ranking, never reviewed", None),
    ("public_w2", "W2 public DTI 0.0942", "user-recorded leaderboard", V,
     "one scored public file; external reference, never a tuning target", None),
    ("w2_disc_geo", "W2 disc 0.242156 / geo 0.105687 (single policy)", "factory CANDIDATES (recorded M1 run)", V,
     "executed nested estimates on mapped faults; the disc proxy withholds whole systems and over-estimated the "
     "public score by ~2.5x", None),
    ("geomoe_v2", "W2 V2 network completion 0.085068 (scenario-selected)", "GeoMoE dgm.1-r1 Colab run", V,
     "executed, but with a scenario-selected cell: not the submitted policy (re-measured with the frozen cell)", None),
    ("binary_optimal", "for a fixed truth set the DTI-optimal map is binary", "p3/curves.py docstring", R_,
     "a property of the metric for KNOWN truth; with uncertain truth a soft map can win (tested in the benchmark)",
     None),
    ("masked_tp_support", "masked (known-fault) predictions do not support TP", "metric audit", H,
     "inferred from an organiser statement; OFFICIAL_EXCLUDE_SUPPORTS_TP is the repository's reading", None),
    ("m7_x2", "M7 ~12.5 % label-free coverage vs X2 82.9 % ranking-limited length", "M7 / P6.1 docs", Q,
     "different pools, budgets and definitions; only comparable under one protocol (GeoMoE reconciliation)", None),
    ("independent_labels", "independent geological interpretations / maps", "repository + Drive search", NA,
     "no independent fault interpretation is known to exist in the project; reviewed atlas candidates are the "
     "first (with provenance)", "independent_sources"),
    ("dem_mosaic", "1 m DEM-derived terrain evidence", "M7 DEM pilot", NA,
     "infrastructure exists; the cached mosaic was absent in the last GeoMoE run", "dem_mosaic"),
]


def verify(R, check: str | None) -> tuple[str, str]:
    """(result, detail) of one Colab-side check; 'NOT_RUN' when no check is defined."""
    if check is None:
        return "NOT_RUN", "no automatic check: classification by method"
    repo = Path(__file__).resolve().parents[2]
    txt = (repo / EDA).read_text(encoding="utf-8") if (repo / EDA).exists() else ""
    if check == "header_grid":
        m = re.search(r"at ([0-9.]+) km/px", txt)
        return ("CONFIRMED", f"working grid {m.group(1)} km/px") if m else ("NOT_FOUND", "header not found")
    if check == "bound_method_repr":
        n = txt.count("<bound method Series.transform")
        return ("CONFIRMED_BUG" if n else "NOT_FOUND", f"{n} malformed transform cells in {EDA}")
    if check == "saturated_evidence":
        m = re.search(r"Median max-evidence by geometry: (.*?)\n", txt, re.S)
        vals = re.findall(r"([0-9.]+)", m.group(1)) if m else []
        sat = bool(vals) and all(abs(float(v) - 1.0) < 1e-9 for v in vals)
        return ("CONFIRMED_SATURATION" if sat else "NOT_SATURATED"), f"values {vals}"
    if check == "segment_lengths":
        from scipy import ndimage

        from ..geology.skeleton import skeletonize

        sk = skeletonize(np.asarray(R.ctx.data.labels, bool))
        lab, n = ndimage.label(sk, structure=np.ones((3, 3), bool))
        km = np.bincount(lab.ravel())[1:] * 0.1 * 1.12
        return "MEASURED", f"100 m skeleton components {n}, median {np.median(km):.2f} km, p90 " \
                           f"{np.quantile(km, 0.9):.2f} km (component, not branch, lengths)"
    if check == "independent_sources":
        hits = []
        for root in (repo / "data", R.drive / "data", R.drive / "external"):
            if Path(root).exists():
                for pat in ("*geolog*", "*contact*", "*lineament*", "*interpret*", "*.shp", "*.gpkg"):
                    hits += [str(p) for p in Path(root).rglob(pat)][:20]
        return ("FOUND_FILES_NOT_IMPORTED" if hits else "NONE_FOUND"), "; ".join(sorted(set(hits))[:20])
    if check == "dem_mosaic":
        from ..m7.dem_morph import mosaic_path

        p = mosaic_path(R.gm.m7)
        on_drive = (R.drive / p.relative_to(R.gm.m7.local)).exists()
        return ("PRESENT" if p.exists() or on_drive else "ABSENT"), str(p)
    return "NOT_RUN", f"unknown check {check}"


def prior_findings(R) -> pd.DataFrame:
    rows = []
    for fid, claim, src, cls, why, chk in PRIOR:
        try:
            res, det = verify(R, chk)
        except Exception as e:                       # a failed check is recorded, never hidden
            res, det = "CHECK_FAILED", f"{type(e).__name__}: {e}"
        rows.append({"finding_id": fid, "claim": claim, "source": src, "classification": cls, "reason": why,
                     "colab_check": chk, "check_result": res, "check_detail": str(det)[:600]})
    return pd.DataFrame(rows)


def source_inventory(R) -> pd.DataFrame:
    d = R.ctx.data
    rows = [{"kind": "band", "name": b, "present": b in d.band_names, "role": role}
            for role, b in R.rc["bands"].items()]
    rows += [{"kind": "band", "name": b, "present": True, "role": "other GeoDAWN band"} for b in d.band_names
             if b not in R.rc["bands"].values()]
    w2 = R.bst.w2_inputs().get("maps_sha256") or {}
    for su in R.units:
        rows.append({"kind": "honest_map", "name": f"W2 {su.base.task}_{su.base.zone}",
                     "present": f"{su.base.task}_{su.base.zone}.npy" in w2, "role": su.scenario})
    g = R.bst.g02()
    rows.append({"kind": "g02", "name": "G02 unit rasters", "present": g is not None,
                 "role": ";".join(g.raster_units()) if g is not None else "absent"})
    fz = R.gm.frozen_policy()
    rows.append({"kind": "frozen_policy", "name": "submitted W2 cell", "present": "fallback" not in fz["source"],
                 "role": f"{fz['cell']} ({fz['source']})"})
    ref = R.rc["benchmark"].get("reference_prediction")
    rows.append({"kind": "reference", "name": "organiser reference prediction", "present": bool(
        ref and (R.drive / ref).exists()), "role": str(ref)})
    rev = R.load_reviews()
    rows.append({"kind": "independent_labels", "name": "reviewed atlas annotations", "present": bool(len(rev)),
                 "role": f"{len(rev)} review events"})
    return pd.DataFrame(rows)


def validation_contract() -> pd.DataFrame:
    return pd.DataFrame([
        {"validation": "V1 withheld systems", "truth": "whole mapped systems withheld (discovery splits)",
         "measures": "discovery of complete systems far from known context",
         "known_bias": "under-represents corrections / tips near known faults; optimistic for W2 (2.5x public)",
         "representativeness": "proxy"},
        {"validation": "V2 network completion", "truth": "real mapped tips / branches / splays / subsections",
         "measures": "completion of partially mapped networks",
         "known_bias": "class mix is a design choice; extra retained context benefits network features",
         "representativeness": "diagnostic"},
        {"validation": "V3 geographic folds", "truth": "every mapped fault of a held-out fold",
         "measures": "rediscovery of mapped faults in unseen geography (no known mask)",
         "known_bias": "truth = mapped faults, not new ones", "representativeness": "generalisation check"},
        {"validation": "reviewed atlas candidates (EVAL regions)", "truth": "human interpretations with provenance",
         "measures": "agreement with independent geological judgement",
         "known_bias": "small; diagnostic reviews are not blind", "representativeness": "independent when "
                                                                                          "EVIDENCE_ONLY"},
        {"validation": "public leaderboard", "truth": "hidden new faults", "measures": "the target",
         "known_bias": "one score per file; never used for tuning", "representativeness": "external reference"}])


LIMITATIONS = """# Known limitations

* Mapped faults are incomplete: unmapped pixels are unlabelled, not negative (PU). Every offline score counts some
  true faults as false positives.
* V1 / V2 / V3 score MAPPED faults; the leaderboard scores NEW faults. A model that reproduces the mapping style can
  score well offline and poorly publicly (W2: disc 0.242 vs public 0.0942).
* Potential-field responses are kilometres wide; the metric tolerance is 300 m. Broad evidence may indicate corridors
  rather than traces (measured: physics/localization_error.csv).
* Geophysical edges are not faults: contacts, intrusions, basin margins and survey seams create the same gradients.
  The physics table records contrast vs disruption, and alternative explanations, as observations only.
* Atlas reviews made with predictions visible are not blind labels (label_class DIAGNOSTIC_REVIEWED).
* DEM evidence exists only where the cached / pilot mosaic covers; missing coverage is not evidence of absence.
* The official treatment of predictions on masked known faults is inferred (OFFICIAL_EXCLUDE_SUPPORTS_TP).
"""
