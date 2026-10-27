"""W0: a small number of leaderboard probes around the B0/B1 fallback, a SHA-keyed log and an honest reading.

    A  existing fallback ``m9_fallback_fusion_thin.tif`` (fusion, 1-px crest, ring 0, area from the geo choice)
    B  A's rule and A's number of selected pixels, + 300 m suppression ring around known faults
    C  A's rule with the area fraction nested-selected on ``disc_official`` (skipped when it equals A's)
    D  diagnostic: 1 on every template-finite, non-known pixel (bounds the public new-fault density)

Nothing is uploaded. Files are written by the M7 / GEMS3 writers (template authority, eligible budget, area ledger,
``gems3.submission.validate``) into the P6 workspace and never overwrite an earlier file.

Ring semantics (``gems3.representation.op_known_ring``): the ring REMOVES predictions within ``ring_px`` of known
faults; B keeps A's pixel count, so the budget moves away from known traces. Hence
B > A  -> near-known predictions cost more FP than they earn TP on the public truth -> broad discovery;
B < A  -> hidden new truth lies next to known traces (corrections / tips) -> near-known completion (W5).
One public delta is evidence about the public subset only, never proof of where undiscovered faults lie.

Probe D algebra (official scorer, docs/GEMS_OFFICIAL_METRIC_AUDIT.md). With p = 1 on the scored non-known pixels
E (|E| = N) and new truth T inside E (truth is new faults, so never on masked known pixels):
TP = |T| (each truth pixel is covered by its own prediction, k(0) = 1), FN = 0,
FP = sum_{x in E} (1 - max_g k(x, g)) = N - S with |T| <= S <= c_max |T|, c_max = sum of the kernel weights
(an isolated truth pixel). DTI_D = |T| / (|T| + 0.2 (N - S)) gives
rho = |T| / N = 0.2 / (1 / DTI_D - 1 + 0.2 c), c in [1, c_max]  ->  [rho_lo, rho_hi].
Valid only if: masked (known) pixels carry no penalty (organizer), the scorer pools one index over the public
subset, NaN-free predictions cover every template-finite pixel, eps is negligible. ``density_bounds`` therefore
reports a bracket, not a point estimate, and labels it DIAGNOSTIC.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

LOG_COLUMNS = ["probe", "submission_file", "sha256", "model_source", "area_frac_eligible", "representation",
               "ring_px", "valid_px", "positive_px", "positive_frac_valid", "date_utc", "public_score",
               "offline_disc_dti", "offline_geo_dti", "notes"]
VERDICTS = ("broad_discovery", "near_known_completion", "different_budget", "inconclusive", "PENDING")


# =============================================================================== pure helpers
def kernel_mass(radius_m: float = 300.0, pixel_m=(100.0, 100.0)) -> float:
    """sum_x k(x) of the official triangular kernel around one pixel (c_max of the probe-D bracket)."""
    from ..metric import kernel_offsets

    return float(sum(k for _, _, k in kernel_offsets(radius_m, pixel_m)))


def density_bounds(dti_all: float, c_max: float | None = None) -> dict:
    """Bracket of the public new-truth density rho = |T| / N from the probe-D score (see module docstring)."""
    c_max = kernel_mass() if c_max is None else float(c_max)
    if not (dti_all is not None and np.isfinite(dti_all) and 0 < dti_all < 1):
        return {"rho_lo": np.nan, "rho_hi": np.nan, "c_max": c_max, "status": "no valid probe-D score"}
    a = 1.0 / float(dti_all) - 1.0
    return {"rho_lo": 0.2 / (a + 0.2 * c_max), "rho_hi": 0.2 / (a + 0.2), "c_max": c_max,
            "status": "DIAGNOSTIC (bracket; assumptions in probes.py docstring)"}


def all_eligible_map(template_valid: np.ndarray, known: np.ndarray) -> np.ndarray:
    """Probe D raster: 1 on template-finite non-known pixels, 0 on known pixels, NaN where the template is NaN."""
    tv, kn = np.asarray(template_valid, bool), np.asarray(known, bool)
    out = np.where(tv, np.where(kn, 0.0, 1.0), np.nan).astype(np.float32)
    return out


def raster_facts(a: np.ndarray) -> dict:
    """Valid (finite) pixel count, positive count and positive fraction of one submission array."""
    a = np.asarray(a, np.float32)
    fin = np.isfinite(a)
    n = int(fin.sum())
    pos = int((np.nan_to_num(a, nan=0.0) > 0).sum())
    return {"valid_px": n, "positive_px": pos, "positive_frac_valid": pos / n if n else np.nan}


def probe_plan(sel: dict, prim: str, champion: str, include, frac_a: float) -> list[dict]:
    """Which probes to build and why (pure). ``frac_a`` = A's area fraction of eligible pixels."""
    out = []
    pa = ((sel.get(prim) or {}).get(champion) or {}).get("params_all") or {}
    if "A" in include:
        out.append({"probe": "A", "ring_px": 0, "frac": frac_a, "match": None,
                    "question": "real public score of the fallback"})
    if "B" in include:
        out.append({"probe": "B", "ring_px": None, "frac": None, "match": "A",
                    "question": "does removing predictions within the ring help at the same pixel count?"})
    if "C" in include:
        fc = pa.get("frac")
        if fc is None:
            out.append({"probe": "C", "skip": f"no nested {prim} fraction for {champion}"})
        elif math.isclose(float(fc), float(frac_a), rel_tol=1e-6, abs_tol=1e-9):
            out.append({"probe": "C", "skip": f"nested {prim} fraction {fc} equals A's: C would duplicate A"})
        else:
            out.append({"probe": "C", "ring_px": 0, "frac": float(fc), "match": None,
                        "question": f"is A's budget {frac_a:g} right? C uses the {prim} choice {float(fc):g}"})
    if "D" in include:
        out.append({"probe": "D", "ring_px": 0, "frac": 1.0, "match": None,
                    "question": "diagnostic bracket of the public new-fault density (not a candidate)"})
    return out


def merge_log(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Append rows by sha256; a row already present keeps its manually entered public score and notes."""
    old = old if old is not None and len(old) else pd.DataFrame(columns=LOG_COLUMNS)
    new = new.reindex(columns=LOG_COLUMNS)
    keep = new[~new.sha256.isin(set(old.sha256.astype(str)))]
    out = pd.concat([old.reindex(columns=LOG_COLUMNS), keep], ignore_index=True) if len(keep) else \
        old.reindex(columns=LOG_COLUMNS)
    return out.reset_index(drop=True)


def _latest(log: pd.DataFrame, probe: str) -> pd.Series | None:
    g = log[(log.probe.astype(str) == probe) & pd.to_numeric(log.public_score, errors="coerce").notna()]
    return None if not len(g) else g.sort_values("date_utc").iloc[-1]


def interpret(log: pd.DataFrame, tie: float = 0.005) -> dict:
    """Reading of the logged probes. Only scored rows count; a missing score is PENDING, never imputed."""
    log = log if log is not None and len(log) else pd.DataFrame(columns=LOG_COLUMNS)
    sc = {p: _latest(log, p) for p in "ABCD"}
    val = {p: (float(r.public_score) if r is not None else None) for p, r in sc.items()}
    findings, missing = [], [p for p in "ABC" if val[p] is None]
    ring = budget = None
    if val["A"] is not None and val["B"] is not None:
        d = val["B"] - val["A"]
        ring = d
        if abs(d) < tie:
            findings.append(f"ring: B - A = {d:+.4f} within +-{tie} -> inconclusive (keep ring 0)")
        elif d > 0:
            findings.append(f"ring: B - A = {d:+.4f}: predictions within the ring were net FP on the public truth "
                            "-> spend the budget away from known traces (broad discovery); keep W5 off")
        else:
            findings.append(f"ring: B - A = {d:+.4f}: removing near-known predictions lost TP -> public new truth "
                            "lies next to known traces -> test near-known completion (W5)")
    if val["A"] is not None and val["C"] is not None:
        d = val["C"] - val["A"]
        budget = d
        fa, fc = (float(sc[p].area_frac_eligible) for p in "AC")
        direction = "smaller" if fc < fa else "larger"
        if abs(d) < tie:
            findings.append(f"budget: C - A = {d:+.4f} within +-{tie} -> keep A's fraction {fa:g}")
        else:
            better = direction if d > 0 else ("larger" if direction == "smaller" else "smaller")
            findings.append(f"budget: C ({fc:g}) - A ({fa:g}) = {d:+.4f} -> a {better} budget than {fa:g} scores "
                            "better on the public subset (one point on a curve, not its optimum)")
    dens = density_bounds(val["D"]) if val["D"] is not None else None
    if dens is not None and np.isfinite(dens["rho_lo"]):
        findings.append(f"density (D, diagnostic): public new-truth pixels / scored non-known pixels in "
                        f"[{dens['rho_lo']:.2e}, {dens['rho_hi']:.2e}]")
    if ring is None and budget is None:
        verdict = "PENDING"
    else:
        cand = [(abs(x), k) for x, k in ((ring, "ring"), (budget, "budget")) if x is not None and abs(x) >= tie]
        if not cand:
            verdict = "inconclusive"
        else:
            k = max(cand)[1]
            verdict = ("broad_discovery" if ring > 0 else "near_known_completion") if k == "ring" else "different_budget"
    return {"verdict": verdict, "findings": findings, "missing_scores": missing, "scores": val,
            "density": dens, "w5_enabled": verdict == "near_known_completion", "tie_delta": tie,
            "caveat": "public-subset evidence from single submissions; not proof of the hidden-test distribution"}


# =============================================================================== Colab side
def _fallback_manifest(p6) -> tuple[Path | None, dict]:
    name = p6.cfg["probes"]["fallback_file"]
    p6.stage_m9([f"submissions/{name}", f"submissions/{Path(name).stem}.json"])
    p = p6.m9_path(f"submissions/{name}")
    j = p.with_suffix(".json")
    man = json.loads(j.read_text(encoding="utf-8")) if j.exists() else {}
    return (p if p.exists() else None), man


def _offline(sel: dict, task: str, method: str) -> float:
    s = (sel.get(task) or {}).get(method) or {}
    v = s.get("dti_crossfit")
    return float(v) if v is not None else np.nan


def _row(p6, probe: str, path: Path, info: dict, sel: dict, notes: str) -> dict:
    import rasterio

    from ..artifacts.atomic import sha256_file

    with rasterio.open(path) as src:
        facts = raster_facts(src.read(1))
    par = info.get("parameters") or {}
    led = info.get("area_ledger") or []
    champ = p6.cfg["probes"]["champion"]
    return {"probe": probe, "submission_file": path.name, "sha256": sha256_file(path),
            "model_source": info.get("model_id") or info.get("method") or f"s01_{champ}",
            "area_frac_eligible": info.get("final_frac_of_eligible", led[-1]["frac_of_eligible"] if led else np.nan),
            "representation": par.get("rep", info.get("representation", "")), "ring_px": int(par.get("ring", 0) or 0),
            **facts, "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "public_score": np.nan,
            "offline_disc_dti": _offline(sel, p6.prim, champ) if probe in "AB" else np.nan,
            "offline_geo_dti": _offline(sel, "geo", champ) if probe == "A" else np.nan, "notes": notes}


def _write_diagnostic(p6, name: str) -> tuple[Path, dict]:
    """Probe D: bypasses ``finalise`` (which zeroes template-finite pixels outside the data footprint)."""
    from ..artifacts.atomic import atomic_save
    from ..gems3 import submission as S
    from ..submission import write_submission

    data = p6.ctx.data
    arr = all_eligible_map(data.template_valid, data.labels)
    p = p6.ws.root / "submissions" / f"{name}.tif"
    if p.exists() and not p6.cfg.get("overwrite_existing"):
        p = p.with_name(f"{name}__{time.strftime('%Y%m%dT%H%M%S')}.tif")
    atomic_save(p, lambda t: write_submission(arr, data, Path(t)))
    p6.ws._track(p)
    chk = S.validate(p, data, "template")
    info = {"model_id": "diagnostic_all_eligible", "parameters": {"rep": "all_eligible", "ring": 0},
            "area_ledger": [S.area_ledger(arr, data, 0, "all template-finite non-known pixels = 1")],
            "validation": chk, "uploaded": False, "role": "DIAGNOSTIC (never a candidate)"}
    info["final_frac_of_eligible"] = info["area_ledger"][-1]["frac_of_eligible"]
    p6.ws.save_json(json.loads(json.dumps(info, default=str)), f"submissions/{p.stem}.json")
    return p, info


def run(p6) -> pd.DataFrame:
    """Build the probe files, append them to both leaderboard logs (Drive + repository copy), save the plan."""
    from ..gems3 import submission as S
    from ..m7 import export as E

    pc = p6.cfg["probes"]
    sel = p6.selection()
    champ = pc["champion"]
    data = p6.ctx.data
    a_path, a_man = _fallback_manifest(p6)
    rows, files = [], {}
    if a_path is not None:
        k_a = int(a_man.get("k_px") or (a_man.get("area_ledger") or [{}])[0].get("positive_px", 0))
        frac_a = float((a_man.get("parameters") or {}).get("frac", np.nan))
        files["A"] = (a_path, a_man, "existing fallback (read-only)")
    else:
        info = E.export_baseline(p6.view, "p6_probe_A_fusion_thin_r0", champ, ring_px=0)
        k_a, frac_a = int(info["k_px"]), float(info["parameters"]["frac"])
        files["A"] = (p6.ws.root / "submissions" / info["submission"], info,
                      "fallback missing on Drive: rebuilt with the identical rule (NOT the original file)")
    if not k_a:
        raise ValueError("probe A: selected pixel count unknown (fallback manifest has no ledger)")
    plan = probe_plan(sel, p6.prim, champ, pc["include"], frac_a)
    for pr in plan:
        if pr.get("skip") or pr["probe"] == "A":
            continue
        if pr["probe"] == "B":
            info = E.export_baseline(p6.view, f"p6_probe_B_fusion_thin_r{int(pc['ring_px'])}", champ,
                                     ring_px=int(pc["ring_px"]), k_px=k_a)
            files["B"] = (p6.ws.root / "submissions" / info["submission"], info,
                          f"A's rule and pixel count ({k_a} px) + {int(pc['ring_px'])} px suppression ring")
        elif pr["probe"] == "C":
            n_el = int(S.eligible_mask(data, 0, "eligible").sum())
            k_c = max(int(math.ceil(float(pr["frac"]) * n_el)), 1)
            info = E.export_baseline(p6.view, f"p6_probe_C_fusion_thin_f{float(pr['frac']):g}", champ, ring_px=0,
                                     k_px=k_c)
            files["C"] = (p6.ws.root / "submissions" / info["submission"], info,
                          f"{p6.prim} nested all-zone fraction {pr['frac']:g} ({k_c} px)")
        elif pr["probe"] == "D":
            p, info = _write_diagnostic(p6, "p6_probe_D_all_eligible")
            files["D"] = (p, info, "DIAGNOSTIC: all template-finite non-known pixels = 1; never a candidate")
    for k, (path, info, note) in files.items():
        rows.append(_row(p6, k, Path(path), info, sel, note))
    new = pd.DataFrame(rows, columns=LOG_COLUMNS)
    log = update_logs(p6, new)
    p6.ws.save_table(pd.DataFrame(plan), "probe_plan", "probes")
    p6.ws.save_json({"k_px_A": k_a, "frac_A": frac_a, "c_max": kernel_mass(),
                     "upload": "manual; at most 3 scored submissions per rolling 7 days; record the score in "
                               "leaderboard_log.csv (public_score) and rerun w0_interpret"}, "probes/probe_meta.json")
    return log


def log_paths(p6) -> list[Path]:
    """Drive copy (P6 workspace) and the repository copy (``probes.log``, relative to the repository root)."""
    repo = Path(__file__).resolve().parents[2]
    return [p6.ws.root / "probes" / "leaderboard_log.csv", repo / p6.cfg["probes"]["log"]]


def read_log(p6) -> pd.DataFrame:
    """Union of both copies (manual scores may have been typed into either), deduplicated by sha256."""
    parts = [pd.read_csv(p) for p in log_paths(p6) if p.exists() and p.stat().st_size]
    if not parts:
        return pd.DataFrame(columns=LOG_COLUMNS)
    t = pd.concat([x.reindex(columns=LOG_COLUMNS) for x in parts], ignore_index=True)
    t["_has"] = pd.to_numeric(t.public_score, errors="coerce").notna()
    t = t.sort_values("_has", ascending=False).drop_duplicates("sha256").drop(columns="_has")
    return t.sort_values(["date_utc", "probe"]).reset_index(drop=True)


def update_logs(p6, new: pd.DataFrame) -> pd.DataFrame:
    from ..artifacts.atomic import atomic_save

    log = merge_log(read_log(p6), new)
    for p in log_paths(p6):
        p.parent.mkdir(parents=True, exist_ok=True)
        atomic_save(p, lambda t: log.to_csv(t, index=False))
    return log


def interpret_log(p6) -> dict:
    out = interpret(read_log(p6), float(p6.cfg["probes"]["tie_delta"]))
    p6.ws.save_json(json.loads(json.dumps(out, default=float)), "probes/interpretation.json")
    return out
