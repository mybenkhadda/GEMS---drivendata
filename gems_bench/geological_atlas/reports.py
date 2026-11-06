"""Markdown reports generated from the computed numbers only (``04_eda/reports/results.json``).

Every statement carries its number and is tagged OBSERVATION (measured on the data) or HYPOTHESIS (a geological
reading of a measurement). If a quantity is missing, the sentence says so instead of guessing.
"""
from __future__ import annotations

import json
import time

import numpy as np

from .cache import atomic_write_json, atomic_write_text, read_json
from .circular_statistics import axial_difference


def _f(v, fmt="{:.2f}", na="n/a"):
    try:
        return na if v is None or not np.isfinite(float(v)) else fmt.format(float(v))
    except (TypeError, ValueError):
        return na


def load(paths) -> dict:
    return read_json(paths.d("04_eda", "reports") / "results.json", {}) or {}


def save(paths, results: dict):
    atomic_write_json(paths.d("04_eda", "reports") / "results.json", results)


def section(results: dict, key: str) -> str:
    r = results.get(key)
    if not r:
        return f"_No results for `{key}` yet - run its stage._"
    if key == "faults":
        s = r["summary"]
        return (f"- OBSERVATION: {s['label_pixels']:,} mapped fault pixels, {_f(s['total_trace_km'], '{:,.0f}')} km of "
                f"skeleton trace in {s['n_traces']:,} 8-connected traces (median {_f(s['median_trace_km'], '{:.1f}')} km, "
                f"p90 {_f(s['p90_trace_km'], '{:.1f}')} km; {_f(s['traces_per_100km'], '{:.1f}')} traces per 100 km).\n"
                f"- OBSERVATION: length-weighted mean strike {_f(s['length_weighted_mean_strike_deg'], '{:.0f}')} deg "
                f"(axial R = {_f(s['strike_resultant_R'])}, circular sd {_f(s['strike_circular_std_deg'], '{:.0f}')} deg, "
                f"Rayleigh p = {_f(s['rayleigh_trace_p'], '{:.2g}')}).\n"
                f"- HYPOTHESIS: a high R indicates one dominant structural grain; low R a mixed population.\n"
                "- LIMITATION: mapped faults are an incomplete reference; fragmentation partly reflects mapping practice.")
    if key in ("magnetic", "gravity"):
        return (f"- OBSERVATION: dominant {key} boundary strike {_f(r.get('strike_deg'), '{:.0f}')} deg "
                f"(coherence-weighted axial R = {_f(r.get('strike_R'))}).\n"
                f"- OBSERVATION: {_f(100 * r.get('persistent_share', np.nan), '{:.1f}')} % of the footprint has a boundary "
                f"that is strong (rank >= 0.9) at >= 60 % of the continuation levels.\n"
                f"- OBSERVATION: wavelength energy shares: {r.get('band_energy_text', 'n/a')}.\n"
                f"- HYPOTHESIS: boundaries persisting through 2-5 km continuation have deep, laterally continuous sources "
                "(major contacts / basin-bounding structures); boundaries only in the residual are shallow.\n"
                "- LIMITATION: no source depth is inferred from the spectrum slope.")
    if key == "terrain":
        return (f"- OBSERVATION: dominant 1 km terrain lineament strike {_f(r.get('strike_deg'), '{:.0f}')} deg "
                f"(R = {_f(r.get('strike_R'))}).\n- LIMITATION: 100 m data resolve range fronts and valleys, not scarps.")
    if key == "dem":
        return (f"- OBSERVATION: {r.get('mode')} DEM statistics cover {_f(100 * r.get('covered_share_of_footprint', np.nan), '{:.1f}')} % "
                f"of the footprint ({r.get('covered_cells', 0):,} cells; {r.get('seam_cells', 0):,} cells touch a project seam).\n"
                "- LIMITATION: DEM evidence exists only where tiles were downloaded; elsewhere it is absent, not zero.")
    if key == "cross_modal":
        return (f"- OBSERVATION: strike difference magnetic vs gravity {_f(r.get('mag_grav_strike_diff'), '{:.0f}')} deg, "
                f"magnetic vs terrain {_f(r.get('mag_terr_strike_diff'), '{:.0f}')} deg, gravity vs terrain "
                f"{_f(r.get('grav_terr_strike_diff'), '{:.0f}')} deg.\n"
                f"- OBSERVATION: {_f(100 * r.get('two_plus_share', np.nan), '{:.2f}')} % of the footprint has strong edges "
                f"in >= 2 independent groups.\n"
                "- HYPOTHESIS: places with strong, co-oriented edges in several independent groups are the best-supported "
                "structural boundaries.\n- LIMITATION: terrain and DEM are counted as ONE topography group.")
    if key == "lineaments":
        return (f"- OBSERVATION: {r.get('n', 0)} candidate lineaments (rules fixed before looking at faults), "
                f"{_f(r.get('total_km'), '{:,.0f}')} km total; {r.get('n_multi', 0)} with >= 2 evidence groups; "
                f"{r.get('n_far', 0)} farther than 1 km from any mapped fault.\n"
                f"- OBSERVATION: length-weighted strike {_f(r.get('strike_deg'), '{:.0f}')} deg (R = {_f(r.get('strike_R'))}); "
                f"vs mapped-fault strike difference {_f(r.get('strike_diff_vs_faults'), '{:.0f}')} deg.\n"
                "- HYPOTHESIS: multi-group candidates far from mapped faults and parallel to the regional grain are the "
                "most plausible unmapped structures.\n- LIMITATION: UNVERIFIED; not a fault probability.")
    if key == "proximity":
        top = r.get("top_contrast", [])
        return ("- OBSERVATION (descriptive, block-sampled): largest near (0-300 m) vs far (> 3 km) median contrasts: "
                + ", ".join(f"{t['feature_id']} {t['contrast']:+.2f} IQR" for t in top[:6]) + ".\n"
                "- LIMITATION: association with mapped faults only; not causal, not a selection criterion.")
    if key == "quality":
        return (f"- OBSERVATION: {r.get('n', 0)} scalar features: " + ", ".join(f"{k} {v}" for k, v in
                                                                         r.get("classes", {}).items())
                + f"; {r.get('retain', 0)} retained.\n- Rules are label-free (FEATURE_ENGINEERING.md).")
    return json.dumps(r, indent=1, default=str)[:2000]


def findings(results: dict) -> list[dict]:
    out = []
    f = results.get("faults", {}).get("summary", {})
    if f:
        out.append({"rank": 0, "kind": "OBSERVATION", "text": f"Mapped faults: {_f(f.get('total_trace_km'), '{:,.0f}')} km, "
                    f"mean strike {_f(f.get('length_weighted_mean_strike_deg'), '{:.0f}')} deg (R {_f(f.get('strike_resultant_R'))})",
                    "weight": float(f.get("strike_resultant_R") or 0)})
    cm = results.get("cross_modal", {})
    for k, lab in (("mag_grav_strike_diff", "magnetic and gravity"), ("mag_terr_strike_diff", "magnetic and terrain"),
                   ("grav_terr_strike_diff", "gravity and terrain")):
        if cm.get(k) is not None and np.isfinite(cm[k]):
            out.append({"rank": 0, "kind": "OBSERVATION", "text": f"Dominant strikes of {lab} boundaries differ by "
                        f"{cm[k]:.0f} deg", "weight": 1 - cm[k] / 90})
    li = results.get("lineaments", {})
    if li.get("n"):
        out.append({"rank": 0, "kind": "HYPOTHESIS", "text": f"{li.get('n_far', 0)} candidate lineaments lie > 1 km from "
                    f"mapped faults; {li.get('n_multi_far', 0)} of them carry >= 2 independent evidence groups and are the "
                    "best-supported unmapped-structure candidates", "weight": float(li.get("n_multi_far", 0)) / max(li["n"], 1) + 0.5})
    for k in ("magnetic", "gravity"):
        r = results.get(k, {})
        if r.get("persistent_share") is not None:
            out.append({"rank": 0, "kind": "OBSERVATION", "text": f"{k}: {100 * r['persistent_share']:.1f} % of the "
                        "footprint carries scale-persistent boundaries", "weight": r["persistent_share"]})
    out.sort(key=lambda x: -x["weight"])
    for i, o in enumerate(out, 1):
        o["rank"] = i
    return out


def write_all(paths, cfg, results: dict, artifacts: list[dict]) -> list:
    rep = paths.reports
    now = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    keys = ["faults", "magnetic", "gravity", "terrain", "dem", "cross_modal", "proximity", "quality", "lineaments"]
    summ = [f"# GEMS Geological Data Atlas V1 - summary\n\nGenerated {now} from computed results only.\n"]
    for k in keys:
        summ.append(f"## {k.replace('_', ' ').title()}\n\n{section(results, k)}\n")
    fnd = ["# Geological findings (ranked)\n", "OBSERVATION = measured; HYPOTHESIS = geological reading.\n"]
    fnd += [f"{o['rank']}. **{o['kind']}** - {o['text']}" for o in findings(results)] or ["_no results yet_"]
    q = results.get("quality", {})
    rec = ["# Feature recommendations\n", "Label-free classification (04_eda/tables/feature_quality.csv).\n",
           f"Retained: {q.get('retain', 0)} of {q.get('n', 0)} scalar features.\n", "## Retained features\n"]
    rec += [f"- `{f}`" for f in q.get("retained", [])] or ["_none yet_"]
    rec += ["\n## Not retained (reason)\n"] + [f"- `{f}`: {why}" for f, why in q.get("dropped", {}).items()]
    lim = ["# Data limitations\n"] + [f"- {x}" for x in results.get("limitations", [])] + [
        "- 100 m GeoDAWN resolution: structures narrower than a few hundred metres are not resolved.",
        "- Mapped faults are an incomplete reference; proximity statistics are descriptive associations.",
        "- Candidate lineaments are UNVERIFIED; their rules are fixed and label-free.",
        "- DEM coverage is partial: absent DEM is missing evidence, never zero relief."]
    outs = [atomic_write_text(rep / "atlas_summary.md", "\n".join(summ)),
            atomic_write_text(rep / "geological_findings.md", "\n".join(fnd) + "\n"),
            atomic_write_text(rep / "feature_recommendations.md", "\n".join(rec) + "\n"),
            atomic_write_text(rep / "data_limitations.md", "\n".join(lim) + "\n"),
            atomic_write_json(rep / "run_manifest.json", {"generated_utc": now, "config": cfg.as_dict(),
                                                          "artifacts": artifacts,
                                                          "stages": results.get("stages", {})})]
    return outs


def strike_diff(a, b) -> float:
    if a is None or b is None or not (np.isfinite(a) and np.isfinite(b)):
        return float("nan")
    return float(axial_difference(a, b))
