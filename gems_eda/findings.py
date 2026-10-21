"""Assemble EDA_FINDINGS.md from computed tables (numbers are filled in at run time)."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from .config import EVIDENCE_GROUPS
from .context import EDAContext


def _f(x, nd=2) -> str:
    try:
        return "n/a" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"
    except TypeError:
        return str(x)


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(_f(v) if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(lines)


def modality_ranking(prof: dict, enh_res: dict, cm: dict) -> pd.DataFrame:
    eff = prof["effects"]
    uniq = cm["unique"].set_index("group")
    rows = []
    for m in EVIDENCE_GROUPS:
        e = eff[eff.modality == m]
        b = enh_res["best"][enh_res["best"].modality == m]
        rows.append({
            "modality": m,
            "best_profile_signal": e.iloc[0].signal if len(e) else "-",
            "profile_auc": e.iloc[0].best_auc if len(e) else np.nan,
            "best_pixel_transform": f"{b.iloc[0]['transform']}@{b.iloc[0]['scale_km']:g}km" if len(b) else "-",
            "pixel_auc": b.iloc[0].pixel_auc if len(b) else np.nan,
            "station_support": uniq.support_share.get(m, np.nan),
            "unique_support": uniq.unique_share.get(m, np.nan),
        })
    df = pd.DataFrame(rows)
    df["score"] = df[["profile_auc", "pixel_auc"]].mean(axis=1)
    return df.sort_values("score", ascending=False, ignore_index=True)


def recommended_patch(ctx: EDAContext, prof: dict, morph: dict) -> dict:
    eff = prof["effects"].head(8)
    fwhm = float(np.nanpercentile(eff.fwhm_km, 75)) if eff.fwhm_km.notna().any() else 4 * ctx.grid.pixel_km
    ctx_km = max(4 * fwhm, morph["summary"]["median_segment_km"], 2 * ctx.cfg.profile_half_km)
    px = ctx_km / ctx.grid.pixel_km
    patch = int(2 ** np.ceil(np.log2(max(px, 64))))
    return {"fwhm_p75_km": fwhm, "context_km": ctx_km, "patch_px": patch, "patch_km": patch * ctx.grid.pixel_km,
            "stride_px": patch // 2}


def strike_evidence(arc: dict, bins: int = 6) -> pd.DataFrame:
    a = arc["arch"]
    edges = np.linspace(0, 180, bins + 1)
    a = a.assign(strike_bin=pd.cut(a.azimuth % 180, edges, include_lowest=True))
    return a.groupby("strike_bin", observed=False).agg(trace_km=("length_km", "sum"), median_max_ev=("max_ev", "median")).reset_index()


def write_findings(ctx: EDAContext, morph: dict, prof: dict, enh_res: dict, cm: dict, arc: dict, cres: dict) -> str:
    g, s = ctx.grid, morph["summary"]
    rank = modality_ranking(prof, enh_res, cm)
    patch = recommended_patch(ctx, prof, morph)
    counts = arc["counts"].set_index("primary_archetype")
    geom = arc["geometry"].set_index("geometry_class")
    val = cres["validation"]
    se = strike_evidence(arc)
    kinds_km = s["component_km_by_kind"]
    total_km = sum(kinds_km.values()) or np.nan
    weak_share = counts.share_km.get("weak_ambiguous", 0.0)
    top_raw = [r for r in prof["effects"].signal if ":" not in r][:8]
    best_tf = [f"{r.modality}:{r.transform}@{r.scale_km:g}km" for r in enh_res["best"].itertuples()]
    scale_pref = enh_res["auc"].loc[enh_res["auc"].groupby(["modality", "transform"]).pixel_auc.idxmax()]
    scale_counts = scale_pref.scale_km.value_counts().sort_index()
    fams = morph["families"]
    weak_strike = se.sort_values("median_max_ev").iloc[0] if len(se) else None
    top_cands = cres["cands"][cres["cands"].status == "unmapped"].nsmallest(5, "rank_unmapped") if len(cres["cands"]) else pd.DataFrame()

    def geom_ev(k):
        return _f(geom.max_ev.get(k, np.nan))

    md = f"""# GEMS — Geological EDA findings

_Generated {dt.date.today().isoformat()} by `notebooks/01_geological_eda.ipynb`. Every number below is computed from
the data; reload paths for all artifacts are in `outputs/eda/manifest.json`._

## Data frame of reference
- Working grid {g.height} x {g.width} px at {_f(g.pixel_km, 3)} km/px (native {ctx.stack.native_shape[0]} x {ctx.stack.native_shape[1]}),
  CRS `{g.crs}`, {len(ctx.stack.names)} geophysical bands in 7 modalities.
- Mapped fault pixels: {_f(100 * ctx.fault_info['positive_frac'], 3)} % of the grid; median label width {_f(s['label_width_px_median'], 1)} px.
- {s['n_segments']} skeleton segments ({_f(s['total_trace_km'], 0)} km of trace), {s['n_junctions']} branch points,
  median segment {_f(s['median_segment_km'], 1)} km (p90 {_f(s['p90_segment_km'], 1)} km).
- Dominant strike families: {', '.join(f'{f:.0f}°' for f in fams) or 'none resolved'}.

## 1. Most informative modalities
{_md_table(rank.drop(columns='score'))}

- Ranking (mean of profile and pixel AUC): **{' > '.join(rank.modality)}**.
- {_f(100 * cm['unique'].set_index('group').support_share.get('none', np.nan), 0)} % of fault stations have no modality
  above the 10 %-false-alarm threshold: these are the label positives a model will struggle to justify from the data.
- Cross-modal edge coincidence: pixels where ≥3 modalities carry a top-10 % edge are
  {_f(cm['lift'][cm['lift'].n_modalities_with_edge >= 3].lift.max(), 1)}x enriched near mapped faults (best case).

## 2. Useful transformations and spatial scales
- Best transform per modality (pixel AUC, near vs far): {', '.join(f'`{b}`' for b in best_tf)}.
- Scale preference across all modality x transform pairs (count of pairs whose AUC peaks at that scale):
  {', '.join(f'{k:g} km: {v}' for k, v in scale_counts.items())}.
- Fault response width (FWHM of the aligned profile response, top-8 signals, 75th percentile): {_f(patch['fwhm_p75_km'], 1)} km.
- The multi-modal fused lineament score recovers {_f(100 * val['recall_known_within_km'], 0)} % of mapped trace within
  {val['match_km']:g} km; its detected traces fall near mapped faults {_f(val['lift'], 1)}x more often than random placement.

## 3. Fault patterns likely to be difficult
- Weak / ambiguous segments: {_f(100 * weak_share, 0)} % of mapped trace length has no modality above 0.6 calibrated evidence.
- Median max-evidence by geometry: straight {geom_ev('straight')}, curved {geom_ev('curved')},
  intersection {geom_ev('intersection')}, short {geom_ev('short')} (0.5 = background level).
- Weakest strike sector: {weak_strike.strike_bin if weak_strike is not None else 'n/a'}
  (median max-evidence {_f(weak_strike.median_max_ev if weak_strike is not None else np.nan)}).
- Regional contrast in mapped density (km/100 km²): min {_f(morph['regional'].density_km_per_100km2.min(), 1)},
  max {_f(morph['regional'].density_km_per_100km2.max(), 1)} — part of this is mapping effort, not geology.

## 4. Limitations of the existing labels
- Raster components are not fault objects: by trace length {', '.join(f'{k} {_f(100 * v / total_km, 0)} %' for k, v in kinds_km.items())}.
  Networks merge many faults through junctions; fragments split single faults.
- {s['bridgeable_gaps']} collinear endpoint pairs across components lie within {ctx.cfg.gap_km:g} km
  ({_f(100 * s['endpoint_share_bridgeable'], 0)} % of fault tips) — mapped traces are interrupted, so pixel-exact
  negatives along strike are unreliable.
- {val['n_unmapped']} coherent lineaments ≥ {ctx.cfg.candidate_min_km:g} km have no mapped fault within
  {3 * ctx.cfg.candidate_match_km:g} km. Some are contacts or dykes, but their existence means the negatives contain
  unlabelled positives (positive-unlabelled setting).
- Labels are thin lines while geophysical responses are {_f(patch['fwhm_p75_km'], 1)} km wide: evaluate and train with a
  distance tolerance, not pixel-exact overlap.

## 5. Recommended feature sets and patch sizes
- Raw bands (profile-ranked): {', '.join(f'`{b}`' for b in top_raw)}.
- Enhanced layers: per-modality best transforms above, tilt of RTP and isostatic gravity at 2 and 10 km
  (`features/enhanced_best.tif`), fused lineament score/support/coherence (`features/lineament_maps.tif`),
  structure-tensor coherence per modality (`features/crossmodal_orientation.tif`).
- Context only: strain and seismicity bands are regional (tens of km); use them as low-frequency context channels.
- Never feed `features/label_context.tif` (distance to mapped faults, trace density) to a model: it is derived from labels.
- Patch: **{patch['patch_px']} px ({_f(patch['patch_km'], 0)} km)**, stride {patch['stride_px']} px; needs ≥ {_f(patch['context_km'], 0)} km
  of context (4 x response width, median fault length, profile span).
- Validation: spatial blocks = the {ctx.cfg.region_grid[0]}x{ctx.cfg.region_grid[1]} regions used here (hold out whole regions).

## 6. Three immediate baseline experiments
1. **Zero-training lineament floor** — threshold the fused lineament score (and the ≥3-modality edge-agreement map)
   into the submission grid; tune only the threshold on held-out regions with a {ctx.cfg.candidate_match_km:g} km tolerant F1.
   This is the reference every learned model must beat.
2. **Pixel GBM with ablation** — LightGBM on raw bands vs raw + enhanced layers at 2/5/10 km, target = labels dilated
   by {ctx.cfg.near_tol_px} px, negatives sampled > {ctx.cfg.far_km:g} km from labels, spatial-block CV. The raw→enhanced gain
   measures the value of Section 4; SHAP by modality checks Section 1.
3. **Thin-structure U-Net** — {patch['patch_px']} px patches of the recommended channels, loss = BCE + soft-Dice/clDice on a
   distance-tolerant target, down-weighted negatives on top-ranked unmapped lineaments (PU-aware), region hold-out.

## Top unmapped candidates (exploratory, not confirmed faults)
{_md_table(top_cands[['rank_unmapped', 'length_km', 'azimuth', 'structural_score', 'evidence_score', 'n_supporting_modalities', 'x_mid', 'y_mid']]) if len(top_cands) else 'none'}
"""
    ctx.paths.findings.write_text(md, encoding="utf-8")
    ctx.manifest.add("eda_findings", ctx.paths.findings, "report", "Concise EDA findings and modelling implications")
    return md
