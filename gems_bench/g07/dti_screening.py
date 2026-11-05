"""DTI-first, post-hoc G07 research diagnostics (not a training or promotion stage).

Scores use the competition-style 300 m / alpha=.2 / beta=.8 GEMS scorer and
the canonical V1/V3 units. DEM-only comparisons are LOCAL RESTRICTED PROTOCOLS;
W2+DEM blends are scored over ORIGINAL full units. Both are SCREENING_ONLY:
held-out truths are read to measure, never to choose signs, weights or budgets.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from ..gems3.official import FastScorer
from .evaluation import unit_protocol
from .representation import exact_k_mask

# Directions are declared in advance from the geometric meaning of the feature.
# Do NOT infer a sign from held-out AUC/DTI. Orientation cos/sin are excluded.
FEATURE_DIRECTIONS = {
    "dem_scarp_s10": "high",
    "dem_scarp_s150": "high",
    "dem_scarp_persist_fine": "high",
    "dem_slope_max_s300": "high",
    "dem_lin_s10": "high",
    "dem_dircoh_fine": "high",
    "dem_dircoh_coarse": "high",
    "dem_aniso_s30": "high",
    "dem_hs_range": "high",
    "dem_curvprof_s10": "abs",
}


def _rank_map(values: np.ndarray, eligible: np.ndarray, direction: str = "high") -> np.ndarray:
    """Deterministic, label-free midrank in [0,1]; NaN outside valid domain.

    A direction like 'abs' is fixed before looking at evaluation labels.
    This is a ranking transformation, NOT probability calibration.
    """
    a = np.asarray(values, np.float64)
    ok = np.asarray(eligible, bool) & np.isfinite(a)
    result = np.full(a.shape, np.nan, np.float32)
    if not ok.any():
        return result
    vals = np.abs(a[ok]) if direction == "abs" else a[ok]
    if direction == "low":
        vals = -vals
    if direction not in ("high", "low", "abs"):
        raise ValueError(f"Unapproved feature direction: {direction}")
    result[ok] = (rankdata(vals, method="average") / len(vals)).astype(np.float32)
    return result


def _component_row(comp, **kwargs):
    return {**kwargs, "dti": float(comp.dti), "tp_w": float(comp.tp),
            "fp_w": float(comp.fp), "fn_w": float(comp.fn),
            "n_truth": int(comp.n_truth), "prediction_mass": float(comp.pred_mass)}


def _pooled(rows: pd.DataFrame, groups: tuple[str, ...], expected: dict | None = None) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    records = []
    for key, g in rows.groupby(list(groups), dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        meta = dict(zip(groups, key))
        tp, fp, fn = (float(g[c].sum()) for c in ("tp_w", "fp_w", "fn_w"))
        n = int(g.n_truth.sum())
        n_units = int(g.unit.nunique())
        n_expected = expected.get(meta["scenario"], n_units) if expected else n_units
        dti = tp / (tp + 0.2 * fp + 0.8 * fn + 1e-6) if n else np.nan
        records.append({**meta, "dti": dti, "tp_w": tp, "fp_w": fp,
                        "fn_w": fn, "n_truth": n, "n_units": n_units,
                        "n_expected_units": n_expected,
                        "complete": n_units == n_expected, "result_tier": "SCREENING_ONLY"})
    return pd.DataFrame(records)


def run_feature_dti(
    P,
    *,
    features: dict[str, str] | None = None,
    fracs: tuple[float, ...] = (0.005, 0.01, 0.02),
    fusion_weights: tuple[float, ...] = (0.15, 0.35),
    scenarios: tuple[str, ...] = ("V1", "V3"),
) -> dict:
    """Compare DEM vs W2 on identical local cells, and W2+DEM vs W2 full-region.

    All feature directions, weights and fractions MUST be fixed independently of
    held-out truth; this routine does not select a winning method/weight.
    Local DTI is conditional on coverage and is NOT an official full-region score.
    Blends keep W2 ranks outside DEM coverage and preserve the global exact-k budget.
    """
    if any(not (0 < float(f) < 1) for f in fracs):
        raise ValueError("fracs must be in (0,1)")
    if any(not (0 <= float(a) <= 1) for a in fusion_weights):
        raise ValueError("fusion_weights must be in [0,1]")

    st = P.dem_store()
    if st is None:
        raise FileNotFoundError("DEM feature store missing: run DEM_FEATURES first")
    features = dict(FEATURE_DIRECTIONS if features is None else features)
    features = {f: d for f, d in features.items() if f in st.names}
    if not features:
        raise ValueError("No configured DTI features exist in DEM feature store")
    if any(f.endswith(("_cos2", "_sin2")) for f in features):
        raise ValueError("Directional orientation components need an independently fixed geological reference")

    U = {k: v for k, v in P.units().items() if v.scenario in scenarios}
    w2 = P.w2_maps()
    grid = P.grid()
    template = P.engine().template_valid()
    coverage_threshold = float(P.cfg["dem"].get("min_coverage", 0.5))
    coverage_rows, local_rows, full_rows, system_rows = [], [], [], []
    missing = []
    expected = {s: sum(v.scenario == s for v in U.values()) for s in scenarios}
    ef = float(P.cfg["evaluation"].get("system_recovered_frac", 0.5))
    es = float(P.cfg["evaluation"].get("system_recovered_support", 0.5))

    from ..geomoe.scenarios import system_recovery

    def system_set(vu, q):
        r = system_recovery(vu.unit, q, vu.sysmap, es, ef)
        return set(r.loc[r.recovered, "system_id"].astype(int)) if len(r) else set()

    for key, vu in sorted(U.items()):
        if key not in w2:
            missing.append(f"{key}: missing W2 map")
            continue
        u = vu.unit
        W = np.asarray(w2[key], np.float32)
        if W.shape != u.ev.shape:
            missing.append(f"{key}: W2 shape {W.shape} != {u.ev.shape}")
            continue
        win = (vu.sl[0].start, vu.sl[0].stop, vu.sl[1].start, vu.sl[1].stop)
        proto = unit_protocol(vu, np.asarray(template[vu.sl], bool))
        eligible = proto["elig"]
        coverage = np.asarray(st.coverage(win), np.float32)
        cov = eligible & np.isfinite(coverage) & (coverage >= coverage_threshold)
        truth_cov = u.eval_truth() & cov
        system_ids = np.unique(vu.sysmap[truth_cov & (vu.sysmap >= 0)])
        coverage_rows.append({
            "unit": key, "scenario": vu.scenario, "eligible_px": int(eligible.sum()),
            "dem_covered_px": int(cov.sum()), "dem_truth_px": int(truth_cov.sum()),
            "dem_fault_systems": int(len(system_ids)),
            "dem_share": float(cov.sum() / max(int(eligible.sum()), 1)),
            "validity": "DESCRIPTIVE_ONLY" if len(system_ids) < 5 else "REQUIRES_HELDOUT_AUDIT",
        })

        # Ranking is label-free. No training or feature selection occurs here.
        full_w2_rank = _rank_map(W, eligible, "high")
        base_by_frac = {}
        base_systems = {}
        for frac in fracs:
            q, info = exact_k_mask(W, eligible, frac, offset=(win[0], win[2]), width_full=grid.width)
            base_by_frac[frac] = q
            comp = u.components(q.astype(np.float32))
            full_rows.append(_component_row(comp, model="W2", unit=key, scenario=vu.scenario, frac=frac,
                                            selected_px=int(info["actual_px"]), shortfall=int(info["shortfall"]),
                                            local_feature_px=int(cov.sum()), protocol="FULL_V1_V3"))
            if np.isclose(frac, 0.01):
                base_systems[frac] = system_set(vu, q.astype(np.float32))
        for feature, direction in features.items():
            a = np.asarray(st.load([feature], win)[0], np.float32)
            finite = cov & np.isfinite(a)
            if not finite.any():
                missing.append(f"{key}: {feature} has no finite coverage")
                # Still evaluate the no-op fusion, so full-region comparisons remain complete.
            rank_local = _rank_map(a, finite, direction)
            for frac in fracs:
                if finite.any():
                    local_scorer = FastScorer(u.truth, u.ev & finite, u.exclude,
                                              exclude_supports_tp=u.supports_tp)
                    q_w, rw = exact_k_mask(W, finite, frac, offset=(win[0], win[2]), width_full=grid.width)
                    q_f, rf = exact_k_mask(rank_local, finite, frac, offset=(win[0], win[2]),
                                           width_full=grid.width)
                    cw = local_scorer.components(q_w.astype(np.float32))
                    cf = local_scorer.components(q_f.astype(np.float32))
                    local_rows.extend([
                        _component_row(cw, model="W2", feature=feature, unit=key, scenario=vu.scenario,
                                       frac=frac, finite_px=int(finite.sum()), selected_px=int(rw["actual_px"]),
                                       shortfall=int(rw["shortfall"]), protocol="DEM_COVERAGE_RESTRICTED"),
                        _component_row(cf, model=feature, feature=feature, unit=key, scenario=vu.scenario,
                                       frac=frac, finite_px=int(finite.sum()), selected_px=int(rf["actual_px"]),
                                       shortfall=int(rf["shortfall"]), protocol="DEM_COVERAGE_RESTRICTED"),
                    ])
                for alpha in fusion_weights:
                    model = f"W2+{feature}@{alpha:.2f}"
                    fusion = full_w2_rank.copy()
                    if finite.any():
                        # Blend two LABEL-FREE ranks inside coverage; W2 stays unchanged elsewhere.
                        fusion[finite] = ((1.0 - alpha) * full_w2_rank[finite] +
                                          alpha * rank_local[finite])
                    q_mix, rm = exact_k_mask(fusion, eligible, frac, offset=(win[0], win[2]),
                                              width_full=grid.width)
                    comp = u.components(q_mix.astype(np.float32))
                    full_rows.append(_component_row(comp, model=model, unit=key, scenario=vu.scenario,
                                                    frac=frac, selected_px=int(rm["actual_px"]),
                                                    shortfall=int(rm["shortfall"]),
                                                    local_feature_px=int(finite.sum()),
                                                    protocol="FULL_V1_V3"))
                    if frac in base_systems:
                        current = system_set(vu, q_mix.astype(np.float32))
                        ref = base_systems[frac]
                        system_rows.append({
                            "scenario": vu.scenario, "unit": key, "model": model,
                            "frac": frac, "n_recovered": len(current),
                            "n_new_vs_w2": len(current - ref),
                            "n_lost_vs_w2": len(ref - current),
                            "new_ids": ";".join(map(str, sorted(current - ref))),
                            "lost_ids": ";".join(map(str, sorted(ref - current))),
                        })
        # Avoid cached scorers keeping large arrays alive across units.
        u.release()

    local = pd.DataFrame(local_rows)
    full = pd.DataFrame(full_rows)
    coverage_df = pd.DataFrame(coverage_rows)
    # Local samples vary per feature. Pool only within one feature and scenario.
    local_sum = _pooled(local, ("feature", "model", "scenario", "frac")) if len(local) else pd.DataFrame()
    full_sum = _pooled(full, ("model", "scenario", "frac"), expected) if len(full) else pd.DataFrame()

    if len(local_sum):
        ref = local_sum[local_sum.model == "W2"][["feature", "scenario", "frac", "dti"]].rename(
            columns={"dti": "w2_matched_dti"})
        local_sum = local_sum.merge(ref, on=["feature", "scenario", "frac"], how="left")
        local_sum["delta_dti"] = local_sum.dti - local_sum.w2_matched_dti
        local_sum["protocol_note"] = "Restricted DEM-coverage protocol; never comparable to full-region DTI."
    if len(full_sum):
        ref = full_sum[full_sum.model == "W2"][["scenario", "frac", "dti"]].rename(
            columns={"dti": "w2_dti"})
        full_sum = full_sum.merge(ref, on=["scenario", "frac"], how="left")
        full_sum["delta_dti"] = full_sum.dti - full_sum.w2_dti
        full_sum["usable_for_promotion"] = False  # labels used post-hoc; fixed settings not nested-selected
        full_sum["protocol_note"] = "Original V1/V3 unit; post-hoc fixed-policy screening only."
    out = "evaluation/dti_screening"
    for name, frame in [
        ("coverage.csv", coverage_df), ("feature_local_rows.csv", local),
        ("feature_local_summary.csv", local_sum), ("fusion_full_rows.csv", full),
        ("fusion_full_summary.csv", full_sum), ("system_recovery.csv", pd.DataFrame(system_rows)),
        ("missing_inputs.csv", pd.DataFrame({"reason": missing})),
    ]:
        P.ws.save_table(frame, f"{out}/{name}")
    P.ws.save_json({
        "status": "SCREENING_ONLY",
        "feature_directions": features,
        "primary_frac": 0.01,
        "fracs": list(fracs),
        "fusion_weights": list(fusion_weights),
        "note": "Held-out truth is used ONLY for post-hoc scoring; no feature/weight selected.",
        "local_metric": "Restricted-domain 300 m DTI (not official full-region DTI).",
        "full_metric": "Canonical 300 m DTI with full V1/V3 masks and exact-k policy.",
        "continuous_metric": "Not computed for uncalibrated raw feature/rank maps.",
        "missing_inputs": missing[:40],
    }, f"{out}/manifest.json")
    P.push([out])
    return {"coverage": coverage_df, "local": local_sum, "full": full_sum,
            "system_recovery": pd.DataFrame(system_rows), "missing": missing}


def cached_models_dti(P, fracs: tuple[float, ...] = (0.01,)) -> dict:
    """Score cached G07 model maps and W2; NEVER pool incomplete scenario folds.

    This avoids the separate promotion-stage failure and does not retrain models.
    """
    from . import evaluation as EV

    U = P.units()
    maps, strict, kinds = P._score_maps()
    scenarios = [s for s in ("V1", "V3") if s in P.cfg["validation"]["scenarios"]]
    status = EV.evaluation_status(maps, U, strict, scenarios)
    templates = {k: P.engine().template_valid()[vu.sl] for k, vu in U.items()}
    data = []
    for scenario in scenarios:
        score_maps = {m: v for m, v in maps.items() if kinds.get(m) == "score"}
        rows, _ = EV.evaluate_scenario(
            U, score_maps, scenario, {"top": {"select": "top", "post": []}},
            fracs, templates, P.cfg, strict, width_full=P.grid().width,
        )
        if len(rows):
            data.append(rows)
    unit_rows = pd.concat(data, ignore_index=True) if data else pd.DataFrame()
    if not unit_rows.empty:
        full_pairs = {(r.model, r.scenario) for r in status.itertuples() if r.status == "FULLY_EVALUATED"}
        complete = unit_rows.loc[[((r.model, r.scenario) in full_pairs) for r in unit_rows.itertuples()]]
        pooled = EV.aggregate_components(complete) if len(complete) else pd.DataFrame()
        paired = EV.compare_to_w2(pooled) if len(pooled) else pd.DataFrame()
    else:
        pooled, paired = pd.DataFrame(), pd.DataFrame()
    out = "evaluation/dti_screening"
    for name, frame in [("cached_model_status.csv", status),
                        ("cached_model_unit_rows.csv", unit_rows),
                        ("cached_model_complete_pooled.csv", pooled),
                        ("cached_model_paired_vs_w2.csv", paired)]:
        P.ws.save_table(frame, f"{out}/{name}")
    P.push([out])
    return {"status": status, "unit_rows": unit_rows, "pooled": pooled, "paired": paired}


def save_dti_figures(results: dict, root) -> list[str]:
    """PNG/SVG figures with explicit SCREENING_ONLY semantics; no synthetic scores."""
    import matplotlib.pyplot as plt

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    saved = []
    for key, fname, desc in [
        ("local", "dem_covered_delta_dti", "Local DEM-covered ΔDTI vs matched W2 (diagnostic)"),
        ("full", "full_region_fusion_delta_dti", "Full-region W2+DEM ΔDTI (post-hoc screening)"),
    ]:
        df = results.get(key)
        if df is None or df.empty:
            continue
        s = df[(df.scenario == "V1") & np.isclose(df.frac, .01) & (df.model != "W2")]
        if s.empty:
            continue
        s = s.sort_values("delta_dti").tail(20)
        fig, ax = plt.subplots(figsize=(10, max(3, len(s) * .32 + 1.5)))
        ax.barh(s.model.astype(str), s.delta_dti.astype(float))
        ax.axvline(0, color="black", linewidth=1)
        ax.set_xlabel("Paired ΔDTI (positive = better than W2)")
        ax.set_title(desc)
        fig.text(.01, .01, "SCREENING_ONLY · one/few DEM fault systems · not a promotion result",
                 fontsize=8, alpha=.7)
        fig.tight_layout(rect=(0, .04, 1, 1))
        for ext in ("png", "svg"):
            path = root / f"{fname}.{ext}"
            fig.savefig(path, dpi=180, bbox_inches="tight")
            saved.append(str(path))
        plt.close(fig)
    return saved



def continuous_probability_dti(P, model_names: tuple[str, ...] = ("G07-B0", "G07-B1")) -> dict:
    """Official-form DTI of explicit [0,1] model probability maps, no area cap.

    Does NOT convert feature ranks to probabilities and does NOT presume W2 ranks
    are calibrated probabilities. No paired W2 delta is claimed on this track.
    Complete scenario coverage is required for a pooled DTI. Calibration remains
    an independent training-fold-only question; all results are diagnostic.
    """
    maps, strict, kind = P._score_maps()
    U = {k: v for k, v in P.units().items() if v.scenario in ("V1", "V3")}
    expected = {s: sum(v.scenario == s for v in U.values()) for s in ("V1", "V3")}
    rows, rejected = [], []
    for model in model_names:
        if model not in maps or kind.get(model) != "score":
            rejected.append({"model": model, "unit": "*", "reason": "model unavailable or not a score map"})
            continue
        for key, vu in sorted(U.items()):
            if key not in maps[model]:
                rejected.append({"model": model, "unit": key, "reason": "missing held-out prediction"})
                continue
            u = vu.unit
            raw = np.asarray(maps[model][key], np.float64)
            if raw.shape != u.ev.shape:
                rejected.append({"model": model, "unit": key, "reason": "shape mismatch"})
                continue
            eligible = u.scored_mask()
            values = raw[eligible & np.isfinite(raw)]
            if not values.size:
                rejected.append({"model": model, "unit": key, "reason": "no finite eligible predictions"})
                continue
            lo, hi = float(values.min()), float(values.max())
            if lo < -1e-6 or hi > 1 + 1e-6:
                rejected.append({"model": model, "unit": key,
                                 "reason": f"not probability-valued: range [{lo}, {hi}]"})
                continue
            # Undefined prediction pixels count as zero, as in the canonical scorer.
            q = np.clip(np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
            comp = u.components(q)
            rows.append(_component_row(comp, model=model, unit=key, scenario=vu.scenario,
                                       frac=np.nan, protocol="FULL_CONTINUOUS_NO_AREA_CAP",
                                       prediction_kind="probability_uncalibrated",
                                       nonfinite_px=int((eligible & ~np.isfinite(raw)).sum()),
                                       positive_area_px=int(((q > 0) & eligible).sum()),
                                       strictness=strict.get(model, "SCREENING_ONLY")))
            u.release()
    frame = pd.DataFrame(rows)
    pooled = _pooled(frame, ("model", "scenario"), expected) if len(frame) else pd.DataFrame()
    if len(pooled):
        # Even when sources are strict by split, calibration/promotional eligibility
        # cannot be inferred from numeric ranges alone.
        pooled["prediction_kind"] = "probability_uncalibrated"
        pooled["metric"] = "official_form_dti_without_area_cap"
        pooled["usable_for_promotion"] = False
    out = "evaluation/dti_screening"
    P.ws.save_table(frame, f"{out}/continuous_probability_unit_rows.csv")
    P.ws.save_table(pooled, f"{out}/continuous_probability_pooled.csv")
    P.ws.save_table(pd.DataFrame(rejected, columns=["model", "unit", "reason"]),
                    f"{out}/continuous_probability_exclusions.csv")
    P.push([out])
    return {"units": frame, "pooled": pooled, "rejected": pd.DataFrame(rejected)}
