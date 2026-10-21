"""Benchmark report: summary tables, comparison figure and BENCHMARK_FINDINGS.md.

Every recommendation in the findings file is a *rule applied to saved numbers*; when an input
is missing the corresponding decision is reported as "not evaluated" instead of guessed.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import BenchPaths
from .persist import ExperimentRun, load_registry

MIN_GAIN = 0.005          # DTI gain treated as meaningful only if consistent across folds too


def fold_table(run: ExperimentRun) -> pd.DataFrame:
    m = run.table("metrics_fold")
    m = m[m.fold.astype(str) != "pooled"].copy()
    m["fold"] = m.fold.astype(int)
    return m


def paired_delta(a: ExperimentRun | None, b: ExperimentRun | None, col: str = "dti") -> dict:
    """b - a per fold. Verdict: 'gain' (all folds > 0 and mean > MIN_GAIN), 'loss', or 'inconclusive'."""
    if a is None or b is None or not a.done or not b.done:
        return {"verdict": "not evaluated"}
    fa, fb = fold_table(a).set_index("fold")[col], fold_table(b).set_index("fold")[col]
    d = (fb - fa).dropna()
    if d.empty:
        return {"verdict": "not evaluated"}
    verdict = "gain" if (d > 0).all() and d.mean() > MIN_GAIN else "loss" if (d < 0).all() and d.mean() < -MIN_GAIN \
        else "inconclusive"
    return {"verdict": verdict, "mean_delta": float(d.mean()), "min_delta": float(d.min()), "max_delta": float(d.max()),
            "per_fold": {int(k): round(float(v), 4) for k, v in d.items()}}


def benchmark_summary(paths: BenchPaths, runs: dict[str, ExperimentRun],
                      discovery: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    for label, r in runs.items():
        if not r.done:
            rows.append({"model": label, "exp_id": r.exp_id, "status": r.state.get("status")})
            continue
        s = r.state.get("summary", {})
        ft = fold_table(r)
        rows.append({
            "model": label, "exp_id": r.exp_id, "family": r.family, "status": "done",
            "input": r.config.get("group") or r.config.get("input_config"),
            "n_inputs": s.get("n_features"),
            "dti_pooled_raw": s.get("dti_pooled"), "dti_pooled_crossfit_tf": s.get("dti_pooled_crossfit_tf"),
            "dti_fold_mean": s.get("dti_fold_mean"), "dti_fold_std": s.get("dti_fold_std"), "dti_fold_min": s.get("dti_fold_min"),
            "roc_auc": s.get("roc_auc_pooled"), "avg_precision": s.get("ap_pooled"),
            "recall@100m": ft["recall@100m"].mean() if "recall@100m" in ft else np.nan,
            "precision@100m": ft["precision@100m"].mean() if "precision@100m" in ft else np.nan,
            "recall@300m": s.get("recall300_pooled"),
            "train_seconds": s.get("train_seconds"),
            "gpu_peak_mb": ft["gpu_peak_mb"].max() if "gpu_peak_mb" in ft else np.nan,
            "rss_mb_max": max(r.state.get("rss_mb", {}).values(), default=np.nan),
        })
    df = pd.DataFrame(rows)
    if discovery is not None and len(discovery) and "model" in df:
        d = discovery.groupby("model").agg(pseudo_discovery_dti=("dti_excl_retained", "mean"),
                                           pseudo_discovery_recall_w=("weighted_recall", "mean"),
                                           pseudo_discovery_recall300=("recall@300m", "mean")).reset_index()
        df = df.merge(d, on="model", how="left")
    return df


def comparison_figure(summary: pd.DataFrame, fold_tables: dict[str, pd.DataFrame]):
    import matplotlib.pyplot as plt

    s = summary[summary.status == "done"].sort_values("dti_pooled_crossfit_tf")
    fig, axs = plt.subplots(1, 2, figsize=(14, 0.55 * len(s) + 2), constrained_layout=True)
    y = np.arange(len(s))
    axs[0].barh(y - 0.2, s.dti_pooled_raw, 0.4, label="pooled OOF (raw p)", color="#9aa5b1")
    axs[0].barh(y + 0.2, s.dti_pooled_crossfit_tf, 0.4, label="pooled OOF (cross-fitted output transform)", color="#1f618d")
    for i, m in enumerate(s.model):
        ft = fold_tables.get(m)
        if ft is not None:
            axs[0].scatter(ft.dti, np.full(len(ft), i), color="k", s=12, zorder=3, label="per fold" if i == 0 else None)
    axs[0].set_yticks(y, s.model)
    axs[0].set_xlabel("distance-weighted Tversky index (OOF, geographic folds)")
    axs[0].legend(fontsize=8, loc="lower right")
    if "pseudo_discovery_recall_w" in s:
        axs[1].barh(y, s.pseudo_discovery_recall_w.fillna(0), color="#b03a2e")
        axs[1].set_yticks(y, s.model)
        axs[1].set_xlabel("pseudo-discovery weighted recall of withheld systems")
    else:
        axs[1].set_axis_off()
    return fig


def _fmt(x, nd=4):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def write_findings(path: Path, summary: pd.DataFrame, runs: dict[str, ExperimentRun], *,
                   archetype: pd.DataFrame | None = None, discovery: pd.DataFrame | None = None,
                   ensembles: pd.DataFrame | None = None, roles: dict[str, str] | None = None) -> Path:
    """``roles`` maps role -> model label: G1, G2, G3, G4, CNN_A, CNN_B (any subset)."""
    roles = roles or {}
    R = lambda k: runs.get(roles.get(k, ""), None)
    done = summary[summary.get("status", pd.Series(dtype=str)) == "done"].copy() if len(summary) else summary
    L = [f"# GEMS — Phase 2 benchmark findings", "",
         f"_Generated {time.strftime('%Y-%m-%d %H:%M')} by `notebooks/02_fault_benchmark.ipynb` from saved artifacts. "
         "All scores are out-of-fold on geographic folds (not leaderboard scores)._", ""]
    if not len(done):
        L += ["No finished experiment found — run the notebook first."]
        path.write_text("\n".join(L), encoding="utf-8")
        return path

    L += ["## Results", "", done[["model", "dti_pooled_raw", "dti_pooled_crossfit_tf", "dti_fold_mean", "dti_fold_std",
                                  "dti_fold_min", "recall@100m", "precision@100m"]
                                 + [c for c in ("pseudo_discovery_recall_w",) if c in done]].round(4).to_markdown(index=False), ""]

    # 1. primary baseline: best cross-fitted DTI; ties (within the leader's fold std) go to the higher fold minimum
    d = done.sort_values("dti_pooled_crossfit_tf", ascending=False)
    lead = d.iloc[0]
    close = d[d.dti_pooled_crossfit_tf >= lead.dti_pooled_crossfit_tf - (lead.dti_fold_std or 0)]
    primary = close.sort_values("dti_fold_min", ascending=False).iloc[0]
    L += ["## Decisions", "",
          f"**1. Primary baseline:** `{primary.model}` (pooled cross-fitted DTI {_fmt(primary.dti_pooled_crossfit_tf)}, "
          f"worst fold {_fmt(primary.dti_fold_min)}). Candidates within one fold-std of the leader: "
          f"{', '.join(close.model)}; the most fold-robust of them was chosen.", ""]

    # 2. modalities
    g12 = paired_delta(R("G1"), R("G2"))
    L += [f"**2. Do non-terrain modalities add value (G2 raw-all vs G1 terrain-only)?** verdict = **{g12['verdict']}** "
          f"(mean ΔDTI {_fmt(g12.get('mean_delta'))}, per fold {g12.get('per_fold', '-')}).",
          "   - gain → keep all modalities; inconclusive → keep them but test modality-drop ablations on G3; "
          "loss → terrain-only is the stronger prior, drop low-SHAP modalities before Phase 3.", ""]
    g23 = paired_delta(R("G2"), R("G3"))
    g34 = paired_delta(R("G3"), R("G4"))
    L += [f"**3. Is multi-scale feature engineering worth keeping (G3 vs G2)?** verdict = **{g23['verdict']}** "
          f"(mean ΔDTI {_fmt(g23.get('mean_delta'))}). Regional context (G4 vs G3): **{g34['verdict']}** "
          f"(mean ΔDTI {_fmt(g34.get('mean_delta'))}).",
          "   - keep engineered layers only on a 'gain' verdict; on 'inconclusive' keep the top-SHAP subset only.", ""]
    best_gbm = done[done.family == "gbm"].sort_values("dti_pooled_crossfit_tf").tail(1)
    best_cnn = done[done.family == "cnn"].sort_values("dti_pooled_crossfit_tf").tail(1)
    if len(best_gbm) and len(best_cnn):
        a, b = best_gbm.iloc[0], best_cnn.iloc[0]
        loc = (b["precision@100m"] - a["precision@100m"], b["recall@100m"] - a["recall@100m"])
        dt = b.dti_pooled_crossfit_tf - a.dti_pooled_crossfit_tf
        verdict = "yes" if dt > MIN_GAIN and loc[0] > 0 else "no" if dt < -MIN_GAIN else "mixed"
        L += [f"**4. Does U-Net improve fine localisation over LightGBM?** **{verdict}** — `{b.model}` vs `{a.model}`: "
              f"ΔDTI {_fmt(dt)}, Δprecision@100m {_fmt(loc[0])}, Δrecall@100m {_fmt(loc[1])}.", ""]
    else:
        L += ["**4. U-Net vs LightGBM:** not evaluated (one family missing).", ""]
    if archetype is not None and len(archetype):
        a = archetype[(archetype.n_segments >= 20)].sort_values("weighted_recall").head(5)
        L += ["**5. Poorly recovered fault types (primary model, strata with ≥ 20 segments):**", "",
              a[["stratum", "value", "n_segments", "trace_km", "weighted_recall"]].round(3).to_markdown(index=False), ""]
    else:
        L += ["**5. Poorly recovered fault types:** not evaluated.", ""]

    # 6. next phase
    geo_rec = primary.get("recall@300m", np.nan)
    disc_rec = primary.get("pseudo_discovery_recall300", np.nan) if "pseudo_discovery_recall300" in primary else np.nan
    ens_gain = np.nan
    if ensembles is not None and len(ensembles):
        ens_gain = ensembles.dti_crossfit_tf.max() - primary.dti_pooled_crossfit_tf
    cnn_gain = (best_cnn.iloc[0].dti_pooled_crossfit_tf - best_gbm.iloc[0].dti_pooled_crossfit_tf) \
        if len(best_gbm) and len(best_cnn) else np.nan
    if np.isfinite(disc_rec) and np.isfinite(geo_rec) and disc_rec < 0.7 * geo_rec:
        nxt = "**PU learning / label-incompleteness handling** — withheld systems are rediscovered far less than mapped faults in held-out regions (pseudo-discovery recall@300m {:.3f} vs geographic {:.3f})".format(disc_rec, geo_rec)
    elif np.isfinite(cnn_gain) and cnn_gain > 0.02:
        nxt = f"**architecture** — the CNN leads the best GBM by {cnn_gain:.3f} DTI"
    elif np.isfinite(ens_gain) and ens_gain > 0.01:
        nxt = f"**ensembling** — the best OOF ensemble adds {ens_gain:.3f} DTI over the primary baseline"
    else:
        nxt = "**targets and post-processing** (distance-aware targets, line thinning, output calibration): no family or ensemble shows a clear margin"
    L += [f"**6. Next research phase:** {nxt}.",
          "   Rule order: PU first if pseudo-discovery recall < 0.7 × geographic recall; else architecture if CNN − GBM > 0.02; "
          "else ensembling if ensemble gain > 0.01; else targets / post-processing.", ""]
    if discovery is not None and len(discovery):
        L += ["## Pseudo-discovery (withheld fault systems)", "",
              discovery.groupby("model")[["dti_excl_retained", "dti_no_exclusion", "weighted_recall", "recall@100m",
                                          "recall@300m", "extra_confident_km2"]].mean().round(4).to_markdown(), ""]
    if ensembles is not None and len(ensembles):
        L += ["## Ensembles (OOF, cross-fitted)", "", ensembles.round(4).to_markdown(index=False), ""]
    L += ["## Caveats", "",
          "- Geographic OOF DTI scores mapped USGS faults in held-out regions; the hidden test scores expert-mapped *new* faults.",
          "- Phase-1 transform/scale choices used all labels (mild selection effect on G3/G4).",
          "- Filters wider than the fold buffer see validation-area inputs (never labels): see `tables/leakage_audit.csv`.",
          "- Known-fault policy for submissions (`model` vs `model_max_known`) is unresolved until leaderboard feedback."]
    path.write_text("\n".join(L), encoding="utf-8")
    return path
