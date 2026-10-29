"""Stage 13 figures: every PNG is drawn from persisted tables and saved with ``<name>_data.csv`` (Workspace.save_fig).
A figure whose inputs are absent is skipped and named in ``figures/MISSING.txt``; it never fails the run and never
plots placeholder values."""
from __future__ import annotations

import numpy as np
import pandas as pd

FAM = ("E1_learned", "E2_potential", "E3_structural", "E4_terrain_network")
REF = "A_submitted_W2"


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _bars(ax, D: pd.DataFrame, x: str, hue: str, y: str):
    xs, hs = list(dict.fromkeys(D[x])), list(dict.fromkeys(D[hue]))
    w = 0.8 / max(len(hs), 1)
    for i, h in enumerate(hs):
        d = D[D[hue] == h].set_index(x).reindex(xs)
        ax.bar(np.arange(len(xs)) + i * w, d[y].to_numpy(float), w, label=str(h))
    ax.set_xticks(np.arange(len(xs)) + 0.4 - w / 2)
    ax.set_xticklabels(xs)
    ax.legend(fontsize=7)


def baseline_comparison(G):
    S = G.ws.load_table("gating/model_summary.csv")
    Sd = G.ws.load_table("gating/model_summary_scenario_selected.csv")
    if not len(S):
        return None
    keep = [m for m in (REF, "A_W2_policy_reselected", "A_seg", "B", "C_all", "D") if m in set(S.model)]
    D = S[S.model.isin(keep)][["scenario", "model", "dti", "worst_zone_dti", "protocol"]]
    if len(Sd):
        D = pd.concat([D, Sd[Sd.model == "A_W2_scenario_selected_diagnostic"][["scenario", "model", "dti",
                                                                               "worst_zone_dti"]].assign(
            protocol="scenario_selected_diagnostic")], ignore_index=True)
    plt = _plt()
    fig, ax = plt.subplots(figsize=(10, 4))
    _bars(ax, D, "scenario", "model", "dti")
    ax.set(ylabel="pooled held-out DTI", title="Frozen submitted W2 vs fixed-policy models (diagnostic W2 hatched "
                                               "in data only)")
    return fig, D


def oracle_vs_actual(G):
    K = G.ws.load_table("candidates/oracle_curves.csv")
    S = G.ws.load_table("validation/frozen_w2_baseline.csv")
    if not len(K):
        return None
    g = K.groupby(["scenario", "order", "frac"])[["tp_w", "fp_w", "fn_w"]].sum().reset_index()
    g["dti"] = g.tp_w / (g.tp_w + 0.2 * g.fp_w + 0.8 * g.fn_w + 1e-6)
    D = g.assign(kind=g.order)
    if len(S):
        f = S.groupby(["scenario", "frac"])[["tp_w", "fp_w", "fn_w"]].sum().reset_index()
        f["dti"] = f.tp_w / (f.tp_w + 0.2 * f.fp_w + 0.8 * f.fn_w + 1e-6)
        D = pd.concat([D, f.assign(kind="frozen_W2_actual")], ignore_index=True)
    plt = _plt()
    scs = list(dict.fromkeys(D.scenario))
    fig, axs = plt.subplots(1, len(scs), figsize=(4.5 * len(scs), 4), squeeze=False, sharey=True)
    for ax, sc in zip(axs[0], scs):
        for k, d in D[D.scenario == sc].groupby("kind"):
            ax.plot(d.frac, d.dti, "o-" if k != "frozen_W2_actual" else "k*", label=k)
        ax.set(title=sc, xlabel="area fraction of eligible", ylabel="official DTI")
    axs[0][0].legend(fontsize=7)
    fig.suptitle("Kernel-weighted oracle (DIAGNOSTIC, truth-ordered) vs label-free W2 order vs frozen W2 actual")
    return fig, D


def unique_geology_recovery(G):
    C = G.ws.load_table("candidates/oracle_coverage.csv")
    if not len(C):
        return None
    cols = [c for c in C if c.startswith("unique_")] + [c for c in ("independent_not_E1", "independent_not_W2")
                                                         if c in C]
    D = C.melt(id_vars=["scenario"], value_vars=cols, var_name="quantity", value_name="systems")
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 4))
    _bars(ax, D, "scenario", "quantity", "systems")
    ax.set(ylabel="systems", title="Systems covered only by one family / by independent geology and no E1 / no W2")
    return fig, D


def v2_fault_subtypes(G):
    V = G.ws.load_table("validation/v2_by_subtype.csv")
    if not len(V):
        V = G.ws.load_table("validation/v2_coverage_by_subtype.csv")
        if not len(V):
            return None
        D = V.melt(id_vars=["subtype"], value_vars=[c for c in V if c.startswith(("inventory_", "w2_", "coverage_"))],
                   var_name="quantity", value_name="share")
        y, title = "share", "V2 subtypes: geometric coverage and frozen W2 recovery (no model comparison yet)"
    else:
        D = V[[c for c in ("subtype", "model", "dti", "recovered_share", "tp_w", "fp_w", "fn_w") if c in V]]
        D = D.rename(columns={"model": "quantity", "dti": "share"})
        y, title = "share", "V2 subtypes: official DTI under each subtype's own mask, per model"
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 4))
    _bars(ax, D, "subtype", "quantity", y)
    ax.set(title=title)
    return fig, D


def expert_complementarity(G):
    U = G.load_parquet("candidates/unique_parts.parquet")
    if not len(U):
        return None
    rows = []
    for a in FAM:
        for b in FAM:
            A, B = U.get(f"covered_{a}"), U.get(f"covered_{b}")
            if A is None or B is None:
                continue
            A, B = A.astype(bool), B.astype(bool)
            rows.append({"family_a": a, "family_b": b, "jaccard": float((A & B).sum() / max(int((A | B).sum()), 1))})
    D = pd.DataFrame(rows)
    if not len(D):
        return None
    M = D.pivot(index="family_a", columns="family_b", values="jaccard").reindex(index=FAM, columns=FAM)
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(M.to_numpy(float), vmin=0, vmax=1, cmap="viridis")
    ax.set(xticks=range(4), yticks=range(4), xticklabels=[f[:2] for f in FAM], yticklabels=list(FAM),
           title="Covered-system overlap between expert families (Jaccard)")
    fig.colorbar(im, ax=ax)
    return fig, D


def gating_comparison(G):
    P = G.ws.load_table("gating/paired_comparison.csv")
    if not len(P):
        return None
    D = P[["scenario", "method", "reference", "delta_dti", "ci_lo", "ci_hi", "p_delta_positive"]].copy()
    D["pair"] = D.method + " - " + D.reference
    plt = _plt()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, (sc, g) in enumerate(D.groupby("scenario")):
        x = np.arange(len(g)) + i * 0.25
        ax.errorbar(x, g.delta_dti, yerr=[g.delta_dti - g.ci_lo, g.ci_hi - g.delta_dti], fmt="o", label=sc)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(np.arange(D.pair.nunique()))
    ax.set_xticklabels(list(dict.fromkeys(D.pair)), rotation=45, ha="right", fontsize=7)
    ax.set(ylabel="paired DTI delta (90 % block bootstrap)", title="Fusion / gating vs frozen W2 and vs static fusion")
    ax.legend(fontsize=7)
    return fig, D


def geographic_robustness(G):
    P = G.ws.load_table("validation/per_zone_metrics.csv")
    if not len(P):
        return None
    D = P[[c for c in ("scenario", "model", "zone", "rep", "frac", "dti_heldout", "dev_rule") if c in P]]
    plt = _plt()
    scs = list(dict.fromkeys(D.scenario))
    fig, axs = plt.subplots(1, len(scs), figsize=(4 * len(scs), 4), sharey=True, squeeze=False)
    for ax, sc in zip(axs[0], scs):
        for m, g in D[D.scenario == sc].groupby("model"):
            ax.plot(g.zone, g.dti_heldout, "o-", label=m)
        ax.set(title=sc, xlabel="held-out zone", ylabel="DTI")
    axs[0][-1].legend(fontsize=6)
    fig.suptitle("Per-zone held-out DTI (frozen W2 + fixed-policy models)")
    return fig, D


FIGURES = {"baseline_comparison": baseline_comparison, "oracle_vs_actual": oracle_vs_actual,
           "unique_geology_recovery": unique_geology_recovery, "v2_fault_subtypes": v2_fault_subtypes,
           "expert_complementarity": expert_complementarity, "gating_comparison": gating_comparison,
           "geographic_robustness": geographic_robustness}


def write_all(G) -> list[str]:
    missing = []
    for name, fn in FIGURES.items():
        try:
            out = fn(G)
        except Exception as e:                           # a figure never decides anything
            missing.append(f"{name}: {type(e).__name__}: {e}")
            continue
        if out is None:
            missing.append(f"{name}: inputs absent")
            continue
        fig, data = out
        G.ws.save_fig(fig, f"figures/{name}.png", data)
    G.ws.save_text("\n".join(missing) + "\n" if missing else "none\n", "figures/MISSING.txt")
    return missing
