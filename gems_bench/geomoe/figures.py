"""Stage 13 figures: every PNG is drawn from persisted tables and saved with ``<name>_data.csv`` (Workspace.save_fig).
A figure whose inputs are absent is skipped and named in ``figures/MISSING.txt``; it never fails the run."""
from __future__ import annotations

import numpy as np
import pandas as pd

FAM = ("E1_learned", "E2_potential", "E3_structural", "E4_terrain_network")


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def candidate_coverage(G):
    S = G.ws.load_table("candidates/oracle_coverage.csv")
    if not len(S):
        return None
    cols = [c for c in S if c.startswith("cov_fam:") or c in ("cov_label_free", "cov_inventory",
                                                                "w2_recovered_share")]
    D = S.melt(id_vars=["scenario"], value_vars=cols, var_name="pool", value_name="share")
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 4))
    pools = list(dict.fromkeys(D.pool))
    sc = list(dict.fromkeys(D.scenario))
    w = 0.8 / max(len(pools), 1)
    for i, p in enumerate(pools):
        d = D[D.pool == p].set_index("scenario").reindex(sc)
        ax.bar(np.arange(len(sc)) + i * w, d.share, w, label=p.replace("cov_", "").replace("fam:", ""))
    ax.set(xticks=np.arange(len(sc)) + 0.4 - w / 2, xticklabels=sc, ylabel="share of truth systems", ylim=(0, 1),
           title="Candidate coverage (300 m, >= 50 % of a system) vs W2 deployed recovery")
    ax.legend(fontsize=7, ncol=2)
    return fig, D


def unique_recovery(G):
    S = G.ws.load_table("candidates/oracle_coverage.csv")
    if not len(S):
        return None
    cols = [c for c in S if c.startswith("unique_")] + ["independent_not_E1"]
    D = S.melt(id_vars=["scenario"], value_vars=[c for c in cols if c in S], var_name="family", value_name="systems")
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (f, d) in enumerate(D.groupby("family", sort=False)):
        ax.bar(np.arange(d.scenario.nunique()) + i * 0.15, d.systems, 0.15, label=f)
    ax.set(xticks=np.arange(S.scenario.nunique()) + 0.3, xticklabels=list(S.scenario), ylabel="systems",
           title="Systems covered by one family only (unique) / by independent geology and no learned source")
    ax.legend(fontsize=7)
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
    M = D.pivot(index="family_a", columns="family_b", values="jaccard").reindex(index=FAM, columns=FAM)
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(M.to_numpy(float), vmin=0, vmax=1, cmap="viridis")
    ax.set(xticks=range(4), yticks=range(4), xticklabels=[f[:2] for f in FAM], yticklabels=list(FAM),
           title="Covered-system overlap between expert families (Jaccard)")
    fig.colorbar(im, ax=ax)
    return fig, D


def gate_weights_by_geology(G):
    W = G.load_parquet("gating/gating_weights.parquet")
    if len(W):
        W["known_band"] = pd.cut(W.known_dist_km, [-1, 1, 5, 1e9], labels=["near<=1km", "mid<=5km", "far"])
        D = W.groupby(["scenario", "known_band"], observed=True)[[f"g_{f}" for f in FAM]].mean().reset_index()
        title = "Learned gate: mean routing weight by distance to the permitted known faults"
    else:
        S = G.ws.load_table("gating/static_weights.csv")
        if not len(S):
            return None
        D = S.rename(columns={f"w_{f}": f"g_{f}" for f in FAM}).assign(known_band="all")
        D["scenario"] = D.unit.str.split(":").str[0]
        D = D.groupby(["scenario", "known_band"])[[f"g_{f}" for f in FAM]].mean().reset_index()
        title = "Static fusion weights (no learned gate was trained)"
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 4))
    lab = [f"{a}|{b}" for a, b in zip(D.scenario, D.known_band)]
    bottom = np.zeros(len(D))
    for f in FAM:
        ax.bar(lab, D[f"g_{f}"].to_numpy(float), bottom=bottom, label=f)
        bottom += np.nan_to_num(D[f"g_{f}"].to_numpy(float))
    ax.set(ylabel="weight", title=title)
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.legend(fontsize=7)
    return fig, D


def discovery_vs_geography(G):
    S = G.ws.load_table("gating/model_summary.csv")
    if not len(S):
        return None
    D = S.pivot_table(index="model", columns="scenario", values="dti").reset_index()
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 5))
    for r in D.itertuples():
        x = getattr(r, "V3", np.nan)
        for sc, mk in (("V1", "o"), ("V2", "s")):
            y = getattr(r, sc, np.nan)
            ax.scatter(x, y, marker=mk, label=f"{r.model} {sc}")
            ax.annotate(r.model, (x, y), fontsize=7)
    lim = [0, max(0.05, float(np.nanmax(D.drop(columns="model").to_numpy(float))) * 1.1)]
    ax.plot(lim, lim, "k:", lw=0.8)
    ax.set(xlabel="V3 geographic DTI (nested, pooled)", ylabel="V1 (o) / V2 (s) discovery DTI", xlim=lim, ylim=lim,
           title="Discovery vs geographic transfer (same cross-fitted protocol)")
    return fig, D


def per_zone_dti(G):
    P = G.ws.load_table("validation/per_zone_metrics.csv")
    if not len(P):
        return None
    D = P[["scenario", "model", "zone", "rep", "frac", "dti_heldout"]]
    plt = _plt()
    scs = list(dict.fromkeys(D.scenario))
    fig, axs = plt.subplots(1, len(scs), figsize=(4 * len(scs), 4), sharey=True, squeeze=False)
    for ax, sc in zip(axs[0], scs):
        d = D[D.scenario == sc]
        for m, g in d.groupby("model"):
            ax.plot(g.zone, g.dti_heldout, "o-", label=m)
        ax.set(title=sc, xlabel="held-out zone", ylabel="DTI")
    axs[0][-1].legend(fontsize=7)
    fig.suptitle("Per-zone held-out DTI (nested rep x frac)")
    return fig, D


def prediction_case_studies(G, n: int = 4, pad: int = 25):
    Y = G.load_parquet("validation/system_recovery.parquet")
    sel = (G.ws.read_json("gating/selection.json") or {}).get("selected")
    if not len(Y) or G.units is None:
        return None
    m = sel if sel not in (None, "RETAIN_W2") else next((x for x in ("D", "C_all", "B") if x in set(Y.model)), None)
    if m is None:
        return None
    a = Y[Y.model == m].set_index(["unit", "system_id"]).recovered
    b = Y[Y.model == "A_pixel"].set_index(["unit", "system_id"]).recovered
    j = pd.concat([a.rename("model"), b.rename("w2")], axis=1).dropna()
    gained = j[(j.model.astype(bool)) & ~(j.w2.astype(bool))].reset_index()
    lost = j[~(j.model.astype(bool)) & (j.w2.astype(bool))].reset_index()
    cases = pd.concat([gained.head(n - n // 2).assign(kind=f"recovered by {m}, missed by W2"),
                       lost.head(n // 2).assign(kind=f"recovered by W2, missed by {m}")], ignore_index=True)
    if not len(cases):
        return None
    plt = _plt()
    fig, axs = plt.subplots(1, len(cases), figsize=(4 * len(cases), 4), squeeze=False)
    rows = []
    for ax, r in zip(axs[0], cases.itertuples()):
        su = next(s for s in G.units if s.key == r.unit)
        gr, gc = su.unit.scorer.g_rc
        ids = su.sysmap[gr, gc]
        k = ids == int(r.system_id)
        h, w = su.unit.ev.shape
        r0, r1 = max(int(gr[k].min()) - pad, 0), min(int(gr[k].max()) + pad, h)
        c0, c1 = max(int(gc[k].min()) - pad, 0), min(int(gc[k].max()) + pad, w)
        img = np.ones((r1 - r0, c1 - c0, 3)) * 0.95
        tr = np.zeros(su.unit.ev.shape, bool)
        tr[gr[k], gc[k]] = True
        kn = su.known
        for arr, col in ((kn, (0.6, 0.6, 0.6)), (tr, (0.0, 0.0, 0.0))):
            img[arr[r0:r1, c0:c1]] = col
        for mm, col in (("A_pixel", (0.1, 0.3, 0.9)), (m, (0.9, 0.1, 0.1))):
            p = G.ws.path(f"predictions/heldout/{su.key.replace(':', '_')}_{mm}.npz")
            if p.exists():
                with np.load(p) as z:
                    q = np.unpackbits(z["q"])[:int(np.prod(z["shape"]))].reshape(z["shape"]).astype(bool)
                img[q[r0:r1, c0:c1]] = col
        ax.imshow(img, interpolation="nearest")
        ax.set(title=f"{r.unit} sys {int(r.system_id)}\n{r.kind}", xticks=[], yticks=[])
        rows.append({"unit": r.unit, "system_id": int(r.system_id), "kind": r.kind, "row0": r0 + su.unit.sl[0].start,
                     "col0": c0 + su.unit.sl[1].start, "rows": r1 - r0, "cols": c1 - c0, "model": m})
    fig.suptitle("black = held-out truth, grey = permitted known, blue = W2, red = model (overlap: red on top)")
    return fig, pd.DataFrame(rows)


FIGURES = {"candidate_coverage": candidate_coverage, "unique_recovery": unique_recovery,
           "expert_complementarity": expert_complementarity, "gate_weights_by_geology": gate_weights_by_geology,
           "discovery_vs_geography": discovery_vs_geography, "per_zone_dti": per_zone_dti,
           "prediction_case_studies": prediction_case_studies}


def write_all(G) -> list[str]:
    missing = []
    for name, fn in FIGURES.items():
        try:
            out = fn(G)
        except Exception as e:                           # a figure never decides anything
            out = None
            missing.append(f"{name}: {type(e).__name__}: {e}")
        if out is None:
            if not any(m.startswith(name) for m in missing):
                missing.append(f"{name}: inputs absent")
            continue
        fig, data = out
        G.ws.save_fig(fig, f"figures/{name}.png", data)
    G.ws.save_text("\n".join(missing) + "\n" if missing else "none\n", "figures/MISSING.txt")
    return missing
