"""Part E (F048-F059): cross-modal evidence. Part F (F060-F069): incomplete labels and mapping bias. Part G
(F070-F078): regional generalisation. Independent evidence families: magnetic, gravity, terrain, basement,
conductivity (derivatives of one survey are never counted as independent)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

from . import viz as V
from .core import (DESCRIPTIVE, INCONCLUSIVE, PALETTE, SUPPORTED, Result, axial_diff_deg, axial_mean_deg,
                   cliffs_delta, cluster_effect, effect_result)
from .data import lineaments, zscore
from .parts_ab import TEST_OUT, domain_windows, rose, sp
from .parts_cd import crests1, effects


def lin_data(E):
    """Label-free lineament inventory (1 km crests of the independent families) + label raster + family count map."""
    if "LIN" in E.D.cache:
        return E.D.cache["LIN"], E.D.cache["LINLAB"], E.D.cache["FAMCOUNT"]
    p = E.ws.path("datasets/structural_lineaments.parquet")
    if p.exists() and E.current_stage != "e07_crossmodal":
        L = pd.read_parquet(p)
        with np.load(E.ws.path("rasters/evidence_maps/lineaments.npz")) as z:
            lab, cnt = z["labels"], z["family_count"]
    else:
        cr = crests1(E)
        L, lab = lineaments(E.D, cr, 1.0)
        cnt = np.zeros(E.D.grid.shape, np.int8)
        for m in cr.values():
            cnt += ndimage.binary_dilation(m, iterations=3).astype(np.int8)
        g = {}
        for k in ("rtp", "iso_grav_anom"):
            if E.D.has(k):
                g[k] = E.D.enh.thdr(E.D.band(k), 1.0)
        for k, a in g.items():
            z, _, _ = zscore(a, E.D.valid)
            ok = np.isfinite(z)
            if len(L):
                ids = L.lineament_id.to_numpy()
                s_ = ndimage.sum(np.where(ok, z, 0.0), lab, ids)
                n_ = ndimage.sum(ok.astype(np.float64), lab, ids)
                L[f"mean_{k}_grad_z"] = np.where(n_ > 0, s_ / np.maximum(n_, 1), np.nan)
        E.save_parquet(L, "datasets/structural_lineaments.parquet")
        E.save_npz("rasters/evidence_maps/lineaments.npz", labels=lab, family_count=cnt)
    E.D.cache.update(LIN=L, LINLAB=lab, FAMCOUNT=cnt)
    return L, lab, cnt


def station_family_counts(E) -> tuple[pd.DataFrame, list[str]]:
    """Per station: distance (km) to the nearest 1 km crest of each independent family and the number of families
    with a crest within 300 m (computed from the label-free crests, independent of the mode's scale list)."""
    if "FAMST" in E.D.cache:
        return E.D.cache["FAMST"]
    S, *_ = E.stations_data()
    out = S[["sample_id", "kind", "system", "region"]].copy()
    fams = []
    for fam, m in crests1(E).items():
        if not m.any():
            continue
        d = ndimage.distance_transform_edt(~m) * E.D.grid.pixel_km
        out[fam] = d[S.r.to_numpy(), S.c.to_numpy()]
        fams.append(fam)
    out["n_families_300m"] = (out[fams] <= 0.3).sum(axis=1) if fams else 0
    E.D.cache["FAMST"] = (out, fams)
    return out, fams


# =============================================================================== PART E
def f048(E):
    D = E.D
    L, lab, cnt = lin_data(E)
    X, fams = station_family_counts(E)
    s = sp("F048", "family_coincidence", "Independent-family crest coincidence", "E", "Are faults enriched at "
           "physically corroborated structures?", "Number of independent families with a 1 km crest within 300 m: "
           "faults vs matched controls", "Independent physics agreeing is stronger evidence than one strong edge",
           "Cliff's delta of the count, cluster CI", TEST_OUT,
           {SUPPORTED: "Multi-family coincidence is a candidate feature.", INCONCLUSIVE: "Undecided.",
            "NOT_SUPPORTED": "Coincidence does not separate faults."}, "Crest = top 3 % of 1 km line strength.", 1)
    f, c = X[X.kind == "fault"], X[X.kind == "control"]
    e = cluster_effect(f.n_families_300m, f.system, c.n_families_300m, c.system, E.n_boot, E.seed)
    p = V.plt()
    fig, ax = p.subplots(1, 2, figsize=(12, 4.5))
    bins = np.arange(len(fams) + 2) - 0.5
    ax[0].hist([f.n_families_300m, c.n_families_300m], bins=bins, density=True, label=["faults", "controls"],
               color=[PALETTE["faults"], PALETTE["control"]])
    ax[0].set(xlabel="families with a crest within 300 m", ylabel="share", title=f"delta {e['effect']:+.2f}")
    ax[0].legend()
    d, r, cc = domain_windows(E, 1, 250)[0]
    V.show(ax[1], D, cnt.astype(float), "family count (dilated crests)", sl=V.win(D, r, cc, 250), cmap="viridis",
           faults=True, vmin=0, vmax=len(fams))
    E.reg.save(s, fig, X, effect_result(e, E.min_effect, None, "family count:"))


def f049(E):
    X, fams = station_family_counts(E)
    s = sp("F049", "agreement_vs_single", "Does agreement add information beyond the best single family?", "E",
           "Is multimodal agreement more discriminative than one family?", "Cliff's delta of 'crest within 300 m' "
           "per family vs of '>= 2 families'", "Agreement could add specificity", "Deltas with cluster CI",
           TEST_OUT, {SUPPORTED: "Agreement (>= 2) separates faults; compare its delta with the best single family "
                                 "in the summary.", INCONCLUSIVE: "Undecided.", "NOT_SUPPORTED": "No separation."},
           "Binary indicators lose strength information.", 2)
    rows = []
    f, c = X[X.kind == "fault"], X[X.kind == "control"]
    for m in fams:
        rows.append({"label": m, **cluster_effect((f[m] <= 0.3).astype(float), f.system, (c[m] <= 0.3).astype(float),
                                                  c.system, E.n_boot, E.seed)})
    rows.append({"label": ">= 2 families", **cluster_effect((f.n_families_300m >= 2).astype(float), f.system,
                                                            (c.n_families_300m >= 2).astype(float), c.system,
                                                            E.n_boot, E.seed)})
    R = pd.DataFrame(rows)
    fig, ax = V.grid(1, 1, 7, 3.5)
    V.effect_bars(ax[0], R, title="single family vs agreement", min_effect=E.min_effect)
    ag = R[R.label == ">= 2 families"].iloc[0].to_dict()
    res = effect_result(ag, E.min_effect, None, ">= 2 families:")
    res.summary += f"; best single family delta {R[R.label != '>= 2 families'].effect.max():+.3f}"
    E.reg.save(s, fig, R, res)


def f050(E):
    S, M, *_, ST = E.stations_data()
    X = ST.pivot_table(index="sample_id", columns="field", values="local_strike_deg").reset_index()
    X = X.merge(S[["sample_id", "kind", "system", "strike"]], on="sample_id")
    s = sp("F050", "orientation_agreement", "Magnetic / gravity fabric orientation agreement", "E", "Do aligned "
           "lineaments form meaningful structural evidence?", "Axial difference between magnetic and gravity "
           "structure-tensor strikes (2 km), faults vs controls; and vs the mapped strike", "Aligned fabrics of "
           "independent fields suggest a common structure", "Cliff's delta of -difference (positive = better "
           "aligned at faults)", TEST_OUT, {SUPPORTED: "Orientation agreement is a candidate feature.",
                                            INCONCLUSIVE: "Undecided.", "NOT_SUPPORTED": "No separation."},
           "Fabric orientation is noisy in low-gradient areas.", 2)
    if not {"magnetic", "gravity"} <= set(X.columns):
        return E.reg.skip(s, "magnetic or gravity structure metrics absent")
    X["mg_diff"] = axial_diff_deg(X.magnetic, X.gravity)
    X["mag_vs_fault"] = axial_diff_deg(X.magnetic, X.strike)
    f, c = X[X.kind == "fault"], X[X.kind == "control"]
    e = cluster_effect(-f.mg_diff, f.system, -c.mg_diff, c.system, E.n_boot, E.seed)
    p = V.plt()
    fig, ax = p.subplots(1, 2, figsize=(11, 4))
    V.ecdf(ax[0], f.mg_diff, "faults")
    V.ecdf(ax[0], c.mg_diff, "controls")
    ax[0].set(xlabel="|mag - grav| strike (deg)", ylabel="ECDF")
    ax[0].legend()
    V.ecdf(ax[1], f.mag_vs_fault, "magnetic fabric vs mapped strike")
    ax[1].set(xlabel="deg", title=f"share within 20 deg {(f.mag_vs_fault <= 20).mean():.2f}")
    E.reg.save(s, fig, X, effect_result(e, E.min_effect, None, "mag-grav alignment:"))


def f051(E):
    S, M, *_ = E.stations_data()
    a = M[M.signal == "rtp:thdr1km"].set_index("sample_id").grad_peak_offset_km
    b = M[M.signal == "iso_grav_anom:thdr1km"].set_index("sample_id").grad_peak_offset_km
    X = S[["sample_id", "kind", "system"]].assign(d=lambda x: (x.sample_id.map(a) - x.sample_id.map(b)).abs())
    s = sp("F051", "crossmodal_offset", "Spatial offset between magnetic and gravity gradient maxima", "E",
           "Do modalities identify the same trace or only nearby broad contrasts?", "|offset(mag max) - offset(grav "
           "max)| along the fault normal; faults vs controls", "Coincident maxima point to one structure; km offsets "
           "to different sources / depths", "Median |offset difference| and share within 300 m; Cliff's delta of "
           "-difference", TEST_OUT, {SUPPORTED: "Maxima coincide more at faults than at controls.",
                                     INCONCLUSIVE: "Undecided.", "NOT_SUPPORTED": "No difference."},
           "Maxima within +-5 km only.", 2)
    f, c = X[X.kind == "fault"], X[X.kind == "control"]
    e = cluster_effect(-f.d, f.system, -c.d, c.system, E.n_boot, E.seed)
    p = V.plt()
    fig, ax = p.subplots(figsize=(6, 4))
    V.ecdf(ax, f.d, "faults")
    V.ecdf(ax, c.d, "controls")
    ax.axvline(0.3, color="r", lw=0.6)
    ax.set(xlabel="|mag - grav| peak offset (km)", ylabel="ECDF", title=f"median at faults {f.d.median():.2f} km")
    ax.legend()
    E.reg.save(s, fig, X, effect_result(e, E.min_effect, None, "coincidence:"))


def lro_auc(X: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """Leave-region-out logistic AUC (fault vs control) for a feature set."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    rows = []
    for k in sorted(X.region.unique()):
        tr, te = X[X.region != k], X[X.region == k]
        if te.y.nunique() < 2 or tr.y.nunique() < 2 or len(te) < 50:
            continue
        mu, sd = tr[feats].mean(), tr[feats].std().replace(0, 1)
        m = LogisticRegression(max_iter=300, class_weight="balanced")
        m.fit(((tr[feats] - mu) / sd).fillna(0), tr.y)
        p = m.predict_proba(((te[feats] - mu) / sd).fillna(0))[:, 1]
        rows.append({"region": int(k), "auc": float(roc_auc_score(te.y, p)), "n": len(te)})
    return pd.DataFrame(rows)


def station_features(E) -> pd.DataFrame:
    S, M, *_, ST = E.stations_data()
    W = M.pivot_table(index="sample_id", columns="signal", values=["abs_step", "abs_peak"])
    W.columns = [f"{s}|{m}" for m, s in W.columns]
    T = ST.pivot_table(index="sample_id", columns="field", values=["side_corr", "along_persistence", "coherence"])
    T.columns = [f"{f}:structure|{m}" for m, f in T.columns]
    X = S[["sample_id", "kind", "system", "region"]].merge(W.reset_index(), on="sample_id").merge(
        T.reset_index(), on="sample_id", how="left")
    X["y"] = (X.kind == "fault").astype(int)
    return X[X.kind.isin(["fault", "control"])]


FAMILY_PREFIX = {"magnetic": ("rtp", "tmi_hg", "magnetic:"), "gravity": ("iso_grav_anom", "gravity:"),
                 "terrain": ("det_elev", "terrain:"), "basement": ("depth_to_base_surf",), "conductivity": ("cond_surf",)}


def fam_cols(X, fam):
    return [c for c in X.columns if any(c.startswith(p) for p in FAMILY_PREFIX[fam])]


def f052(E):
    X = station_features(E)
    s = sp("F052", "conditional_information", "Incremental information of each family (leave-region-out)", "E",
           "Does gravity add information after magnetics, and vice versa?", "Held-out-region AUC of logistic models "
           "on magnetic, gravity, terrain and combined station features", "Complementary families raise held-out "
           "AUC; redundant ones do not", "AUC per held-out region; incremental AUC (combined - single)",
           {SUPPORTED: "Combining magnetic and gravity raises held-out AUC in >= 2/3 of regions.",
            INCONCLUSIVE: "Gains in some regions only.", "NOT_SUPPORTED": "Gains in <= 1/3 of regions."},
           {SUPPORTED: "Keep both families as separate inputs.", INCONCLUSIVE: "Undecided.",
            "NOT_SUPPORTED": "No incremental information from the second family."},
           "Small linear models on station summaries; controls may contain unmapped faults.", 1)
    sets = {f: fam_cols(X, f) for f in ("magnetic", "gravity", "terrain")}
    sets = {k: v for k, v in sets.items() if v}
    sets["magnetic+gravity"] = sets.get("magnetic", []) + sets.get("gravity", [])
    sets["magnetic+gravity+terrain"] = sets["magnetic+gravity"] + sets.get("terrain", [])
    rows = []
    for name, cols in sets.items():
        if cols:
            rows.append(lro_auc(X, cols).assign(feature_set=name))
    R = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not len(R):
        return E.reg.skip(s, "no feature set")
    P = R.pivot_table(index="region", columns="feature_set", values="auc")
    inc = (P.get("magnetic+gravity") - P[[c for c in ("magnetic", "gravity") if c in P]].max(axis=1)).dropna()
    p = V.plt()
    fig, ax = p.subplots(figsize=(8, 4))
    for c in P.columns:
        ax.plot(P.index, P[c], "o-", label=c)
    ax.axhline(0.5, color="k", lw=0.5)
    ax.set(xlabel="held-out region", ylabel="AUC", title=f"incremental AUC (mag+grav - best single): mean "
                                                         f"{inc.mean():+.3f}")
    ax.legend(fontsize=6)
    share = float((inc > 0.01).mean()) if len(inc) else np.nan
    st = SUPPORTED if share >= 2 / 3 else ("NOT_SUPPORTED" if share <= 1 / 3 else INCONCLUSIVE)
    E.reg.save(s, fig, R, Result(st, f"mag+grav exceeds the best single family by > 0.01 AUC in {share:.0%} of "
                                     f"{len(inc)} held-out regions (mean {inc.mean():+.3f})", {"share": share}, len(R)))


def f053(E):
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.stats import spearmanr

    from .data import PROVENANCE

    D = E.D
    rng = np.random.default_rng(E.seed)
    rr, cc = np.nonzero(D.valid)
    i = rng.choice(len(rr), min(60_000, len(rr)), replace=False)
    names = list(D.stack.names)
    X = np.column_stack([D.band(n)[rr[i], cc[i]] for n in names])
    ok = np.isfinite(X).all(1)
    C = spearmanr(X[ok]).correlation
    order = leaves_list(linkage(1 - np.abs(C[np.triu_indices(len(names), 1)]), "average"))
    s = sp("F053", "redundancy", "Band dependency structure (Spearman, clustered)", "E", "Which apparent features "
           "are redundant?", "|rho| between all bands on a pixel sample; provenance marked", "Correlated bands of one "
           "survey are one evidence family", "Pairs with |rho| >= 0.9", {DESCRIPTIVE: "Redundancy map."},
           {DESCRIPTIVE: "Merge |rho| >= 0.9 groups into one family before counting independent evidence."},
           "Pixel-level monotone association only.", 2)
    p = V.plt()
    fig, ax = p.subplots(figsize=(9, 8))
    Co = np.abs(C)[np.ix_(order, order)]
    im = ax.imshow(Co, cmap="viridis", vmin=0, vmax=1)
    lab = [f"{names[k]} [{PROVENANCE.get(names[k], '?')[:4]}]" for k in order]
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(lab, rotation=90, fontsize=6)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(lab, fontsize=6)
    fig.colorbar(im, ax=ax, label="|Spearman rho|")
    R = pd.DataFrame(C, index=names, columns=names)
    hi = [(names[a], names[b], C[a, b]) for a in range(len(names)) for b in range(a + 1, len(names))
          if abs(C[a, b]) >= 0.9]
    E.reg.save(s, fig, R.reset_index(), Result(DESCRIPTIVE, f"{len(hi)} pairs with |rho| >= 0.9: "
                                                            f"{[(a, b) for a, b, _ in hi][:8]}", {}, int(ok.sum())))


def f054(E):
    D = E.D
    cr = crests1(E)
    s = sp("F054", "contradictions", "Where magnetic and gravity disagree", "E", "Does disagreement reflect depth, "
           "ambiguity or artifacts?", "Magnetic crests without a gravity crest within 300 m (and vice versa); rate "
           "near vs far from mapped faults", "Shallow magnetic sources (dykes, volcanics) often lack a density "
           "contrast", "Contradiction share of crest pixels, within 300 m vs > 2 km of mapped traces",
           {DESCRIPTIVE: "Disagreement statistics."}, {DESCRIPTIVE: "Contradiction-rich areas are hard-negative "
                                                                    "candidates (contacts / intrusions)."},
           "Crest thresholds per family are relative (top 3 %).", 2)
    if not {"magnetic", "gravity"} <= set(cr):
        return E.reg.skip(s, "magnetic or gravity crests absent")
    gm = ndimage.binary_dilation(cr["gravity"], iterations=3)
    mm = ndimage.binary_dilation(cr["magnetic"], iterations=3)
    mag_only, grav_only = cr["magnetic"] & ~gm, cr["gravity"] & ~mm
    near, far = D.net.distance_km <= 0.3, D.net.distance_km > 2
    rows = [{"type": t, "near_share": float(m[near & base].sum() / max((near & base).sum(), 1)),
             "far_share": float(m[far & base].sum() / max((far & base).sum(), 1))}
            for t, m, base in (("magnetic_without_gravity", mag_only, cr["magnetic"]),
                               ("gravity_without_magnetic", grav_only, cr["gravity"]))]
    p = V.plt()
    fig, ax = p.subplots(1, 2, figsize=(12, 5))
    d, r, c = domain_windows(E, 1, 250)[0]
    sl = V.win(D, r, c, 250)
    rgb = np.dstack([mag_only[sl], grav_only[sl], (cr["magnetic"] & gm)[sl]]).astype(float)
    ax[0].imshow(rgb, extent=V.extent_km(D, sl))
    V.overlay_faults(ax[0], D, sl)
    ax[0].set(title="R: magnetic only, G: gravity only, B: both")
    R = pd.DataFrame(rows)
    ax[1].bar(np.arange(2) - 0.2, R.near_share, 0.4, label="within 300 m of mapped")
    ax[1].bar(np.arange(2) + 0.2, R.far_share, 0.4, label="> 2 km from mapped")
    ax[1].set_xticks(range(2))
    ax[1].set_xticklabels(R.type, fontsize=7)
    ax[1].legend()
    E.reg.save(s, fig, R, Result(DESCRIPTIVE, "; ".join(f"{r.type}: near {r.near_share:.2f}, far {r.far_share:.2f}"
                                                       for r in R.itertuples()), {}, 2))


def f055(E):
    S, M, *_, ST = E.stations_data()
    X = ST.merge(S[["sample_id", "kind", "system"]], on="sample_id")
    X["offset_like"] = ((X.side_lag_km.abs() >= 0.2) & (X.side_corr >= 0.5)).astype(float)
    X["truncation_like"] = (X.side_corr < 0.3).astype(float)
    s = sp("F055", "structural_offsets", "Offset- and truncation-like side patterns", "E", "Do detected disruptions "
           "behave like faults?", "Share of stations with an apparent lateral offset (lag >= 0.2 km, r >= 0.5) or a "
           "truncation (r < 0.3) of the side profiles; faults vs controls", "Faults offset or truncate pre-existing "
           "patterns; contacts do not", "Cliff's delta of the indicators (cluster CI)", TEST_OUT,
           {SUPPORTED: "Offset / truncation indicators are candidate features.", INCONCLUSIVE: "Undecided.",
            "NOT_SUPPORTED": "No separation."}, "Lag search +-0.8 km; along-strike profiles of 4 km.", 1)
    rows = []
    for (fld, ind), g in X.melt(id_vars=["sample_id", "kind", "system", "field"],
                                value_vars=["offset_like", "truncation_like"]).groupby(["field", "variable"]):
        f, c = g[g.kind == "fault"], g[g.kind == "control"]
        rows.append({"label": f"{fld} {ind}", "fault_share": f.value.mean(), "control_share": c.value.mean(),
                     **cluster_effect(f.value, f.system, c.value, c.system, E.n_boot, E.seed)})
    R = pd.DataFrame(rows)
    fig, ax = V.grid(1, 1, 7, 3.5)
    V.effect_bars(ax[0], R, title="offset / truncation indicators", min_effect=E.min_effect)
    best = R.iloc[R.effect.argmax()].to_dict()
    E.reg.save(s, fig, R, effect_result(best, E.min_effect, None, f"best {best['label']}:"))


def f056(E):
    from scipy.stats import spearmanr

    D = E.D
    L, lab, cnt = lin_data(E)
    sk = lab > 0
    nb = ndimage.convolve(sk.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant") - sk
    lj = sk & (nb >= 3)
    mj = D.net.skeleton & ((ndimage.convolve(D.net.skeleton.astype(np.uint8), np.ones((3, 3), np.uint8),
                                             mode="constant") - D.net.skeleton) >= 3)
    bp = 100
    H, W = D.grid.shape
    rows = []
    for i in range(0, H, bp):
        for j in range(0, W, bp):
            v = D.valid[i:i + bp, j:j + bp]
            if v.mean() < 0.5:
                continue
            rows.append({"bi": i // bp, "bj": j // bp, "lineament_junctions": int(lj[i:i + bp, j:j + bp].sum()),
                         "mapped_junctions": int(mj[i:i + bp, j:j + bp].sum())})
    R = pd.DataFrame(rows)
    rho = spearmanr(R.lineament_junctions, R.mapped_junctions).correlation if len(R) > 5 else np.nan
    s = sp("F056", "intersections", "Lineament intersections vs mapped-fault intersections (10 km blocks)", "E",
           "Does topology supply evidence beyond gradient amplitude?", "Junction counts of the label-free lineament "
           "skeleton vs mapped junctions per block", "Fault networks intersect; contacts often do not", "Spearman rho "
           "across blocks", {DESCRIPTIVE: "Topology association."},
           {DESCRIPTIVE: "A positive association motivates network / intersection features."},
           "Pixel junctions of a thresholded skeleton are noisy.", 2)
    p = V.plt()
    fig, ax = p.subplots(figsize=(5.5, 4.5))
    ax.hexbin(R.lineament_junctions, R.mapped_junctions, gridsize=25, bins="log", cmap="Greys")
    ax.set(xlabel="label-free lineament junctions", ylabel="mapped junctions", title=f"rho {rho:.2f}")
    E.reg.save(s, fig, R, Result(DESCRIPTIVE, f"block Spearman rho {rho:.2f} over {len(R)} blocks", {"rho": rho},
                                 len(R)))


def f057(E):
    D = E.D
    L, lab, cnt = lin_data(E)
    U = L[L.status == "unmapped"]
    pick = pd.concat([U[U.n_families >= 2].head(3), U[U.n_families == 1].head(3)])
    s = sp("F057", "candidate_attribution", "Which independent families support individual candidates?", "E",
           "Why was a candidate generated, and what alternatives remain?", "Unmapped lineaments with >= 2 and with 1 "
           "supporting families", "Single-family support leaves a contact / intrusion alternative open",
           "Families with a crest on >= 50 % of the lineament", {DESCRIPTIVE: "Attribution examples."},
           {DESCRIPTIVE: "Candidates are reported with their families and alternatives (part I atlas)."},
           "Examples only.", 3)
    fig, ax = V.grid(max(len(pick), 1), 3, 4, 3.6)
    for a_, r in zip(ax, pick.itertuples()):
        sl = V.win(D, r.r_mid, r.c_mid, 60)
        a_.imshow((lab[sl] == r.lineament_id).astype(float), extent=V.extent_km(D, sl), cmap="Reds", alpha=0.9)
        a_.imshow(np.where(cnt[sl] > 0, cnt[sl], np.nan), extent=V.extent_km(D, sl), cmap="viridis", alpha=0.4)
        a_.set(title=f"L{r.lineament_id}: {r.families or 'none'} ({r.length_km:.1f} km)")
        a_.set_aspect("equal")
    E.reg.save(s, fig, pick, Result(DESCRIPTIVE, f"{len(U)} unmapped lineaments; {int((U.n_families >= 2).sum())} "
                                                 "with >= 2 families", {}, len(pick)))


def f058(E):
    L, lab, cnt = lin_data(E)
    s = sp("F058", "evidence_vs_mapping", "Physical evidence on mapped, near-mapped and unmapped lineaments", "E",
           "Do mapped structures represent only part of the available physical evidence?", "Number of supporting "
           "families and gradient strength by mapping status", "If unmapped lineaments carry evidence comparable to "
           "mapped ones, the label set is incomplete in a way models can exploit", "Cliff's delta mapped vs unmapped, "
           "cluster bootstrap over regions", TEST_OUT,
           {SUPPORTED: "Mapped lineaments carry more evidence than unmapped ones (direction in summary).",
            "NOT_SUPPORTED": "Comparable evidence on mapped and unmapped lineaments: strong label incompleteness "
                             "signal.", INCONCLUSIVE: "Undecided."}, "Status by distance to mapped traces only.", 1)
    a, b = L[L.status == "mapped"], L[L.status == "unmapped"]
    e = cluster_effect(a.n_families, a.region, b.n_families, b.region, E.n_boot, E.seed)
    p = V.plt()
    fig, ax = p.subplots(1, 2, figsize=(11, 4))
    for st, g in L.groupby("status"):
        V.ecdf(ax[0], g.n_families, st)
        if "mean_rtp_grad_z" in g:
            V.ecdf(ax[1], g.mean_rtp_grad_z, st)
    ax[0].set(xlabel="supporting families", ylabel="ECDF")
    ax[1].set(xlabel="mean magnetic 1 km gradient (z)")
    ax[0].legend()
    E.reg.save(s, fig, L.groupby("status")[["n_families", "length_km"]].describe().reset_index(),
               effect_result(e, E.min_effect, None, "mapped vs unmapped n_families:", "regions",
                             ("mapped", "unmapped lineaments")))


CROSSMODAL_TESTS = ("F048", "F049", "F050", "F051", "F052", "F055", "F058")
CROSSMODAL_COLUMNS = ("figure", "title", "status", "effect", "ci_lo", "ci_hi", "summary")


def crossmodal_table(E) -> pd.DataFrame:
    """Stage e07 product (written after the figures, independent of F059 rendering): one row per cross-modal test,
    NOT_RUN when its interpretation record does not exist (a failed figure is visible, never silently dropped)."""
    rows = []
    for fid in CROSSMODAL_TESTS:
        j = E.ws.read_json(f"figures/interpretations/{fid}.json") or {}
        m = j.get("measured", {})
        rows.append({"figure": fid, "title": j.get("title"), "status": m.get("status", "NOT_RUN"),
                     "effect": m.get("effect"), "ci_lo": m.get("ci_lo"), "ci_hi": m.get("ci_hi"),
                     "summary": j.get("interpretation", "")[:200]})
    R = pd.DataFrame(rows, columns=list(CROSSMODAL_COLUMNS))
    E.ws.save_table(R, "tables/crossmodal_complementarity.csv")
    return R


def f059(E):
    R = crossmodal_table(E)
    s = sp("F059", "crossmodal_synthesis", "Cross-modal synthesis", "E", "Which complementary families are most "
           "promising?", "Results of F048-F058 side by side", "Synthesis without forcing a weighted fusion score",
           "Status and effect per test", {DESCRIPTIVE: "Synthesis table."}, {DESCRIPTIVE: "Feeds the hypothesis "
                                                                                          "register."}, "Inherits the "
                                                                                                        "limitations "
                                                                                                        "of each test.",
           1)
    p = V.plt()
    fig, ax = p.subplots(figsize=(11, 0.5 * len(R) + 1.2))
    ax.axis("off")
    tb = ax.table(cellText=R[["figure", "title", "status", "effect"]].astype(str).values, colLabels=["figure", "title",
                                                                                                   "status", "effect"],
                  loc="center")
    tb.auto_set_font_size(False)
    tb.set_fontsize(7)
    E.reg.save(s, fig, R, Result(DESCRIPTIVE, f"statuses {R.status.value_counts().to_dict()}", {}, len(R)))


PART_E = [f048, f049, f050, f051, f052, f053, f054, f055, f056, f057, f058, f059]


# =============================================================================== PART F
def f060(E):
    from scipy.stats import spearmanr

    from gems_eda.morphology import trace_density

    D = E.D
    dom = E.domains()
    L, lab, cnt = lin_data(E)
    mdens = trace_density(D.net, D.grid, 10.0)
    ldens = ndimage.gaussian_filter((lab > 0).astype(np.float32) * D.grid.pixel_km, 100) / (0.01) * 100
    rows = []
    for d in np.unique(dom[dom >= 0]):
        m = dom == d
        rows.append({"domain": int(d), "mapped_density": float(np.nanmean(mdens[m])),
                     "lineament_density": float(np.nanmean(ldens[m])), "area_px": int(m.sum())})
    R = pd.DataFrame(rows)
    rho = spearmanr(R.mapped_density, R.lineament_density).correlation if len(R) > 3 else np.nan
    s = sp("F060", "density_vs_context", "Mapped density vs label-free lineament density by domain", "F", "Does "
           "apparent mapping density depend on environment?", "Per domain: mapped trace density vs label-free "
           "lineament density", "Domains with many label-free lineaments but few mapped faults are candidate "
           "under-mapped areas", "Ratio and Spearman rho across domains", {DESCRIPTIVE: "Density contrast."},
           {DESCRIPTIVE: "Low mapped / lineament ratio domains: prioritise review there; expect PU bias."},
           "Lineaments include non-faults; densities smoothed at 10 km.", 1)
    p = V.plt()
    fig, ax = p.subplots(figsize=(6, 4.5))
    ax.scatter(R.lineament_density, R.mapped_density, s=R.area_px / R.area_px.max() * 300)
    for r in R.itertuples():
        ax.annotate(f"d{r.domain}", (r.lineament_density, r.mapped_density))
    ax.set(xlabel="label-free lineament density", ylabel="mapped density (km / 100 km2)", title=f"rho {rho:.2f}")
    R["mapped_to_lineament"] = R.mapped_density / R.lineament_density.replace(0, np.nan)
    E.reg.save(s, fig, R, Result(DESCRIPTIVE, f"mapped / lineament ratio range {R.mapped_to_lineament.min():.2f}-"
                                              f"{R.mapped_to_lineament.max():.2f}; rho {rho:.2f}", {}, len(R)))


LABEL_BIAS_COLUMNS = ("label", "effect", "ci_lo", "ci_hi", "n_groups", "n_fault", "n_control")


def label_bias_table(E) -> pd.DataFrame:
    """Stage e08 product: mapped vs unmapped label-free lineaments per attribute (Cliff's delta, cluster bootstrap over
    regions; n_fault / n_control = mapped / unmapped lineaments)."""
    if "LABELBIAS" in E.D.cache:
        return E.D.cache["LABELBIAS"]
    L, *_ = lin_data(E)
    a, b = L[L.status == "mapped"], L[L.status == "unmapped"]
    cols = [c for c in ("length_km", "linearity", "n_families", "mean_rtp_grad_z", "mean_iso_grav_anom_grad_z")
            if c in L]
    rows = [{"label": c, **cluster_effect(a[c], a.region, b[c], b.region, E.n_boot, E.seed)} for c in cols]
    R = pd.DataFrame(rows).reindex(columns=list(LABEL_BIAS_COLUMNS))
    E.ws.save_table(R, "tables/label_selection_bias.csv")
    E.D.cache["LABELBIAS"] = R
    return R


def f061(E):
    s = sp("F061", "mapped_vs_unmapped_lineaments", "Mapped vs unmapped lineament characteristics", "F", "Do "
           "unmapped proposals occupy a different evidence distribution?", "Length, linearity, families, gradient "
           "strength: mapped vs unmapped lineaments", "A different distribution tells what kind of structures the "
           "map leaves out", "Cliff's delta (cluster over regions)", TEST_OUT,
           {SUPPORTED: "Unmapped lineaments differ (see the largest attribute): the label set is selective.",
            "NOT_SUPPORTED": "Indistinguishable: unmapped lineaments look like mapped ones.",
            INCONCLUSIVE: "Undecided."}, "Unmapped proposals are not confirmed faults.", 1)
    R = label_bias_table(E)
    if not len(R) or R.effect.notna().sum() == 0:
        return E.reg.skip(s, "no mapped / unmapped lineament attribute measurable")
    fig, ax = V.grid(1, 1, 7, 3)
    V.effect_bars(ax[0], R, title="mapped (positive) vs unmapped", min_effect=E.min_effect)
    best = R.loc[R.effect.abs().idxmax()].to_dict()
    E.reg.save(s, fig, R, effect_result(best, E.min_effect, None, f"largest: {best['label']}:", "regions",
                                        ("mapped", "unmapped lineaments")))


def f062(E):
    L, *_ = lin_data(E)
    D = E.D
    edges = [0, 0.3, 1, 2, 5, 10, 1e9]
    dist = D.net.distance_km
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        area = float(((dist >= lo) & (dist < hi) & D.valid).sum() * 0.01)
        n = int(((L.dist_known_median_km >= lo) & (L.dist_known_median_km < hi)).sum())
        rows.append({"band_km": f"{lo}-{hi if hi < 1e9 else 'inf'}", "lineaments": n, "area_km2": area,
                     "per_1000km2": 1000 * n / area if area else np.nan})
    R = pd.DataFrame(rows)
    s = sp("F062", "distance_to_mapped", "Lineament abundance vs distance to mapped faults", "F", "Are lineaments "
           "(and later model predictions) concentrated near known networks?", "Label-free lineaments per 1000 km2 by "
           "distance band to mapped traces", "Physical structure density should not collapse away from mapped "
           "networks if the map is complete", "Area-normalised abundance", {DESCRIPTIVE: "Abundance by distance."},
           {DESCRIPTIVE: "Abundant far lineaments = unexplored structural candidates (see F086 for W2)."},
           "Lineament status uses the median distance.", 2)
    p = V.plt()
    fig, ax = p.subplots(figsize=(6, 4))
    ax.bar(R.band_km, R.per_1000km2, color="0.4")
    ax.set(xlabel="distance to mapped trace (km)", ylabel="lineaments / 1000 km2")
    E.reg.save(s, fig, R, Result(DESCRIPTIVE, "; ".join(f"{r.band_km}: {r.per_1000km2:.2f}" for r in R.itertuples()),
                                 {}, len(L)))


def f063(E):
    L, *_ = lin_data(E)
    near, far = L[L.dist_known_min_km < 1], L[L.dist_known_min_km > 5]
    s = sp("F063", "novelty_vs_evidence", "Candidate novelty vs independent evidence", "F", "Do plausible structures "
           "exist away from the existing maps?", "Supporting families: lineaments > 5 km vs < 1 km from mapped "
           "faults", "Far lineaments with near-like evidence are discovery targets", "Cliff's delta (near minus far), "
           "cluster over regions", TEST_OUT,
           {SUPPORTED: "Evidence differs with novelty (direction in summary).",
            "NOT_SUPPORTED": "Far lineaments carry evidence similar to near ones: plausible unexplored structures.",
            INCONCLUSIVE: "Undecided."}, "Families only (no fault verification).", 1)
    e = cluster_effect(near.n_families, near.region, far.n_families, far.region, E.n_boot, E.seed)
    p = V.plt()
    fig, ax = p.subplots(figsize=(6, 4))
    ax.hexbin(np.log10(L.dist_known_min_km + 0.05), L.n_families, gridsize=25, bins="log", cmap="Greys")
    ax.set(xlabel="log10 distance to mapped (km)", ylabel="supporting families")
    E.reg.save(s, fig, L[["lineament_id", "dist_known_min_km", "n_families", "length_km", "region"]],
               effect_result(e, E.min_effect, None, "near vs far:", "regions", ("near", "far lineaments")))


def f064(E):
    from scipy.stats import ks_2samp

    L, *_ = lin_data(E)
    S = E.D.net.segments
    U = L[L.status == "unmapped"]
    ks = ks_2samp(S.length_km, U.length_km) if len(U) else None
    s = sp("F064", "length_bias", "Mapped fault lengths vs unmapped lineament lengths", "F", "May short structures be "
           "under-represented?", "ECDF of mapped branch lengths vs unmapped lineament lengths", "A mapping practice "
           "that skips short faults leaves them for the hidden test", "KS statistic and medians",
           {DESCRIPTIVE: "Length comparison."}, {DESCRIPTIVE: "If unmapped lineaments are shorter: generators and "
                                                              "metrics must keep short structures."},
           "Lineament length is limited by the 1 km crest threshold.", 2)
    p = V.plt()
    fig, ax = p.subplots(figsize=(6, 4))
    V.ecdf(ax, S.length_km, "mapped branches")
    V.ecdf(ax, U.length_km, "unmapped lineaments")
    ax.set_xscale("log")
    ax.set(xlabel="length (km)", ylabel="ECDF")
    ax.legend()
    E.reg.save(s, fig, pd.DataFrame({"mapped_median": [S.length_km.median()], "unmapped_median": [U.length_km.median()],
                                     "ks": [ks.statistic if ks else np.nan]}),
               Result(DESCRIPTIVE, f"medians mapped {S.length_km.median():.2f} vs unmapped {U.length_km.median():.2f} "
                                   f"km; KS {ks.statistic if ks else np.nan:.3f}", {}, len(U)))


def f065(E):
    L, *_ = lin_data(E)
    S = E.D.net.segments
    p = V.plt()
    fig, ax = p.subplots(2, 5, subplot_kw={"projection": "polar"}, figsize=(16, 6.5))
    ax = ax.ravel()
    rows = []
    U = L[(L.status == "unmapped") & L.azimuth.notna()]
    for k in range(9):
        a, b = S[S.region == k], U[U.region == k]
        if len(a) < 10 or len(b) < 10:
            ax[k].set_axis_off()
            continue
        rose(ax[k], a.azimuth, a.length_km, PALETTE["faults"], "mapped")
        rose(ax[k], b.azimuth, b.length_km, PALETTE["w2"], "unmapped")
        ma, _ = axial_mean_deg(a.azimuth, a.length_km)
        mb, _ = axial_mean_deg(b.azimuth, b.length_km)
        ax[k].set_title(f"region {k}: diff {axial_diff_deg(ma, mb):.0f} deg", fontsize=7)
        rows.append({"region": k, "mapped_mean_az": ma, "unmapped_mean_az": mb, "diff_deg": float(axial_diff_deg(ma,
                                                                                                                 mb))})
    ax[9].set_axis_off()
    R = pd.DataFrame(rows)
    s = sp("F065", "orientation_bias", "Mapped vs unmapped orientations by region", "F", "Are some directions more "
           "commonly labelled?", "Length-weighted roses, mapped (black) vs unmapped lineaments (gold)", "Mapping or "
           "model priors can favour the dominant strike", "Axial mean difference per region",
           {DESCRIPTIVE: "Orientation comparison."}, {DESCRIPTIVE: "Large differences: avoid strike priors learned "
                                                                   "from labels."},
           "Lineament orientations include survey artifacts.", 2)
    E.reg.save(s, fig, R, Result(DESCRIPTIVE, f"median regional difference {R.diff_deg.median():.0f} deg" if len(R)
                                 else "insufficient", {}, len(R)))


def f066(E):
    D = E.D
    L, lab, _ = lin_data(E)
    from gems_eda.morphology import regional_summary

    R = regional_summary(D.net, D.grid, D.valid, (3, 3))
    R["lineament_km"] = [float(L[L.region == k].length_km.sum()) for k in R.region]
    R["mapped_to_lineament_km"] = R.trace_km / R.lineament_km.replace(0, np.nan)
    s = sp("F066", "regional_label_heterogeneity", "Mapping density vs physical evidence by region", "F", "Where may "
           "regional label-selection effects occur?", "Mapped km vs label-free lineament km per region",
           "Uneven ratios point to uneven mapping (or uneven geology)", "Ratio per region",
           {DESCRIPTIVE: "Regional ratios."}, {DESCRIPTIVE: "Low-ratio regions: expect more hidden faults; weight "
                                                            "evaluation accordingly."}, "Not causal.", 2)
    p = V.plt()
    fig, ax = p.subplots(figsize=(7, 4))
    ax.bar(R.region_name, R.mapped_to_lineament_km, color="0.4")
    ax.set(ylabel="mapped km / lineament km")
    E.reg.save(s, fig, R, Result(DESCRIPTIVE, f"ratio range {R.mapped_to_lineament_km.min():.2f}-"
                                              f"{R.mapped_to_lineament_km.max():.2f}", {}, len(R)))


def f067(E):
    S, M, *_ = E.stations_data()
    X, fams = station_family_counts(E)
    strong = set(X[(X.kind == "control") & (X.n_families_300m >= 2)].sample_id)
    s = sp("F067", "pu_sensitivity", "Positive-unlabelled sensitivity of the fault / control separation", "F",
           "How strongly do conclusions depend on treating unlabelled locations as background?", "|step| deltas with "
           "all controls vs excluding controls within 300 m of >= 2-family lineaments", "If many controls are "
           "unmapped faults, excluding plausible structures raises the measured separation", "Change in Cliff's "
           "delta per signal", {DESCRIPTIVE: "Sensitivity of effects."},
           {DESCRIPTIVE: "Large increases: PU contamination is material; use PU-aware weighting in any supervised "
                         "step."}, "Exclusion uses label-free evidence only (no labels).", 1)
    rows = []
    for sig, g in M.groupby("signal"):
        F = g[g.kind == "fault"]
        C = g[g.kind == "control"]
        C2 = C[~C.sample_id.isin(strong)]
        rows.append({"signal": sig, "delta_all": cliffs_delta(F.abs_step, C.abs_step),
                     "delta_excl_plausible": cliffs_delta(F.abs_step, C2.abs_step), "n_excluded": int(len(C) -
                                                                                                    len(C2))})
    R = pd.DataFrame(rows)
    R["change"] = R.delta_excl_plausible - R.delta_all
    p = V.plt()
    fig, ax = p.subplots(figsize=(7, 0.3 * len(R) + 1.5))
    y = np.arange(len(R))
    ax.plot(R.delta_all, y, "o", label="all controls")
    ax.plot(R.delta_excl_plausible, y, "s", label="plausible structures excluded")
    ax.set_yticks(y)
    ax.set_yticklabels(R.signal, fontsize=6)
    ax.legend()
    ax.set(xlabel="Cliff's delta (|step|)")
    E.reg.save(s, fig, R, Result(DESCRIPTIVE, f"median change {R.change.median():+.3f} (max {R.change.max():+.3f}); "
                                              f"{len(strong)} controls excluded", {}, len(R)))


def f068(E):
    D = E.D
    L, lab, _ = lin_data(E)
    U = L[(L.status == "unmapped") & (L.n_families == 1)]
    mag_only = U[U.families == "magnetic"].head(3)
    ter_only = U[U.families == "terrain"].head(3)
    pick = pd.concat([mag_only.assign(alt="magnetic-only: dyke / intrusion / contact possible"),
                      ter_only.assign(alt="terrain-only: drainage / range-front / erosional edge possible")])
    s = sp("F068", "suspected_nonfault", "Suspected non-fault lineaments (single-family)", "F", "What hard-negative "
           "categories may future supervised models need?", "Unmapped single-family lineaments with their plausible "
           "alternative explanation", "Single-family edges are where contacts and geomorphic edges hide",
           "Examples by rule (not truth)", {DESCRIPTIVE: "Examples."},
           {DESCRIPTIVE: "Review these in the atlas; confirmed ones become hard negatives."}, "Rule-based examples.",
           3)
    fig, ax = V.grid(max(len(pick), 1), 3, 4, 3.6)
    for a_, r in zip(ax, pick.itertuples()):
        b = "rtp" if "magnetic" in r.families else "det_elev"
        sl = V.win(D, r.r_mid, r.c_mid, 60)
        V.show(a_, D, D.band(b), f"L{r.lineament_id} {b}", sl=sl, band=b, colorbar=False)
        rr, cc = np.nonzero(lab[sl] == r.lineament_id)
        x, y = D.grid.rc_to_xy(rr + sl[0].start, cc + sl[1].start)
        a_.plot(np.asarray(x) / 1e3, np.asarray(y) / 1e3, ",", color="yellow")
    E.reg.save(s, fig, pick, Result(DESCRIPTIVE, f"{len(U)} single-family unmapped lineaments", {}, len(pick)))


def f069(E):
    rows = []
    for fid in ("F060", "F061", "F062", "F063", "F064", "F065", "F066", "F067"):
        j = E.ws.read_json(f"figures/interpretations/{fid}.json") or {}
        rows.append({"figure": fid, "title": j.get("title"), "status": (j.get("measured") or {}).get("status"),
                     "measured": j.get("interpretation", "")[:240]})
    R = pd.DataFrame(rows)
    s = sp("F069", "label_bias_synthesis", "Label-bias synthesis", "F", "What is the evidence for selection bias?",
           "F060-F067 side by side", "Synthesis", "Statuses and measured summaries", {DESCRIPTIVE: "Synthesis."},
           {DESCRIPTIVE: "Feeds MODELING_DECISION question 8."}, "Inherits each test's limitations.", 1)
    p = V.plt()
    fig, ax = p.subplots(figsize=(12, 0.55 * len(R) + 1))
    ax.axis("off")
    tb = ax.table(cellText=R[["figure", "title", "status"]].astype(str).values, colLabels=["figure", "title", "status"],
                  loc="center")
    tb.auto_set_font_size(False)
    tb.set_fontsize(7)
    E.reg.save(s, fig, R, Result(DESCRIPTIVE, f"{len(R)} label-bias analyses", {}, len(R)))


PART_F = [f060, f061, f062, f063, f064, f065, f066, f067, f068, f069]


# =============================================================================== PART G
def regional_effects(E) -> pd.DataFrame:
    T = effects(E)
    rows = []
    for r in T.itertuples():
        for tok in str(r.regional or "").split(";"):
            if ":" in tok:
                k, v = tok.split(":")
                rows.append({"signal": r.signal, "metric": r.metric, "region": int(k), "effect": float(v),
                             "overall": r.effect})
    return pd.DataFrame(rows)


def f070(E):
    D = E.D
    dom = E.domains()
    S = D.net.segments.copy()
    S["domain"] = dom[S.r_mid, S.c_mid]
    G = S.groupby("domain").agg(n=("seg_id", "size"), km=("length_km", "sum"), med_len=("length_km", "median"),
                                med_sinuosity=("sinuosity", "median")).reset_index()
    s = sp("F070", "domains_vs_faults", "Label-free domains vs mapped-fault morphology", "G", "Do fault "
           "morphologies differ between physical domains?", "Domain map with mapped traces; per-domain geometry",
           "Context for regional generalisation", "Counts, km, median length, sinuosity", {DESCRIPTIVE: "Overlay."},
           {DESCRIPTIVE: "Different morphologies per domain: report validation per domain."}, "Domains statistical.",
           2)
    p = V.plt()
    fig, ax = p.subplots(1, 2, figsize=(13, 5.5))
    V.show(ax[0], D, np.where(dom >= 0, dom, np.nan).astype(float), "domains + mapped faults", cmap="tab10", step=2,
           faults=True, vmin=-0.5, vmax=9.5)
    ax[1].axis("off")
    tb = ax[1].table(cellText=G.round(2).astype(str).values, colLabels=G.columns, loc="center")
    tb.auto_set_font_size(False)
    tb.set_fontsize(7)
    E.reg.save(s, fig, G, Result(DESCRIPTIVE, f"{len(G)} domains with faults", {}, len(S)))


def f071(E):
    from .parts_cd import agg

    S, *_ = E.stations_data()
    sig = "rtp:thdr1km" if "rtp:thdr1km" in E.stations_data()[4] else E.stations_data()[4][0]
    fig, ax = V.grid(9, 3, 3.6, 2.8)
    rows = []
    for k in range(9):
        m = (S.region == k).to_numpy()
        f, c = m & (S.kind == "fault").to_numpy(), m & (S.kind == "control").to_numpy()
        if f.sum() < 30:
            ax[k].set_axis_off()
            continue
        for mask, lab, col in ((f, "faults", PALETTE["faults"]), (c, "controls", PALETTE["control"])):
            A = agg(E, sig, mask)
            ax[k].plot(A.offset_km, A.q50, color=col, label=lab)
            rows.append(A.assign(region=k, population=lab))
        ax[k].set_title(f"region {k} (n={int(f.sum())})", fontsize=7)
    s = sp("F071", "regional_profiles", f"Regional fault-centred profiles ({sig})", "G", "Do fault signatures look "
           "alike across regions?", "Median aligned profiles per 3 x 3 region", "Signatures that change shape between "
           "regions will not generalise", "Medians per region", {DESCRIPTIVE: "Regional profiles."},
           {DESCRIPTIVE: "Shape changes: domain-aware features or validation."}, "Regions with < 30 stations "
                                                                                  "omitted.", 2)
    E.reg.save(s, fig, pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(),
               Result(DESCRIPTIVE, f"{len(set(r.region.iloc[0] for r in rows)) if rows else 0} regions", {},
                      len(rows)))


def f072(E):
    R = regional_effects(E)
    s = sp("F072", "regional_separability", "Fault vs control separation by region", "G", "Where does each signal "
           "separate faults?", "Cliff's delta of |step| per region and signal", "Regional effects reveal where "
           "evidence works", "Heatmap", {DESCRIPTIVE: "Regional effects."}, {DESCRIPTIVE: "See F078 classification."},
           "No CI per cell (see F077 for tested regional transfer).", 2)
    X = R[R.metric == "abs_step"].pivot_table(index="signal", columns="region", values="effect")
    if not len(X):
        return E.reg.skip(s, "no regional effects")
    p = V.plt()
    fig, ax = p.subplots(figsize=(8, 0.35 * len(X) + 2))
    im = ax.imshow(X.to_numpy(), cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
    ax.set_yticks(range(len(X)))
    ax.set_yticklabels(X.index, fontsize=6)
    ax.set_xticks(range(X.shape[1]))
    ax.set_xticklabels(X.columns)
    fig.colorbar(im, ax=ax)
    E.reg.save(s, fig, X.reset_index(), Result(DESCRIPTIVE, f"{X.shape[0]} signals x {X.shape[1]} regions", {},
                                               X.size))


def f073(E):
    from scipy.stats import wasserstein_distance

    D = E.D
    regmap = D.cache.get("region_map")
    if regmap is None:
        regmap = D.region(*np.indices(D.grid.shape)).astype(np.int8)
        D.cache["region_map"] = regmap
    rng = np.random.default_rng(E.seed)
    rows = []
    for n in D.stack.names:
        z, _, _ = zscore(D.band(n), D.valid)
        allv = z[D.valid & np.isfinite(z)]
        allv = rng.choice(allv, min(50_000, len(allv)), replace=False)
        for k in range(9):
            v = z[D.valid & (regmap == k) & np.isfinite(z)]
            if len(v) < 1000:
                continue
            rows.append({"band": n, "region": k, "w1": float(wasserstein_distance(rng.choice(v, min(20_000, len(v)),
                                                                                             replace=False), allv))})
    R = pd.DataFrame(rows)
    X = R.pivot_table(index="band", columns="region", values="w1")
    s = sp("F073", "distribution_shift", "Region-to-footprint distribution shift (Wasserstein, robust z)", "G",
           "How much do feature distributions shift between regions?", "W1 distance of each region's band "
           "distribution to the footprint distribution", "Large shifts break global thresholds", "W1 in robust-z "
           "units", {DESCRIPTIVE: "Shift matrix."}, {DESCRIPTIVE: "High-shift bands: regional normalisation."},
           "Marginal distributions only.", 2)
    p = V.plt()
    fig, ax = p.subplots(figsize=(8, 7))
    im = ax.imshow(X.to_numpy(), cmap="magma", aspect="auto")
    ax.set_yticks(range(len(X)))
    ax.set_yticklabels(X.index, fontsize=6)
    ax.set_xticks(range(X.shape[1]))
    ax.set_xticklabels(X.columns)
    fig.colorbar(im, ax=ax, label="W1 (robust z)")
    E.reg.save(s, fig, R, Result(DESCRIPTIVE, f"largest shift {R.loc[R.w1.idxmax(), 'band']} region "
                                              f"{int(R.loc[R.w1.idxmax(), 'region'])} W1 {R.w1.max():.2f}", {}, len(R)))


def f074(E):
    R = E.table("tables/scale_regional.csv")
    s = sp("F074", "regional_scales", "Regional scale preferences", "G", "Do detection / localisation scales change "
           "between regions?", "Best detect and best localisation scale per region and modality", "Region-specific "
           "scales argue against one global filter bank", "argmax per region", {DESCRIPTIVE: "Scale preferences."},
           {DESCRIPTIVE: "Stable scales -> one filter bank; unstable -> multi-scale inputs."}, "Point estimates.", 2)
    if not len(R):
        return E.reg.skip(s, "no regional scale table")
    rows = []
    for (m, k), g in R.groupby(["modality", "region"]):
        rows.append({"modality": m, "region": int(k), "best_detect_km": float(g.loc[g.detect_delta.idxmax(),
                                                                                    "scale_km"]),
                     "best_localise_km": float(g.loc[g.loc_within_300m.idxmax(), "scale_km"])})
    X = pd.DataFrame(rows)
    p = V.plt()
    fig, ax = p.subplots(1, 2, figsize=(12, 4))
    for m, g in X.groupby("modality"):
        ax[0].plot(g.region, g.best_detect_km, "o-", label=m, color=PALETTE.get(m))
        ax[1].plot(g.region, g.best_localise_km, "o-", label=m, color=PALETTE.get(m))
    for a_, t in zip(ax, ("best detection scale", "best localisation scale")):
        a_.set_yscale("log")
        a_.set(xlabel="region", ylabel="km", title=t)
    ax[0].legend()
    E.reg.save(s, fig, X, Result(DESCRIPTIVE, "; ".join(f"{m}: detect scales {sorted(set(g.best_detect_km))}"
                                                       for m, g in X.groupby("modality")), {}, len(X)))


def f075(E):
    from gems_eda.morphology import endpoint_gaps, regional_summary

    D = E.D
    R = regional_summary(D.net, D.grid, D.valid, (3, 3))
    G = endpoint_gaps(D.net, D.grid, 5.0, 25.0)
    if len(G):
        G["region"] = D.region(G.r_a.to_numpy(), G.c_a.to_numpy())
        R["aligned_gaps"] = R.region.map(G.region.value_counts()).fillna(0)
    s = sp("F075", "geometry_by_region", "Fault geometry by region", "G", "Do length, orientation, fragmentation and "
           "curvature vary by region?", "Regional summary of mapped geometry", "Geometric priors learned in one "
           "region may mislead in another", "Medians, strike R, aligned gaps", {DESCRIPTIVE: "Regional geometry."},
           {DESCRIPTIVE: "Varying geometry: region-aware validation of geometric post-processing."},
           "Grid regions.", 2)
    fig, ax = V.grid(4, 4, 3.6, 3.0)
    for a_, c in zip(ax, ("median_length_km", "strike_R", "median_curv_deg_km", "density_km_per_100km2")):
        a_.bar(R.region_name, R[c], color="0.4")
        a_.set(title=c)
        a_.tick_params(axis="x", rotation=45)
    E.reg.save(s, fig, R, Result(DESCRIPTIVE, f"median length range {R.median_length_km.min():.2f}-"
                                              f"{R.median_length_km.max():.2f} km", {}, len(R)))


def f076(E):
    from scipy.stats import spearmanr

    R = regional_effects(E)
    X = R[R.metric == "abs_step"].pivot_table(index="signal", columns="region", values="effect")
    rho = []
    for i in X.columns:
        for j in X.columns:
            if i < j:
                ok = X[i].notna() & X[j].notna()
                if ok.sum() >= 4:
                    rho.append({"a": i, "b": j, "rho": spearmanr(X[i][ok], X[j][ok]).correlation})
    P = pd.DataFrame(rho)
    s = sp("F076", "importance_stability", "Stability of univariate signal ranking across regions", "G", "Is "
           "apparent feature importance spatially stable?", "Spearman rho between regions of the per-signal |step| "
           "effects", "Unstable rankings mean importance is region-specific", "Pairwise rho distribution",
           {DESCRIPTIVE: "Ranking stability."}, {DESCRIPTIVE: "Low rho: do not trust global feature rankings."},
           "Few signals per region.", 2)
    p = V.plt()
    fig, ax = p.subplots(figsize=(6, 4))
    if len(P):
        ax.hist(P.rho, bins=15, color="0.4")
    ax.set(xlabel="between-region Spearman rho", ylabel="pairs")
    E.reg.save(s, fig, P, Result(DESCRIPTIVE, f"median between-region rho {P.rho.median():.2f}" if len(P) else
                                 "insufficient", {}, len(P)))


LRO_COLUMNS = ("held_out_region", "signal", "metric", "dev_effect", "effect", "ci_lo", "ci_hi", "n_groups", "n_fault",
               "n_control", "supported_same_direction")


def lro_table(E) -> pd.DataFrame:
    """Stage e09 product: for each region, the (signal, metric) with the largest |delta| on the OTHER regions, tested
    on the held-out region (cluster bootstrap over fault systems). Zero rows (no region with min_region_stations) is
    a valid, explicitly typed empty result: the header is always written."""
    if "LRO" in E.D.cache:
        return E.D.cache["LRO"]
    S, M, *_ = E.stations_data()
    rows = []
    mets = ("abs_step", "abs_peak", "local_contrast")
    for k in sorted(S.region.unique()):
        dev = M[M.region != k]
        best, bv = None, 0.0
        for sig, g in dev.groupby("signal"):
            for m in mets:
                d = cliffs_delta(g[g.kind == "fault"][m], g[g.kind == "control"][m])
                if np.isfinite(d) and abs(d) > abs(bv):
                    best, bv = (sig, m), d
        if best is None:
            continue
        h = M[(M.region == k) & (M.signal == best[0])]
        f, c = h[h.kind == "fault"], h[h.kind == "control"]
        if len(f) < int(E.c["statistics"]["min_region_stations"]):
            continue
        e = cluster_effect(f[best[1]], f.system, c[best[1]], c.system, E.n_boot, E.seed)
        sup = (e["ci_lo"] > E.min_effect) if bv > 0 else (e["ci_hi"] < -E.min_effect)
        rows.append({"held_out_region": int(k), "signal": best[0], "metric": best[1], "dev_effect": bv, **e,
                     "supported_same_direction": bool(sup)})
    R = pd.DataFrame(rows, columns=list(LRO_COLUMNS))
    E.ws.save_table(R, "tables/regional_robustness.csv")
    E.D.cache["LRO"] = R
    return R


def f077(E):
    s = sp("F077", "leave_region_out", "Leave-region-out robustness of the best signature", "G", "Does the most "
           "promising signature persist in regions not used to select it?", "For each region: choose the "
           "(signal, metric) with the largest |delta| on the OTHER regions; test it on the held-out region",
           "Avoids circular selection-and-validation on the same data", "Held-out delta with cluster CI; share of "
           "regions where it is supported in the selected direction", TEST_OUT,
           {SUPPORTED: "The selected signature transfers to held-out regions.", "NOT_SUPPORTED": "It does not "
                                                                                                 "transfer.",
            INCONCLUSIVE: "Mixed transfer."}, "Selection among the profile metrics only.", 1)
    R = lro_table(E)
    if not len(R):
        return E.reg.skip(s, "no region with enough stations")
    fig, ax = V.grid(1, 1, 7, 3.5)
    V.effect_bars(ax[0], R.assign(label=R.held_out_region.astype(str) + ": " + R.signal + " " + R.metric),
                  title="held-out region effect of the signature selected elsewhere", min_effect=E.min_effect)
    share = float(R.supported_same_direction.mean())
    st = SUPPORTED if share >= 2 / 3 else ("NOT_SUPPORTED" if share <= 1 / 3 else INCONCLUSIVE)
    E.reg.save(s, fig, R, Result(st, f"selected signature supported in {share:.0%} of {len(R)} held-out regions",
                                 {"share": share}, len(R)))


def f078(E):
    R = regional_effects(E)
    X = R[R.metric == "abs_step"].copy()
    X["class"] = np.select([(np.sign(X.effect) == np.sign(X.overall)) & (X.effect.abs() >= 0.2),
                            X.effect.abs() < 0.2], ["useful", "uncertain"], "unreliable")
    P = X.pivot_table(index="signal", columns="region", values="class", aggfunc="first")
    s = sp("F078", "regional_dashboard", "Where each measurement is useful, uncertain or unreliable", "G",
           "Regional synthesis", "Per signal and region: useful (|delta| >= 0.2, same sign as overall), uncertain "
           "(< 0.2), unreliable (opposite sign)", "One view of regional reliability", "Classes from regional deltas",
           {DESCRIPTIVE: "Regional reliability."}, {DESCRIPTIVE: "Only signals useful in most regions are global "
                                                                 "feature candidates."}, "Point estimates.", 1)
    code = {"useful": 2, "uncertain": 1, "unreliable": 0}
    p = V.plt()
    fig, ax = p.subplots(figsize=(8, 0.35 * len(P) + 2))
    ax.imshow(P.replace(code).to_numpy(dtype=float), cmap="RdYlGn", vmin=0, vmax=2, aspect="auto")
    ax.set_yticks(range(len(P)))
    ax.set_yticklabels(P.index, fontsize=6)
    ax.set_xticks(range(P.shape[1]))
    ax.set_xticklabels(P.columns)
    ax.set_title("green useful / yellow uncertain / red unreliable")
    share = (P == "useful").mean(axis=1).sort_values(ascending=False)
    E.reg.save(s, fig, X, Result(DESCRIPTIVE, f"signals useful in >= 2/3 of regions: "
                                              f"{list(share[share >= 2 / 3].index) or 'none'}", {}, len(X)))


PART_G = [f070, f071, f072, f073, f074, f075, f076, f077, f078]
