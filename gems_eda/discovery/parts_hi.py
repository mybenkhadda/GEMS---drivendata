"""Part H (F079-F090): forensics of the frozen B0 / B1 / W2 predictions (no training, no inference beyond the cached
held-out maps). Part I (F091-F098): reviewable discovery-candidate atlas. PART_DEM: opt-in cached DEM pilot.

The honest maps are the zone-stitched held-out V3 maps of ``gems_bench.reset.atlas.honest_maps`` (each geo zone scored
by the model that never saw it) and the frozen submitted W2 binary map (policy of the submitted W2 file). Error
classes per mapped branch (distance from the branch to the nearest frozen-W2 pixel):
  detected                - >= 50 % of the branch within 300 m (the official radius)
  localisation_failure    - not detected, but >= 50 % within 2 km (the model saw the structure, misplaced it)
  missed_ranked_high      - otherwise, and the mean held-out W2 rank on the branch >= 0.9 (signal present, cut by
                            the area budget)
  detection_failure       - otherwise (the model does not rank the branch)
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import ndimage

from . import viz as V
from .core import DESCRIPTIVE, INCONCLUSIVE, NOT_AVAILABLE, PALETTE, SUPPORTED, Result, cluster_effect, effect_result
from .data import PRIMARY
from .parts_ab import TEST_OUT, rose, sp
from .parts_efg import lin_data

CLASSES = ("detected", "localisation_failure", "missed_ranked_high", "detection_failure")
CLASS_COLOR = {"detected": "#1b9e77", "localisation_failure": "#e6ab02", "missed_ranked_high": "#7570b3",
               "detection_failure": "#d95f02"}
FP_CLASSES = ("near_fp_0.3_2km", "far_fp_with_lineament", "far_fp_no_evidence")


# =============================================================================== shared forensics data
def error_class(hit_share: float, share_2km: float, w2_rank: float) -> str:
    """Error class of one mapped branch (module docstring)."""
    if hit_share >= 0.5:
        return "detected"
    if share_2km >= 0.5:
        return "localisation_failure"
    return "missed_ranked_high" if np.isfinite(w2_rank) and w2_rank >= 0.9 else "detection_failure"


def fp_class(median_dist_km: float, lineament_share: float) -> str:
    if median_dist_km <= 2.0:
        return "near_fp_0.3_2km"
    return "far_fp_with_lineament" if lineament_share >= 0.5 else "far_fp_no_evidence"


def maps(E) -> dict:
    M = E.D.cache.get("maps")
    if M is None:
        raise RuntimeError("honest maps unavailable (frozen engine not attached / 03_scenarios not COMPLETE)")
    if tuple(M["w2_frozen"].shape) != tuple(E.D.grid.shape):
        raise ValueError(f"honest-map grid {M['w2_frozen'].shape} != EDA grid {E.D.grid.shape}")
    return M


def eval_mask(E) -> np.ndarray:
    M = maps(E)
    return (M["zone_id"] >= 0) & np.isfinite(M["w2"].astype(np.float32)) & E.D.valid


def segment_errors(E) -> pd.DataFrame:
    """One row per mapped branch inside the evaluated zones: error class + covariates (cached)."""
    if "SEGERR" in E.D.cache:
        return E.D.cache["SEGERR"]
    D, M = E.D, maps(E)
    ev = eval_mask(E)
    fro = M["w2_frozen"] & ev
    dw = (ndimage.distance_transform_edt(~fro) * D.grid.pixel_km).astype(np.float32) if fro.any() else \
        np.full(fro.shape, np.inf, np.float32)
    L, lab, cnt = lin_data(E)
    dom = E.domains()
    ranks = {k: M[k].astype(np.float32) for k in ("w2", "b0", "b1")}
    rows = []
    for s, p in zip(D.net.segments.itertuples(), D.net.paths):
        rr, cc = p[:, 0], p[:, 1]
        inside = ev[rr, cc]
        if inside.mean() < 0.8:
            continue
        d = dw[rr, cc]
        hit, near = float((d <= 0.3).mean()), float((d <= 2.0).mean())
        rk = {k: float(np.nanmean(a[rr, cc])) for k, a in ranks.items()}
        cls = error_class(hit, near, rk["w2"])
        rows.append({"sample_id": f"seg{s.seg_id}", "seg_id": s.seg_id, "component": s.component,
                     "zone": int(np.bincount(M["zone_id"][rr, cc].clip(0)).argmax()), "region": s.region,
                     "domain": int(dom[s.r_mid, s.c_mid]), "length_km": s.length_km, "azimuth": s.azimuth,
                     "sinuosity": s.sinuosity, "n_junction_ends": s.n_junction_ends, "r_mid": s.r_mid,
                     "c_mid": s.c_mid, "x_mid": s.x_mid, "y_mid": s.y_mid, "hit_share_300m": hit,
                     "share_within_2km": near, "median_dist_to_w2_km": float(np.median(d)),
                     "w2_rank": rk["w2"], "b0_rank": rk["b0"], "b1_rank": rk["b1"],
                     "family_count_mean": float(cnt[rr, cc].mean()), "error_class": cls})
    S = pd.DataFrame(rows)
    E.D.cache["SEGERR"] = S
    return S


def fp_components(E) -> pd.DataFrame:
    """Connected components of the frozen W2 map farther than 300 m from every mapped trace."""
    if "FPERR" in E.D.cache:
        return E.D.cache["FPERR"]
    from gems_bench.reset.atlas import components

    D, M = E.D, maps(E)
    ev = eval_mask(E)
    L, lab, cnt = lin_data(E)
    linnear = ndimage.binary_dilation(lab > 0, iterations=3)
    rows = []
    for i, (rr, cc) in enumerate(components(M["w2_frozen"] & ev & (D.net.distance_km > 0.3), 2)):
        d = D.net.distance_km[rr, cc]
        lin_share = float(linnear[rr, cc].mean())
        cls = fp_class(float(np.median(d)), lin_share)
        r, c = int(np.median(rr)), int(np.median(cc))
        rows.append({"sample_id": f"fp{i}", "n_px": len(rr), "length_km": len(rr) * 0.112, "r_mid": r, "c_mid": c,
                     "region": int(D.region(np.array([r]), np.array([c]))[0]), "zone": int(M["zone_id"][r, c]),
                     "median_dist_known_km": float(np.median(d)), "lineament_share": lin_share,
                     "family_count_mean": float(cnt[rr, cc].mean()), "error_class": cls})
    F = pd.DataFrame(rows)
    E.D.cache["FPERR"] = F
    return F


def official_rows(E) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Official components of the frozen W2 map per V3 zone (E.score) and per 10 km block (m7 block_components)."""
    if "OFFICIAL" in E.D.cache:
        return E.D.cache["OFFICIAL"]
    from gems_bench.m7.protocol import block_components

    M = maps(E)
    rows, blocks = [], []
    for su in [s for s in E.units if s.scenario == "V3"]:
        u = su.unit
        q = (M["w2_frozen"][u.sl] & (M["zone_id"][u.sl] == su.zone)).astype(np.float32)
        r = E.score(u, q)
        rows.append({"zone": su.zone, **{k: r[k] for k in ("tp_w", "fp_w", "fn_w", "n_truth", "dti", "area_px")}})
        B = block_components(u, q, 100)
        br = (B.block // 100000) * 100 + 50
        bc = (B.block % 100000) * 100 + 50
        H, W = E.D.grid.shape
        B["region"] = E.D.region(np.clip(br, 0, H - 1).to_numpy(), np.clip(bc, 0, W - 1).to_numpy())
        blocks.append(B)
    out = (pd.DataFrame(rows), pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame())
    E.D.cache["OFFICIAL"] = out
    return out


def class_shares(S: pd.DataFrame, by: str) -> pd.DataFrame:
    X = S.groupby([by, "error_class"]).length_km.sum().unstack(fill_value=0.0)
    X = X.reindex(columns=list(CLASSES), fill_value=0.0)
    return X.div(X.sum(axis=1), axis=0)


def stacked(ax, X: pd.DataFrame, xlabel: str):
    bottom = np.zeros(len(X))
    for c in X.columns:
        ax.bar(np.arange(len(X)), X[c], bottom=bottom, color=CLASS_COLOR.get(c), label=c)
        bottom += X[c].to_numpy()
    ax.set_xticks(np.arange(len(X)))
    ax.set_xticklabels([str(i) for i in X.index], rotation=45, fontsize=6)
    ax.set(xlabel=xlabel, ylabel="share of mapped length")


# =============================================================================== PART H
def f079(E):
    D, M = E.D, maps(E)
    s = sp("F079", "honest_maps", "Held-out B0 / B1 / W2 ranks and the frozen submitted W2 map", "H", "What do the "
           "frozen models predict where they were not trained?", "Zone-stitched held-out rank maps and the binary "
           "frozen W2 policy map, mapped faults overlaid", "Forensics must use honest (held-out) maps",
           "Area share of the frozen map in the evaluated zones", {DESCRIPTIVE: "Map overview."},
           {DESCRIPTIVE: "Basis of F080-F090."}, "Only V3 geo zones are evaluated; ranks are per zone.", 1)
    p = V.plt()
    fig, ax = p.subplots(2, 2, figsize=(14, 12))
    for a_, k in zip(ax.ravel()[:3], ("w2", "b0", "b1")):
        V.show(a_, D, M[k].astype(np.float32), f"{k.upper()} held-out rank", cmap="viridis", step=3, vmin=0, vmax=1)
    ev = eval_mask(E)
    V.show(ax[1, 1], D, np.where(ev, M["w2_frozen"].astype(float), np.nan), "frozen W2 map + mapped faults",
           cmap="Greys", step=3, faults=True, vmin=0, vmax=1, colorbar=False)
    share = float(M["w2_frozen"][ev].mean()) if ev.any() else np.nan
    E.reg.save(s, fig, V.map_data("F079", {k: M[k].astype(np.float32) for k in ("w2", "b0", "b1")}, D),
               Result(DESCRIPTIVE, f"frozen W2 covers {share:.4f} of {int(ev.sum())} evaluated pixels", {"share": share},
                      int(ev.sum())))


def f080(E):
    S = segment_errors(E)
    s = sp("F080", "error_taxonomy", "Detection vs localisation failures of the frozen W2 map", "H", "Are W2 errors "
           "mostly detection or localisation failures?", "Error class of each mapped branch in the evaluated zones "
           "(module docstring)", "Localisation failures call for thinning / snapping; detection failures for new "
           "evidence", "Length-weighted class shares", {DESCRIPTIVE: "Taxonomy."},
           {DESCRIPTIVE: "The largest failure class sets the next modelling lever (MODELING_DECISION Q3)."},
           "300 m / 2 km / rank 0.9 cut-offs are analysis choices.", 1)
    if not len(S):
        return E.reg.skip(s, "no mapped branch inside the evaluated zones")
    sh = S.groupby("error_class").length_km.sum().reindex(list(CLASSES), fill_value=0.0)
    sh = sh / sh.sum()
    p = V.plt()
    fig, ax = p.subplots(1, 2, figsize=(14, 5.5))
    ax[0].bar(sh.index, sh.values, color=[CLASS_COLOR[c] for c in sh.index])
    ax[0].set(ylabel="share of mapped length")
    ax[0].tick_params(axis="x", rotation=20)
    for c in CLASSES:
        g = S[S.error_class == c]
        ax[1].scatter(g.x_mid / 1e3, g.y_mid / 1e3, s=np.clip(g.length_km * 2, 1, 30), color=CLASS_COLOR[c], label=c)
    ax[1].set(xlabel="E (km)", ylabel="N (km)", title="branch midpoints by class")
    ax[1].set_aspect("equal")
    ax[1].legend(fontsize=6)
    E.reg.save(s, fig, S, Result(DESCRIPTIVE, "; ".join(f"{k} {v:.2f}" for k, v in sh.items()), sh.to_dict(),
                                 len(S)))


def f081(E):
    S = segment_errors(E)
    S = S.assign(length_bin=pd.cut(S.length_km, [0, 1, 2, 5, 10, 1e9], labels=["<1", "1-2", "2-5", "5-10", ">10"]))
    X = class_shares(S, "length_bin")
    s = sp("F081", "errors_by_length", "W2 error class by branch length", "H", "Are short faults missed more?",
           "Length-weighted class shares per length bin; Cliff's delta of length, detected vs not detected",
           "Short structures may be under-ranked or lost to thinning", "Shares; delta clustered by fault system",
           TEST_OUT, {SUPPORTED: "Detection depends on length (direction in summary).", INCONCLUSIVE: "Undecided.",
                      "NOT_SUPPORTED": "No length dependence."}, "Branch length depends on skeleton tracing.", 2)
    a, b = S[S.error_class == "detected"], S[S.error_class != "detected"]
    e = cluster_effect(a.length_km, a.component, b.length_km, b.component, E.n_boot, E.seed)
    fig, ax = V.grid(1, 1, 7, 4)
    stacked(ax[0], X, "branch length (km)")
    ax[0].legend(fontsize=6)
    E.reg.save(s, fig, X.reset_index(), effect_result(e, E.min_effect, None, "length detected vs not:",
                                                      "fault systems", ("detected", "not detected branches")))


def f082(E):
    S = segment_errors(E).dropna(subset=["azimuth"])
    p = V.plt()
    fig, ax = p.subplots(1, 2, subplot_kw={"projection": "polar"}, figsize=(10, 5))
    det = S[S.error_class == "detected"]
    mis = S[S.error_class != "detected"]
    rose(ax[0], det.azimuth, det.length_km, CLASS_COLOR["detected"], "detected")
    rose(ax[1], mis.azimuth, mis.length_km, CLASS_COLOR["detection_failure"], "not detected")
    ax[0].set_title("detected")
    ax[1].set_title("not detected")
    S = S.assign(az_bin=(S.azimuth // 30 * 30).astype(int))
    X = class_shares(S, "az_bin")
    s = sp("F082", "errors_by_orientation", "W2 error class by strike", "H", "Does W2 prefer some orientations?",
           "Length-weighted roses of detected vs not detected branches; class shares per 30 deg bin",
           "A strike preference learned from labels transfers mapping bias", "Detected share per strike bin",
           {DESCRIPTIVE: "Orientation dependence."}, {DESCRIPTIVE: "A strong strike preference: check against F065."},
           "Axial binning only.", 2)
    det_share = X["detected"] if "detected" in X else pd.Series(dtype=float)
    E.reg.save(s, fig, X.reset_index(), Result(DESCRIPTIVE, f"detected share by 30 deg bin: "
                                                            f"{det_share.round(2).to_dict()}", {}, len(S)))


def f083(E):
    S = segment_errors(E)
    X = class_shares(S, "domain")
    s = sp("F083", "errors_by_domain", "W2 error class by label-free physical domain", "H", "Do failures concentrate "
           "in particular geological settings?", "Length-weighted class shares per domain", "Domain-specific failures "
           "point to missing evidence there", "Detected share range across domains", {DESCRIPTIVE: "By domain."},
           {DESCRIPTIVE: "Domains with low detected share: candidate targets for domain-aware evidence."},
           "Domains are statistical, not geological units.", 2)
    fig, ax = V.grid(1, 1, 7, 4)
    stacked(ax[0], X, "domain")
    ax[0].legend(fontsize=6)
    d = X.get("detected", pd.Series(dtype=float))
    E.reg.save(s, fig, X.reset_index(), Result(DESCRIPTIVE, f"detected share {d.min():.2f}-{d.max():.2f} across "
                                                            f"{len(d)} domains" if len(d) else "no domain", {}, len(S)))


def f084(E):
    S = segment_errors(E)
    s = sp("F084", "errors_vs_evidence", "Do missed faults lack independent physical evidence?", "H", "Are W2 "
           "detection failures faults without physical expression in the features?", "Mean independent-family count "
           "on detected vs detection-failure branches", "If misses lack evidence, no model of these inputs will find "
           "them; if misses carry evidence, the model under-uses it", "Cliff's delta (detected minus detection "
           "failure), clustered by fault system", TEST_OUT,
           {SUPPORTED: "Evidence differs between detected and missed branches (direction in summary).",
            "NOT_SUPPORTED": "Missed branches carry evidence like detected ones: the model under-uses available "
                             "evidence.", INCONCLUSIVE: "Undecided."}, "Family count uses 1 km crests only.", 1)
    a, b = S[S.error_class == "detected"], S[S.error_class == "detection_failure"]
    e = cluster_effect(a.family_count_mean, a.component, b.family_count_mean, b.component, E.n_boot, E.seed)
    p = V.plt()
    fig, ax = p.subplots(figsize=(6.5, 4))
    for c in CLASSES:
        V.ecdf(ax, S[S.error_class == c].family_count_mean, c, CLASS_COLOR[c])
    ax.set(xlabel="mean independent families with a crest (dilated)", ylabel="ECDF")
    ax.legend(fontsize=6)
    E.reg.save(s, fig, S[["sample_id", "error_class", "family_count_mean", "component"]],
               effect_result(e, E.min_effect, None, "families detected vs missed:", "fault systems",
                             ("detected", "missed branches")))


def f085(E):
    S = segment_errors(E)
    s = sp("F085", "rank_vs_budget", "Held-out W2 rank of missed branches", "H", "Are misses ranked just below the "
           "area budget or not ranked at all?", "ECDF of the mean held-out W2 rank per branch, by class", "High-rank "
           "misses are budget / representation losses; low-rank misses are evidence losses", "Share of not-detected "
           "length with rank >= 0.9", {DESCRIPTIVE: "Rank distribution."},
           {DESCRIPTIVE: "A large high-rank share favours representation / budget work over new models."},
           "Rank is per zone.", 1)
    p = V.plt()
    fig, ax = p.subplots(figsize=(6.5, 4))
    for c in CLASSES:
        V.ecdf(ax, S[S.error_class == c].w2_rank, c, CLASS_COLOR[c])
    ax.axvline(0.9, color="k", lw=0.5)
    ax.set(xlabel="mean held-out W2 rank on the branch", ylabel="ECDF")
    ax.legend(fontsize=6)
    nd = S[S.error_class != "detected"]
    sh = float(nd.length_km[nd.w2_rank >= 0.9].sum() / max(nd.length_km.sum(), 1e-9))
    E.reg.save(s, fig, S[["sample_id", "error_class", "w2_rank", "length_km"]],
               Result(DESCRIPTIVE, f"{sh:.2f} of not-detected length has held-out W2 rank >= 0.9", {"share": sh},
                      len(nd)))


def f086(E):
    F = fp_components(E)
    s = sp("F086", "false_positives", "Frozen-W2 predictions away from mapped faults", "H", "Are W2 false positives "
           "near misses, plausible unmapped structures or unsupported?", "Components of the frozen map > 300 m from "
           "mapped traces, classed by distance and label-free lineament support", "Far, lineament-supported FPs are "
           "discovery candidates (part I), not necessarily errors", "Length shares per FP class",
           {DESCRIPTIVE: "FP taxonomy."}, {DESCRIPTIVE: "Far unsupported FPs are the cheapest to remove."},
           "Lineament support is not fault verification.", 1)
    if not len(F):
        return E.reg.skip(s, "no frozen-W2 component > 300 m from mapped traces")
    sh = F.groupby("error_class").length_km.sum().reindex(list(FP_CLASSES), fill_value=0.0)
    sh = sh / sh.sum()
    p = V.plt()
    fig, ax = p.subplots(1, 2, figsize=(12, 4.5))
    ax[0].bar(sh.index, sh.values, color="0.4")
    ax[0].tick_params(axis="x", rotation=15)
    ax[0].set(ylabel="share of FP length")
    V.ecdf(ax[1], F.median_dist_known_km, "FP components")
    ax[1].set_xscale("log")
    ax[1].set(xlabel="median distance to mapped trace (km)")
    E.reg.save(s, fig, F, Result(DESCRIPTIVE, "; ".join(f"{k} {v:.2f}" for k, v in sh.items()), sh.to_dict(), len(F)))


def f087(E):
    Z, B = official_rows(E)
    s = sp("F087", "official_decomposition", "Official TP / FP / FN decomposition of the frozen W2 map", "H", "Which "
           "metric component dominates the loss, and where?", "Official scorer components per V3 zone and per 3 x 3 "
           "region (10 km block sums)", "The metric penalises FN 4x more than FP: losses are read in weighted units",
           "0.8 FN_w and 0.2 FP_w per region", {DESCRIPTIVE: "Decomposition."},
           {DESCRIPTIVE: "FN-dominated regions need recall; FP-dominated ones precision."},
           "Zone-stitched map, official components (R = 300 m, alpha 0.2, beta 0.8).", 1)
    if not len(Z):
        return E.reg.skip(s, "no V3 unit")
    R = B.groupby("region")[["tp_w", "fp_w", "fn_w"]].sum().reset_index() if len(B) else pd.DataFrame()
    p = V.plt()
    fig, ax = p.subplots(1, 2, figsize=(13, 4.5))
    x = np.arange(len(Z))
    ax[0].bar(x - 0.25, Z.tp_w, 0.25, label="TP_w", color="#1b9e77")
    ax[0].bar(x, 0.2 * Z.fp_w, 0.25, label="0.2 FP_w", color="#d95f02")
    ax[0].bar(x + 0.25, 0.8 * Z.fn_w, 0.25, label="0.8 FN_w", color="#7570b3")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([f"zone {z}\nDTI {d:.3f}" for z, d in zip(Z.zone, Z.dti)], fontsize=6)
    ax[0].legend()
    if len(R):
        xr = np.arange(len(R))
        ax[1].bar(xr - 0.2, 0.2 * R.fp_w, 0.4, label="0.2 FP_w", color="#d95f02")
        ax[1].bar(xr + 0.2, 0.8 * R.fn_w, 0.4, label="0.8 FN_w", color="#7570b3")
        ax[1].set_xticks(xr)
        ax[1].set_xticklabels(R.region)
        ax[1].set(xlabel="region")
        ax[1].legend()
    tp, fp, fn = Z.tp_w.sum(), Z.fp_w.sum(), Z.fn_w.sum()
    pooled = tp / (tp + 0.2 * fp + 0.8 * fn + 1e-6)
    E.reg.save(s, fig, pd.concat([Z.assign(level="zone"), R.assign(level="region")], ignore_index=True),
               Result(DESCRIPTIVE, f"pooled V3 DTI {pooled:.4f}; weighted loss shares FP {0.2 * fp / (0.2 * fp + 0.8 * fn + 1e-9):.2f}"
                                   f" / FN {0.8 * fn / (0.2 * fp + 0.8 * fn + 1e-9):.2f}",
                      {"pooled_dti": pooled}, len(Z)))


def f088(E):
    S = segment_errors(E)
    s = sp("F088", "model_agreement", "B0 / B1 / W2 held-out ranks on detected and missed branches", "H", "Do the "
           "frozen models fail on the same faults?", "Mean held-out ranks per branch, by W2 error class", "Shared "
           "failures point at the inputs; disjoint ones at the models", "Share of W2 detection failures where B0 or "
           "B1 rank >= 0.9", {DESCRIPTIVE: "Agreement."},
           {DESCRIPTIVE: "A high share: the base models hold signal W2 discards (fusion / representation)."},
           "Ranks only (no thresholded B0 / B1 maps).", 2)
    p = V.plt()
    fig, ax = p.subplots(1, 2, figsize=(12, 4.5))
    for c in CLASSES:
        g = S[S.error_class == c]
        ax[0].scatter(g.b0_rank, g.b1_rank, s=6, color=CLASS_COLOR[c], label=c, alpha=0.6)
    ax[0].set(xlabel="B0 rank", ylabel="B1 rank")
    ax[0].legend(fontsize=6)
    df = S[S.error_class == "detection_failure"]
    rescue = float(((df.b0_rank >= 0.9) | (df.b1_rank >= 0.9)).mean()) if len(df) else np.nan
    X = S.groupby("error_class")[["w2_rank", "b0_rank", "b1_rank"]].median().reindex(list(CLASSES))
    X.plot.bar(ax=ax[1], color=[PALETTE.get("w2", "C0"), "C1", "C2"])
    ax[1].tick_params(axis="x", rotation=20)
    E.reg.save(s, fig, S[["sample_id", "error_class", "w2_rank", "b0_rank", "b1_rank"]],
               Result(DESCRIPTIVE, f"{rescue:.2f} of W2 detection failures have B0 or B1 rank >= 0.9", {"share": rescue},
                      len(df)))


def gallery(E, cases: pd.DataFrame, title: str, fid: str, name: str):
    D, M = E.D, maps(E)
    fig, ax = V.grid(max(len(cases), 1), 4, 4, 3.6)
    for a_, r in zip(ax, cases.itertuples()):
        sl = V.win(D, int(r.r_mid), int(r.c_mid), 50)
        V.show(a_, D, D.band(PRIMARY["magnetic"]), f"{r.sample_id} {r.error_class}", sl=sl, band="rtp",
               colorbar=False)
        fr, fc = np.nonzero(M["w2_frozen"][sl])
        x, y = D.grid.rc_to_xy(fr + sl[0].start, fc + sl[1].start)
        a_.plot(np.asarray(x) / 1e3, np.asarray(y) / 1e3, ",", color=PALETTE.get("w2", "gold"))
        V.overlay_faults(a_, D, sl)
    return fig


def f089(E):
    S, F = segment_errors(E), fp_components(E)
    n = max(int(E.c["sampling"]["n_model_error_examples"]) // 8, 2)
    rng = np.random.default_rng(E.seed)
    pick = []
    for c in ("localisation_failure", "missed_ranked_high", "detection_failure"):
        g = S[S.error_class == c]
        pick.append(g.iloc[rng.permutation(len(g))[:n]])
    for c in ("far_fp_with_lineament", "far_fp_no_evidence"):
        g = F[F.error_class == c] if len(F) else F
        pick.append(g.iloc[rng.permutation(len(g))[:n]] if len(g) else g)
    P = pd.concat([p for p in pick if len(p)], ignore_index=True) if any(len(p) for p in pick) else pd.DataFrame()
    s = sp("F089", "error_gallery", "Gallery of frozen-W2 error cases (RTP, W2 gold, mapped black)", "H", "What do "
           "the failure classes look like?", "Seeded random examples per class", "Visual check of the taxonomy",
           "Examples (no statistic)", {DESCRIPTIVE: "Gallery."}, {DESCRIPTIVE: "Review alongside the atlas."},
           "Random examples; not representative of rare cases.", 3)
    if not len(P):
        return E.reg.skip(s, "no error case")
    fig = gallery(E, P, s.title, s.fid, s.name)
    E.reg.save(s, fig, P, Result(DESCRIPTIVE, f"{len(P)} examples", {}, len(P)))


def f090(E):
    S, F = segment_errors(E), fp_components(E)
    rows = [{"side": "mapped_branch", "error_class": c, "n": int((S.error_class == c).sum()),
             "length_km": float(S.length_km[S.error_class == c].sum()),
             "median_family_count": float(S.family_count_mean[S.error_class == c].median()),
             "median_w2_rank": float(S.w2_rank[S.error_class == c].median())} for c in CLASSES]
    rows += [{"side": "prediction", "error_class": c, "n": int((F.error_class == c).sum()) if len(F) else 0,
              "length_km": float(F.length_km[F.error_class == c].sum()) if len(F) else 0.0,
              "median_family_count": float(F.family_count_mean[F.error_class == c].median()) if len(F) else np.nan,
              "median_w2_rank": np.nan} for c in FP_CLASSES]
    T = pd.DataFrame(rows)
    s = sp("F090", "failure_taxonomy", "Frozen-W2 failure taxonomy", "H", "Synthesis of the model forensics",
           "Counts, length and evidence per error class on both sides of the metric", "One table for the modeling "
           "decision", "Table", {DESCRIPTIVE: "Taxonomy table."}, {DESCRIPTIVE: "MODELING_DECISION Q3 / Q4."},
           "Class definitions in the module docstring.", 1)
    p = V.plt()
    fig, ax = p.subplots(figsize=(11, 0.45 * len(T) + 1.2))
    ax.axis("off")
    tb = ax.table(cellText=T.round(3).astype(str).values, colLabels=T.columns, loc="center")
    tb.auto_set_font_size(False)
    tb.set_fontsize(7)
    E.ws.save_table(T, "tables/model_failure_taxonomy.csv")
    cases = pd.concat([S.assign(side="mapped_branch"), F.assign(side="prediction")], ignore_index=True)
    E.save_parquet(cases, "datasets/model_error_cases.parquet")
    E.reg.save(s, fig, T, Result(DESCRIPTIVE, "; ".join(f"{r.error_class} {r.length_km:.0f} km" for r in
                                                       T.itertuples()), {}, len(cases)))


PART_H = [f079, f080, f081, f082, f083, f084, f085, f086, f087, f088, f089, f090]


# =============================================================================== PART I: candidate atlas
CATEGORIES = {"A_unmapped_multifamily": 0.25, "B_unmapped_single_family": 0.15, "C_w2_far_supported": 0.15,
              "D_missed_mapped": 0.15, "E_suspected_nonfault": 0.10, "F_short_fragment": 0.10,
              "G_regional_random": 0.10}
CAT_SOURCES = {"A_unmapped_multifamily": ("structural_lineaments",), "B_unmapped_single_family": (
    "structural_lineaments",), "C_w2_far_supported": ("w2_frozen", "structural_lineaments"), "D_missed_mapped": (
    "mapped_faults", "w2_frozen"), "E_suspected_nonfault": ("structural_lineaments",), "F_short_fragment": (
    "mapped_faults",), "G_regional_random": ()}


def candidate_pools(E) -> dict:
    """{category: [(rows, cols), ...]} - lineament / branch / FP pixel sets. W2-derived categories only when the honest
    maps exist (otherwise recorded as unavailable)."""
    from gems_bench.reset.atlas import components

    D = E.D
    L, lab, _ = lin_data(E)
    objs = ndimage.find_objects(lab)

    def px(i):
        sl = objs[i - 1]
        rr, cc = np.nonzero(lab[sl] == i)
        return rr + sl[0].start, cc + sl[1].start

    U = L[L.status == "unmapped"]
    pools = {"A_unmapped_multifamily": [px(i) for i in U[U.n_families >= 2].lineament_id],
             "B_unmapped_single_family": [px(i) for i in U[(U.n_families == 1) &
                                                           ~U.families.isin(["terrain"])].lineament_id],
             "E_suspected_nonfault": [px(i) for i in U[(U.n_families == 1) & U.families.isin(["terrain"])].lineament_id],
             "F_short_fragment": [(p[:, 0], p[:, 1]) for s, p in zip(D.net.segments.itertuples(), D.net.paths)
                                  if s.length_km < 1.0 and len(p) >= 3]}
    if D.cache.get("maps") is not None:
        S, F = segment_errors(E), fp_components(E)
        seg_px = {f"seg{s.seg_id}": p for s, p in zip(D.net.segments.itertuples(), D.net.paths)}
        pools["D_missed_mapped"] = [(seg_px[i][:, 0], seg_px[i][:, 1]) for i in
                                    S[S.error_class == "detection_failure"].sample_id if i in seg_px]
        M = maps(E)
        far = components(M["w2_frozen"] & eval_mask(E) & (D.net.distance_km > 2.0), 3)
        linnear = ndimage.binary_dilation(lab > 0, iterations=3)
        pools["C_w2_far_supported"] = [c for c in far if linnear[c].mean() >= 0.5]
    rr, cc = np.nonzero(D.valid)
    rng = np.random.default_rng(E.seed + 7)
    pick = rng.choice(len(rr), min(2000, len(rr)), replace=False)
    pools["G_regional_random"] = [(np.array([rr[i]]), np.array([cc[i]])) for i in pick]
    return pools


def build_atlas(E) -> tuple[pd.DataFrame, dict, dict]:
    from gems_bench.reset import atlas as AT

    if "ATLAS" in E.D.cache:
        return E.D.cache["ATLAS"]
    D = E.D
    L, lab, cnt = lin_data(E)
    pools = candidate_pools(E)
    q = AT.quotas({k: v for k, v in CATEGORIES.items() if pools.get(k)}, int(E.c["sampling"]["n_regional_examples"]))
    reg = (lambda r, c: int(D.region(np.array([int(r)]), np.array([int(c)]))[0]))
    kept = AT.select(pools, q, reg, 5.0 / D.grid.pixel_km, np.random.default_rng(E.seed))
    T = D.grid.transform
    eval_regions = E.rc["atlas"]["eval_regions"]
    rows, geom = [], {}
    for i, k in enumerate(kept):
        cid = f"GDE{i:03d}_{k['category'][0]}"
        r, c = int(round(k["row"])), int(round(k["col"]))
        rr, cc = np.asarray(k["rr"]), np.asarray(k["cc"])
        geom[cid] = (rr, cc)
        x, y = D.grid.rc_to_xy(np.array([r]), np.array([c]))
        sl = V.win(D, r, c, 60)
        l_, r_, b_, t_ = D.grid.window_extent(sl[0].start, sl[0].stop, sl[1].start, sl[1].stop)
        lid = np.unique(lab[rr, cc])
        lid = lid[lid > 0]
        fam = ";".join(sorted(set(";".join(L.set_index("lineament_id").loc[lid].families).split(";")) - {""})) if \
            len(lid) else ""
        src = CAT_SOURCES[k["category"]]
        rows.append({"candidate_id": cid, "category": k["category"], "region_id": k["region_id"],
                     "review_split": AT.review_split(k["region_id"], eval_regions), "row": r, "col": c,
                     "x_utm": float(np.asarray(x)[0]), "y_utm": float(np.asarray(y)[0]),
                     "window_extent": json.dumps([l_, b_, r_, t_]), "length_km": float(len(rr) * 0.112) if len(rr) > 1
                     else 0.0, "n_px": int(len(rr)), "source_models": ";".join(s for s in src if s.startswith("w2")),
                     "source_layers": ";".join(s for s in src if not s.startswith("w2")), "supporting_families": fam,
                     "n_supporting_families": len([f for f in fam.split(";") if f]),
                     "mean_family_count": float(cnt[rr, cc].mean()),
                     "distance_to_known_fault_km": float(D.net.distance_km[r, c]),
                     "review_status": "NOT_REVIEWED", "interpretation": "NOT_REVIEWED", "confidence": "",
                     "supporting_evidence": "", "alternative_explanation": alternative(k["category"], fam),
                     "reviewer_notes": "", "geometry_wkt": AT.wkt(geom[cid], T)})
    C = pd.DataFrame(rows)
    meta = {"quota": q, "pool_sizes": {k: len(v) for k, v in pools.items()},
            "categories_unavailable": [k for k in CATEGORIES if not pools.get(k)]}
    E.D.cache["ATLAS"] = (C, geom, meta)
    return C, geom, meta


def alternative(cat: str, fam: str) -> str:
    """Rule-based alternative explanation to keep open during review (not an interpretation)."""
    if cat == "G_regional_random":
        return "random control location"
    if fam == "terrain":
        return "drainage / range-front / erosional edge"
    if fam == "magnetic":
        return "dyke, intrusion margin or lithological contact"
    if fam == "gravity":
        return "density contact / basin margin"
    if cat == "D_missed_mapped":
        return "mapped fault without expression in the inputs"
    return "lithological contact or survey artefact"


def panel_layers(E, sl) -> dict:
    from gems_bench.reset.physics import multi_hillshade

    D = E.D
    L, lab, cnt = lin_data(E)
    out = {}
    for k, b in (("Magnetic RTP", PRIMARY["magnetic"]), ("Gravity", PRIMARY["gravity"])):
        if D.has(b):
            out[k] = (D.band(b)[sl], "RdBu_r", b)
    if D.has(PRIMARY["magnetic"]):
        out["Magnetic |grad| 1 km"] = (D.enh.thdr(D.band(PRIMARY["magnetic"])[sl], 1.0), "magma", None)
    if D.has(PRIMARY["gravity"]):
        out["Gravity |grad| 1 km"] = (D.enh.thdr(D.band(PRIMARY["gravity"])[sl], 1.0), "magma", None)
    if D.has(PRIMARY["terrain"]):
        out["Terrain hillshade"] = (multi_hillshade(D.band(PRIMARY["terrain"])[sl]), "gray", None)
    out["Independent-family count"] = (cnt[sl].astype(np.float32), "viridis", None)
    return out


def render(E, row, geom, mode: str):
    D = E.D
    sl = V.win(D, int(row.row), int(row.col), 60)
    ext = V.extent_km(D, sl)
    lay = panel_layers(E, sl)
    if mode == "DIAGNOSTIC" and D.cache.get("maps") is not None:
        M = maps(E)
        keep = list(lay.items())[:3]
        lay = dict(keep + [("W2 held-out rank", (M["w2"][sl].astype(np.float32), "viridis", None)),
                           ("B0 held-out rank", (M["b0"][sl].astype(np.float32), "viridis", None)),
                           ("B1 held-out rank", (M["b1"][sl].astype(np.float32), "viridis", None))])
    p = V.plt()
    fig, axs = p.subplots(2, 3, figsize=(13, 8.5))
    norm = {}
    for ax, (title, (a, cmap, band)) in zip(axs.ravel(), list(lay.items())[:6]):
        st = V.show(ax, D, _full(D, a, sl), title, cmap=cmap, sl=sl, band=band,
                    colorbar=True, vmin=0 if "rank" in title else None, vmax=1 if "rank" in title else None)
        norm[title] = st
        if mode == "DIAGNOSTIC":
            V.overlay_faults(ax, D, sl)
            if D.cache.get("maps") is not None:
                fr, fc = np.nonzero(maps(E)["w2_frozen"][sl])
                x, y = D.grid.rc_to_xy(fr + sl[0].start, fc + sl[1].start)
                ax.plot(np.asarray(x) / 1e3, np.asarray(y) / 1e3, ",", color="gold")
            gx, gy = D.grid.rc_to_xy(geom[0], geom[1])
            ax.plot(np.asarray(gx) / 1e3, np.asarray(gy) / 1e3, ".", ms=1.2, color="cyan")
        x0, y0 = D.grid.rc_to_xy(np.array([int(row.row)]), np.array([int(row.col)]))
        ax.plot(np.asarray(x0) / 1e3, np.asarray(y0) / 1e3, "+", color="lime", ms=10, mew=1.5)
    for ax in axs.ravel()[len(lay):]:
        ax.set_axis_off()
    fig.suptitle(f"{row.candidate_id} | {row.category} | region {row.region_id} | {mode} | UTM 11N "
                 f"({row.x_utm:.0f}, {row.y_utm:.0f})" + ("" if mode == "DIAGNOSTIC" else
                                                         " | mapped faults and model outputs hidden"), fontsize=9)
    return fig, norm, ext


def _full(D, a, sl):
    """Embed a window array in a full-grid NaN array so ``viz.show`` can slice it uniformly."""
    out = np.full(D.grid.shape, np.nan, np.float32)
    out[sl] = a
    return out


def f091(E):
    C, geom, meta = build_atlas(E)
    s = sp("F091", "candidate_overview", "Discovery-candidate locations by category", "I", "Where are the candidates "
           "for review?", "Seeded, region-balanced selection per category (>= 5 km separation)", "Review effort spread "
           "over regions and evidence types", "Counts per category vs quota", {DESCRIPTIVE: "Overview."},
           {DESCRIPTIVE: "Reviews feed the hypothesis register; unreviewed candidates are not negatives."},
           "Categories are selection rules, not interpretations.", 1)
    if not len(C):
        return E.reg.skip(s, "no candidate selected")
    D = E.D
    p = V.plt()
    fig, ax = p.subplots(figsize=(10, 9))
    V.show(ax, D, np.where(D.valid, 0.0, np.nan), "candidates", cmap="Greys", step=4, faults=True, vmin=0, vmax=1,
           colorbar=False)
    for k, g in C.groupby("category"):
        ax.scatter(g.x_utm / 1e3, g.y_utm / 1e3, s=25, label=f"{k} ({len(g)})")
    ax.legend(fontsize=6)
    E.reg.save(s, fig, C.drop(columns=["geometry_wkt"]),
               Result(DESCRIPTIVE, f"{len(C)} candidates; quota {meta['quota']}; unavailable categories "
                                   f"{meta['categories_unavailable'] or 'none'}", meta, len(C)))


def f092(E):
    """Write the review atlas: catalog, index, geometries, panels, template, html. Reviews are never overwritten."""
    from gems_bench.artifacts.atomic import atomic_save
    from gems_bench.reset import atlas as AT

    C, geom, meta = build_atlas(E)
    s = sp("F092", "review_panels", "Evidence-only and diagnostic review panels", "I", "Can a reviewer judge each "
           "candidate from evidence alone first?", "2 x 3 panels per candidate: EVIDENCE_ONLY (no mapped faults or "
           "model output), DIAGNOSTIC (mapped faults, frozen W2, B0 / B1 ranks)", "Blind review avoids confirming the "
           "labels", "Panels written", {DESCRIPTIVE: "Atlas written."},
           {DESCRIPTIVE: "Review EVIDENCE_ONLY first; record with E.record_review or the widget."},
           "Panels show 12 km windows at native resolution.", 2)
    if not len(C):
        return E.reg.skip(s, "no candidate")
    p = V.plt()
    P = []
    for row in C.itertuples():
        out = {"candidate_id": row.candidate_id}
        for mode in AT.REVIEW_MODES:
            fig, norm, ext = render(E, row, geom[row.candidate_id], mode)
            rel = f"atlas/panels/{row.candidate_id}_{mode.lower()}.png"
            atomic_save(E.ws.path(rel), lambda t, f=fig: f.savefig(t, dpi=110, bbox_inches="tight", format="png"))
            p.close(fig)
            out[f"panel_{mode.lower()}"] = rel
            E.ws.save_json({"candidate_id": row.candidate_id, "mode": mode, "extent_km": ext, "normalization": norm},
                           f"atlas/panels_data/{row.candidate_id}_{mode.lower()}.json")
        P.append(out)
    P = pd.DataFrame(P)
    ids = list(geom)
    off = np.cumsum([0] + [len(geom[i][0]) for i in ids])
    E.save_npz("atlas/cache/geometry.npz", ids=np.asarray(ids), offsets=off,
               rr=np.concatenate([geom[i][0] for i in ids]).astype(np.int32),
               cc=np.concatenate([geom[i][1] for i in ids]).astype(np.int32))
    E._geom = None
    E.save_parquet(C, "atlas/candidate_catalog.parquet")
    E.ws.save_table(C, "atlas/atlas_index.csv")
    gpkg(E, C)
    tpl = C[[c for c in AT.CATALOG_FIELDS if c in C]].copy()
    for k in ("reviewer", "review_mode", "evidence_sources", "data_license"):
        tpl[k] = ""
    E.ws.save_table(tpl, "atlas/review_template.csv")
    if not E.ws.path("atlas/reviewed_annotations.csv").exists():          # never overwritten
        E.ws.save_table(pd.DataFrame(columns=list(AT.REVIEW_FIELDS)), "atlas/reviewed_annotations.csv")
    E.ws.save_text(AT.overview_html(C, P, E.load_reviews()), "atlas/atlas_index.html")
    fig, ax = V.grid(1, 1, 7, 3.5)
    cnt = C.category.value_counts()
    ax[0].barh(cnt.index, cnt.values, color="0.4")
    ax[0].set(xlabel="candidates")
    E.reg.save(s, fig, P, Result(DESCRIPTIVE, f"{len(P)} candidates x 2 panels written", {}, len(P)))


def gpkg(E, C):
    try:
        import geopandas as gpd
        from shapely import wkt

        g = gpd.GeoDataFrame(C.drop(columns=["geometry_wkt"]), geometry=[wkt.loads(x) for x in C.geometry_wkt],
                             crs="EPSG:32611")
        p = E.ws.path("atlas/candidate_geometries.gpkg")
        p.parent.mkdir(parents=True, exist_ok=True)
        g.to_file(p, driver="GPKG")
    except Exception as e:                                   # optional format, recorded
        E.errors.append({"stage": "e12_candidates", "error": f"gpkg skipped: {type(e).__name__}: {e}"})


def f093(E):
    C, *_ = build_atlas(E)
    s = sp("F093", "candidate_evidence", "Independent evidence per candidate category", "I", "Which categories carry "
           "the most physical support?", "Mean independent-family count by category", "Prioritises review; random "
           "locations calibrate the scale", "Medians per category vs the random category", {DESCRIPTIVE: "Evidence."},
           {DESCRIPTIVE: "Categories indistinguishable from random locations deserve the least review time."},
           "Family count is a selection covariate for A / B (circular there).", 2)
    if not len(C):
        return E.reg.skip(s, "no candidate")
    p = V.plt()
    fig, ax = p.subplots(figsize=(8, 4))
    cats = sorted(C.category.unique())
    ax.boxplot([C[C.category == k].mean_family_count for k in cats], labels=cats)
    ax.tick_params(axis="x", rotation=30, labelsize=6)
    ax.set(ylabel="mean family count")
    m = C.groupby("category").mean_family_count.median()
    E.reg.save(s, fig, m.reset_index(), Result(DESCRIPTIVE, f"medians {m.round(2).to_dict()}", {}, len(C)))


def f094(E):
    C, *_ = build_atlas(E)
    s = sp("F094", "candidate_novelty", "Distance of candidates to mapped faults", "I", "How novel are the candidates?",
           "ECDF of distance to the nearest mapped trace per category", "Novel candidates are the discovery targets",
           "Medians", {DESCRIPTIVE: "Novelty."}, {DESCRIPTIVE: "Distance is recorded in the catalog for review."},
           "Distance from the candidate centre.", 3)
    if not len(C):
        return E.reg.skip(s, "no candidate")
    p = V.plt()
    fig, ax = p.subplots(figsize=(7, 4))
    for k, g in C.groupby("category"):
        V.ecdf(ax, g.distance_to_known_fault_km, k)
    ax.set_xscale("symlog")
    ax.set(xlabel="distance to mapped trace (km)")
    ax.legend(fontsize=6)
    E.reg.save(s, fig, C[["candidate_id", "category", "distance_to_known_fault_km"]],
               Result(DESCRIPTIVE, f"median {C.distance_to_known_fault_km.median():.2f} km", {}, len(C)))


def f095(E):
    C, geom, _ = build_atlas(E)
    s = sp("F095", "candidate_examples", "Example evidence-only panels (one per category)", "I", "What does a "
           "candidate look like without labels?", "First candidate of each category, evidence-only layers",
           "Illustrates the review task", "Examples", {DESCRIPTIVE: "Examples."}, {DESCRIPTIVE: "See the atlas."},
           "One example per category.", 3)
    pick = C.groupby("category").head(1) if len(C) else C
    if not len(pick):
        return E.reg.skip(s, "no candidate")
    D = E.D
    fig, ax = V.grid(len(pick), 4, 4, 3.6)
    for a_, r in zip(ax, pick.itertuples()):
        sl = V.win(D, int(r.row), int(r.col), 60)
        lay = panel_layers(E, sl)
        k = "Magnetic |grad| 1 km" if "Magnetic |grad| 1 km" in lay else next(iter(lay))
        V.show(a_, D, _full(D, lay[k][0], sl), f"{r.candidate_id}\n{k}", cmap=lay[k][1], sl=sl, colorbar=False)
    E.reg.save(s, fig, pick.drop(columns=["geometry_wkt"]), Result(DESCRIPTIVE, f"{len(pick)} examples", {},
                                                                   len(pick)))


def f096(E):
    R = E.load_reviews()
    s = sp("F096", "review_status", "Review status of the atlas", "I", "How many candidates have been reviewed, and "
           "how?", "Latest review per candidate: interpretation x review mode", "Tracks independent (evidence-only) vs "
           "diagnostic reviews", "Counts", {DESCRIPTIVE: "Review progress.", NOT_AVAILABLE: "No review yet."},
           {DESCRIPTIVE: "Only INDEPENDENT_REVIEWED labels may train or validate future models."},
           "Reviews are human judgements, not ground truth.", 2)
    if not len(R):
        return E.reg.save(s, None, None, Result(NOT_AVAILABLE, "no review recorded yet (atlas/reviewed_annotations.csv "
                                                               "is empty)"))
    from gems_bench.reset.atlas import latest_reviews

    Lr = latest_reviews(R)
    X = pd.crosstab(Lr.interpretation, Lr.review_mode)
    fig, ax = V.grid(1, 1, 7, 4)
    X.plot.barh(ax=ax[0])
    E.reg.save(s, fig, X.reset_index(), Result(DESCRIPTIVE, f"{len(Lr)} candidates reviewed; label classes "
                                                            f"{Lr.label_class.value_counts().to_dict()}", {}, len(Lr)))


def f097(E):
    C, *_ = build_atlas(E)
    R = E.load_reviews()
    s = sp("F097", "review_vs_evidence", "Reviewed interpretation vs independent evidence", "I", "Do reviewers' "
           "likely faults carry more independent evidence than their non-faults?", "Cliff's delta of the family count, "
           "LIKELY / POSSIBLE_FAULT vs non-fault interpretations (evidence-only reviews)", "Consistency check between "
           "blind review and the measured evidence", "Delta clustered by region", TEST_OUT,
           {SUPPORTED: "Reviews and evidence agree (direction in summary).", INCONCLUSIVE: "Too few reviews.",
            "NOT_SUPPORTED": "Reviews do not track the family count."}, "Needs reviews; small samples.", 3)
    if not len(R) or not len(C):
        return E.reg.save(s, None, None, Result(NOT_AVAILABLE, "no review recorded yet"))
    from gems_bench.reset.atlas import latest_reviews

    Lr = latest_reviews(R)
    Lr = Lr[Lr.review_mode == "EVIDENCE_ONLY"].merge(C[["candidate_id", "mean_family_count", "region_id"]],
                                                     on="candidate_id")
    fa = Lr[Lr.interpretation.isin(["LIKELY_FAULT", "POSSIBLE_FAULT"])]
    nf = Lr[~Lr.interpretation.isin(["LIKELY_FAULT", "POSSIBLE_FAULT", "INSUFFICIENT_EVIDENCE", "NOT_REVIEWED"])]
    if len(fa) < 5 or len(nf) < 5:
        return E.reg.save(s, None, Lr, Result(INCONCLUSIVE, f"{len(fa)} fault / {len(nf)} non-fault evidence-only "
                                                            "reviews (< 5 each)"))
    e = cluster_effect(fa.mean_family_count, fa.region_id, nf.mean_family_count, nf.region_id, E.n_boot, E.seed)
    fig, ax = V.grid(1, 1, 6, 4)
    V.ecdf(ax[0], fa.mean_family_count, "reviewed fault")
    V.ecdf(ax[0], nf.mean_family_count, "reviewed non-fault")
    ax[0].legend()
    E.reg.save(s, fig, Lr, effect_result(e, E.min_effect, None, "fault vs non-fault reviews:", "regions",
                                         ("fault", "non-fault reviews")))


def f098(E):
    C, _, meta = build_atlas(E)
    s = sp("F098", "atlas_summary", "Atlas summary", "I", "Synthesis of the candidate atlas", "Pool sizes, quotas, "
           "selected counts", "Shows how selective the atlas is", "Table", {DESCRIPTIVE: "Summary."},
           {DESCRIPTIVE: "Larger review effort: increase sampling.n_regional_examples."}, "Pools are rule-based.", 3)
    T = pd.DataFrame([{"category": k, "pool": meta["pool_sizes"].get(k, 0), "quota": meta["quota"].get(k, 0),
                       "selected": int((C.category == k).sum()) if len(C) else 0} for k in CATEGORIES])
    p = V.plt()
    fig, ax = p.subplots(figsize=(9, 0.45 * len(T) + 1.2))
    ax.axis("off")
    tb = ax.table(cellText=T.astype(str).values, colLabels=T.columns, loc="center")
    tb.auto_set_font_size(False)
    tb.set_fontsize(7)
    E.reg.save(s, fig, T, Result(DESCRIPTIVE, f"{int(T.selected.sum())} selected from {int(T.pool.sum())} pooled",
                                 {}, len(T)))


PART_I = [f091, f092, f093, f094, f095, f096, f097, f098]


# =============================================================================== PART_DEM (opt-in)
def fdem(E):
    from gems_bench.reset import terrain as TR

    s = sp("F099", "dem_pilot", "Cached 1 m DEM pilot: terrain evidence on covered held-out pixels", "DEM",
           "Does high-resolution terrain add evidence where it exists?", "Official components of frozen W2, terrain "
           "only and W2 + terrain on DEM-covered pixels (restricted units, a new protocol)", "Fault scarps may be "
           "visible only at metre scale", "DTI per model on covered pixels; unique terrain structures",
           {DESCRIPTIVE: "Pilot measurements.", NOT_AVAILABLE: "No cached DEM: coverage not evaluated."},
           {DESCRIPTIVE: "Continue only if the stop rule says CONTINUE.", NOT_AVAILABLE: "No conclusion about "
                                                                                         "terrain."},
           "Pilot coverage only; restricted units are not comparable to full-zone scores.", 3)
    if E.F is None or not getattr(E, "units", None):
        return E.reg.save(s, None, None, Result(NOT_AVAILABLE, "frozen engine not attached (units unavailable)"))
    r = TR.run(E)
    T = E.ws.load_table("terrain/pilot_comparison.csv")
    if r.get("status") != "COMPLETE" or not len(T):
        E.ws.save_table(pd.DataFrame([{"status": r.get("status"), "reason": r.get("reason", r.get("note"))}]),
                        "tables/dem_pilot.csv")
        return E.reg.save(s, None, None, Result(NOT_AVAILABLE, str(r.get("reason", r.get("note")))))
    E.ws.save_table(T, "tables/dem_pilot.csv")
    G = T.groupby("model")[["tp_w", "fp_w", "fn_w"]].sum()
    G["dti"] = G.tp_w / (G.tp_w + 0.2 * G.fp_w + 0.8 * G.fn_w + 1e-6)
    fig, ax = V.grid(1, 1, 6, 4)
    ax[0].bar(G.index, G.dti, color="0.4")
    ax[0].set(ylabel="pooled DTI on DEM-covered pixels")
    E.reg.save(s, fig, G.reset_index(), Result(DESCRIPTIVE, f"{r.get('note')}; pooled DTI {G.dti.round(4).to_dict()}",
                                               {}, len(T)))


PART_DEM = [fdem]
