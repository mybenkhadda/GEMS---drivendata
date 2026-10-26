"""Section 14: candidate-level geological features (one row per traced candidate, Parquet + GeoPackage).

Base descriptors are M7's (``m7.cand_features.describe`` -> ``gems3.descriptors.describe``): geometry, provenance,
evidence of the M7 candidate layers, magnetic / gravity orientation consistency, terrain side contrast and slope
asymmetry, 1 m DEM evidence where the M7 mosaic covers the line, honest B0 / B1 support (ranks), known-fault
distances and network context, and the distance-aware PU label (1 near scored truth, 0 far, -1 unlabelled; label
columns never enter a feature group). M9 adds:

    tracing       beam score, evidence mean / min / p10 / p90, alignment with the predicted strike, orientation
                  confidence, mean turn, gap share, endpoint / junction head support, branch flag, tracer one-hots
    geophysics    fault-normal contrast of magnetic / gravity / conductivity / basement fields at 200 / 400 / 800 m
    spectral      wavelet / FFT anisotropy, coherence, persistence along the line + spectral-strike agreement
    crossmodal    mean of the new tolerant magnetic-gravity / terrain-magnetic alignment layers along the line
    neural_m9     support of the M9 dense map (production context and no-context) + disagreement with B0 / B1
    context       seismicity / strain percentiles along the line (regional descriptors; low seismicity never rejects)
    montecarlo    MC-A detection frequency / positional variability / fragmentation, MC-B occupancy / branch freq
                  (``has_mc`` = 0 where section 12 did not run: missing, never silently imputed)
Candidate ids, zone / fold ids, seed positions and regional identifiers are excluded from every feature group.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EXCLUDE = {"cand_id", "zone", "uid", "seed_r", "seed_c", "r_full", "c_full", "r_mid", "c_mid", "network_id", "parent",
           "label", "kd_mean", "kd_max", "near300", "dist_median_km", "unit", "task", "strike_deg", "n_px"}


def _along(a: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.asarray(a)[p[:, 0], p[:, 1]]


def _crop_band(m9, band: str, u) -> np.ndarray | None:
    if band not in m9.ctx.data.band_names:
        return None
    return m9.ctx.data.read_window(band, u.sl[0].start, u.sl[0].stop, u.sl[1].start, u.sl[1].stop)


def describe(m9, u, cs, pred_prod: dict | None, pred_noctx: dict | None, *, with_labels: bool = True,
             mc: pd.DataFrame | None = None, legacy: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(feature table, provenance rows) of one unit's traced candidates."""
    from ..gems3.candidates import CandidateSet
    from ..gems3.representation import rank01
    from ..m7.cand_features import describe as m7_describe
    from ..m7.cand_features import side_contrast
    from .features import available, read

    if cs is None or not len(cs):
        return pd.DataFrame(), pd.DataFrame()
    t = cs.table.copy()
    t["kind"] = "trace"
    cs = CandidateSet(t, cs.paths)
    scores = dict(legacy or {})
    if pred_prod is not None:
        scores["m9"] = np.nan_to_num(pred_prod["seg"])
    if pred_noctx is not None and u.task != "geo":
        scores["m9nc"] = np.nan_to_num(pred_noctx["seg"])
    df = m7_describe(m9.m7view, u, cs, scores, with_labels=with_labels, dem=m9.dem_mosaic())
    prov = [{"feature": c, "group": "m7_descriptor", "source": "m7.cand_features / gems3.descriptors"} for c in df.columns]
    paths = cs.paths
    strike = df.strike_deg.to_numpy()
    for k in t.columns:                                         # tracing descriptors (kept from section 11)
        if k not in df and pd.api.types.is_numeric_dtype(t[k]):
            df[k] = t[k].to_numpy()
    for s in ("beam", "beam_branch", "beam_stoch", "crest", "mcp_bridge", "mc_only"):
        df[f"src_{s}"] = (t.source.astype(str) == s).astype(int).to_numpy()
    df["is_branch"] = (t.get("parent", pd.Series(-1, index=t.index)).to_numpy() >= 0).astype(int)
    fc = m9.cfg["candidates"]
    for mod, band in (fc["fields"] or {}).items():
        f = _crop_band(m9, band, u)
        if f is None:
            continue
        for off in fc["side_offsets_m"]:
            col = f"fn_{mod}_{int(off)}m"
            df[col] = side_contrast(f, paths, strike, float(off) / 100.0)
            prov.append({"feature": col, "group": "geophysics", "source": f"{band}: |mean(+n) - mean(-n)| / IQR"})
    cols = [c for c in df if c.startswith("fn_") and c.endswith(f"_{int(fc['side_offsets_m'][1])}m")]
    if cols:
        med = df[cols].median()
        df["n_modal_contrast"] = (df[cols] > med).sum(1)
    spec = [n for n in m9.spectral_selected() if available(m9, n)]
    for n in spec:
        if any(tok in n for tok in ("_c2", "_s2")):
            continue
        a = np.asarray(read(m9, n))[u.sl]
        col = "spec_" + n.split(":")[-1].replace("@", "_").replace(".", "p")
        df[col] = [float(np.nanmean(_along(a, p))) for p in paths]
        prov.append({"feature": col, "group": "spectral", "source": n})
    for fam in ("magnetic", "gravity"):
        c2, s2 = f"m9:spec:{fam}_wav_strike_c2@1.6km", f"m9:spec:{fam}_wav_strike_s2@1.6km"
        if available(m9, c2) and available(m9, s2):
            a, b = np.asarray(read(m9, c2))[u.sl], np.asarray(read(m9, s2))[u.sl]
            th = np.radians(2 * strike)
            df[f"spec_strike_agree_{fam}"] = [float(np.nanmean(_along(a, p) * np.cos(t2) + _along(b, p) * np.sin(t2)))
                                              for p, t2 in zip(paths, th)]
    for n in [x for x in m9.struct_selected() if x.startswith("m9:xmod:")]:
        a = np.asarray(read(m9, n))[u.sl]
        col = "xmod_" + n.split(":")[-1].replace("@", "_").replace(".", "p")
        df[col] = [float(np.nanmean(_along(a, p))) for p in paths]
        prov.append({"feature": col, "group": "crossmodal", "source": n})
    if "m9_mean" in df:
        for b in ("b0", "b1"):
            if f"{b}_mean" in df:
                df[f"m9_minus_{b}"] = df.m9_mean - df[f"{b}_mean"]
        if "m9nc_mean" in df:
            df["m9_context_gain"] = df.m9_mean - df.m9nc_mean
    for band in fc["context_bands"]:
        a = _crop_band(m9, band, u)
        if a is None:
            continue
        r = rank01(np.nan_to_num(a, nan=float(np.nanmedian(a)) if np.isfinite(a).any() else 0.0), u.ev)
        df[f"ctx_{band}"] = [float(_along(r, p).mean()) for p in paths]
        prov.append({"feature": f"ctx_{band}", "group": "context", "source": f"{band} (unit percentile)"})
    if mc is not None and len(mc) and "uid" in mc:
        m = mc.set_index("uid")
        for c in ("mc_a_detection", "mc_a_pos_var_px", "mc_a_fragmentation", "mc_b_occupancy", "mc_b_branch_freq"):
            if c in m:
                df[c] = df.uid.map(m[c]).to_numpy() if "uid" in df else np.nan
        df["has_mc"] = df.uid.isin(m.index).astype(int) if "uid" in df else 0
    else:
        df["has_mc"] = 0
    df["unit"] = u.key
    return df, pd.DataFrame(prov).drop_duplicates("feature")


def groups(df: pd.DataFrame) -> dict[str, list[str]]:
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in EXCLUDE]
    g = {"geometry": [c for c in num if c in ("length_km", "chord_km", "sinuosity", "strike_cos2", "strike_sin2",
                                              "curvature_deg_per_km", "straightness", "n_members", "endpoint_connectivity",
                                              "orient_persistence")],
         "tracing": [c for c in num if c.startswith(("e_", "src_")) or c in ("beam_score", "align_mean", "orient_conf_mean",
                                                                             "turn_mean_deg", "gap_frac", "ep_support",
                                                                             "jn_support", "is_branch", "confidence_lf")],
         "geophysics": [c for c in num if c.startswith(("fn_", "side_magnetic", "side_gravity", "gradcoh_magnetic",
                                                        "gradcoh_gravity", "orient_consistency_", "ev_mag", "ev_grav",
                                                        "ev_cond", "ev_base")) or c == "n_modal_contrast"],
         "spectral": [c for c in num if c.startswith("spec_")],
         "crossmodal": [c for c in num if c.startswith("xmod_") or c.startswith("ev_eda")],
         "terrain_dem": [c for c in num if c.startswith(("topo_", "dem_", "ev_terr", "normal_asymmetry"))],
         "neural_m9": [c for c in num if c.startswith(("m9_", "m9nc_"))],
         "neural_legacy": [c for c in num if c.startswith(("b0_", "b1_", "neural_"))],
         "network": [c for c in num if c.startswith(("dist_known_", "cont_angle", "parallel_known"))],
         "context": [c for c in num if c.startswith("ctx_")],
         "montecarlo": [c for c in num if c.startswith("mc_") or c == "has_mc"]}
    return {k: v for k, v in g.items() if v}


def feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    """Ablation sets of the GBM benchmark (identical across libraries)."""
    g = groups(df)
    core = g.get("geometry", []) + g.get("tracing", []) + g.get("geophysics", []) + g.get("crossmodal", []) + \
        g.get("terrain_dem", []) + g.get("network", []) + g.get("context", [])
    out = {"physics": core, "physics_m9": core + g.get("neural_m9", []),
           "all": core + g.get("neural_m9", []) + g.get("neural_legacy", []) + g.get("spectral", [])}
    if g.get("montecarlo"):
        out["all_mc"] = out["all"] + g["montecarlo"]
    return {k: list(dict.fromkeys(v)) for k, v in out.items() if v}
