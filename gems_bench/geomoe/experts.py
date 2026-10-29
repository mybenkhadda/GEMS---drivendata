"""Experiment C: four geological expert families, label-free source layers, cross-fitted score normalisation.

An expert is an evidence family, not a network:
    E1_learned          B0 / B1 honest CNN maps, the frozen W2 held-out map (recognition of mapped-fault patterns)
    E2_potential        magnetic / gravity horizontal gradients, their strike agreement, two-scale persistence
    E3_structural       G02 continuity (G02-C), G02 apparent offset (G02-A), G01 wavelet persistence
    E4_terrain_network  permitted-known-network continuation (tip distance x along-strike cosine) and cached DEM
                        morphology (M7 100 m mosaic) when it exists

Every source layer is label-free. Continuous maps are ranked inside the base unit's evaluation mask (``stack.rank_in``,
the W2 convention), NaN where the source does not exist. A missing source is NaN and stays NaN: an expert whose members
are all missing is UNAVAILABLE for that candidate (never zero evidence). DEM absence is explicit (``dem_observed``).

Expert scores (segment means of the member layers) are not comparable across families, so each is mapped to [0, 1] by
an ECDF fitted ONLY on the candidates of the training zones of the same scenario (``fit_ecdf`` / ``apply_ecdf``); the
deployed construction uses exactly the same fit (zone k <- knots fitted without zone k).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FAMILIES = ("E1_learned", "E2_potential", "E3_structural", "E4_terrain_network")
SHORT = {"E1_learned": "E1", "E2_potential": "E2", "E3_structural": "E3", "E4_terrain_network": "E4"}
SOURCE_FAMILY = {"cnn_b0": "E1_learned", "cnn_b1": "E1_learned", "w2": "E1_learned", "fusion": "E1_learned",
                 "mag_hgrad": "E2_potential", "grav_hgrad": "E2_potential", "mg_cross": "E2_potential",
                 "pf_persistence": "E2_potential", "g02c": "E3_structural", "g02a": "E3_structural",
                 "g01_persistence": "E3_structural", "network": "E4_terrain_network", "dem": "E4_terrain_network",
                 "tips": "E4_terrain_network"}
# pool generation order (dedup priority) and the label-free descriptor layers kept per candidate
POOL_SOURCES = ("w2", "fusion", "cnn_b0", "cnn_b1", "g02c", "g02a", "g01_persistence", "mag_hgrad", "grav_hgrad",
                "mg_cross", "pf_persistence", "dem", "network")
DESCRIPTOR_LAYERS = ("cnn_disagree", "mg_orient_agree", "known_dist_km", "known_density", "tip_dist_km",
                     "tip_along_cos", "dem_observed", "network_w2ctx")
MIN_FINITE_SHARE = 0.5                  # a member layer counts on a segment when >= this share of it is finite


# =============================================================================== pure: layers
def network_evidence(tip_dist_km: np.ndarray, tip_cos: np.ndarray, scale_km: float) -> np.ndarray:
    """Continuation evidence of the permitted known network in [0, 1]: straight ahead of a tip and close to it."""
    d = np.asarray(tip_dist_km, np.float32)
    c = np.clip(np.nan_to_num(np.asarray(tip_cos, np.float32), nan=0.0), 0.0, 1.0)
    return (c * np.exp(-d / float(scale_km))).astype(np.float32)


def mg_cross(mag_rank: np.ndarray, grav_rank: np.ndarray, agree: np.ndarray) -> np.ndarray:
    """Cross-modal agreement: both gradients high AND strikes parallel (geometric mean x doubled-angle agreement)."""
    return (np.sqrt(np.clip(mag_rank, 0, 1) * np.clip(grav_rank, 0, 1)) * np.asarray(agree, np.float32)).astype(
        np.float32)


def persistence(r1: list[np.ndarray], r3: list[np.ndarray]) -> np.ndarray | None:
    """Scale persistence: per field min(rank at 1 px, rank at 3 px), averaged over the fields present."""
    parts = [np.fmin(a, b) for a, b in zip(r1, r3) if a is not None and b is not None]
    if not parts:
        return None
    with np.errstate(invalid="ignore"):
        return np.nanmean(np.stack(parts), 0).astype(np.float32) if len(parts) > 1 else parts[0]


# =============================================================================== pure: normalisation
def fit_ecdf(v: np.ndarray, knots: int = 101) -> np.ndarray | None:
    """Quantile knots of the finite values (None when fewer than 2 finite values)."""
    x = np.asarray(v, float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return None
    return np.quantile(x, np.linspace(0.0, 1.0, int(knots)))


def apply_ecdf(knots: np.ndarray | None, v: np.ndarray) -> np.ndarray:
    """ECDF value in [0, 1] (piecewise linear between the knots); NaN stays NaN; no knots -> all NaN."""
    x = np.asarray(v, float)
    out = np.full(x.shape, np.nan)
    if knots is None:
        return out
    k = np.asarray(knots, float)
    ok = np.isfinite(x)
    lev = np.linspace(0.0, 1.0, len(k))
    ku = np.unique(k)                                  # flat stretches of tied knots map to their mean level
    lv = np.array([lev[k == a].mean() for a in ku])
    out[ok] = np.interp(x[ok], ku, lv, left=0.0, right=1.0) if len(ku) > 1 else 0.5
    return out


def fit_normalizer(T: pd.DataFrame, cols: list[str], knots: int) -> dict:
    """{col: knots} fitted on the given (training) rows only."""
    return {c: (fit_ecdf(T[c].to_numpy(), knots) if c in T else None) for c in cols}


def normalize(T: pd.DataFrame, norm: dict, suffix: str = "") -> pd.DataFrame:
    return pd.DataFrame({f"{c}{suffix}": apply_ecdf(k, T[c].to_numpy() if c in T else np.full(len(T), np.nan))
                         for c, k in norm.items()}, index=T.index)


def norm_summary(norm: dict) -> dict:
    """JSON-able record of a fit (knot count, 5 / 50 / 95 % knots), absent fits named."""
    out = {}
    for c, k in norm.items():
        if k is None:
            out[c] = {"status": "UNAVAILABLE (no finite training value)"}
        else:
            n = len(k)
            out[c] = {"status": "FITTED", "knots": n, "q05": float(k[int(0.05 * (n - 1))]),
                      "q50": float(k[(n - 1) // 2]), "q95": float(k[int(0.95 * (n - 1))])}
    return out


# =============================================================================== pure: expert scores
def expert_raw(T: pd.DataFrame, experts: dict) -> pd.DataFrame:
    """Raw expert score per candidate = mean of its available member layers (``mean_<layer>`` where
    ``fin_<layer>`` >= MIN_FINITE_SHARE); NaN when no member is available. Columns ``raw_<E>``, ``avail_<E>``,
    ``members_<E>`` (number of available members)."""
    out = {}
    for fam, members in experts.items():
        vals, n = [], np.zeros(len(T), int)
        for m in members:
            if f"mean_{m}" not in T:
                continue
            v = T[f"mean_{m}"].to_numpy(float)
            ok = np.isfinite(v) & (T.get(f"fin_{m}", pd.Series(1.0, index=T.index)).to_numpy(float) >= MIN_FINITE_SHARE)
            vals.append(np.where(ok, v, np.nan))
            n += ok
        with np.errstate(invalid="ignore"):
            raw = np.nanmean(np.stack(vals), 0) if vals else np.full(len(T), np.nan)
        out[f"raw_{fam}"] = raw
        out[f"avail_{fam}"] = np.isfinite(raw)
        out[f"members_{fam}"] = n
    return pd.DataFrame(out, index=T.index)


def availability_table(T: pd.DataFrame, experts: dict, by: list[str]) -> pd.DataFrame:
    """Share of candidates where each expert / member is available, per ``by`` group (no candidate is dropped)."""
    rows = []
    for key, g in T.groupby(by):
        key = key if isinstance(key, tuple) else (key,)
        base = dict(zip(by, key))
        for fam, members in experts.items():
            a = g.get(f"avail_{fam}")
            rows.append({**base, "expert": fam, "member": "*", "n_candidates": len(g),
                         "available_share": float(a.mean()) if a is not None else 0.0})
            for m in members:
                f = g.get(f"fin_{m}")
                rows.append({**base, "expert": fam, "member": m, "n_candidates": len(g),
                             "available_share": float((f >= MIN_FINITE_SHARE).mean()) if f is not None else 0.0,
                             "status": "present" if f is not None and (f >= MIN_FINITE_SHARE).any() else "ABSENT"})
    return pd.DataFrame(rows)


# =============================================================================== Colab side: layers of one unit
def unit_layers(G, su) -> tuple[dict, dict]:
    """(layers, provenance) of one scenario unit, cropped to its base unit; every map label-free."""
    from ..p6 import stack as ST

    bst, b = G.bst, su.base
    ev = b.ev
    raw = bst.raw(b)
    L, prov = {}, dict(raw.get("_prov") or {})
    L["cnn_b0"], L["cnn_b1"] = ST.rank_in(raw["cnn_b0"], ev), ST.rank_in(raw["cnn_b1"], ev)
    L["cnn_disagree"] = np.asarray(raw["cnn_disagree"], np.float32)
    L["fusion"] = np.where(ev, bst.fusion(b), np.nan).astype(np.float32)
    try:
        L["w2"] = np.where(ev, bst.stack_rank(b, bst.w2_map(b)), np.nan).astype(np.float32)
        prov["w2"] = f"stack/maps/{b.task}_{b.zone}.npy (cross-fitted held-out)"
    except (FileNotFoundError, ValueError) as e:
        prov["w2"] = f"absent: {type(e).__name__}: {e}"
    r1 = {}
    for k in ("mag_hgrad", "grav_hgrad"):
        if k in raw:
            r1[k] = L[k] = ST.rank_in(raw[k], ev)
    if "mg_orient_agree" in raw:
        L["mg_orient_agree"] = np.asarray(raw["mg_orient_agree"], np.float32)
        if len(r1) == 2:
            L["mg_cross"] = np.where(ev, mg_cross(r1["mag_hgrad"], r1["grav_hgrad"], raw["mg_orient_agree"]), np.nan)
    s3 = G.pf_scale2()
    r3 = {k: ST.rank_in(np.asarray(s3[k])[b.sl], ev) for k in r1 if k in s3}
    p = persistence([r1.get(k) for k in r1], [r3.get(k) for k in r1])
    if p is not None:
        L["pf_persistence"] = p
    for k in ("g02c", "g02a", "g01_persistence"):
        if k in raw and np.isfinite(raw[k][ev]).any():
            L[k] = ST.rank_in(raw[k], ev)
    if su.scenario == "V2":
        from .scenarios import v2_full_known

        sc = G.p6.cfg["stack"]
        kn = ST.known_features(v2_full_known(G, su), G.ctx.data.valid, 0.1, float(sc["known_cap_km"]),
                               float(sc["known_density_radius_km"]), float(sc["endpoint_max_km"]))
        kn = {k: v[b.sl] for k, v in kn.items()}
        prov["known"] = "faults outside fold k + retained fold-k faults (V2 context; withheld pixels excluded)"
    else:
        kn = {k: raw[k] for k in ("known_dist_km", "known_density", "tip_dist_km", "tip_along_cos") if k in raw}
        prov["known"] = "stack.permitted_known (W2 convention)"
    for k, v in kn.items():
        L[k] = np.where(ev, v, np.nan).astype(np.float32)
    s = float(G.cfg["network_scale_km"])
    if "tip_dist_km" in L and "tip_along_cos" in L:
        L["network"] = np.where(ev, network_evidence(L["tip_dist_km"], L["tip_along_cos"], s), np.nan)
    # context-fairness ablation: the network evidence W2 itself could see (W2's permitted known map; for V2 it
    # excludes the retained fold-k faults, for V1 / V3 it equals ``network``)
    if "tip_dist_km" in raw and "tip_along_cos" in raw:
        L["network_w2ctx"] = np.where(ev, network_evidence(raw["tip_dist_km"], raw["tip_along_cos"], s), np.nan)
        prov["network_w2ctx"] = "W2 permitted-known context (p6.stack.permitted_known of the base unit)"
    dem = G.dem_layers(b)
    if dem is not None:
        L["dem"], L["dem_observed"] = dem["dem"], dem["observed"]
        prov["dem"] = dem["source"]
    else:
        L["dem_observed"] = np.where(ev, 0.0, np.nan).astype(np.float32)
        prov["dem"] = "absent (cached mosaic not found; DEM evidence UNAVAILABLE, never zero evidence)"
    return L, prov
