"""Rule-based fault archetypes from calibrated segment evidence and geometry."""
from __future__ import annotations

import numpy as np
import pandas as pd

ARCHETYPES = ("multi_modal", "strong_magnetic", "gravity_related", "terrain_visible",
              "weak_ambiguous", "intersection", "curved")

ARCHETYPE_LABELS = {
    "multi_modal": "Multi-modal faults",
    "strong_magnetic": "Strong magnetic faults",
    "gravity_related": "Gravity-related faults",
    "terrain_visible": "Terrain-visible faults",
    "weak_ambiguous": "Weak / ambiguous faults",
    "intersection": "Fault intersections",
    "curved": "Curved structures",
}

# At most four geophysical signals per gallery example ("@fused" = lineament map from section 7).
ARCHETYPE_SIGNALS = {
    "multi_modal": ("rtp", "iso_grav_anom", "det_elev", "cond_surf"),
    "strong_magnetic": ("rtp", "tmi_hg", "magnetic:tilt", "det_elev"),
    "gravity_related": ("iso_grav_anom", "iso_grav_anom_hg", "depth_to_base_surf", "rtp"),
    "terrain_visible": ("det_elev", "det_elev_slope", "rtp", "iso_grav_anom"),
    "weak_ambiguous": ("rtp", "iso_grav_anom", "det_elev", "cond_surf"),
    "intersection": ("rtp", "iso_grav_anom_hg", "det_elev", "@fused"),
    "curved": ("rtp", "magnetic:tilt", "det_elev", "@fused"),
}


def assign_archetypes(segments: pd.DataFrame, seg_ev: pd.DataFrame, strong: float, weak: float,
                      min_len_km: float) -> pd.DataFrame:
    """Flag each segment with evidence archetypes (exclusive primary label) and geometric archetypes."""
    df = segments.merge(seg_ev, on="seg_id", how="inner")
    g = {k: df.get(f"ev_{k}", pd.Series(np.nan, index=df.index)).fillna(0)
         for k in ("magnetic", "gravity", "terrain", "basement", "conductivity")}
    pot = pd.concat([g["gravity"], g["basement"]], axis=1).max(axis=1)
    other_mag = pd.concat([pot, g["terrain"], g["conductivity"]], axis=1).max(axis=1)
    other_grav = pd.concat([g["magnetic"], g["terrain"]], axis=1).max(axis=1)
    other_ter = pd.concat([g["magnetic"], pot], axis=1).max(axis=1)
    allmax = pd.concat(list(g.values()), axis=1).max(axis=1)
    top3 = pd.concat(list(g.values()), axis=1).apply(lambda r: np.sort(r.to_numpy())[-3:].mean(), axis=1)

    df["score_multi_modal"] = np.where(df.n_strong_groups >= 3, top3, np.nan)
    df["score_strong_magnetic"] = np.where((g["magnetic"] >= strong) & (other_mag < 0.75), g["magnetic"] - other_mag, np.nan)
    df["score_gravity_related"] = np.where((pot >= strong) & (g["magnetic"] < 0.75), pot - other_grav, np.nan)
    df["score_terrain_visible"] = np.where((g["terrain"] >= strong) & (pd.concat([g["magnetic"], pot], axis=1).max(axis=1) < 0.75),
                                           g["terrain"] - other_ter, np.nan)
    df["score_weak_ambiguous"] = np.where(allmax < weak, 1 - allmax, np.nan)
    df["score_intersection"] = np.where(df.n_junction_ends >= 1, df.n_junction_ends + np.log1p(df.length_km) / 10, np.nan)
    curved = (df.sinuosity >= 1.15) | (df.curv_mean_deg_km >= df.curv_mean_deg_km.quantile(0.9))
    df["score_curved"] = np.where(curved & (df.length_km >= 2 * min_len_km), df.sinuosity * np.log1p(df.length_km), np.nan)

    order = ["multi_modal", "strong_magnetic", "gravity_related", "terrain_visible", "weak_ambiguous"]
    primary = pd.Series("moderate_mixed", index=df.index)
    for name in reversed(order):
        primary = primary.mask(df[f"score_{name}"].notna(), name)
    df["primary_archetype"] = primary
    df["geometry_class"] = np.select(
        [df.n_junction_ends >= 1, curved, df.length_km < 2 * min_len_km], ["intersection", "curved", "short"], "straight")
    return df


def archetype_counts(arch: pd.DataFrame) -> pd.DataFrame:
    a = arch.groupby("primary_archetype").agg(n_segments=("seg_id", "size"), trace_km=("length_km", "sum"))
    a["share_km"] = a.trace_km / a.trace_km.sum()
    return a.sort_values("trace_km", ascending=False).reset_index()


def evidence_by_geometry(arch: pd.DataFrame) -> pd.DataFrame:
    ev_cols = [c for c in arch.columns if c.startswith("ev_")]
    return arch.groupby("geometry_class")[ev_cols + ["max_ev", "n_strong_groups"]].median().reset_index()


def pick_examples(arch: pd.DataFrame, name: str, n: int, min_len_km: float, min_sep_px: float) -> pd.DataFrame:
    """Top-scoring, spatially separated, sufficiently long, non-edge examples of an archetype."""
    s = arch[arch[f"score_{name}"].notna() & (arch.length_km >= min_len_km) & ~arch.touches_edge]
    s = s.sort_values(f"score_{name}", ascending=False)
    chosen = []
    for _, row in s.iterrows():
        if all(np.hypot(row.r_mid - c.r_mid, row.c_mid - c.c_mid) > min_sep_px for c in chosen):
            chosen.append(row)
        if len(chosen) == n:
            break
    return pd.DataFrame(chosen)
