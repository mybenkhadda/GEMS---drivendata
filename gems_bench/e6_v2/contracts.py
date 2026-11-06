"""Raw-data contract and the feature schema of the three encoder branches.

The channel lists below are the schema. Their hash is part of every dataset / checkpoint identity, so a
change of any list invalidates the datasets and every model trained on them.
"""
from __future__ import annotations

from .config import stable_hash

ALL_GEODAWN_BANDS = ("mag_anom", "rtp", "tmi", "tmi_hg", "tmi_vg", "tc", "iso_grav_anom", "iso_grav_anom_hg",
                     "iso_grav_anom_vg", "iso_grav_anom_slope", "det_elev", "det_elev_slope", "depth_to_base_surf",
                     "cond_surf", "geod_2ndinv", "geod_shearrate", "geod_dilaterate", "deq_n100a15", "ieq_n100a15")

# kind: scalar = invariant under 90-degree rotation and flips; axial_cos / axial_sin = (cos 2t, sin 2t) pair
SCALAR, AXIAL_COS, AXIAL_SIN = "scalar", "axial_cos", "axial_sin"


def required_bands(cfg: dict) -> list[str]:
    t, p = cfg["terrain"], cfg["potential"]
    bands = [t["elevation_band"], t["slope_band"]] + [f["band"] for f in p["fields"].values()]
    return list(dict.fromkeys(bands + list(p["subsurface_bands"])))


def terrain_channels(cfg: dict) -> list[str]:
    t = cfg["terrain"]
    ch = ["elev", "slope_raw"]
    ch += [f"grad_{s}m" for s in t["derivative_scales_m"]]
    ch += [f"log_{s}m" for s in t["derivative_scales_m"]]
    ch += [f"relief_{w}m" for w in t["relief_windows_m"]]
    ch += [f"tpi_{r}m" for r in t["tpi_radii_m"]]
    ch += [f"dog_{a}_{b}m" for a, b in t["dog_bands_m"]]
    return ch


def field_channels(cfg: dict, prefix: str) -> list[str]:
    hs = cfg["potential"]["continuation_heights_m"]
    ch = [f"{prefix}_field", f"{prefix}_vd", f"{prefix}_thd", f"{prefix}_tilt", f"{prefix}_as"]
    ch += [f"{prefix}_uc{h}" for h in hs]
    ch += [f"{prefix}_res{h}" for h in hs]
    ch += [f"{prefix}_thd_uc{h}" for h in hs]
    return ch


def subsurface_channels(cfg: dict) -> list[str]:
    return [f"raw_{b}" for b in cfg["potential"]["subsurface_bands"]]


DEM_CHANNELS = ("edge_energy_p50", "edge_energy_p90", "edge_energy_max", "edge_anisotropy", "coherence_p90_5m",
                "coherence_p90_20m", "orientation_cos2", "orientation_sin2", "orientation_R", "slope_p50",
                "slope_p90", "slope_std", "curvature_min", "curvature_max", "tpi_50m", "tpi_200m", "scarp_strength",
                "crest_response", "toe_response", "edge_fraction", "roughness", "roughness_p90")


def channel_kind(name: str) -> str:
    if name == "orientation_cos2":
        return AXIAL_COS
    if name == "orientation_sin2":
        return AXIAL_SIN
    return SCALAR


def schema(cfg: dict) -> dict:
    """Full input schema: {group: [channel, ...]} plus axial pairs per group."""
    groups = {
        "terrain": terrain_channels(cfg),
        "field": field_channels(cfg, "mag") + field_channels(cfg, "grav") + subsurface_channels(cfg),
        "dem": list(DEM_CHANNELS),
    }
    axial = {g: [(ch.index("orientation_cos2"), ch.index("orientation_sin2"))] if "orientation_cos2" in ch else []
             for g, ch in groups.items()}
    return {"groups": groups, "axial_pairs": axial, "context_groups": ["terrain", "field"],
            "coverage": "dem_coverage", "pool": cfg["model"]["context_pool"]}


def schema_hash(cfg: dict) -> str:
    return stable_hash(schema(cfg))


def check_band_contract(cfg: dict, names: list[str]) -> list[str]:
    """Blocker messages for the GeoDAWN band list read from the raster header."""
    out = []
    missing = [b for b in required_bands(cfg) if b not in names]
    if missing:
        out.append(f"GeoDAWN raster lacks required bands {missing} (has {names})")
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        out.append(f"GeoDAWN raster has duplicate band names {dup}")
    clash = set(cfg["potential"]["subsurface_bands"]) & set(cfg["potential"]["excluded_orientation_dependent"])
    if clash:
        out.append(f"config lists {sorted(clash)} both as subsurface input and as orientation-dependent")
    return out
