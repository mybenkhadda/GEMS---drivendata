"""Mandatory stage products (core scientific tables), produced independently of figure rendering.

gde.1-r1 declared ``tables/band_characterization.csv`` (e02) and ``tables/fault_geometry.csv`` (e03) in the stage
output contract but never wrote them, so both stages FAILED after rendering all their figures and blocked every
downstream stage. Each producer here:
* computes only what the inputs support (no invented measurement; unavailable values stay NaN and are named in
  ``measured_in`` / ``limitations``),
* validates its own schema before writing (``validate``), and
* writes atomically through the workspace, so an interrupted write never leaves a valid-looking partial file.

Columns are documented in ``BAND_COLUMNS`` / ``GEOMETRY_COLUMNS`` (also in docs/GEMS_EXHAUSTIVE_EDA_HANDOFF.md).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

from .data import PRIMARY, PROVENANCE

# provider derivatives and their parent field (as used by F008)
PARENT = {"tmi_hg": "tmi", "tmi_vg": "tmi", "iso_grav_anom_hg": "iso_grav_anom", "iso_grav_anom_vg": "iso_grav_anom",
          "iso_grav_anom_slope": "iso_grav_anom", "det_elev_slope": "det_elev"}

BAND_COLUMNS = (
    "band", "band_index", "modality", "provenance_class", "units", "description", "is_primary_of_modality",
    "is_derived", "parent_band", "n_footprint_px", "n_valid_px", "valid_frac_footprint", "p01", "p05", "p25", "p50",
    "p75", "p95", "p99", "mean", "std", "robust_scale_mad", "iqr", "dynamic_range_p01_p99", "robust_cv",
    "local_regional_std_ratio", "n_sample_ratio", "corr_length_km", "anisotropy_ratio_2km", "anisotropy_max_dir_deg",
    "short_wavelength_power_share", "spearman_vs_recomputed_from_parent", "n_unique_sample", "measured_in",
    "limitations")
GEOMETRY_COLUMNS = (
    "seg_id", "system_id", "length_km", "chord_km", "sinuosity", "strike_deg", "linearity", "curv_mean_deg_km",
    "curv_p90_deg_km", "n_px", "start_node", "end_node", "start_degree", "end_degree", "n_junction_ends",
    "n_free_ends", "is_loop", "touches_edge", "is_short_lt_2km", "aligned_gap_partners", "system_n_segments",
    "system_length_km", "system_kind", "region", "region_name", "domain", "trace_density_km_per_100km2",
    "r_mid", "c_mid", "x_mid_utm", "y_mid_utm", "in_footprint", "crs")


class ProductInvalid(ValueError):
    """A mandatory product violates its schema: the stage must FAIL (never be written as if valid)."""


def validate(df: pd.DataFrame, columns, rules: dict, name: str) -> None:
    miss = [c for c in columns if c not in df.columns]
    if miss:
        raise ProductInvalid(f"{name}: columns {miss} absent")
    bad = [k for k, ok in rules.items() if not bool(ok(df))]
    if bad:
        raise ProductInvalid(f"{name}: schema rules violated {bad}")


# =============================================================================== e02: band characterisation
def _local_regional_ratio(a: np.ndarray, rr: np.ndarray, cc: np.ndarray) -> np.ndarray:
    """std in 1 km windows / std in 10 km windows at sampled pixels (the F004 measurement)."""
    f = np.nan_to_num(a, nan=float(np.nanmedian(a)))

    def lstd(w):
        m1 = ndimage.uniform_filter(f, w)
        return np.sqrt(np.maximum(ndimage.uniform_filter(f * f, w) - m1 * m1, 0))
    return lstd(11)[rr, cc] / (lstd(101)[rr, cc] + 1e-12)


def band_characterization(E) -> pd.DataFrame:
    """Per band: identity, provenance, units, coverage, robust distribution, variability and the local / regional
    variance ratio, all measured on the native footprint. Spatial-scale columns (correlation length, anisotropy,
    spectra, derivative redundancy) are left NaN here and filled by ``enrich_band_characterization`` from the
    figure data of F005 - F008 when those figures ran."""
    D = E.D
    meta = E.table("audit/band_metadata.csv")
    meta = meta.set_index("band") if len(meta) and "band" in meta else pd.DataFrame()
    fp = D.valid
    rng = np.random.default_rng(E.seed)
    rr, cc = np.nonzero(fp)
    pick = rng.choice(len(rr), min(200_000, len(rr)), replace=False) if len(rr) else np.zeros(0, int)
    primary = set(PRIMARY.values())
    rows = []
    for i, n in enumerate(D.stack.names, start=1):
        a = np.asarray(D.band(n), np.float64)
        v = a[fp & np.isfinite(a)]
        q = np.quantile(v, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]) if v.size else np.full(7, np.nan)
        mad = float(1.4826 * np.median(np.abs(v - q[3]))) if v.size else np.nan
        ratio = _local_regional_ratio(a, rr[pick], cc[pick]) if v.size and len(pick) else np.zeros(0)
        m = meta.loc[n] if len(meta) and n in meta.index else {}
        rows.append({
            "band": n, "band_index": int(m.get("band_index", i)) if len(meta) and n in meta.index else i,
            "modality": D.modality(n), "provenance_class": PROVENANCE.get(n, "unknown (verify)"),
            "units": str(m.get("units", "")) if len(meta) and n in meta.index else "not available (e01 metadata absent)",
            "description": str(m.get("description", "")) if len(meta) and n in meta.index else "",
            "is_primary_of_modality": n in primary, "is_derived": n in PARENT, "parent_band": PARENT.get(n, ""),
            "n_footprint_px": int(fp.sum()), "n_valid_px": int(v.size),
            "valid_frac_footprint": float(v.size / max(int(fp.sum()), 1)),
            "p01": q[0], "p05": q[1], "p25": q[2], "p50": q[3], "p75": q[4], "p95": q[5], "p99": q[6],
            "mean": float(v.mean()) if v.size else np.nan, "std": float(v.std()) if v.size else np.nan,
            "robust_scale_mad": mad, "iqr": float(q[4] - q[2]), "dynamic_range_p01_p99": float(q[6] - q[0]),
            "robust_cv": float(mad / abs(q[3])) if v.size and abs(q[3]) > 1e-12 else np.nan,
            "local_regional_std_ratio": float(np.nanmedian(ratio)) if ratio.size else np.nan,
            "n_sample_ratio": int(ratio.size),
            "corr_length_km": np.nan, "anisotropy_ratio_2km": np.nan, "anisotropy_max_dir_deg": np.nan,
            "short_wavelength_power_share": np.nan, "spearman_vs_recomputed_from_parent": np.nan,
            "measured_in": "e02 core (native 100 m footprint)",
            "n_unique_sample": float(m.get("n_unique_sample", np.nan)) if len(meta) and n in meta.index else np.nan})
        del a, v
    T = pd.DataFrame(rows)
    T["limitations"] = [limitations(r) for r in T.to_dict("records")]
    return T


def limitations(r: dict) -> str:
    """Data-quality caveats derived from the measured row only."""
    out = []
    if r.get("valid_frac_footprint", 1) < 0.99:
        out.append(f"coverage {r['valid_frac_footprint']:.1%} of the footprint")
    pc = r.get("provenance_class", "")
    if pc == "model_product":
        out.append("model product (inverted / interpolated), not a direct measurement")
    elif pc == "catalog_density":
        out.append("event-catalog kernel density: regional context, not trace-scale")
    elif pc.startswith("unknown"):
        out.append("provenance unknown: verify the band description")
    if r.get("is_derived"):
        out.append(f"derivative of {r.get('parent_band')}: same evidence family as its parent")
    cl = r.get("corr_length_km")
    if cl is not None and np.isfinite(cl) and cl > 1.0:
        out.append(f"effective support ~{cl:.1f} km (> 1 km): corridor context only")
    lr = r.get("local_regional_std_ratio")
    if lr is not None and np.isfinite(lr) and lr < 0.1:
        out.append(f"little km-scale variance (1 km / 10 km std ratio {lr:.3f})")
    nu = r.get("n_unique_sample")
    if nu is not None and np.isfinite(nu) and nu < 1000:
        out.append(f"only {int(nu)} distinct values in the inventory sample (quantised / interpolated)")
    rho = r.get("spearman_vs_recomputed_from_parent")
    if rho is not None and np.isfinite(rho) and rho >= 0.9:
        out.append(f"redundant with its recomputed parent derivative (rho {rho:.2f})")
    return "; ".join(out) or "none identified by these measurements"


BAND_RULES = {"unique_band": lambda d: d.band.is_unique,
              "fractions_in_0_1": lambda d: d.valid_frac_footprint.between(0, 1).all(),
              "ordered_quantiles": lambda d: ((d.p01 <= d.p50) & (d.p50 <= d.p99) | d.p50.isna()).all(),
              "non_negative_spread": lambda d: ((d.robust_scale_mad >= 0) | d.robust_scale_mad.isna()).all(),
              "n_valid_le_footprint": lambda d: (d.n_valid_px <= d.n_footprint_px).all()}


def enrich_band_characterization(E, T: pd.DataFrame) -> pd.DataFrame:
    """Fill the spatial-scale columns from the persisted figure data of F005 - F008 of THIS stage (only for the bands
    those figures measured); everything else stays NaN. Returns the enriched table (also recomputes limitations)."""
    T = T.copy()
    src = []
    F5 = E.table("figures/data/F005_autocorrelation.csv")
    if len(F5) and {"band", "corr_length_km"} <= set(F5):
        T["corr_length_km"] = T.band.map(F5.groupby("band").corr_length_km.median())
        src.append("F005")
    F6 = E.table("figures/data/F006_anisotropy.csv")
    if len(F6) and "modality" in F6:
        b6 = F6.assign(band=F6.modality.map(PRIMARY)).set_index("band")
        T["anisotropy_ratio_2km"] = T.band.map(b6.anisotropy_ratio)
        T["anisotropy_max_dir_deg"] = T.band.map(b6.max_dir_deg)
        src.append("F006")
    F7 = E.table("figures/data/F007_power_spectra.csv")
    if len(F7) and "modality" in F7:
        T["short_wavelength_power_share"] = T.band.map(
            F7.assign(band=F7.modality.map(PRIMARY)).set_index("band").short_wavelength_power_share)
        src.append("F007 (200 m)")
    F8 = E.table("figures/data/F008_raw_vs_derived.csv")
    if len(F8) and {"derived", "spearman"} <= set(F8):
        T["spearman_vs_recomputed_from_parent"] = T.band.map(F8.set_index("derived").spearman)
        src.append("F008")
    if src:
        T["measured_in"] = "e02 core (native 100 m footprint); " + ", ".join(src)
    T["limitations"] = [limitations(r) for r in T.to_dict("records")]
    return T


def write_band_characterization(E, T: pd.DataFrame) -> pd.DataFrame:
    out = T.reindex(columns=list(BAND_COLUMNS))
    validate(out, BAND_COLUMNS, BAND_RULES, "tables/band_characterization.csv")
    E.ws.save_table(out, "tables/band_characterization.csv")
    return out


# =============================================================================== e03: fault geometry
GEOMETRY_RULES = {"unique_seg_id": lambda d: d.seg_id.is_unique,
                  "positive_length": lambda d: (d.length_km > 0).all(),
                  "chord_le_length": lambda d: (d.chord_km <= d.length_km + 1e-6).all(),
                  "strike_in_0_180": lambda d: (d.strike_deg.between(0, 180, inclusive="left") | d.strike_deg.isna()).all(),
                  "linearity_in_0_1": lambda d: (d.linearity.between(0, 1) | d.linearity.isna()).all(),
                  "valid_system_id": lambda d: (d.system_id > 0).all()}


def fault_geometry(E) -> pd.DataFrame:
    """One row per traced mapped-fault branch: geometry (km, degrees clockwise from north, axial 0-180), topology
    (node degrees, junction / free ends, loops), fragmentation (short branch, aligned gaps to other systems), system
    context (connected component of the mapped raster), local mapped-trace density (10 km kernel), label-free domain,
    region and UTM midpoint. Built from ``gems_eda.morphology`` (the same network every figure of part B uses)."""
    from gems_eda.morphology import component_diagnostics, endpoint_gaps, region_name, trace_density

    D = E.D
    S = D.net.segments
    if not len(S):
        return pd.DataFrame(columns=list(GEOMETRY_COLUMNS))
    C = component_diagnostics(D.net, D.grid, 2.0)
    G = endpoint_gaps(D.net, D.grid, 5.0, 25.0)
    gaps = pd.concat([G.seg_a, G.seg_b]).value_counts() if len(G) else pd.Series(dtype=int)
    dens = trace_density(D.net, D.grid, 10.0)
    dom = E.domains()
    r, c = S.r_mid.to_numpy(), S.c_mid.to_numpy()
    T = pd.DataFrame({
        "seg_id": S.seg_id.astype(int), "system_id": S.component.astype(int), "length_km": S.length_km,
        "chord_km": S.chord_km, "sinuosity": S.sinuosity, "strike_deg": S.azimuth, "linearity": S.linearity,
        "curv_mean_deg_km": S.curv_mean_deg_km, "curv_p90_deg_km": S.get("curv_p90_deg_km", np.nan),
        "n_px": S.n_px.astype(int), "start_node": S.start_node.astype(int), "end_node": S.end_node.astype(int),
        "start_degree": S.start_deg.astype(int), "end_degree": S.end_deg.astype(int),
        "n_junction_ends": S.n_junction_ends.astype(int),
        "n_free_ends": (S.start_deg == 1).astype(int) + (S.end_deg == 1).astype(int),
        "is_loop": S.is_loop.astype(bool), "touches_edge": S.touches_edge.astype(bool),
        "is_short_lt_2km": S.length_km < 2.0, "aligned_gap_partners": S.seg_id.map(gaps).fillna(0).astype(int),
        "region": S.region.astype(int), "region_name": [region_name(int(k), D.region_grid) for k in S.region],
        "domain": dom[r, c].astype(int), "trace_density_km_per_100km2": dens[r, c].astype(float),
        "r_mid": r.astype(int), "c_mid": c.astype(int), "x_mid_utm": S.x_mid, "y_mid_utm": S.y_mid,
        "in_footprint": D.valid[r, c], "crs": "EPSG:32611"})
    if len(C):
        sysd = C.set_index("component")
        T["system_n_segments"] = T.system_id.map(sysd.n_segments).astype("Int64")
        T["system_length_km"] = T.system_id.map(sysd.length_km)
        T["system_kind"] = T.system_id.map(sysd.kind).fillna("unknown")
    else:
        T["system_n_segments"], T["system_length_km"], T["system_kind"] = pd.NA, np.nan, "unknown"
    return T.reindex(columns=list(GEOMETRY_COLUMNS))


def write_fault_geometry(E, T: pd.DataFrame) -> pd.DataFrame:
    validate(T, GEOMETRY_COLUMNS, GEOMETRY_RULES if len(T) else {}, "tables/fault_geometry.csv")
    E.ws.save_table(T, "tables/fault_geometry.csv")
    return T
