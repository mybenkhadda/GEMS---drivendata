"""The atlas figure families (numbering = brief section 11). Each function takes the pipeline, reads saved
products only, draws, saves (PNG; charts also SVG) and shows. A missing source yields a limitation panel.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import visualization as vz
from .cache import read_geotiff


def _feat(P, fid):
    A = P.atlas()
    return A.load_feature(fid) if fid in set(A.registry.feature_id) else None


def _mp(P):
    return P.cfg["figures"]["max_display_px"]


def _new(n, ncols=2, h=4.6):
    import matplotlib.pyplot as plt

    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(vz.MAP_W * ncols / 2 * 1.35, h * nrows), squeeze=False)
    return fig, axes.ravel()


def _panel(P, ax, a, kind, title, unit):
    if a is None:
        vz.limitation_panel(ax, title, "source not available in this run")
    else:
        vz.map_panel(ax, a, P.base()["grid"], kind, title, unit, _mp(P))


# 1 ---------------------------------------------------------------------------------------------------------------
def study_area(P):
    b = P.base()
    cov_p = P.paths.d("02_geospatial", "coverage") / "dem_coverage_catalog_tiles.tif"
    cov = read_geotiff(cov_p, 1) if cov_p.exists() else None
    mm_p = P.paths.d("02_geospatial", "masks") / "multimodal_coverage.tif"
    fig, ax = _new(4)
    _panel(P, ax[0], np.where(b["footprint"], 1.0, np.nan), "fraction", "GeoDAWN valid numerical footprint", "valid")
    _panel(P, ax[1], np.where(b["template"], 1.0, np.nan), "fraction", "Official template valid area", "valid")
    _panel(P, ax[2], np.where(b["label_footprint"], b["labels"].astype(float), np.nan), "rank",
           "Fault-label raster: footprint and mapped faults", "fault (1) / none (0)")
    if mm_p.exists():
        _panel(P, ax[3], read_geotiff(mm_p, 1).astype(float), "count", "Multimodal coverage bitmask",
               "1 GeoDAWN + 2 labels + 4 DEM")
    else:
        _panel(P, ax[3], cov, "fraction", "DEM coverage", "share")
    fig.tight_layout()
    P._fig(fig, "overview", "01_study_area_and_coverage")


# 2 ---------------------------------------------------------------------------------------------------------------
def dem_coverage(P):
    c = P.paths.d("02_geospatial", "coverage")
    names = [("dem_coverage_catalog_tiles", "fraction", "Measured DEM valid-pixel share (decimated read)", "share"),
             ("dem_overlap_count", "count", "Overlapping DEM tiles", "tiles"),
             ("dem_project_id", "categorical", "Preferred DEM project", "project id"),
             ("dem_seam_mask", "fraction", "Cells on a project seam", "seam")]
    fig, ax = _new(5)
    for i, (n, kind, t, u) in enumerate(names):
        p = c / f"{n}.tif"
        a = read_geotiff(p, 1).astype(np.float32) if p.exists() else None
        if a is not None and n in ("dem_project_id",):
            a = np.where(a < 0, np.nan, a)
        if a is not None and n in ("dem_overlap_count", "dem_seam_mask"):
            a = np.where(a <= 0, np.nan, a)
        _panel(P, ax[i], a, kind, t, u)
    cat = P.paths.d("02_geospatial", "footprints") / "dem_catalog_nominal_footprint.tif"
    _panel(P, ax[4], np.where(read_geotiff(cat, 1) > 0, 1.0, np.nan) if cat.exists() else None, "fraction",
           "TNM catalog nominal tile rectangles (not coverage)", "listed")
    ax[5].set_axis_off()
    fig.tight_layout()
    P._fig(fig, "overview", "02_dem_acquisition_and_coverage")


# 3 ---------------------------------------------------------------------------------------------------------------
def band_atlas(P):
    reg = pd.read_csv(P.paths.root / "03_features/raw_bands/raw_band_registry.csv")
    n = len(reg)
    fig, ax = _new(n, ncols=4, h=3.2)
    for i, r in enumerate(reg.itertuples()):
        a = _feat(P, r.band_name)
        kind = "terrain" if r.band_name == "det_elev" else "signed" if r.modality in ("magnetic", "gravity") and \
            not r.band_name.endswith(("_hg", "_slope")) else "gradient"
        _panel(P, ax[i], a, kind, f"{r.band_name} ({r.modality})", str(r.unit))
    for j in range(n, len(ax)):
        ax[j].set_axis_off()
    fig.tight_layout()
    P._fig(fig, "overview", "03_raw_band_atlas")


# 4 ---------------------------------------------------------------------------------------------------------------
def fault_network(P):
    import matplotlib.pyplot as plt

    b = P.base()
    tr = pd.read_csv(P.paths.d("04_eda", "tables") / "mapped_fault_traces.csv")
    reg = pd.read_csv(P.paths.d("04_eda", "tables") / "mapped_fault_density_by_region.csv")
    fig = plt.figure(figsize=(12, 5))
    a1 = fig.add_subplot(1, 3, 1)
    sk = read_geotiff(P.paths.d("04_eda", "geological_evidence") / "mapped_fault_skeleton.tif", 1).astype(float)
    vz.map_panel(a1, np.where(b["footprint"], np.where(sk > 0, 1.0, 0.15), np.nan), b["grid"], "rank",
                 "Mapped fault network (skeleton)", "fault", _mp(P), cbar=False)
    a2 = fig.add_subplot(1, 3, 2, projection="polar")
    vz.rose(a2, tr.strike_deg, tr.length_km, title="Strike (length-weighted, axial)")
    a3 = fig.add_subplot(1, 3, 3)
    a3.hist(tr.length_km.clip(upper=tr.length_km.quantile(0.99)), bins=40, color=vz.CATEG[0])
    a3.set_xlabel("trace length (km)")
    a3.set_ylabel("traces")
    a3.set_title("Trace length distribution", loc="left")
    fig.tight_layout()
    P._fig(fig, "fault_discovery", "04_mapped_fault_network_rose", svg=False)
    import matplotlib.pyplot as plt2

    fig, ax = plt2.subplots(figsize=(6, 3.2))
    ax.bar(reg.region, reg.km_per_1000km2, color=vz.CATEG[0])
    ax.set_ylabel("trace km per 1000 km2")
    ax.set_title("Mapped-fault density by subregion (3 x 3)", loc="left")
    P._fig(fig, "fault_discovery", "04b_fault_density_by_region", svg=True)


# 5 / 6 -----------------------------------------------------------------------------------------------------------
def field_maps(P, prefix: str):
    unit = P.cfg["potential"]["magnetic_unit" if prefix == "mag" else "gravity_unit"]
    fig, ax = _new(4)
    _panel(P, ax[0], _feat(P, f"{prefix}_field"), "signed", f"{prefix}: raw field", unit)
    _panel(P, ax[1], _feat(P, f"{prefix}_thd"), "gradient", f"{prefix}: total horizontal derivative", f"{unit}/m")
    _panel(P, ax[2], _feat(P, f"{prefix}_tilt"), "signed", f"{prefix}: tilt derivative", "rad")
    _panel(P, ax[3], _feat(P, f"{prefix}_analytic_signal"), "gradient", f"{prefix}: analytic signal", f"{unit}/m")
    fig.tight_layout()
    P._fig(fig, "geophysics", f"{'05' if prefix == 'mag' else '06'}_{prefix}_thd_tilt")


def spectra(P):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for i, f in enumerate(("magnetic", "gravity")):
        p = P.paths.d("04_eda", "tables") / f"{f}_radial_spectrum.csv"
        if p.exists():
            s = pd.read_csv(p)
            ax[0].plot(s.k_rad_per_km, s.log10_power - s.log10_power.max(), color=vz.CATEG[i], lw=2, label=f)
        e = P.paths.d("04_eda", "tables") / f"{f}_band_energy.csv"
        if e.exists():
            be = pd.read_csv(e)
            x = np.arange(len(be)) + (i - 0.5) * 0.38
            ax[1].bar(x, be.energy_share * 100, width=0.38, color=vz.CATEG[i], label=f)
            ax[1].set_xticks(np.arange(len(be)))
            ax[1].set_xticklabels([f"{a:g}-{b:g}" for a, b in zip(be.wavelength_from_km, be.wavelength_to_km)])
    ax[0].set_xlabel("angular wavenumber (rad/km)")
    ax[0].set_ylabel("log10 power (normalised)")
    ax[0].set_title("Radial power spectra (padded, tapered)", loc="left")
    ax[1].set_xlabel("wavelength band (km)")
    ax[1].set_ylabel("energy share (%)")
    ax[1].set_title("Energy by spatial scale", loc="left")
    for a in ax:
        a.legend(frameon=False)
    fig.tight_layout()
    P._fig(fig, "geophysics", "06b_radial_spectra", svg=True)


# 7 / 8 -----------------------------------------------------------------------------------------------------------
def continuation(P):
    hs = P.cfg["potential"]["continuation_m"]
    fig, ax = _new(2 * len(hs), ncols=len(hs), h=3.4)
    for j, pre in enumerate(("mag", "grav")):
        for i, h in enumerate(hs):
            _panel(P, ax[j * len(hs) + i], _feat(P, f"{pre}_thd_uc{h}"), "gradient", f"{pre} THD after {h} m", "per m")
    fig.tight_layout()
    P._fig(fig, "geophysics", "07_upward_continuation_comparison")


def persistence(P):
    fig, ax = _new(4)
    _panel(P, ax[0], _feat(P, "mag_edge_persistence"), "fraction", "Magnetic boundary persistence", "share of scales")
    _panel(P, ax[1], _feat(P, "grav_edge_persistence"), "fraction", "Gravity boundary persistence", "share of scales")
    _panel(P, ax[2], _feat(P, "mag_edge_stability"), "rank", "Magnetic cross-scale stability", "rank")
    _panel(P, ax[3], _feat(P, "grav_edge_stability"), "rank", "Gravity cross-scale stability", "rank")
    fig.tight_layout()
    P._fig(fig, "structures", "08_cross_scale_persistence")


def strike_roses(P):
    """Coherence-weighted axial strike roses per modality (block sample) next to the mapped-fault roses."""
    import matplotlib.pyplot as plt

    from .circular_statistics import axial_mean, from_vectors
    from .magnetic import block_sample

    b = P.base()
    rr, cc = block_sample(b["footprint"], 10, np.random.default_rng(P.cfg["seed"]), 40000)
    sets = [("magnetic 1 km", "mag_strike_c2_1000m", "mag_strike_s2_1000m", "mag_coherence_1000m"),
            ("gravity 1 km", "grav_strike_c2_1000m", "grav_strike_s2_1000m", "grav_coherence_1000m"),
            ("terrain 1 km", "terrain_strike_c2_1000m", "terrain_strike_s2_1000m", "terrain_coherence_1000m"),
            ("USGS DEM", "dem_orient_c2", "dem_orient_s2", "dem_orient_R")]
    fig = plt.figure(figsize=(14, 3.6))
    rows = []
    for i, (lab, c2, s2, w) in enumerate(sets):
        ax = fig.add_subplot(1, 5, i + 1, projection="polar")
        C, S, W = (_feat(P, f) for f in (c2, s2, w))
        if C is None:
            ax.set_axis_off()
            ax.set_title(f"{lab}: not available", fontsize=8)
            continue
        c, s, ww = C[rr, cc], S[rr, cc], W[rr, cc]
        ok = np.isfinite(c) & np.isfinite(s) & np.isfinite(ww)
        az = from_vectors(c[ok], s[ok])
        vz.rose(ax, az, ww[ok], color=vz.CATEG[i], title=lab)
        m, R = axial_mean(az, ww[ok])
        rows.append({"modality": lab, "mean_strike_deg": m, "R": R, "n_blocks": int(ok.sum())})
    tr = pd.read_csv(P.paths.d("04_eda", "tables") / "mapped_fault_traces.csv")
    ax = fig.add_subplot(1, 5, 5, projection="polar")
    vz.rose(ax, tr.strike_deg, tr.length_km, color=vz.INK, title="mapped faults")
    m, R = axial_mean(tr.strike_deg, tr.length_km)
    rows.append({"modality": "mapped faults (length-weighted)", "mean_strike_deg": m, "R": R, "n_blocks": len(tr)})
    fig.tight_layout()
    P._fig(fig, "structures", "09_strike_roses_by_modality", svg=True)
    df = pd.DataFrame(rows)
    P._table(df, "tables", "strike_by_modality")
    return df


# 9 ---------------------------------------------------------------------------------------------------------------
def terrain(P):
    fig, ax = _new(6, ncols=3, h=3.8)
    for i, (f, k, t, u) in enumerate([("terrain_elevation", "terrain", "Elevation", "m"),
                                      ("terrain_slope_1000m", "gradient", "Slope (1 km)", "degree"),
                                      ("terrain_slope_break_1000m", "gradient", "Slope break (1 km)", "degree/m"),
                                      ("terrain_ridge_2000m", "gradient", "Ridge strength (2 km)", "1/m"),
                                      ("terrain_valley_2000m", "gradient", "Valley strength (2 km)", "1/m"),
                                      ("terrain_coherence_1000m", "coherence", "Topographic coherence (1 km)", "1")]):
        _panel(P, ax[i], _feat(P, f), k, t, u)
    fig.tight_layout()
    P._fig(fig, "terrain", "09_regional_terrain_structure")


# 10 / 11 ---------------------------------------------------------------------------------------------------------
def dem_examples(P):
    import matplotlib.pyplot as plt
    import rasterio

    ids = P.results.get("dem_examples", {}).get("windows", [])
    rows = []
    for w in ids:
        f10 = P.paths.d("07_cache", "dem_windows") / "FAST_10M" / "examples" / f"{w}.tif"
        f2 = P.paths.d("07_cache", "dem_windows") / "DETAILED_2M" / "examples" / f"{w}.tif"
        if f10.exists() and f2.exists():
            rows.append((w, f10, f2))
    if not rows:
        fig, ax = plt.subplots(figsize=(7, 2.5))
        vz.limitation_panel(ax, "DEM example windows", "no example windows were processed (no DEM, or section 08 "
                            "example cell not run)")
        P._fig(fig, "terrain", "10_dem_fine_coarse_examples")
        return
    fig, ax = plt.subplots(len(rows), 4, figsize=(13, 3.3 * len(rows)), squeeze=False)
    for i, (w, f10, f2) in enumerate(rows):
        for j, (f, res) in enumerate(((f10, "10 m"), (f2, "2 m"))):
            with rasterio.open(f) as s:
                desc = list(s.descriptions)
                ext = (s.bounds.left / 1e3, s.bounds.right / 1e3, s.bounds.bottom / 1e3, s.bounds.top / 1e3)
                slope = s.read(desc.index("slope") + 1)
                scarp = s.read(desc.index("scarp") + 1)
            vz.map_panel(ax[i, 2 * j], slope, P.base()["grid"], "gradient", f"{w} slope {res}", "degree", 900, extent=ext)
            vz.map_panel(ax[i, 2 * j + 1], scarp, P.base()["grid"], "gradient", f"{w} scarp step {res}", "m", 900,
                         extent=ext)
    fig.tight_layout()
    P._fig(fig, "terrain", "10_dem_fine_coarse_examples")


def dem_maps(P):
    fig, ax = _new(4)
    s = P.cfg.dem_params["structure_scales_m"][1]
    _panel(P, ax[0], _feat(P, "dem_edge_p90"), "gradient", f"DEM edge energy p90 ({P.cfg.dem_mode})", "m/m")
    _panel(P, ax[1], _feat(P, f"dem_coherence_p90_{s}m"), "coherence", f"DEM coherence p90 ({s} m)", "1")
    _panel(P, ax[2], _feat(P, "dem_scarp_p90"), "gradient", "DEM scarp step p90", "m")
    _panel(P, ax[3], P.dem_coverage(), "fraction", "Measured DEM coverage", "share")
    fig.tight_layout()
    P._fig(fig, "terrain", "11_dem_gradient_coherence_scarp")


# 12 / 16 ---------------------------------------------------------------------------------------------------------
def orientation_agreement(P):
    fig, ax = _new(4)
    for i, (f, t) in enumerate([("xm_orient_magn_grav", "Strike agreement magnetic-gravity"),
                                ("xm_orient_magn_terr", "Strike agreement magnetic-terrain"),
                                ("xm_orient_grav_terr", "Strike agreement gravity-terrain"),
                                ("xm_orient_magn_dem", "Strike agreement magnetic-DEM")]):
        _panel(P, ax[i], _feat(P, f), "signed", t, "cos 2(da)  (+1 parallel)")
    fig.tight_layout()
    P._fig(fig, "cross_modal", "12_structural_orientation_agreement")


def boundary_agreement(P):
    fig, ax = _new(4)
    _panel(P, ax[0], _feat(P, "xm_edge_magn_grav"), "rank", "Magnetic-gravity edge agreement", "sqrt(rank x rank)")
    _panel(P, ax[1], _feat(P, "xm_evidence_count"), "count", "Independent strong-edge evidence (0-3)", "groups")
    _panel(P, ax[2], _feat(P, "xm_available_modalities"), "count", "Evidence groups available", "groups")
    _panel(P, ax[3], _feat(P, "xm_continuity"), "rank", "Structural continuity along strike", "mean evidence")
    fig.tight_layout()
    P._fig(fig, "cross_modal", "16_multimodal_boundary_agreement")


# 13 / 14 / 15 ----------------------------------------------------------------------------------------------------
def profiles(P):
    import matplotlib.pyplot as plt

    p = P.paths.d("04_eda", "profiles") / "fault_normal_profiles.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    feats = [f for f in P.PROFILE_FEATURES if f in set(d.feature_id)]
    fig, ax = plt.subplots(int(np.ceil(len(feats) / 3)), 3, figsize=(12, 2.8 * np.ceil(len(feats) / 3)), squeeze=False)
    for a, f in zip(ax.ravel(), feats):
        g = d[d.feature_id == f]
        for k, (tid, gg) in enumerate(g.groupby("trace_id")):
            v = gg.value - np.nanmedian(gg.value)
            a.plot(gg.offset_m / 1e3, v, color=vz.CATEG[k % len(vz.CATEG)], lw=1.5, alpha=0.85,
                   label=f"trace {tid} ({gg.trace_km.iloc[0]:.0f} km)")
        a.axvline(0, color=vz.MUTED, lw=0.8, ls="--")
        a.set_title(f, loc="left")
        a.set_xlabel("distance across mapped trace (km)")
    for a in ax.ravel()[len(feats):]:
        a.set_axis_off()
    ax.ravel()[0].legend(frameon=False, fontsize=6)
    fig.suptitle("Fault-normal profiles across training-visible mapped traces (descriptive examples, median-centred)",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    P._fig(fig, "fault_discovery", "13_fault_normal_profiles", svg=True)


def response(P):
    import matplotlib.pyplot as plt

    p = P.paths.d("04_eda", "tables") / "feature_response_vs_distance.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    fs = d.feature_id.unique()
    fig, ax = plt.subplots(int(np.ceil(len(fs) / 3)), 3, figsize=(12, 3 * np.ceil(len(fs) / 3)), squeeze=False)
    for a, f in zip(ax.ravel(), fs):
        g = d[d.feature_id == f]
        a.fill_between(g.dist_m / 1e3, g.q25, g.q75, color=vz.CATEG[0], alpha=0.2, lw=0)
        a.plot(g.dist_m / 1e3, g["median"], color=vz.CATEG[0], lw=2)
        a.set_title(f, loc="left")
        a.set_xlabel("distance to mapped fault (km)")
    for a in ax.ravel()[len(fs):]:
        a.set_axis_off()
    fig.suptitle("Feature response vs distance to mapped faults (block-sampled median, IQR; descriptive)", fontsize=10,
                 x=0.01, ha="left")
    fig.tight_layout()
    P._fig(fig, "fault_discovery", "14_feature_response_vs_distance", svg=True)


def redundancy(P):
    import matplotlib.pyplot as plt

    p = P.paths.d("04_eda", "tables") / "feature_spearman_matrix.csv"
    q = P.paths.d("04_eda", "tables") / "feature_quality.csv"
    if not p.exists():
        return
    m = pd.read_csv(p).set_index("feature_id")
    qq = pd.read_csv(q).set_index("feature_id")
    order = qq.sort_values(["cluster", "feature_id"]).index.intersection(m.index)
    m = m.loc[order, order]
    fig, ax = plt.subplots(figsize=(12, 11))
    im = ax.imshow(m.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    step = max(len(order) // 60, 1)
    ax.set_xticks(range(0, len(order), step))
    ax.set_xticklabels(order[::step], rotation=90, fontsize=5)
    ax.set_yticks(range(0, len(order), step))
    ax.set_yticklabels(order[::step], fontsize=5)
    plt.colorbar(im, ax=ax, fraction=0.03).set_label("Spearman rho (block sample)")
    ax.set_title("Feature correlation, ordered by redundancy cluster", loc="left")
    P._fig(fig, "structures", "15_feature_correlation_redundancy")
    fig, ax = plt.subplots(figsize=(6, 3))
    c = qq["class"].value_counts()
    ax.barh(c.index, c.values, color=vz.CATEG[0])
    ax.set_xlabel("features")
    ax.set_title("Feature classes (label-free)", loc="left")
    P._fig(fig, "structures", "15b_feature_classes", svg=True)


# 17 / 18 ---------------------------------------------------------------------------------------------------------
def lineament_map(P, gdf):
    import matplotlib.pyplot as plt

    b = P.base()
    fig, ax = plt.subplots(1, 2, figsize=(14, 6.5))
    vz.map_panel(ax[0], _feat(P, "xm_evidence_mean"), b["grid"], "rank", "Combined evidence + candidate lineaments",
                 "mean rank", _mp(P))
    if gdf is not None and len(gdf):
        vz.lines_on(ax[0], gdf[gdf.evidence_count < 2], color="#c9d1d9", lw=0.6, label="1 evidence group")
        vz.lines_on(ax[0], gdf[gdf.evidence_count >= 2], color="#e0782f", lw=1.0, label=">= 2 groups")
        ax[0].legend(loc="upper right", fontsize=7, frameon=True)
    sk = read_geotiff(P.paths.d("04_eda", "geological_evidence") / "mapped_fault_skeleton.tif", 1)
    vz.map_panel(ax[1], np.where(b["footprint"], np.where(sk > 0, 1.0, 0.1), np.nan), b["grid"], "rank",
                 "Candidates (orange) vs mapped faults (white): provenance view", "", _mp(P), cbar=False)
    if gdf is not None and len(gdf):
        vz.lines_on(ax[1], gdf, color="#e0782f", lw=0.8)
    fig.tight_layout()
    P._fig(fig, "fault_discovery", "17_candidate_lineaments")


def evidence_summary(P, gdf):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    if gdf is None or not len(gdf):
        vz.limitation_panel(ax[1], "Evidence summary", "no candidate lineaments")
        P._fig(fig, "fault_discovery", "18_geological_evidence_summary")
        return
    c = gdf.evidence_count.value_counts().sort_index()
    ax[0].bar(c.index.astype(str), c.values, color=vz.CATEG[0])
    ax[0].set_xlabel("independent evidence groups")
    ax[0].set_ylabel("candidates")
    ax[0].set_title("Candidates by evidence count", loc="left")
    ax[1].hist(np.clip(gdf.mapped_fault_distance_m / 1e3, 0, 20), bins=40, color=vz.CATEG[1])
    ax[1].set_xlabel("median distance to mapped fault (km, clipped 20)")
    ax[1].set_title("Diagnostic: distance to mapped faults", loc="left")
    pax = fig.add_subplot(1, 3, 3, projection="polar")
    ax[2].set_axis_off()
    vz.rose(pax, gdf.strike_deg, gdf.length_km, color=vz.CATEG[1], title="Candidate strike (length-weighted)")
    fig.tight_layout()
    P._fig(fig, "fault_discovery", "18_geological_evidence_summary", svg=True)


def interactive(P, gdf=None):
    b = P.base()
    layers = {}
    for f, k in (("mag_field", "signed"), ("grav_field", "signed"), ("terrain_elevation", "terrain"),
                 ("xm_evidence_mean", "rank"), ("mag_thd", "gradient"), ("terrain_slope_1000m", "gradient")):
        a = _feat(P, f)
        if a is not None:
            layers[f] = (a, k)
    cov = P.dem_coverage()
    if cov is not None:
        layers["dem_coverage"] = (np.where(b["footprint"], cov, np.nan), "fraction")
    rr, cc = np.nonzero(b["labels"])
    sel = slice(None, None, max(len(rr) // 60000, 1))
    return vz.interactive_map(layers, b["grid"], gdf, (rr[sel], cc[sel]))
