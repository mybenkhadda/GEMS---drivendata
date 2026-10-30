"""Feature catalog: hypothesis-driven transformations (restricted formula language of ``lab.featurelang``).

sigma / size in pixels (1 px = 100 m). R01-R21 reproduce the Layer Lab program the user explored on one AOI
(``streamlit/gems_layer_lab/outputs/geological_layer_lab/registry/last_program.py``); sign-flipped duplicates of that
program (R03/R07/R08 "_fault_response", RTP_fault_response) are NOT recomputed: direction is a property of the
evaluation (both directions are always reported), recorded here as ``twin_of``. ``neg_rtp`` is kept explicitly as the
reference requested by the study and is flagged as carrying no new information.

R19-R21 use ``robust_z`` without frozen statistics: in FULL mode they are evaluated on the whole native grid, so the
centre / scale are the label-free full-footprint statistics; the recorded values are frozen into the exported formula
(``formulas/frozen_catalog.py``) so P6 / M9 reproduce them tile by tile.
"""
from __future__ import annotations

RAW_REFERENCE = ("rtp", "mag_anom", "tmi_hg", "tmi_vg", "iso_grav_anom", "iso_grav_anom_hg", "det_elev",
                 "det_elev_slope", "depth_to_base_surf", "cond_surf")

CATALOG = '''
# ---- reference / sign twin ------------------------------------------------------------------------------
neg_rtp = -rtp

# ---- base gradients (shared intermediates, also evaluated) ------------------------------------------------
mag_fine = gradmag(rtp, sigma=2)
mag_broad = gradmag(rtp, sigma=12)
grav_fine = gradmag(iso_grav_anom, sigma=2)
grav_broad = gradmag(iso_grav_anom, sigma=12)
terrain_fine = gradmag(det_elev_slope, sigma=2)
terrain_broad = gradmag(det_elev_slope, sigma=12)
basement_fine = gradmag(depth_to_base_surf, sigma=2)
basement_broad = gradmag(depth_to_base_surf, sigma=12)

# ---- multiscale edge contrasts (fine vs broad) ------------------------------------------------------------
R01_magnetic_scale_change = normdiff(mag_fine, mag_broad, 0.05)
R02_gravity_scale_change = normdiff(grav_fine, grav_broad, 0.05)
R03_terrain_slope_change = normdiff(terrain_fine, terrain_broad, 0.05)
R04_basement_boundary_change = normdiff(basement_fine, basement_broad, 0.05)

# ---- edge relative to its local background ----------------------------------------------------------------
R05_local_magnetic_edge = ratio(mag_fine, gaussian(mag_fine, sigma=15), 0.05)
R06_local_gravity_edge = ratio(grav_fine, gaussian(grav_fine, sigma=15), 0.05)
R07_local_terrain_break = ratio(terrain_fine, gaussian(terrain_fine, sigma=15), 0.05)

# ---- texture contrast (fine / broad local std) ------------------------------------------------------------
R08_terrain_texture_change = ratio(local_std(det_elev_slope, 3), local_std(det_elev_slope, 12), 0.05)
R09_magnetic_texture_change = ratio(local_std(rtp, 3), local_std(rtp, 12), 0.05)
R10_gravity_texture_change = ratio(local_std(iso_grav_anom, 3), local_std(iso_grav_anom, 12), 0.05)

# ---- edge persistence across scales -----------------------------------------------------------------------
R11_terrain_edge_persistence = ratio(minimum(gradmag(det_elev_slope, 2), gradmag(det_elev_slope, 6)), gaussian(terrain_fine, 15), 0.05)
R12_basement_edge_persistence = ratio(minimum(gradmag(depth_to_base_surf, 2), gradmag(depth_to_base_surf, 6)), gaussian(basement_fine, 15), 0.05)

# ---- cross-layer contrasts (robust z: full-footprint statistics in FULL mode, frozen for export) ------------
R19_gravity_terrain_contrast = robust_z(grav_broad) - robust_z(R08_terrain_texture_change)
R20_basement_terrain_contrast = robust_z(depth_to_base_surf) - robust_z(R08_terrain_texture_change)
R21_gravity_slope_contrast = robust_z(grav_broad) - robust_z(R03_terrain_slope_change)

# ---- magnetic hypotheses of this study --------------------------------------------------------------------
R25_magnetic_edge = gradmag(rtp, sigma=3)
R26_magnetic_residual = abs(highpass(rtp, sigma=10))
R27_magnetic_texture = local_std(rtp, 5)
R28_normalized_magnetic_edge = ratio(gradmag(rtp, 3), gaussian(gradmag(rtp, 3), 12), 0.05)
R29_magnetic_multiscale = normdiff(gradmag(rtp, 2), gradmag(rtp, 10), 0.05)

# ---- additional geological hypotheses ---------------------------------------------------------------------
R31_terrain_slope_break = abs(highpass(det_elev_slope, sigma=5))
R32_basement_boundary = gradmag(depth_to_base_surf, sigma=4)
R33_magnetic_tilt_abs = abs(tilt_angle(tmi_vg, tmi_hg))
R34_gravity_tilt_abs = abs(tilt_angle(iso_grav_anom_vg, iso_grav_anom_hg))
R35_magnetic_coherence = coherence(rtp, sigma=5)
R36_gravity_coherence = coherence(iso_grav_anom, sigma=5)
R37_magnetic_anisotropy = gradient_anisotropy(rtp, sigma=5)
R38_coherent_magnetic_edge = coherence(rtp, sigma=5) * mag_fine
R39_magnetic_edge_persistence = multiscale_persistence(rtp, sigma_min=1, sigma_max=8, n=4)
R40_mag_grav_edge_contrast = contrast(gradmag(rtp, 3), gradmag(iso_grav_anom, 3))
R41_basement_terrain_edge_contrast = contrast(basement_fine, terrain_fine)
R42_terrain_ridge = ridge_strength(det_elev, sigma=3, polarity="both")
'''

META = {
    "neg_rtp": {"family": "magnetic", "hypothesis": "reference: reversed RTP (direction only)", "twin_of": "rtp",
                "reference": True},
    "grav_broad": {"family": "gravity", "hypothesis": "broad gravity gradient (reference)", "reference": True},
    "R20_basement_terrain_contrast": {"family": "cross-layer", "hypothesis": "basement depth vs terrain texture "
                                      "(reference)", "reference": True},
    "R08_terrain_texture_change": {"family": "terrain", "hypothesis": "fine / broad slope texture",
                                   "note": "R08_fault_response (= -R08) and R18 (inverse ratio) are its direction twins"},
    "R03_terrain_slope_change": {"family": "terrain", "hypothesis": "fine vs broad slope gradient",
                                 "note": "R03_fault_response = -R03 (direction twin)"},
    "R07_local_terrain_break": {"family": "terrain", "hypothesis": "slope edge vs background",
                                "note": "R07_fault_response = -R07 (direction twin)"},
    "R33_magnetic_tilt_abs": {"family": "magnetic", "hypothesis": "|tilt| from provider vertical / horizontal "
                              "gradients (edges near 0)", "units": "deg"},
    "R34_gravity_tilt_abs": {"family": "gravity", "hypothesis": "|tilt| of gravity (provider gradient bands)",
                             "units": "deg"},
}
REFERENCES = ("neg_rtp", "grav_broad", "R20_basement_terrain_contrast")


def family(name: str) -> str:
    if name in META and "family" in META[name]:
        return META[name]["family"]
    n = name.lower()
    for k, f in (("mag", "magnetic"), ("rtp", "magnetic"), ("tmi", "magnetic"), ("grav", "gravity"),
                 ("terrain", "terrain"), ("elev", "terrain"), ("slope", "terrain"), ("basement", "basement"),
                 ("depth", "basement"), ("cond", "conductivity")):
        if k in n:
            return f
    return "other"
