# -------------------------------------------------
# 1. BASE GEOPHYSICAL GRADIENTS
# -------------------------------------------------

mag_fine = gradmag(rtp, sigma=2)
mag_broad = gradmag(rtp, sigma=12)

grav_fine = gradmag(iso_grav_anom, sigma=2)
grav_broad = gradmag(iso_grav_anom, sigma=12)

terrain_fine = gradmag(det_elev_slope, sigma=2)
terrain_broad = gradmag(det_elev_slope, sigma=12)

basement_fine = gradmag(depth_to_base_surf, sigma=2)
basement_broad = gradmag(depth_to_base_surf, sigma=12)


# -------------------------------------------------
# 2. MULTISCALE EDGE CONTRASTS
# -------------------------------------------------

R01_magnetic_scale_change = normdiff(
    mag_fine, mag_broad, 0.05
)

R02_gravity_scale_change = normdiff(
    grav_fine, grav_broad, 0.05
)

R03_terrain_slope_change = normdiff(
    terrain_fine, terrain_broad, 0.05
)

R04_basement_boundary_change = normdiff(
    basement_fine, basement_broad, 0.05
)


# -------------------------------------------------
# 3. LOCAL EDGE CONTRAST VS SURROUNDINGS
# -------------------------------------------------

R05_local_magnetic_edge = ratio(
    mag_fine,
    gaussian(mag_fine, sigma=15),
    0.05
)

R06_local_gravity_edge = ratio(
    grav_fine,
    gaussian(grav_fine, sigma=15),
    0.05
)

R07_local_terrain_break = ratio(
    terrain_fine,
    gaussian(terrain_fine, sigma=15),
    0.05
)


# -------------------------------------------------
# 4. LOCAL TEXTURE CONTRAST
# -------------------------------------------------

R08_terrain_texture_change = ratio(
    local_std(det_elev_slope, 3),
    local_std(det_elev_slope, 12),
    0.05
)

R09_magnetic_texture_change = ratio(
    local_std(rtp, 3),
    local_std(rtp, 12),
    0.05
)

R10_gravity_texture_change = ratio(
    local_std(iso_grav_anom, 3),
    local_std(iso_grav_anom, 12),
    0.05
)


# -------------------------------------------------
# 5. EDGE PERSISTENCE ACROSS SPATIAL SCALES
# -------------------------------------------------

R11_terrain_edge_persistence = ratio(
    minimum(
        gradmag(det_elev_slope, 2),
        gradmag(det_elev_slope, 6)
    ),
    gaussian(terrain_fine, 15),
    0.05
)

R12_basement_edge_persistence = ratio(
    minimum(
        gradmag(depth_to_base_surf, 2),
        gradmag(depth_to_base_surf, 6)
    ),
    gaussian(basement_fine, 15),
    0.05
)

# A. Test negative-direction signals
R08_fault_response = -R08_terrain_texture_change
R03_fault_response = -R03_terrain_slope_change
R07_fault_response = -R07_local_terrain_break
RTP_fault_response = -rtp

# B. Test inverse terrain texture explicitly
R18_broad_over_fine_texture = ratio(
    local_std(det_elev_slope, 12),
    local_std(det_elev_slope, 3),
    0.05
)

# C. Combine the strongest positive and negative signals
R19_gravity_terrain_contrast = (
    robust_z(grav_broad)
    - robust_z(R08_terrain_texture_change)
)

R20_basement_terrain_contrast = (
    robust_z(depth_to_base_surf)
    - robust_z(R08_terrain_texture_change)
)

R21_gravity_slope_contrast = (
    robust_z(grav_broad)
    - robust_z(R03_terrain_slope_change)
)