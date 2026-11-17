"""Small synthetic GeoDAWN-like example (tests, smoke runs, documentation). NOT real data.

A 400 x 400 px (40 x 40 km) EPSG:32611 grid with the 19 documented band names. Faults are straight-ish segments
that offset a magnetic and a gravity source body and leave a terrain scarp; a share of the faults is *unmapped*
(present in the physics, absent from ``existing_faults.tif``), which is the situation the challenge targets.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage

BANDS = (
    "mag_anom",
    "rtp",
    "tmi",
    "tmi_hg",
    "tmi_vg",
    "tc",
    "iso_grav_anom",
    "iso_grav_anom_hg",
    "iso_grav_anom_vg",
    "iso_grav_anom_slope",
    "det_elev",
    "det_elev_slope",
    "depth_to_base_surf",
    "cond_surf",
    "geod_2ndinv",
    "geod_shearrate",
    "geod_dilaterate",
    "deq_n100a15",
    "ieq_n100a15",
)


def make(root: str | Path, size: int = 400, n_faults: int = 46, unmapped_frac: float = 0.3, seed: int = 7) -> dict:
    import rasterio
    from rasterio.transform import from_origin
    from skimage.draw import line

    rng = np.random.default_rng(seed)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    h = w = size
    yy, xx = np.indices((h, w)).astype(np.float32)
    mapped = np.zeros((h, w), bool)
    all_f = np.zeros((h, w), bool)
    step = np.zeros((h, w), np.float32)
    for i in range(n_faults):
        r0, c0 = rng.integers(10, h - 10), rng.integers(10, w - 10)
        ang = rng.choice([np.radians(20), np.radians(155)]) + rng.normal(0, 0.15)
        L = rng.integers(45, 100)
        r1 = int(np.clip(r0 - L * np.sin(ang), 2, h - 3))
        c1 = int(np.clip(c0 + L * np.cos(ang), 2, w - 3))
        rr, cc = line(int(r0), int(c0), r1, c1)
        f = np.zeros((h, w), bool)
        f[rr, cc] = True
        all_f |= f
        if rng.random() > unmapped_frac:
            mapped |= f
        # signed distance to the fault line -> a step across it (fault offset of a source)
        nx, ny = -np.sin(ang), np.cos(ang)
        sd = (xx - c0) * nx + (-(yy - r0)) * ny
        within = ndimage.distance_transform_edt(~f) < 25
        step += np.where(within, np.tanh(sd / 2.0), 0.0) * rng.uniform(0.5, 1.5)
    base = ndimage.gaussian_filter(rng.normal(size=(h, w)), 20).astype(np.float32) * 30
    near = ndimage.distance_transform_edt(~all_f)
    scarp = np.exp(-near / 1.5)
    rtp = base + 40 * step + rng.normal(0, 2, (h, w))
    grav = ndimage.gaussian_filter(base, 5) * 0.2 + 3 * ndimage.gaussian_filter(step, 2) + rng.normal(0, 0.2, (h, w))
    elev = 200 * ndimage.gaussian_filter(rng.normal(size=(h, w)), 30) + 15 * scarp + 5 * step
    gy, gx = np.gradient(rtp)
    ggy, ggx = np.gradient(grav)
    ey, ex = np.gradient(elev)
    layers = {
        "mag_anom": rtp + rng.normal(0, 1, (h, w)),
        "rtp": rtp,
        "tmi": rtp + 0.3 * np.roll(rtp, 3, 0),
        "tmi_hg": np.hypot(gx, gy),
        "tmi_vg": ndimage.laplace(rtp),
        "tc": rng.normal(0, 1, (h, w)),
        "iso_grav_anom": grav,
        "iso_grav_anom_hg": np.hypot(ggx, ggy),
        "iso_grav_anom_vg": ndimage.laplace(grav),
        "iso_grav_anom_slope": np.hypot(ggx, ggy),
        "det_elev": elev,
        "det_elev_slope": np.hypot(ex, ey),
        "depth_to_base_surf": 1000 - 50 * ndimage.gaussian_filter(step, 3),
        "cond_surf": np.exp(0.2 * step) + rng.normal(0, 0.05, (h, w)),
        "geod_2ndinv": ndimage.gaussian_filter(rng.normal(size=(h, w)), 60),
        "geod_shearrate": ndimage.gaussian_filter(rng.normal(size=(h, w)), 60),
        "geod_dilaterate": ndimage.gaussian_filter(rng.normal(size=(h, w)), 60),
        "deq_n100a15": ndimage.gaussian_filter(all_f.astype(float), 40),
        "ieq_n100a15": ndimage.gaussian_filter(all_f.astype(float), 40),
    }
    stack = np.stack([layers[b] for b in BANDS]).astype(np.float32)
    stack[13, 300:340, 20:120] = np.nan  # conductivity survey gap
    stack[10:12, :, 380:] = np.nan  # DEM gap on the east edge
    tmpl = np.zeros((h, w), np.float32)
    tmpl[:4] = np.nan  # outside competition bounds
    tmpl[:, :3] = np.nan
    stack[:, :4] = np.nan
    stack[:, :, :3] = np.nan
    prof = dict(driver="GTiff", width=w, height=h, crs="EPSG:32611", transform=from_origin(300000, 4300000, 100, 100))
    paths = {
        "raw": root / "gems-geodawn-numerical-features.tif",
        "known": root / "existing_faults.tif",
        "template": root / "example_submission.tif",
    }
    with rasterio.open(paths["raw"], "w", count=len(BANDS), dtype="float32", nodata=np.nan, **prof) as dst:
        dst.write(stack)
        for i, b in enumerate(BANDS, 1):
            dst.set_band_description(i, b)
    with rasterio.open(paths["known"], "w", count=1, dtype="uint8", nodata=None, **prof) as dst:
        dst.write(mapped.astype(np.uint8), 1)
    with rasterio.open(paths["template"], "w", count=1, dtype="float32", nodata=np.nan, **prof) as dst:
        dst.write(tmpl, 1)
    np.save(root / "all_faults_truth.npy", all_f)  # synthetic-only: includes the unmapped faults
    return {k: str(v) for k, v in paths.items()}


def small_config(work: str | Path, data_root: str | Path, mode: str = "FAST_BASELINE", **kw):
    """Config sized for the synthetic grid (fast; used by tests and the smoke run)."""
    from .config import G13Config

    base = dict(
        drive_root=str(data_root),
        work_dir=str(work),
        out_dir="",
        block_km=8.0,
        buffer_km=2.0,
        discovery_frac_km=0.15,
        discovery_eval_km=4.0,
        scales_m=(200, 400, 800),
        upward_continuation_m=(500, 2000),
        gabor_orientations=4,
        n_estimators=40,
        max_positives=4000,
        negative_ratio=4.0,
        n_jobs=2,
        area_fracs=(0.01, 0.03, 0.06),
        known_ring_px=(0, 1),
        deep_tile=64,
        deep_epochs=1,
        figures=True,
    )
    base.update(kw)
    return G13Config.from_mode(mode, **base)
