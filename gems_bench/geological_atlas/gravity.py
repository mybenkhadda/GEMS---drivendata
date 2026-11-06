"""Gravity feature family from the RAW isostatic anomaly (``potential.gravity_band``).

The GeoDAWN raster delivers ``iso_grav_anom_hg`` / ``iso_grav_anom_vg``. Their provenance (exact operator, units
mGal/km) is not documented in the raster, so the derivatives are recomputed from the anomaly with known operators
and compared: when the recomputed THD / VD equal a delivered band (Spearman >= ``potential.duplicate_rho``), the
recomputed copy is not stored (registry status ``alias_of_delivered``). Comparison table:
``04_eda/tables/gravity_delivered_vs_recomputed.csv``.
"""
from __future__ import annotations

from .magnetic import build_potential_family


def build(gd, footprint, cfg, writer, rng=None) -> dict:
    band = cfg["potential"]["gravity_band"]
    if band not in gd.names:
        raise KeyError(f"gravity band {band!r} not in the GeoDAWN raster ({gd.names})")
    delivered = {}
    for key, name in (("thd", f"{band}_hg"), ("vd", f"{band}_vg")):
        if name in gd.names:
            delivered[key] = (name, gd.read(name))
    return build_potential_family(gd.read(band), footprint, prefix="grav", family="gravity", band=band,
                                  unit=cfg["potential"]["gravity_unit"], cfg=cfg, writer=writer,
                                  px=gd.grid.pixel_m, delivered=delivered, rng=rng)
