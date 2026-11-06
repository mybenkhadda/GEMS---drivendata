"""Preflight: read the actual raw headers before any expensive stage and stop with ONE report.

Blockers stop the pipeline (``PreflightBlocked`` lists all of them at once); warnings are recorded. The report
is written to ``manifests/preflight_report.json`` and ``reports/preflight_report.md``.
"""
from __future__ import annotations

import importlib
import shutil
import time

import numpy as np

from .cache import atomic_write_json, atomic_write_text
from .config import CODE_VERSION
from .inventory import build_inventory
from .paths import E6V2Paths

PACKAGES = ("numpy", "scipy", "pandas", "rasterio", "affine", "torch", "sklearn", "skimage", "yaml", "matplotlib")


class PreflightBlocked(RuntimeError):
    pass


def run_preflight(paths: E6V2Paths, cfg: dict, raise_on_block: bool = True) -> dict:
    blockers, warnings, facts = [], [], {"code_version": CODE_VERSION, "config": cfg.get("_source")}

    for pkg in PACKAGES:
        try:
            importlib.import_module(pkg)
        except Exception as exc:
            blockers.append(f"python package {pkg!r} not importable: {exc}")

    if not paths.drive_root.exists():
        blockers.append(f"drive_root {paths.drive_root} does not exist (Drive not mounted?)")
        return _finish(paths, blockers, warnings, facts, raise_on_block, write=False)
    for f in cfg["raw"]["forbidden_prefixes"]:
        if f != cfg["paths"]["namespace"] and cfg["paths"]["namespace"].startswith(f):
            blockers.append(f"namespace {cfg['paths']['namespace']} lies under the forbidden prefix {f}")
    paths.ensure()

    inv = build_inventory(paths, cfg, with_dem=True)
    facts["raw_fingerprint"] = inv["fingerprint"]
    blockers += [f"raw data: {m}" for m in inv["issues"] if not m.startswith("no DEM")]
    if any(m.startswith("no DEM") for m in inv["issues"]):
        blockers.append([m for m in inv["issues"] if m.startswith("no DEM")][0]
                        + " - the DEM branch is part of the one E6 architecture")

    loc = inv["located"]
    grid = None
    if loc["features"]["path"] is not None:
        from .raw_data import GeoDawn

        try:
            gd = GeoDawn(loc["features"]["path"], cfg)
            grid = gd.grid
            ge = cfg["grid_expected"]
            facts["geodawn_grid"] = grid.as_dict()
            facts["geodawn_bands"] = gd.names
            if grid.epsg != ge["epsg"]:
                blockers.append(f"GeoDAWN CRS EPSG:{grid.epsg} != expected EPSG:{ge['epsg']}")
            if abs(grid.pixel_m - ge["pixel_m"]) > 1e-6 or abs(abs(grid.transform[4]) - ge["pixel_m"]) > 1e-6:
                blockers.append(f"GeoDAWN pixel {grid.pixel_m} m != expected {ge['pixel_m']} m")
            if (grid.height, grid.width) != (ge["height"], ge["width"]):
                warnings.append(f"GeoDAWN shape {(grid.height, grid.width)} != documented "
                                f"{(ge['height'], ge['width'])} (the header is used)")
            if abs(grid.transform[1]) > 0 or abs(grid.transform[3]) > 0:
                blockers.append("GeoDAWN grid is rotated - not supported")
        except Exception as exc:
            blockers.append(f"GeoDAWN raster unreadable: {type(exc).__name__}: {exc}")
    if grid is not None and loc["template"]["path"] is not None:
        import rasterio

        from .raw_data import grid_of

        with rasterio.open(loc["template"]["path"]) as src:
            tg = grid_of(src)
            facts["template"] = {"grid": tg.as_dict(), "dtype": src.dtypes[0], "nodata": src.nodata,
                                 "count": src.count}
        if not tg.same_as(grid):
            blockers.append(f"template grid {tg.as_dict()} differs from the GeoDAWN grid")
    if grid is not None and loc["faults"]["path"] is not None:
        import rasterio

        from .raw_data import grid_of

        with rasterio.open(loc["faults"]["path"]) as src:
            fg = grid_of(src)
            facts["faults"] = {"grid": fg.as_dict(), "dtype": src.dtypes[0], "nodata": src.nodata}
        if not fg.same_as(grid):
            warnings.append("fault raster grid differs from the GeoDAWN grid: it is reprojected nearest-neighbour")

    tiles = inv["dem_tiles"]
    if not tiles.empty:
        st = tiles.validation_status.value_counts().to_dict()
        facts["dem_tiles_status"] = st
        blocked = tiles[tiles.validation_status.str.startswith("blocked")]
        if len(blocked):
            blockers.append(f"{len(blocked)} DEM tiles have no determinable vertical unit (e.g. {blocked.name.iloc[0]}):"
                            " read the USGS metadata and set dem.vertical_unit_override")
        rej = tiles[tiles.validation_status.str.startswith("rejected")]
        if len(rej):
            warnings.append(f"{len(rej)} DEM tiles rejected: {rej.validation_status.value_counts().to_dict()}")
        ok = tiles[tiles.validation_status == "ok"]
        facts["dem_vertical_units"] = ok.vertical_unit.value_counts().to_dict() if len(ok) else {}
        facts["dem_crs"] = ok.crs_epsg.value_counts().to_dict() if len(ok) else {}
        if grid is not None and len(ok):
            l, b, r, t = grid.bounds
            inter = ok[(ok.right > l) & (ok.left < r) & (ok.top > b) & (ok.bottom < t)]
            facts["dem_tiles_intersecting_grid"] = int(len(inter))
            if len(inter) == 0:
                blockers.append("no valid DEM tile intersects the GeoDAWN grid")
            area = float(((inter.right.clip(upper=r) - inter.left.clip(lower=l)).clip(lower=0)
                          * (inter.top.clip(upper=t) - inter.bottom.clip(lower=b)).clip(lower=0)).sum())
            facts["dem_tile_area_km2_upper_bound"] = area / 1e6
        elif not len(ok):
            blockers.append("no DEM tile passed the header checks")

    try:
        import torch

        facts["cuda"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            facts["gpu"] = torch.cuda.get_device_name(0)
            facts["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2 ** 30, 1)
        else:
            warnings.append("no CUDA GPU: preparation stages can run, training (sections 09-11) is blocked")
    except Exception:
        pass
    free_local = shutil.disk_usage(str(paths.scratch_root)).free / 2 ** 30
    facts["scratch_free_gb"] = round(free_local, 1)
    if free_local < 15:
        warnings.append(f"only {free_local:.1f} GB free on local scratch (datasets are staged there, ~6 GB)")
    facts["export_policy"] = cfg["export"]["final"]
    warnings.append("export policy is fixed in config (export.final); the competition rules on continuous vs binary "
                    "submissions must be read by you on the official page before uploading")
    return _finish(paths, blockers, warnings, facts, raise_on_block)


def _finish(paths, blockers, warnings, facts, raise_on_block, write=True) -> dict:
    rep = {"written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "status":
           "BLOCKED" if blockers else "READY", "blockers": blockers, "warnings": warnings, "facts": facts}
    if write:
        atomic_write_json(paths.manifests / "preflight_report.json", rep)
        md = [f"# E6 V2 preflight - {rep['status']}", "", f"written {rep['written_utc']}", "", "## Blockers"]
        md += [f"- {b}" for b in blockers] or ["- none"]
        md += ["", "## Warnings"] + ([f"- {w}" for w in warnings] or ["- none"]) + ["", "## Facts", "```",
                                                                                    _fmt(facts), "```"]
        atomic_write_text(paths.reports / "preflight_report.md", "\n".join(md) + "\n")
    if blockers and raise_on_block:
        raise PreflightBlocked("E6 V2 preflight BLOCKED:\n" + "\n".join(f"  - {b}" for b in blockers))
    return rep


def _fmt(facts: dict) -> str:
    import json

    return json.dumps(facts, indent=2, default=lambda o: o.item() if isinstance(o, np.generic) else str(o))
