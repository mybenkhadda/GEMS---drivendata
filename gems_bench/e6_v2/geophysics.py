"""Stages that read only the raw GeoDAWN raster: base masks, terrain features (branch 1) and potential-field /
subsurface features (branch 2). Each writes a fingerprint-keyed manifest; a matching manifest is reused, a
stale one is recomputed and the reason logged.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import ndimage

from .cache import (atomic_save_npy, atomic_write_csv, atomic_write_json, load_npy, manifest_status, read_json,
                    write_manifest)
from .config import module_hash, section_hash, stable_hash
from .contracts import field_channels, subsurface_channels, terrain_channels
from .fft_features import nearest_fill, potential_field_features
from .inventory import raw_path
from .normalization import robust_stats
from .paths import E6V2Paths
from .raw_data import GeoDawn, Grid, read_labels, read_template

log = logging.getLogger("e6v2.geophysics")


def geodawn_fingerprint(inv: dict) -> str:
    return stable_hash(sorted((r["source_id"], r["checksum"]) for r in inv["rows"]
                              if str(r["source_id"]).startswith("geodawn_")))


def _status(paths, man, fp, name):
    st, why = manifest_status(man, fp, paths.root)
    if st != "complete":
        log.info("%s: %s (%s) -> computing", name, st, why)
    return st == "complete"


# ---------------------------------------------------------------------------
# base masks
# ---------------------------------------------------------------------------
def base_fingerprint(cfg, inv) -> str:
    return stable_hash([geodawn_fingerprint(inv), section_hash(cfg, "raw"), module_hash("geophysics", "raw_data")])


def prepare_base(paths: E6V2Paths, cfg: dict, inv: dict, force: bool = False) -> dict:
    d = paths.base
    man = d / "manifest.json"
    fp = base_fingerprint(cfg, inv)
    if not force and _status(paths, man, fp, "base"):
        return read_json(man)
    gd = GeoDawn(raw_path(inv, "geodawn_features"), cfg)
    valid = gd.footprint()
    labels = read_labels(raw_path(inv, "geodawn_faults"), gd.grid, cfg)
    tmpl, _ = read_template(raw_path(inv, "geodawn_template"), gd.grid)
    outs = [atomic_save_npy(d / "valid.npy", valid), atomic_save_npy(d / "labels.npy", labels),
            atomic_save_npy(d / "template_valid.npy", tmpl)]
    stats = {"valid_px": int(valid.sum()), "label_px": int(labels.sum()), "label_px_in_valid": int((labels & valid).sum()),
             "template_px": int(tmpl.sum()), "template_outside_valid_px": int((tmpl & ~valid).sum())}
    outs.append(atomic_write_json(d / "grid.json", {**gd.grid.as_dict(), "crs_wkt": gd.grid.crs_wkt,
                                                    "bands": gd.names, **stats}))
    return write_manifest(man, stage="base", fingerprint=fp, outputs=outs, root=paths.root, extra={"stats": stats})


def load_base(paths: E6V2Paths) -> dict:
    g = read_json(paths.base / "grid.json")
    if g is None:
        from .cache import MissingArtifact

        raise MissingArtifact("cache/base missing - run section 04")
    grid = Grid(g["crs_wkt"], g["epsg"], tuple(g["transform"]), g["height"], g["width"])
    return {"grid": grid, "valid": load_npy(paths.base / "valid.npy"), "labels": load_npy(paths.base / "labels.npy"),
            "template_valid": load_npy(paths.base / "template_valid.npy"), "meta": g}


# ---------------------------------------------------------------------------
# branch 1: terrain at 100 m
# ---------------------------------------------------------------------------
def terrain_fingerprint(cfg, inv) -> str:
    return stable_hash([base_fingerprint(cfg, inv), section_hash(cfg, "terrain", "normalization"),
                        module_hash("geophysics", "normalization", "contracts")])


def terrain_features(elev: np.ndarray, slope: np.ndarray, footprint: np.ndarray, px: float, cfg: dict):
    t = cfg["terrain"]
    gs = float(t["edge_guard_sigma"])
    ok = np.isfinite(elev) & footprint
    z = nearest_fill(np.where(ok, elev, 0.0), ok)
    dist = ndimage.distance_transform_edt(footprint) * px
    feats, reg = {}, []

    def put(name, arr, unit, guard_m, desc):
        a = np.asarray(arr, np.float32)
        a[~footprint | (dist <= guard_m)] = np.nan
        feats[name] = a
        reg.append({"channel": name, "group": "terrain", "unit": unit, "edge_guard_m": guard_m, "kind": "scalar",
                    "definition": desc})

    put("elev", np.where(ok, elev, np.nan), "m", 0.0, f"{t['elevation_band']} as delivered")
    put("slope_raw", slope, "as delivered", 0.0, f"{t['slope_band']} as delivered")
    for s in t["derivative_scales_m"]:
        sg = s / 2.0 / px
        put(f"grad_{s}m", ndimage.gaussian_gradient_magnitude(z, sg) / px, "m/m", gs * s / 2.0,
            f"gaussian gradient magnitude, sigma {s / 2:g} m")
    for s in t["derivative_scales_m"]:
        sg = s / 2.0 / px
        put(f"log_{s}m", ndimage.gaussian_laplace(z, sg) / px ** 2, "1/m", gs * s / 2.0,
            f"laplacian of gaussian, sigma {s / 2:g} m")
    for w in t["relief_windows_m"]:
        n = max(int(round(w / px)), 3)
        put(f"relief_{w}m", ndimage.maximum_filter(z, n) - ndimage.minimum_filter(z, n), "m", w / 2.0,
            f"max - min in a {w} m window")
    for r in t["tpi_radii_m"]:
        put(f"tpi_{r}m", z - ndimage.gaussian_filter(z, r / 2.0 / px), "m", gs * r / 2.0,
            f"z - gaussian mean, sigma {r / 2:g} m")
    for a, b in t["dog_bands_m"]:
        put(f"dog_{a}_{b}m", ndimage.gaussian_filter(z, a / 2.0 / px) - ndimage.gaussian_filter(z, b / 2.0 / px), "m",
            gs * b / 2.0, f"difference of gaussians sigma {a / 2:g} m - {b / 2:g} m")
    return feats, reg


def prepare_terrain(paths: E6V2Paths, cfg: dict, inv: dict, force: bool = False) -> dict:
    d = paths.terrain
    man = d / "manifest.json"
    fp = terrain_fingerprint(cfg, inv)
    if not force and _status(paths, man, fp, "terrain"):
        return read_json(man)
    base = load_base(paths)
    gd = GeoDawn(raw_path(inv, "geodawn_features"), cfg)
    t = cfg["terrain"]
    feats, reg = terrain_features(gd.read(t["elevation_band"]), gd.read(t["slope_band"]), base["valid"],
                                  base["grid"].pixel_m, cfg)
    names = terrain_channels(cfg)
    if list(feats) != names:
        raise RuntimeError(f"terrain channels {list(feats)} != schema {names}")
    stack = np.stack([feats[n] for n in names])
    del feats
    norm = robust_stats(stack, names, base["valid"], cfg, seed=cfg["splits"]["seed"])
    outs = [atomic_save_npy(d / "feature_stack.npy", stack), atomic_write_csv(d / "feature_registry.csv", pd.DataFrame(reg)),
            atomic_write_json(d / "normalization.json", norm)]
    return write_manifest(man, stage="terrain", fingerprint=fp, outputs=outs, root=paths.root,
                          extra={"channels": names, "shape": list(stack.shape)})


# ---------------------------------------------------------------------------
# branch 2: magnetic / gravity FFT features + raw subsurface bands
# ---------------------------------------------------------------------------
def potential_fingerprint(cfg, inv) -> str:
    return stable_hash([base_fingerprint(cfg, inv), section_hash(cfg, "potential", "normalization"),
                        module_hash("geophysics", "fft_features", "normalization", "contracts")])


def prepare_potential(paths: E6V2Paths, cfg: dict, inv: dict, force: bool = False) -> dict:
    d = paths.potential
    man = d / "manifest.json"
    fp = potential_fingerprint(cfg, inv)
    if not force and _status(paths, man, fp, "potential_fields"):
        return read_json(man)
    base = load_base(paths)
    gd = GeoDawn(raw_path(inv, "geodawn_features"), cfg)
    p = cfg["potential"]
    px = base["grid"].pixel_m
    outs, reg, norm, channels = [], [], {}, {}
    for key, fname, prefix in (("magnetic", "magnetic_features.npy", "mag"), ("gravity", "gravity_features.npy", "grav")):
        f = p["fields"][key]
        feats, r = potential_field_features(gd.read(f["band"]), base["valid"], prefix, f["unit"], px,
                                            p["continuation_heights_m"], p["pad_frac"], p["taper_px"],
                                            p["edge_guard_m"], p["continuation_guard_cap_m"])
        names = field_channels(cfg, prefix)
        if list(feats) != names:
            raise RuntimeError(f"{key} channels {list(feats)} != schema {names}")
        stack = np.stack([feats[n] for n in names])
        del feats
        norm.update(robust_stats(stack, names, base["valid"], cfg, seed=cfg["splits"]["seed"]))
        outs.append(atomic_save_npy(d / fname, stack))
        reg += [{**x, "source_band": f["band"]} for x in r]
        channels[key] = names
        del stack
    names = subsurface_channels(cfg)
    stack = np.stack([np.where(base["valid"], gd.read(b), np.nan).astype(np.float32) for b in p["subsurface_bands"]])
    norm.update(robust_stats(stack, names, base["valid"], cfg, seed=cfg["splits"]["seed"]))
    outs.append(atomic_save_npy(d / "subsurface_features.npy", stack))
    reg += [{"channel": n, "group": "field", "unit": "as delivered", "edge_guard_m": 0.0, "kind": "scalar",
             "definition": "raw GeoDAWN band as delivered", "source_band": b} for n, b in zip(names, p["subsurface_bands"])]
    channels["subsurface"] = names
    outs += [atomic_write_csv(d / "feature_registry.csv", pd.DataFrame(reg)), atomic_write_json(d / "normalization.json", norm)]
    return write_manifest(man, stage="potential_fields", fingerprint=fp, outputs=outs, root=paths.root,
                          extra={"channels": channels,
                                 "excluded_orientation_dependent": p["excluded_orientation_dependent"]})
