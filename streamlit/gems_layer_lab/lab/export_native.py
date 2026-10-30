"""Full-footprint native-resolution GeoTIFF export of saved features (or program variables), tiled with halos.

* Output grid = the feature raster's CRS / transform / shape exactly (profile copied), float32, nodata NaN, tiled
  deflate. Validated after writing (CRS, transform, shape re-read).
* Tiles are evaluated on the tile + the feature's halo and cropped, so the result equals a whole-grid evaluation
  (to float tolerance) - neighbourhood filters do not create seams.
* AOI-dependent features (zscore / percentile_rank / ... without frozen statistics) are REFUSED: their values would
  depend on the tile. Freeze the statistics first (Feature Lab -> Freeze normalisation).
* Never overwrites: a new directory ``exports/<name>_v<version>_<hash>/`` per export; refuses if it exists.
* Labels are never read.

CLI::

    python -m lab.export_native --root /content/drive/MyDrive/GEMS_2 --features mag_grav_ratio feature_A
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window


def export_features(engine, names: list, out_root: Path, tile: int = 512, footprint_mask: bool = False,
                    program=None, progress=None) -> dict:
    """Export saved features ``names`` (or variables of ``program``). Returns the manifest dict."""
    lib = engine.library
    meta = engine.library_meta()
    specs = []
    for n in names:
        if program is not None and n in program.names:
            st = program.statement(n)
            specs.append((n, st.aoi_dependent, f"program:{n}", None))
        elif lib is not None and n in lib.features:
            e = lib.get(n)
            specs.append((n, meta.get(n, (0, e.get("aoi_dependent")))[1], f"library:{n}", e))
        else:
            raise ValueError(f"{n} is neither a saved feature nor a variable of the current program")
    bad = [n for n, aoi, _, _ in specs if aoi]
    if bad:
        raise ValueError(f"AOI-dependent features cannot be exported tile by tile: {bad} - freeze their "
                         "normalisation statistics first")
    tag = "_".join(n for n, *_ in specs)[:60]
    hashes = []
    if lib is not None:
        hashes = [lib.get(n)["formula_hash"] for n, _, src, _ in specs if src.startswith("library:")]
    import hashlib

    h = (hashes[0] if len(hashes) == 1 else hashlib.sha256("|".join(hashes).encode()).hexdigest()[:12]) if hashes \
        else time.strftime("%Y%m%d%H%M%S")
    ver = specs[0][3]["version"] if len(specs) == 1 and specs[0][3] else 0
    out = Path(out_root) / f"{tag}_v{ver}_{h}"
    if out.exists():
        raise FileExistsError(f"{out} exists: exports are never overwritten (choose another feature version)")
    out.mkdir(parents=True)
    paths = {n: out / f"{n}.tif" for n, *_ in specs}
    with rasterio.open(engine.feature_tif) as src:
        prof = src.profile.copy()
        prof.update(driver="GTiff", count=1, dtype="float32", nodata=np.nan, tiled=True, blockxsize=256,
                    blockysize=256, compress="deflate", predictor=3, BIGTIFF="IF_SAFER")
        H, W, crs, transform = src.height, src.width, src.crs, src.transform
    dsts = {n: rasterio.open(p, "w", **prof) for n, p in paths.items()}
    t0 = time.time()
    lib_names = [n for n, _, s, _ in specs if s.startswith("library:")]
    prog_names = [n for n, _, s, _ in specs if s.startswith("program:")]
    try:
        for n, d in dsts.items():
            d.set_band_description(1, n)
        tiles = [(r, c) for r in range(0, H, tile) for c in range(0, W, tile)]
        for i, (r, c) in enumerate(tiles):
            w = Window(c, r, min(tile, W - c), min(tile, H - r))
            res = engine.evaluate(program if prog_names else None, w, targets=prog_names or None,
                                  library_targets=lib_names)
            fp = engine.footprint(w) if footprint_mask else None
            for n, d in dsts.items():
                a = np.asarray(res.arrays[n], np.float32)
                if fp is not None:
                    a = np.where(fp, a, np.nan).astype(np.float32)
                d.write(a, 1, window=w)
            engine.clear_cache()                          # bounded memory across tiles
            if progress:
                progress(i + 1, len(tiles))
    except BaseException as e:
        (out / "EXPORT_INCOMPLETE.json").write_text(json.dumps({"status": "FAILED", "error": f"{type(e).__name__}: "
                                                                f"{e}", "time": time.strftime("%Y-%m-%dT%H:%M:%S")},
                                                               indent=2), encoding="utf-8")
        raise
    finally:
        for d in dsts.values():
            d.close()
    checks = {}
    for n, p in paths.items():
        with rasterio.open(p) as s:
            checks[n] = {"crs_equal": s.crs == crs, "transform_equal": s.transform == transform,
                         "shape_equal": (s.height, s.width) == (H, W)}
    if not all(all(v.values()) for v in checks.values()):
        raise RuntimeError(f"exported grid mismatch: {checks}")
    man = {"status": "COMPLETE", "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "seconds": round(time.time() - t0, 1),
           "source": str(Path(engine.feature_tif).resolve()), "source_stamps": engine.stamps, "grid": {
               "crs": str(crs), "transform": list(transform)[:6], "height": H, "width": W},
           "tile": tile, "footprint_mask": footprint_mask, "layers": {n: str(p) for n, p in paths.items()},
           "features": {n: (lib.get(n) if src.startswith("library:") else {"program_source": program.source})
                        for n, _, src, _ in specs}, "alignment_checks": checks,
           "labels_used": False, "note": "native 100 m grid; values NaN where any input is invalid"}
    (out / "export_manifest.json").write_text(json.dumps(man, indent=2, default=str), encoding="utf-8")
    if lib is not None:
        for n in lib_names:
            lib.record_export(n, str(paths[n]), str(out / "export_manifest.json"))
    return man


def main(argv=None):
    from .integration import FeatureStore

    p = argparse.ArgumentParser(description="Export saved Feature Lab features as native GeoTIFFs")
    p.add_argument("--root", required=True, help="GEMS_2 root (data/raw/... and outputs/geological_layer_lab)")
    p.add_argument("--features", nargs="+", required=True)
    p.add_argument("--tile", type=int, default=512)
    p.add_argument("--footprint-mask", action="store_true")
    a = p.parse_args(argv)
    fs = FeatureStore(a.root)
    man = fs.export(a.features, tile=a.tile, footprint_mask=a.footprint_mask,
                    progress=lambda i, n: print(f"tile {i}/{n}", end="\r"))
    print("\nCOMPLETE:", man["layers"])


if __name__ == "__main__":
    main()
