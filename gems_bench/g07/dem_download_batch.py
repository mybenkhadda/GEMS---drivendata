"""Opt-in coverage-first USGS DEM downloader for G07 Colab. No model training.

Prioritizes marginal study-data coverage without viewing withheld fault labels.
Safeguards: download limits, file-header check, SHA256 Drive copy and audit CSV.
"""
from pathlib import Path
from urllib.parse import urlparse
import json
import shutil
import numpy as np
import pandas as pd

from ..artifacts.atomic import atomic_copy
from ..dem.download import download_one
from ..dem.inventory import choose_per_tile, load_inventory, study_area_intersection
from .dem_io import analysis_window, read_header

HOST = "prd-tnm.s3.amazonaws.com"
PREFIX = "/StagedProducts/Elevation/1m/"


def _usable(path):
    try:
        import rasterio
        p = Path(path)
        if not p.is_file():
            return False
        h = read_header(p)
        if not (h["crs_projected_metric"] and h["transform_valid"] and
                0 < h["res_m"] <= 3.0):
            return False
        with rasterio.open(p) as ds:
            ds.read(1, window=((0, 1), (0, 1)))
        return True
    except Exception:
        return False


def _catalog(P):
    rel = Path("data/raw/dem_1m/tnm_urls.json")
    local = Path(P.local_root) / rel
    drive = Path(P.drive_root) / rel
    local.parent.mkdir(parents=True, exist_ok=True)
    if local.is_file():
        return local
    if drive.is_file():
        shutil.copy2(drive, local)
        return local
    sources = [Path(P.root) / "dem_catalog/dem_inventory.csv",
               Path(P.drive_root) / "outputs/g07_multiscale_fault_discovery" /
               P.run_id / "dem_catalog/dem_inventory.csv"]
    for source in sources:
        if not source.is_file():
            continue
        frame = pd.read_csv(source, usecols=["url"])
        urls = frame["url"].dropna().astype(str).drop_duplicates()
        urls = [u for u in urls if urlparse(u).scheme == "https"
                and urlparse(u).hostname == HOST
                and urlparse(u).path.startswith(PREFIX)]
        if not urls:
            continue
        local.write_text(json.dumps({"items": [{"downloadURL": u} for u in urls]}, indent=2))
        try:
            atomic_copy(local, drive, "size")
        except OSError as e:
            print("[DEM] Catalog Drive mirror unavailable:", str(e)[:150])
        return local
    raise FileNotFoundError("Missing TNM URL catalog and G07 dem_inventory.csv")


def download_next_dem_batch(
    P, max_new_tiles=20, max_gib=9.0, min_study_frac=0.05,
    max_attempts=30, min_free_gib=2.0,
):
    """Add up to max_new_tiles, prioritizing new 100 m grid coverage.

    Existing valid sources are never redownloaded. New files are read-checked
    and SHA256-copied to canonical Drive locations. A damaged original Drive
    file is replaced only after new data is verified; failed copies stay local.
    """
    if not 1 <= max_new_tiles <= 100 or not 0 < max_gib <= 100:
        raise ValueError("Invalid explicit count or GiB limit")
    if max_attempts < max_new_tiles:
        raise ValueError("max_attempts must be >= max_new_tiles")

    root, drive = Path(P.local_root), Path(P.drive_root)
    DLOCAL, DDRIVE = root / "data/raw/dem_1m", drive / "data/raw/dem_1m"
    DLOCAL.mkdir(parents=True, exist_ok=True)
    inventory = study_area_intersection(
        load_inventory(_catalog(P)), P.engine().ctx, min_valid_frac=min_study_frac)
    inventory = choose_per_tile(inventory[inventory.in_study_area], "latest")
    T, sources = P.dem_sources(refresh=True)
    validated_ids = {str(s.tile_id) for s in sources if s.status == "LOCAL"}
    print(f"[DEM] Already validated: {len(validated_ids)} | catalog study tiles: {len(inventory)}")

    grid = P.grid()
    study = np.asarray(P.engine().valid(), bool)
    covered = np.zeros(study.shape, bool)
    for s in sources:
        if s.status != "LOCAL" or s.bounds is None or not s.crs:
            continue
        w = analysis_window(s.bounds, s.crs, grid, pad=0)
        if w:
            r0, r1, c0, c1 = w
            covered[r0:r1, c0:c1] |= study[r0:r1, c0:c1]

    choices = []
    for row in inventory.itertuples():
        if str(row.tile_id) in validated_ids or int(row.zone) != 11:
            continue
        parsed = urlparse(str(row.url))
        if parsed.scheme != "https" or parsed.hostname != HOST or not parsed.path.startswith(PREFIX):
            continue
        win = analysis_window(
            (float(row.west), float(row.south), float(row.east), float(row.north)),
            "EPSG:26911", grid, pad=0)
        if win:
            size = pd.to_numeric(getattr(row, "size_bytes", np.nan), errors="coerce")
            estimate = float(size) if np.isfinite(size) and size > 0 else 350_000_000.0
            choices.append((row, win, estimate))
    if not choices:
        raise RuntimeError("No additional spatially relevant DEM catalog tiles available")

    # Greedy label-free marginal covered-area plan (full study footprint, not withheld truth).
    cover_plan = covered.copy()
    planned, used, centroids = [], set(), []
    for _ in range(min(max_attempts, len(choices))):
        best_i, best_score = None, -1
        for i, (rec, w, size) in enumerate(choices):
            if i in used:
                continue
            r0, r1, c0, c1 = w
            fresh = int((study[r0:r1, c0:c1] & ~cover_plan[r0:r1, c0:c1]).sum())
            if not fresh:
                continue
            center = ((r0 + r1)/2, (c0 + c1)/2)
            spread = 1
            if centroids:
                km = min(np.hypot(center[0]-a, center[1]-b)*grid.pixel_m/1000
                         for a, b in centroids)
                spread += 0.1 * min(km/100, 1.0)
            score = fresh * spread
            if score > best_score:
                best_i, best_score = i, score
        if best_i is None:
            break
        rec, w, size = choices[best_i]
        used.add(best_i)
        r0, r1, c0, c1 = w
        fresh = int((study[r0:r1, c0:c1] & ~cover_plan[r0:r1, c0:c1]).sum())
        planned.append((rec, size, fresh))
        cover_plan[r0:r1, c0:c1] |= study[r0:r1, c0:c1]
        centroids.append(((r0+r1)/2, (c0+c1)/2))

    transferred, complete, attempts = 0, 0, 0
    budget = int(max_gib * 1024**3)
    logs = []
    for rec, est, fresh in planned:
        if complete >= max_new_tiles or attempts >= max_attempts:
            break
        if transferred + est > budget:
            logs.append({"tile_id": rec.tile_id, "status": "BUDGET_SKIP"})
            continue
        if shutil.disk_usage(root).free < max(int(est*1.5), int(min_free_gib*1024**3)):
            print("[DEM] Stopped: low Colab free disk")
            break

        attempts += 1
        rel = Path(str(rec.project_dir)) / str(rec.file)
        loc, remote = DLOCAL / rel, DDRIVE / rel
        if _usable(loc):
            status = "REUSED_LOCAL"
        elif _usable(remote):
            status = "REUSED_DRIVE"
        else:
            if loc.exists():
                quarantine = root / "dem_quarantine" / str(rec.project_dir)
                quarantine.mkdir(parents=True, exist_ok=True)
                loc.replace(quarantine / (loc.name + f".invalid_{attempts}"))
            print(f"[DEM {attempts}/{max_attempts}] {rec.tile_id} "
                  f"(nominal +{fresh*grid.pixel_m**2/1e6:.1f} km2)")
            result = download_one(str(rec.url), loc, retries=3)
            if result.get("status") not in ("downloaded", "exists") or not _usable(loc):
                logs.append({"tile_id": rec.tile_id, "status": "DOWNLOAD_FAILED",
                             "detail": result.get("error", "invalid raster")})
                continue
            transferred += loc.stat().st_size
            status = "LOCAL_ONLY"
            try:
                atomic_copy(loc, remote, "sha256")
                status = "SAVED_TO_DRIVE"
            except Exception as e:
                print("[DEM] Drive save failed; keeping local file:", str(e)[:150])
        complete += 1
        file = remote if status == "REUSED_DRIVE" else loc
        logs.append({"tile_id": rec.tile_id, "status": status, "path": str(file),
                     "bytes": file.stat().st_size, "marginal_cells_planned": fresh})
        print("[DEM]", rec.tile_id, status)

    df = pd.DataFrame(logs)
    P.ws.save_table(df, "dem_catalog/dem_download_next_batch.csv")
    P._dem_cat = None
    P.dem_inventory()
    P.dem_coverage()
    P.push(["dem_catalog"])
    print(f"[DEM] Finished: {complete} tiles handled; {transferred / 1024**3:.2f} GiB downloaded")
    return df
