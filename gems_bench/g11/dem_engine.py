"""Fast, resumable, chunk-persistent DEM statistics engine (default 10 m processing resolution).

Why the previous aggregation was slow: every 6.4 km chunk re-opened every intersecting 1 m tile on Drive through a
warped VRT at 2 m, plus a halo, so each tile was read from Drive several times and 25x more pixels were processed
than at 10 m. This engine:

1. **tile cache** - reads each usable 1 m tile ONCE (windowed, decimated to the processing resolution with
   area-average resampling, full-resolution source by default) and keeps the result on local SSD
   (``<scratch>/dem_tile_cache/r10m/<tile>.tif``); the tile's own CRS is kept (UTM 10 / 11), so nothing is
   resampled twice;
2. **block processing** - the master grid is cut into square chunks of ``dem.engine_chunk_cells`` 100 m cells
   (default 128 = 12.8 km) plus a halo of the largest context scale; only chunks with usable DEM are processed;
   statistics are vectorised over the f x f pixel blocks of each 100 m cell (:func:`dem_reader.block_stats`);
3. **verified persistence** - each chunk is written to local scratch, validated (:func:`validate_feature_chunk`),
   committed to ``03_features/dem_chunks/r10m/`` through a verified temporary file (:func:`commit_feature_chunk`)
   and then given a completion sidecar; a restart skips exactly the committed chunks
   (:func:`resume_feature_processing`);
4. **assembly** - when every chunk is committed, the full-grid ``dem100_*`` layers are assembled from the chunk
   files (each re-hashed) and committed as resolution-tagged layers (``dem100_slope_mean@r10m``).

Nodata, tile provenance (source tile ids per chunk), vertical-unit scaling, the excluded-tile policy and the
processing resolution are recorded in every chunk sidecar and layer manifest row.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .dem_reader import DEMReader, dem_chunk_stats, dem_stat_names, stats_factor
from .feature_versions import res_tag
from .persist import commit_file, write_completion
from .registry import read_json, sha256_file


# ------------------------------------------------------------------------------------------ chunk plan / state
def chunk_plan(grid, covered: np.ndarray, C: int) -> pd.DataFrame:
    H, W = grid.shape
    rows = []
    for r0 in range(0, H, C):
        for q0 in range(0, W, C):
            r1, q1 = min(r0 + C, H), min(q0 + C, W)
            n = int(covered[r0:r1, q0:q1].sum())
            if n:
                rows.append({"chunk_id": f"r{r0:05d}_c{q0:05d}_n{C}", "r0": r0, "r1": r1, "q0": q0, "q1": q1,
                             "covered_cells": n})
    return pd.DataFrame(rows, columns=["chunk_id", "r0", "r1", "q0", "q1", "covered_cells"])


def committed_chunks(chunk_dir, spec_dem: str | None = None, verify_sha: bool = False) -> set:
    """Chunk ids whose completion sidecar exists, matches the spec, and whose data file has the recorded size."""
    d = Path(chunk_dir)
    done = set()
    if not d.is_dir():
        return done
    for s in d.glob("*.json"):
        m = read_json(s)
        if not m or m.get("status") != "COMPLETE" or (spec_dem and m.get("spec_dem") != spec_dem):
            continue
        if m.get("empty"):
            done.add(m["chunk_id"])
            continue
        f = d / m["file"]
        if f.is_file() and f.stat().st_size == m["size"] and (not verify_sha or sha256_file(f) == m["sha256"]):
            done.add(m["chunk_id"])
    return done


# ------------------------------------------------------------------------------------------ tile cache
class TileCache:
    """Each usable DEM tile decimated ONCE to ``res`` m (area average) on local SSD, in the tile's own CRS."""

    def __init__(self, catalog, res: float, cache_dir, use_overviews: bool = False, workers: int = 2, log=print):
        self.cat, self.res = catalog, float(res)
        self.dir = Path(cache_dir) / res_tag(res)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.use_overviews, self.workers, self.log = use_overviews, max(1, int(workers)), log
        self.man_p = self.dir / "cache_manifest.json"
        self.man = read_json(self.man_p) or {}

    def _one(self, r) -> dict:
        import rasterio
        from rasterio.enums import Resampling

        src = Path(r.file_path)
        st = src.stat()
        key = str(r.tile_id)
        out = self.dir / f"{key}.tif"
        ent = self.man.get(key)
        if out.is_file() and ent and ent["src_size"] == st.st_size and ent["src_mtime_ns"] == st.st_mtime_ns:
            return {**ent, "cached": True}
        t0 = time.time()
        opts = {} if self.use_overviews else {"OVERVIEW_LEVEL": "NONE"}
        with rasterio.open(src, **opts) as ds:
            k = self.res / abs(ds.res[0])                              # decimation factor (may be non-integer)
            oh = max(1, int(np.ceil(ds.height * abs(ds.res[1]) / self.res)))
            ow = max(1, int(np.ceil(ds.width * abs(ds.res[0]) / self.res)))
            a = ds.read(1, out_shape=(oh, ow), resampling=Resampling.average if k > 1.0001 else Resampling.nearest,
                        masked=True).astype(np.float32).filled(np.nan)
            a[np.abs(a) > 1e20] = np.nan
            from rasterio.transform import Affine

            t0_, sx, sy = ds.transform, ds.width / ow, ds.height / oh
            T = Affine(t0_.a * sx, t0_.b * sy, t0_.c, t0_.d * sx, t0_.e * sy, t0_.f)
            prof = {"driver": "GTiff", "height": oh, "width": ow, "count": 1, "dtype": "float32", "crs": ds.crs,
                    "transform": T, "nodata": np.nan, "tiled": True, "blockxsize": 256, "blockysize": 256,
                    "compress": "deflate", "predictor": 3}
        tmp = out.with_name(out.stem + ".partial.tif")
        with rasterio.open(tmp, "w", **prof) as o:
            o.write(a[None])
        tmp.replace(out)
        ent = {"tile_id": key, "src": str(src), "src_size": int(st.st_size), "src_mtime_ns": int(st.st_mtime_ns),
               "path": str(out), "decimation": round(float(k), 4), "res_m": self.res, "valid_fraction": float(np.isfinite(a).mean()),
               "overviews_used": self.use_overviews, "seconds": round(time.time() - t0, 2)}
        return {**ent, "cached": False}

    def ensure(self) -> pd.DataFrame:
        from concurrent.futures import ThreadPoolExecutor

        rows, t0 = [], time.time()
        recs = list(self.cat.itertuples())
        with ThreadPoolExecutor(self.workers) as ex:
            for i, r in enumerate(ex.map(self._safe, recs), 1):
                rows.append(r)
                if r.get("path"):
                    self.man[r["tile_id"]] = {k: v for k, v in r.items() if k not in ("cached", "error")}
                if self.log and (i % 25 == 0 or i == len(recs)):
                    self.log(f"[dem-engine] tile cache {i}/{len(recs)} ({time.time() - t0:.0f} s)")
                    self.man_p.write_text(json.dumps(self.man, indent=1))
        self.man_p.write_text(json.dumps(self.man, indent=1))
        return pd.DataFrame(rows)

    def _safe(self, r) -> dict:
        try:
            return self._one(r)
        except Exception as e:                                    # noqa: BLE001 - recorded, tile then unusable
            return {"tile_id": str(r.tile_id), "src": str(r.file_path), "error": f"{type(e).__name__}: {e}"}

    def reader(self, cfg: dict, allow_unknown_vertical: bool) -> DEMReader:
        """A :class:`DEMReader` over the cached tiles (same priority, vertical scale and exclusion policy)."""
        cat = self.cat.copy()
        paths = {k: v["path"] for k, v in self.man.items()}
        cat["source_file_path"] = cat["file_path"]
        cat["file_path"] = cat.tile_id.astype(str).map(paths)
        cat = cat[cat.file_path.notna()]
        rd = DEMReader(cat, cfg, allow_unknown_vertical=allow_unknown_vertical)
        rd.native_res = self.res
        return rd


# ------------------------------------------------------------------------------------------ chunk processing
def validate_feature_chunk(arrays: dict | None, shape: tuple, names: list, min_support: float = 0.25) -> tuple[bool, list]:
    """Structural and semantic checks of one computed chunk before it may be committed."""
    if arrays is None:
        return True, ["empty chunk (no valid DEM pixel in the window)"]
    probs = []
    for n in names:
        a = arrays.get(n)
        if a is None:
            probs.append(f"{n} missing")
            continue
        if a.shape != tuple(shape) or a.dtype != np.float32:
            probs.append(f"{n}: shape {a.shape} / dtype {a.dtype} != {tuple(shape)} / float32")
        if np.isinf(a).any():
            probs.append(f"{n}: infinite values")
    vf = arrays.get("valid_fraction")
    if vf is not None:
        f = vf[np.isfinite(vf)]
        if f.size and (f.min() < -1e-6 or f.max() > 1 + 1e-6):
            probs.append("valid_fraction outside [0, 1]")
        em = arrays.get("elev_mean")
        if em is not None and vf.shape == em.shape:
            sup = np.nan_to_num(vf) >= min_support
            if (np.isfinite(em) != sup).any():
                probs.append("elev_mean finite exactly where valid_fraction >= min support: violated")
    return not probs, probs


def commit_feature_chunk(local_path, chunk_dir, chunk: dict, meta: dict) -> dict:
    """Verified copy of a local chunk file to Drive, then its completion sidecar (the chunk is complete only then)."""
    d = Path(chunk_dir)
    cid = chunk["chunk_id"]
    if local_path is None:                                       # empty chunk: only a sidecar
        side = {**meta, **{k: int(chunk[k]) for k in ("r0", "r1", "q0", "q1")}, "chunk_id": cid, "empty": True,
                "status": "COMPLETE", "file": None, "size": 0, "sha256": None}
        write_completion(d / f"{cid}.json", side)
        return side
    info = commit_file(local_path, d / f"{cid}.npz")
    side = {**meta, **{k: int(chunk[k]) for k in ("r0", "r1", "q0", "q1")}, "chunk_id": cid, "empty": False,
            "status": "COMPLETE", "file": f"{cid}.npz", "size": info["size"], "sha256": info["sha256"]}
    write_completion(d / f"{cid}.json", side)
    return side


def process_missing_feature_chunks(reader, grid, todo: pd.DataFrame, cfg: dict, res: float, chunk_dir, local_dir,
                                   spec_dem: str, log=print, save_every_chunk: bool = True,
                                   max_seconds: float | None = None, code_commit: str | None = None) -> pd.DataFrame:
    """Compute, validate and commit the chunks in ``todo`` (each one independently; a crash loses at most one)."""
    stats_factor(grid, res)
    names = dem_stat_names(cfg)
    ld = Path(local_dir)
    ld.mkdir(parents=True, exist_ok=True)
    t0, rows = time.time(), []
    pending = []
    for i, ch in enumerate(todo.to_dict("records"), 1):
        if max_seconds and time.time() - t0 > max_seconds:
            log(f"[dem-engine] time budget reached after {i - 1} chunk(s); re-run to resume")
            break
        tc = time.time()
        st = dem_chunk_stats(reader, grid, ch["r0"], ch["r1"], ch["q0"], ch["q1"], cfg, res)
        ok, probs = validate_feature_chunk(st, (ch["r1"] - ch["r0"], ch["q1"] - ch["q0"]), names)
        if not ok:
            raise RuntimeError(f"DEM chunk {ch['chunk_id']} failed validation: {probs}")
        tiles = sorted(set(getattr(reader, "_last_tiles", [])))
        meta = {"spec_dem": spec_dem, "processing_resolution_m": float(res), "names": names, "code_commit": code_commit,
                "source_tiles": tiles, "covered_cells": int(ch["covered_cells"]), "validation": probs or ["ok"]}
        loc = None
        if st is not None:
            loc = ld / f"{ch['chunk_id']}.npz"
            tmp = ld / f"{ch['chunk_id']}.partial.npz"
            np.savez_compressed(tmp, **st)
            tmp.replace(loc)
        if save_every_chunk:
            side = commit_feature_chunk(loc, chunk_dir, ch, meta)
            if loc is not None:
                loc.unlink(missing_ok=True)
        else:
            pending.append((loc, ch, meta))
            side = {"chunk_id": ch["chunk_id"], "status": "LOCAL_ONLY"}
        rows.append({"chunk_id": ch["chunk_id"], "status": side["status"], "empty": st is None,
                     "seconds": round(time.time() - tc, 2)})
        if log and (i % 10 == 0 or i == len(todo)):
            el = time.time() - t0
            log(f"[dem-engine] {i}/{len(todo)} chunk(s) at {res:g} m, {el:.0f} s "
                f"(~{el / i * (len(todo) - i) / 60:.1f} min left, ESTIMATE)")
    for loc, ch, meta in pending:                               # save_every_chunk=False: commit at the end
        commit_feature_chunk(loc, chunk_dir, ch, meta)
    return pd.DataFrame(rows)


def assemble_dem_layers(chunk_dir, plan: pd.DataFrame, grid, cfg: dict, res: float, covered: np.ndarray,
                        spec_dem: str) -> dict:
    """Full-grid ``dem100_*`` arrays from the committed chunks (every chunk file re-hashed before use)."""
    names = dem_stat_names(cfg)
    H, W = grid.shape
    full = {n: np.full((H, W), np.nan, np.float32) for n in names}
    d = Path(chunk_dir)
    tiles = set()
    for ch in plan.itertuples():
        m = read_json(d / f"{ch.chunk_id}.json")
        if not m or m.get("status") != "COMPLETE" or m.get("spec_dem") != spec_dem:
            raise RuntimeError(f"DEM chunk {ch.chunk_id} is not committed for this spec - resume processing first")
        tiles |= set(m.get("source_tiles") or [])
        if m.get("empty"):
            continue
        f = d / m["file"]
        if sha256_file(f) != m["sha256"]:
            raise RuntimeError(f"DEM chunk {ch.chunk_id}: sha256 differs from its completion sidecar (corrupt)")
        with np.load(f) as z:
            for n in names:
                full[n][ch.r0:ch.r1, ch.q0:ch.q1] = z[n]
    for n in names:
        full[n][~covered] = np.nan
    full["_source_tiles"] = sorted(tiles)
    return full


def resume_feature_processing(reader_factory, grid, covered: np.ndarray, cfg: dict, res: float, features_root,
                              local_dir, spec_dem: str, log=print, save_every_chunk: bool = True,
                              max_seconds: float | None = None, code_commit: str | None = None) -> dict:
    """Resume the DEM chunk work for ``res``: skip committed chunks, process the rest, report completeness.

    ``reader_factory()`` is called only when there is work to do (building the tile cache is itself resumable).
    """
    C = int(cfg["dem"].get("engine_chunk_cells", 128))
    cdir = Path(features_root) / "dem_chunks" / res_tag(res)
    plan = chunk_plan(grid, covered, C)
    done = committed_chunks(cdir, spec_dem)
    todo = plan[~plan.chunk_id.isin(done)]
    log(f"[dem-engine] {res:g} m: {len(plan)} chunk(s) with usable DEM, {len(done & set(plan.chunk_id))} committed, "
        f"{len(todo)} to compute")
    res_t = pd.DataFrame()
    if len(todo):
        reader = reader_factory()
        res_t = process_missing_feature_chunks(reader, grid, todo, cfg, res, cdir, local_dir, spec_dem, log,
                                               save_every_chunk, max_seconds, code_commit)
    done = committed_chunks(cdir, spec_dem)
    return {"plan": plan, "chunk_dir": cdir, "complete": bool(set(plan.chunk_id) <= done),
            "n_chunks": int(len(plan)), "n_committed": int(len(done & set(plan.chunk_id))), "processed": res_t}
