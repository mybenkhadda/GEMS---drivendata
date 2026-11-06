"""AtlasPipeline - orchestrates every stage of the Geological Data Atlas.

Stage order (notebook sections): health -> discover -> preflight -> snapshot -> geospatial -> dem_inventory
(+ optional acquire) -> raw_bands / magnetic / gravity / terrain -> dem -> cross_modal -> faults / proximity /
profiles / quality / lineaments -> ml_ready -> reports -> artifact_report.

Each stage has a fingerprint (inputs + its config section + its code) and a record in ``07_cache/manifests``.
With ``reuse_valid_cache`` a COMPLETE record with the same fingerprint is reused; otherwise the stage recomputes.
After the snapshot every stage reads ONLY ``00_raw_snapshot/`` and this atlas' own products.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import reports as rep
from . import visualization as vz
from .cache import (atomic_write_csv, atomic_write_geotiff, atomic_write_json, atomic_write_parquet, read_geotiff,
                    read_json, stage_status, write_stage)
from .config import AtlasConfig, module_hash, stable_hash
from .feature_registry import FeatureWriter, Registry
from .paths import AtlasPaths
from .reliability import REMOUNT_HELP, drive_health_check

log = logging.getLogger("atlas")


class PreflightError(RuntimeError):
    pass


def _setup_logging(paths: AtlasPaths):
    lg = logging.getLogger("atlas")
    if lg.handlers:
        return
    lg.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(name)s %(message)s", "%H:%M:%S")
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(fmt)
    lg.addHandler(h)
    fh = logging.FileHandler(paths.reports / "atlas_run.log")
    fh.setFormatter(fmt)
    lg.addHandler(fh)


class AtlasPipeline:
    def __init__(self, config: dict | AtlasConfig | None = None):
        self.cfg = config if isinstance(config, AtlasConfig) else AtlasConfig.from_user(config)
        self.paths = AtlasPaths.from_config(self.cfg)
        self.rng = np.random.default_rng(self.cfg["seed"])
        self._base = None
        self.figs: list[Path] = []
        if self.paths.drive_root.exists():
            self.paths.ensure()
            _setup_logging(self.paths)
        self.results = rep.load(self.paths) if self.paths.root.exists() else {}
        self.results.setdefault("stages", {})
        self.results.setdefault("limitations", [])

    # ------------------------------------------------------------------ helpers
    def _say(self, msg: str):
        log.info(msg)

    def _stage_ok(self, name: str, fp: str) -> bool:
        st, why = stage_status(self.paths.stage_dir, self.paths.root, name, fp)
        if st == "COMPLETE" and self.cfg["reuse_valid_cache"]:
            self._say(f"[{name}] reused (fingerprint {fp})")
            return True
        if st != "MISSING":
            self._say(f"[{name}] {st}: {why} -> recomputing")
        return False

    def _done(self, name: str, fp: str, outputs: list, extra: dict | None = None):
        rec = write_stage(self.paths.stage_dir, self.paths.root, name, fp, [Path(o) for o in outputs], extra)
        self.results["stages"][name] = {"fingerprint": fp, "written_utc": rec["written_utc"],
                                        "outputs": len(rec["outputs"])}
        rep.save(self.paths, self.results)

    def _fig(self, fig, sub: str, name: str, svg: bool = False):
        import matplotlib.pyplot as plt

        if self.cfg["save_figures"]:
            self.figs += vz.save(fig, self.paths.d("05_figures", sub) / name, svg=svg, dpi=self.cfg["figures"]["dpi"])
        plt.show()
        plt.close(fig)

    def _table(self, df: pd.DataFrame, sub: str, name: str) -> list[Path]:
        """CSV is always written (later stages and figures read it); Parquet follows ``export_parquet``.
        ``export_csv`` = False only skips the optional CSV copies of feature rasters' registries."""
        d = self.paths.d("04_eda", sub) if sub in ("tables", "profiles", "geological_evidence") else self.paths.root / sub
        out = [atomic_write_csv(d / f"{name}.csv", df)]
        if self.cfg["export_parquet"]:
            out.append(atomic_write_parquet(d / f"{name}.parquet", df.reset_index(drop=True)))
        return out

    def snapshot_path(self, role: str) -> Path:
        r = self.cfg["raw"]
        sub = {"geodawn_features": ("geodawn", r["geodawn_file"]), "fault_labels": ("faults", r["faults_file"]),
               "official_template": ("official_template", r["template_file"])}[role]
        return self.paths.snapshot / sub[0] / sub[1]

    def raw_fp(self) -> str:
        p = self.paths.inventory / "source_checksums.csv"
        if not p.exists():
            raise PreflightError("no verified snapshot (01_inventory/source_checksums.csv) - run section 01")
        df = pd.read_csv(p)
        core = df[df.role.isin(["geodawn_features", "fault_labels", "official_template"])]
        return stable_hash(sorted(zip(core.role, core.sha256)))

    def dem_fp(self) -> str:
        df = pd.read_csv(self.paths.inventory / "source_checksums.csv")
        d = df[df.role == "dem_tile"]
        dl = self.paths.d("07_cache", "download_state") / "downloads.jsonl"
        extra = dl.read_text() if dl.exists() else ""
        return stable_hash([sorted(zip(d.snapshot_path.map(lambda p: Path(p).name), d.sha256)), stable_hash(extra),
                            self.cfg["dem"]["vertical_unit_override"]])

    # ------------------------------------------------------------------ 01: health, discovery, preflight, snapshot
    def health(self) -> dict:
        h = drive_health_check(self.paths.drive_root, self.paths.d("07_cache", "manifests"))
        if not h["ok"]:
            raise PreflightError(f"Drive health check failed: {h['reason']}\n{REMOUNT_HELP}")
        return h

    def discover(self) -> pd.DataFrame:
        from .inventory import discover_sources

        df = discover_sources(self.cfg, self.paths.drive_root)
        atomic_write_csv(self.paths.inventory / "manifests" / "discovered_sources.csv", df)
        return df

    def preflight(self, sources: pd.DataFrame | None = None) -> pd.DataFrame:
        from .raw_snapshot import snapshot_state

        src = sources if sources is not None else self.discover()
        snap = snapshot_state(self.paths.root)
        led = {}
        sc = self.paths.inventory / "source_checksums.csv"
        if sc.exists():
            led = pd.read_csv(sc).groupby("role").size().to_dict()
        rows = []

        def size_of(m):
            s = pd.to_numeric(src.loc[m, "size_bytes"], errors="coerce").sum()
            return f"{s / 2 ** 30:.2f} GiB" if s >= 2 ** 30 else f"{s / 2 ** 20:.1f} MiB"

        for role, label in (("geodawn_features", "Raw numerical raster"), ("fault_labels", "Fault labels"),
                            ("official_template", "Official template")):
            m = src.role == role
            st = src.loc[m, "access_status"].iloc[0] if m.any() else "MISSING"
            ver = led.get(role, 0) > 0
            status = "Available" if st == "OK" else ("Inaccessible (I/O)" if st.startswith("INACCESSIBLE") else "Missing")
            rows.append({"Artifact": label, "Source": "Drive (original raw)", "Status": status, "Size": size_of(m),
                         "Action": "reuse verified snapshot" if ver else ("snapshot" if st == "OK" else "BLOCKED")})
        m = src.role == "dem_tile"
        n_ok = int((src.loc[m, "access_status"] == "OK").sum())
        n_bad = int(m.sum()) - n_ok
        rows.append({"Artifact": "DEM raw tiles", "Source": "Drive (original raw)",
                     "Status": f"{n_ok} readable / {n_bad} inaccessible" + (" / none found" if not m.any() else ""),
                     "Size": size_of(m), "Action": "snapshot" if n_ok else "no DEM (DEM stages blocked)"})
        cat = src.role == "dem_catalog"
        rows.append({"Artifact": "DEM catalog (TNM)", "Source": "Drive", "Status": f"{int(cat.sum())} file(s)",
                     "Size": size_of(cat), "Action": "snapshot (catalog-only coverage)" if cat.any() else "none"})
        rows.append({"Artifact": "Snapshot checksums", "Source": "New atlas",
                     "Status": ("Valid" if snap.get("complete") else "Incomplete") if snap else "Not built",
                     "Size": f"{sum(led.values())} verified files", "Action": "reuse / verify"})
        reg = self.paths.registry_file
        n_feat = len(pd.read_parquet(reg)) if reg.exists() else 0
        rows.append({"Artifact": "Feature cache", "Source": "New atlas", "Status": "Valid" if n_feat else "Missing",
                     "Size": f"{n_feat} registered features", "Action": "reuse or compute"})
        h = self.health()
        rows.append({"Artifact": "Destination integrity", "Source": "New atlas", "Status": "OK" if h["ok"] else "FAILED",
                     "Size": "-", "Action": h["reason"]})
        table = pd.DataFrame(rows)
        atomic_write_csv(self.paths.inventory / "manifests" / "preflight.csv", table)
        missing = [r["Artifact"] for r in rows[:3] if r["Action"] == "BLOCKED"]
        if missing:
            raise PreflightError(f"required raw sources unavailable: {missing}. Nothing was copied.\n"
                                 + table.to_string(index=False))
        return table

    def snapshot(self, sources: pd.DataFrame | None = None) -> dict:
        from .raw_snapshot import snapshot

        if not self.cfg["snapshot_raw"]:
            self._say("[snapshot] snapshot_raw=False: using the existing verified snapshot only")
            from .raw_snapshot import snapshot_state

            return snapshot_state(self.paths.root)
        src = sources if sources is not None else self.discover()
        rep_ = snapshot(src, self.paths.root, self.cfg, verify=self.cfg["verify_checksums"], progress=self._say)
        self._write_raw_inventory(src)
        return rep_

    def _write_raw_inventory(self, src: pd.DataFrame):
        st = pd.read_csv(self.paths.inventory / "manifests" / "snapshot_status.csv")
        inv = src.merge(st[["source_path", "status", "sha256"]].rename(columns={"status": "snapshot_status"}),
                        on="source_path", how="left")
        atomic_write_csv(self.paths.inventory / "raw_inventory.csv", inv)
        prov = []
        for r in inv.itertuples():
            if r.role.startswith("dem"):
                origin, lic = "USGS 3DEP via The National Map", self.cfg["acquisition"]["license"]
            else:
                origin, lic = "DrivenData GEMS competition (GeoDAWN)", "competition data terms (see competition rules)"
            prov.append({"source_id": r.source_id, "role": r.role, "original_path": r.source_path,
                         "snapshot_path": str(self.paths.root / r.snapshot_rel) if isinstance(r.snapshot_rel, str) else "",
                         "origin": origin, "license": lic, "source_mtime_utc": r.mtime_utc, "sha256": r.sha256,
                         "snapshot_status": r.snapshot_status})
        atomic_write_csv(self.paths.inventory / "provenance.csv", pd.DataFrame(prov))

    def require_snapshot(self, dem: bool = False):
        from .raw_snapshot import snapshot_state

        s = snapshot_state(self.paths.root)
        if not s:
            raise PreflightError("snapshot not built - run section 01")
        if s.get("required_missing_or_unverified"):
            raise PreflightError(f"snapshot lacks verified {s['required_missing_or_unverified']} - downstream stopped")
        if dem and s.get("dem_tiles_verified", 0) < s.get("dem_tiles_planned", 0) and \
                not self.cfg["snapshot"]["allow_partial_dem"]:
            raise PreflightError(f"DEM snapshot incomplete ({s['dem_tiles_verified']}/{s['dem_tiles_planned']} tiles "
                                 "verified); re-run section 01, or set snapshot.allow_partial_dem=True explicitly")

    # ------------------------------------------------------------------ 02: geospatial + DEM inventory / coverage
    def geospatial(self) -> dict:
        from .geospatial import GeoDawn, check_grid, read_labels, read_template
        from .inventory import band_registry

        self.require_snapshot()
        fp = stable_hash([self.raw_fp(), self.cfg.section_hash("raw", "grid_expected"),
                          module_hash("geospatial", "inventory")])
        if not self._stage_ok("geospatial", fp):
            gpath, tpath = self.snapshot_path("geodawn_features"), self.snapshot_path("official_template")
            grid, errors, warns = check_grid(gpath, tpath, self.cfg)
            if errors:
                raise PreflightError("grid contract violated:\n  " + "\n  ".join(errors))
            gd = GeoDawn(gpath, self.cfg)
            foot = gd.footprint()
            labels, lfoot, note = read_labels(self.snapshot_path("fault_labels"), grid, self.cfg)
            tmpl = read_template(tpath)
            m = self.paths.d("02_geospatial", "masks")
            prof = grid.profile()
            outs = [atomic_write_json(self.paths.d("02_geospatial", "grids") / "analysis_grid.json",
                                      {**grid.as_dict(), "warnings": warns, "label_resampling": note})]
            for name, a in (("valid_numerical", foot), ("fault_labels", labels & foot), ("label_footprint", lfoot),
                            ("template_valid", tmpl)):
                outs.append(atomic_write_geotiff(m / f"{name}.tif", a.astype(np.uint8), prof, [name], nodata=None))
            reg = band_registry(gpath, self.cfg)
            rb = self.paths.d("03_features", "raw_bands")
            outs += [atomic_write_csv(rb / "raw_band_registry.csv", reg),          # always: later stages read it
                     atomic_write_parquet(rb / "raw_band_registry.parquet", reg)]
            stats = {"footprint_cells": int(foot.sum()), "footprint_km2": float(foot.sum() * grid.pixel_m ** 2 / 1e6),
                     "fault_pixels": int((labels & foot).sum()), "template_cells": int(tmpl.sum()),
                     "template_outside_footprint": int((tmpl & ~foot).sum()), "bands": len(gd.names),
                     "grid": grid.as_dict(), "warnings": warns}
            self.results["geospatial"] = stats
            self._done("geospatial", fp, outs, {"stats": stats})
        self._base = None
        return self.results.get("geospatial", {})

    def base(self) -> dict:
        if self._base is None:
            from .geospatial import GeoDawn, Grid

            g = read_json(self.paths.d("02_geospatial", "grids") / "analysis_grid.json")
            if g is None:
                raise PreflightError("no analysis grid - run section 02")
            m = self.paths.d("02_geospatial", "masks")
            self._base = {"grid": Grid.from_dict(g), "footprint": read_geotiff(m / "valid_numerical.tif", 1).astype(bool),
                          "labels": read_geotiff(m / "fault_labels.tif", 1).astype(bool),
                          "label_footprint": read_geotiff(m / "label_footprint.tif", 1).astype(bool),
                          "template": read_geotiff(m / "template_valid.tif", 1).astype(bool),
                          "gd": GeoDawn(self.snapshot_path("geodawn_features"), self.cfg)}
        return self._base

    def dem_inventory(self) -> pd.DataFrame:
        from .coverage import build_coverage
        from .inventory import dem_inventory, load_catalog

        self.require_snapshot(dem=True)
        b = self.base()
        fp = stable_hash([self.raw_fp(), self.dem_fp(), self.cfg.section_hash("dem"), module_hash("inventory", "coverage")])
        if not self._stage_ok("dem_inventory", fp):
            cat = load_catalog(self.paths.d("00_raw_snapshot", "source_metadata"))
            inv = dem_inventory(self.paths.d("00_raw_snapshot", "dem_1m"), b["grid"], b["footprint"], self.cfg, cat)
            outs = [atomic_write_csv(self.paths.inventory / "dem_inventory.csv", inv)]
            cov = build_coverage(inv, b["grid"], b["footprint"], b["label_footprint"], progress=self._say)
            c, prof = self.paths.d("02_geospatial", "coverage"), b["grid"].profile()
            outs += [atomic_write_geotiff(c / "dem_coverage_catalog_tiles.tif", cov["coverage"], prof, ["dem_coverage"]),
                     atomic_write_geotiff(c / "dem_overlap_count.tif", cov["count"], prof, ["tiles"], nodata=None),
                     atomic_write_geotiff(c / "dem_project_id.tif", cov["project"], prof, ["project_id"], nodata=-1),
                     atomic_write_geotiff(c / "dem_seam_mask.tif", cov["seam"].astype(np.uint8), prof, ["seam"], nodata=None),
                     atomic_write_geotiff(self.paths.d("02_geospatial", "masks") / "multimodal_coverage.tif",
                                          cov["multimodal"], prof, ["bit1_geodawn_bit2_labels_bit4_dem"], nodata=None),
                     atomic_write_csv(self.paths.d("02_geospatial", "footprints") / "dem_tile_coverage.csv", cov["tile_table"]),
                     atomic_write_json(self.paths.d("02_geospatial", "footprints") / "dem_projects.json", cov["project_ids"])]
            if cat is not None:
                outs.append(self._catalog_footprints(cat, b))
            counts = inv.status.value_counts().to_dict() if len(inv) else {}
            self.results["dem_inventory"] = {"status_counts": counts, **cov["stats"],
                                             "missing_metadata": int((inv.get("has_metadata") == False).sum())  # noqa: E712
                                             if len(inv) else 0}
            if not counts.get("VALID"):
                self.results["limitations"].append("no VALID USGS 1 m DEM tile: DEM sections show limitation panels")
            self._done("dem_inventory", fp, outs, {"stats": self.results["dem_inventory"]})
        return pd.read_csv(self.paths.inventory / "dem_inventory.csv")

    def _catalog_footprints(self, cat: pd.DataFrame, b) -> Path:
        from rasterio.crs import CRS
        from rasterio.warp import transform_bounds

        from .geospatial import window_of_bounds

        g = b["grid"]
        a = np.zeros((g.height, g.width), np.uint8)
        for r in cat.drop_duplicates("tile_id").itertuples():
            try:
                bb = transform_bounds(CRS.from_epsg(26900 + int(r.zone)), g.crs, r.west, r.south, r.east, r.north)
            except Exception:
                continue
            w = window_of_bounds(bb, g)
            if w:
                a[w[0]:w[1], w[2]:w[3]] = 1
        return atomic_write_geotiff(self.paths.d("02_geospatial", "footprints") / "dem_catalog_nominal_footprint.tif",
                                    a, g.profile(), ["catalog_tile_rectangles"], nodata=None)

    def acquire(self) -> pd.DataFrame | None:
        if not self.cfg["enable_new_dem_downloads"]:
            self._say("[acquire] enable_new_dem_downloads=False - skipped (existing tiles are used)")
            return None
        from . import acquisition
        from .inventory import load_catalog

        b = self.base()
        cat = load_catalog(self.paths.d("00_raw_snapshot", "source_metadata"))
        if cat is None:
            raise PreflightError("no TNM catalog in the snapshot (source_metadata) - nothing to select from")
        inv = pd.read_csv(self.paths.inventory / "dem_inventory.csv")
        cov = read_geotiff(self.paths.d("02_geospatial", "coverage") / "dem_coverage_catalog_tiles.tif", 1)
        plan = acquisition.plan_downloads(cat, inv, b["grid"], b["footprint"], cov, self.cfg)
        atomic_write_csv(self.paths.d("07_cache", "download_state") / "download_plan.csv", plan)
        res = acquisition.run(plan, self.paths.d("00_raw_snapshot", "dem_1m"),
                              self.paths.d("07_cache", "download_state"), self.cfg, progress=self._say)
        atomic_write_csv(self.paths.d("07_cache", "download_state") / "download_results.csv", res)
        ver = res[res.status == "VERIFIED"] if len(res) else res
        if len(ver):          # register downloaded tiles as snapshot sources with their checksums
            sc = self.paths.inventory / "source_checksums.csv"
            old = pd.read_csv(sc)
            add = pd.DataFrame({"role": "dem_tile", "source_path": ver.url, "snapshot_path": ver.path,
                                "size_bytes": ver.bytes, "sha256": ver.sha256})
            atomic_write_csv(sc, pd.concat([old[~old.snapshot_path.isin(add.snapshot_path)], add], ignore_index=True))
        return res

    # ------------------------------------------------------------------ 03: feature families
    def _writer(self, stage: str, fp: str) -> FeatureWriter:
        b = self.base()
        return FeatureWriter(self.paths.root, b["grid"], b["footprint"], stage, fp)

    def _register(self, w: FeatureWriter, stage: str):
        reg = Registry(self.paths.registry_file)
        reg.drop_stage(stage)
        reg.upsert(w.rows)
        reg.save()

    def raw_bands(self) -> int:
        b = self.base()
        fp = stable_hash([self.raw_fp(), module_hash("pipeline")])
        if self._stage_ok("raw_bands", fp):
            return self.results.get("raw_bands", 0)
        w = self._writer("raw_bands", fp)
        reg = pd.read_csv(self.paths.root / "03_features/raw_bands/raw_band_registry.csv")
        for r in reg.itertuples():
            w.put(r.band_name, b["gd"].read(r.band_name), family="raw", parent=f"geodawn:band{r.band_index}",
                  unit=str(r.unit), operator="raw band, nodata -> NaN", meaning=str(r.meaning),
                  status="COMPLETE")
        self._register(w, "raw_bands")
        self.results["raw_bands"] = len(reg)
        self._done("raw_bands", fp, w.paths)
        return len(reg)

    def _family(self, name: str, module, sections: tuple):
        b = self.base()
        fp = stable_hash([self.raw_fp(), self.cfg.section_hash(*sections), module_hash(name, "fft_operators",
                                                                                       "structure_tensor",
                                                                                       "feature_registry", "magnetic")])
        if self._stage_ok(name, fp):
            return self.results.get(name, {})
        w = self._writer(name, fp)
        out = module.build(b["gd"], b["footprint"], self.cfg, w, *(() if name == "terrain" else (self.rng,)))
        self._register(w, name)
        outs = list(w.paths)
        res = {}
        if name in ("magnetic", "gravity"):
            outs += self._table(out["spectrum"], "tables", f"{name}_radial_spectrum")
            outs += self._table(out["band_energy"], "tables", f"{name}_band_energy")
            if len(out["duplicates"]):
                outs += self._table(out["duplicates"], "tables", f"{name}_delivered_vs_recomputed")
            from .circular_statistics import axial_mean, from_vectors

            c2, s2, coh = out["orientation"]
            m = b["footprint"] & np.isfinite(c2)
            sub = np.flatnonzero(m.ravel())[:: max(m.sum() // 300000, 1)]
            az = from_vectors(c2.ravel()[sub], s2.ravel()[sub])
            res["strike_deg"], res["strike_R"] = axial_mean(az, coh.ravel()[sub])
            pers = read_geotiff(w.path_for(name, f"{out_prefix(name)}_edge_persistence"), 1)
            res["persistent_share"] = float(np.nanmean(pers[b["footprint"]] >= 0.6))
            be = out["band_energy"]
            res["band_energy_text"] = ", ".join(f"{r.wavelength_from_km:g}-{r.wavelength_to_km:g} km {100 * r.energy_share:.1f} %"
                                                for r in be.itertuples())
        if name == "terrain":
            from .circular_statistics import axial_mean, from_vectors

            c2 = read_geotiff(w.path_for("terrain", "terrain_strike_c2_1000m"), 1)
            s2 = read_geotiff(w.path_for("terrain", "terrain_strike_s2_1000m"), 1)
            coh = read_geotiff(w.path_for("terrain", "terrain_coherence_1000m"), 1)
            m = np.isfinite(c2)
            sub = np.flatnonzero(m.ravel())[:: max(m.sum() // 300000, 1)]
            res["strike_deg"], res["strike_R"] = axial_mean(from_vectors(c2.ravel()[sub], s2.ravel()[sub]),
                                                            coh.ravel()[sub])
        self.results[name] = res
        self._done(name, fp, outs, {"result": res})
        return res

    def magnetic(self):
        from . import magnetic

        return self._family("magnetic", magnetic, ("potential", "evidence"))

    def gravity(self):
        from . import gravity

        return self._family("gravity", gravity, ("potential", "evidence"))

    def terrain(self):
        from . import terrain

        return self._family("terrain", terrain, ("terrain",))

    def dem(self, mode: str | None = None) -> dict:
        from . import dem as dem_stage

        mode = mode or self.cfg.dem_mode
        inv = pd.read_csv(self.paths.inventory / "dem_inventory.csv")
        if not (inv.status == "VALID").any():
            self.results["dem"] = {"status": "NO_DEM", "mode": mode}
            return self.results["dem"]
        b = self.base()
        name = f"dem_{mode}"
        fp = stable_hash([self.dem_fp(), dem_stage.mode_fingerprint(self._mode_cfg(mode))])
        if self._stage_ok(name, fp):
            return self.results.get("dem", {})
        cov_hint = read_geotiff(self.paths.d("02_geospatial", "coverage") / "dem_coverage_catalog_tiles.tif", 1)
        w = self._writer(name, fp)
        res = dem_stage.run(self.paths, self.cfg, b["grid"], b["footprint"], inv, cov_hint, w, mode=mode,
                            progress=self._say)
        if res["status"] != "COMPLETE":
            self._say(f"[dem] {res}")
            return res
        self._register(w, name)
        self.results["dem"] = {k: v for k, v in res.items() if k != "outputs"}
        self._done(name, fp, list(w.paths) + [Path(o) for o in res["outputs"]], {"result": self.results["dem"]})
        return self.results["dem"]

    def dem_examples(self) -> dict:
        """2 m DETAILED example windows (and their 10 m counterparts) for the fine / coarse comparison."""
        from . import dem as dem_stage

        b = self.base()
        inv = pd.read_csv(self.paths.inventory / "dem_inventory.csv")
        ids = dem_stage.pick_example_windows(self.paths, "FAST_10M", int(self.cfg["dem"]["detailed_example_windows"]))
        if not ids:
            return {"status": "NO_WINDOWS"}
        cov_hint = read_geotiff(self.paths.d("02_geospatial", "coverage") / "dem_coverage_catalog_tiles.tif", 1)
        out = {}
        for mode in ("FAST_10M", "DETAILED_2M"):
            out[mode] = dem_stage.run(self.paths, self.cfg, b["grid"], b["footprint"], inv, cov_hint,
                                      self._writer(f"dem_examples_{mode}", "examples"), mode=mode, window_ids=ids,
                                      progress=self._say)
        self.results["dem_examples"] = {"windows": ids}
        rep.save(self.paths, self.results)
        return out

    def _mode_cfg(self, mode):
        d = self.cfg.as_dict()
        d["dem_mode"] = mode
        return AtlasConfig(d)

    def dem_coverage(self) -> np.ndarray | None:
        p = self.paths.d("02_geospatial", "coverage") / f"dem_coverage_{self.cfg.dem_mode}.tif"
        return read_geotiff(p, 1) if p.exists() else None

    def atlas(self):
        from .atlas import GeologicalAtlas

        return GeologicalAtlas.open(self.paths.root)

    def cross_modal(self) -> dict:
        from . import cross_modal

        b = self.base()
        A = self.atlas()
        cov = self.dem_coverage()
        dem_ok = cov is not None and "dem_edge_p90" in set(A.registry.feature_id)
        fp = stable_hash([self.results["stages"].get(s, {}).get("fingerprint") for s in
                          ("magnetic", "gravity", "terrain", f"dem_{self.cfg.dem_mode}")] +
                         [self.cfg.section_hash("evidence"), module_hash("cross_modal")])
        if self._stage_ok("cross_modal", fp):
            return self.results.get("cross_modal", {})
        w = self._writer("cross_modal", fp)
        out = cross_modal.build(A.load_feature, b["footprint"], cov if dem_ok else None, self.cfg, w, b["grid"].pixel_m)
        self._register(w, "cross_modal")
        cnt = out["count"]
        res = {"two_plus_share": float(np.mean(cnt[b["footprint"]] >= 2)), "dem_used": bool(dem_ok)}
        for k, (a_, b_) in {"mag_grav_strike_diff": ("magnetic", "gravity"), "mag_terr_strike_diff": ("magnetic", "terrain"),
                            "grav_terr_strike_diff": ("gravity", "terrain")}.items():
            sa = self.results.get(a_, {}).get("strike_deg")
            sb = self.results.get(b_, {}).get("strike_deg")
            res[k] = rep.strike_diff(sa, sb)
        if not dem_ok:
            self.results["limitations"].append("cross-modal evidence built without DEM (no DEM statistics available)")
        self.results["cross_modal"] = res
        self._done("cross_modal", fp, w.paths, {"result": res})
        return res

    # ------------------------------------------------------------------ 04: EDA
    def scalar_features(self) -> list[str]:
        r = self.atlas().registry
        r = r[r.status.isin(["COMPLETE", "alias_of_delivered"])]
        return [f for f, o in zip(r.feature_id, r.orientation_convention.fillna("")) if not o and f != "terrain_elevation"
                and not f.startswith("dem_elevation")]

    def faults(self) -> dict:
        from . import fault_proximity as fpx

        b = self.base()
        fp = stable_hash([self.raw_fp(), module_hash("fault_proximity", "circular_statistics")])
        if not self._stage_ok("faults", fp):
            net = fpx.network(b["labels"], b["grid"])
            outs = self._table(net["traces"], "tables", "mapped_fault_traces")
            outs += self._table(net["regions"], "tables", "mapped_fault_density_by_region")
            outs.append(atomic_write_geotiff(self.paths.d("04_eda", "geological_evidence") / "mapped_fault_skeleton.tif",
                                             net["skeleton"].astype(np.uint8), b["grid"].profile(), ["skeleton"], nodata=None))
            dist = fpx.distance_m(b["labels"], b["grid"].pixel_m)
            outs.append(atomic_write_geotiff(self.paths.d("04_eda", "geological_evidence") / "distance_to_mapped_fault_m.tif",
                                             dist, b["grid"].profile(), ["distance_m"]))
            self.results["faults"] = {"summary": net["summary"]}
            self._done("faults", fp, outs, {"summary": net["summary"]})
        return self.results["faults"]

    def proximity(self) -> pd.DataFrame:
        from . import fault_proximity as fpx

        b = self.base()
        A = self.atlas()
        ids = self.scalar_features()
        fp = stable_hash([self.raw_fp(), sorted(ids), self.cfg.section_hash("proximity"), module_hash("fault_proximity")])
        if not self._stage_ok("proximity", fp):
            dist = read_geotiff(self.paths.d("04_eda", "geological_evidence") / "distance_to_mapped_fault_m.tif", 1)
            tab = fpx.band_table(ids, A.load_feature, dist, b["footprint"], self.cfg, self.rng)
            outs = self._table(tab, "tables", "feature_by_fault_distance_band")
            top = (tab.drop_duplicates("feature_id").dropna(subset=["near_far_median_contrast_iqr"])
                   .assign(a=lambda d: d.near_far_median_contrast_iqr.abs()).sort_values("a", ascending=False).head(12))
            curves = []
            for fid in top.feature_id.head(6):
                c = fpx.response_curve(A.load_feature(fid), dist, b["footprint"], self.rng)
                curves.append(c.assign(feature_id=fid))
            if curves:
                outs += self._table(pd.concat(curves), "tables", "feature_response_vs_distance")
            self.results["proximity"] = {"top_contrast": [{"feature_id": r.feature_id,
                                                           "contrast": float(r.near_far_median_contrast_iqr)}
                                                          for r in top.itertuples()]}
            self._done("proximity", fp, outs, {"result": self.results["proximity"]})
        return pd.read_csv(self.paths.d("04_eda", "tables") / "feature_by_fault_distance_band.csv")

    PROFILE_FEATURES = ("mag_field", "mag_thd", "mag_tilt", "grav_field", "grav_tilt", "terrain_elevation",
                        "terrain_slope_1000m", "xm_evidence_mean", "dem_edge_p90", "dem_scarp_p90")

    def profiles(self) -> pd.DataFrame:
        from scipy import ndimage

        from . import fault_proximity as fpx

        b = self.base()
        A = self.atlas()
        fp = stable_hash([self.results["stages"].get("faults", {}).get("fingerprint"), self.cfg.section_hash("proximity"),
                          sorted(A.registry.feature_id), module_hash("fault_proximity")])
        if not self._stage_ok("profiles", fp):
            traces = pd.read_csv(self.paths.d("04_eda", "tables") / "mapped_fault_traces.csv")
            sel = fpx.pick_profile_traces(traces, self.cfg, shape=b["labels"].shape)
            sk = read_geotiff(self.paths.d("04_eda", "geological_evidence") / "mapped_fault_skeleton.tif", 1).astype(bool)
            lab = ndimage.label(sk, structure=np.ones((3, 3), bool))
            feats = {f: A.load_feature(f) for f in self.PROFILE_FEATURES if f in set(A.registry.feature_id)}
            prof = fpx.profiles(sel, lab, feats, b["grid"], self.cfg)
            outs = self._table(sel, "profiles", "profile_traces") + self._table(prof, "profiles", "fault_normal_profiles")
            self._done("profiles", fp, outs)
        return pd.read_csv(self.paths.d("04_eda", "profiles") / "fault_normal_profiles.csv")

    def quality(self) -> pd.DataFrame:
        from . import analytics
        from .geospatial import distance_to_edge_m

        b = self.base()
        A = self.atlas()
        ids = self.scalar_features()
        fp = stable_hash([sorted(zip(A.registry.feature_id, A.registry.fingerprint)), self.cfg.section_hash("analytics"),
                          module_hash("analytics")])
        if not self._stage_ok("quality", fp):
            edge = distance_to_edge_m(b["footprint"], b["grid"].pixel_m)
            seam_p = self.paths.d("02_geospatial", "coverage") / "dem_seam_mask.tif"
            seam = read_geotiff(seam_p, 1).astype(bool) if seam_p.exists() else None
            q, rho, sc = analytics.quality(ids, A.load_feature, b["footprint"], edge, seam, self.dem_coverage(), self.cfg,
                                           self.rng)
            outs = self._table(q, "tables", "feature_quality") + self._table(sc, "tables", "scale_consistency")
            outs += self._table(rho.reset_index().rename(columns={"index": "feature_id"}), "tables",
                                "feature_spearman_matrix")
            self.results["quality"] = {"n": int(len(q)), "classes": q["class"].value_counts().to_dict(),
                                       "retain": int(q.retain.sum()), "retained": q[q.retain].feature_id.tolist(),
                                       "dropped": dict(zip(q[~q.retain].feature_id, q[~q.retain].retention_reason))}
            self._done("quality", fp, outs, {"result": {k: v for k, v in self.results["quality"].items()
                                                        if k in ("n", "classes", "retain")}})
        return pd.read_csv(self.paths.d("04_eda", "tables") / "feature_quality.csv")

    def lineaments(self):
        from scipy import ndimage

        from . import lineaments as ln
        from .circular_statistics import axial_mean
        from .feature_registry import pct_rank

        b = self.base()
        A = self.atlas()
        reg_ids = set(A.registry.feature_id)
        fp = stable_hash([self.results["stages"].get("cross_modal", {}).get("fingerprint"),
                          self.cfg.section_hash("lineaments", "evidence"), module_hash("lineaments")])
        if not self._stage_ok("lineaments", fp):
            g = b["grid"]
            E = A.load_feature("xm_evidence_mean")
            lab, n, thr = ln.extract(E, b["footprint"], self.cfg, g.pixel_m)
            segs = ln.segments(lab, n, self.cfg, g)
            cov = self.dem_coverage()
            maps = {"magnetic": A.load_feature("mag_boundary_strength"), "gravity": A.load_feature("grav_boundary_strength"),
                    "terrain": pct_rank(A.load_feature("terrain_slope_1000m"), b["footprint"]),
                    "persistence": A.load_feature("xm_multiscale_persistence"),
                    "continuity": A.load_feature("xm_continuity")}
            maps["topography"] = maps["terrain"]
            if "dem_edge_p90" in reg_ids and cov is not None:
                maps["dem"] = pct_rank(A.load_feature("dem_edge_p90"), b["footprint"] & (cov > 0))
                maps["topography"] = np.where(np.isfinite(maps["dem"]), np.fmax(maps["terrain"], maps["dem"]), maps["terrain"])
            oa = [A.load_feature(f) for f in ("xm_orient_magn_grav", "xm_orient_magn_terr", "xm_orient_grav_terr")
                  if f in reg_ids]
            maps["orientation_agreement"] = np.nanmean(np.stack(oa), 0) if oa else None
            scale_ranks = {}
            for h in [0] + list(self.cfg["potential"]["continuation_m"]):
                fid = "mag_thd" if h == 0 else f"mag_thd_uc{h}"
                if fid in reg_ids:
                    scale_ranks[f"magnetic_{h}m"] = pct_rank(A.load_feature(fid), b["footprint"])
            dist = read_geotiff(self.paths.d("04_eda", "geological_evidence") / "distance_to_mapped_fault_m.tif", 1)
            seam_p = self.paths.d("02_geospatial", "coverage") / "dem_seam_mask.tif"
            seam_d = None
            if seam_p.exists():
                s = read_geotiff(seam_p, 1).astype(bool)
                seam_d = (ndimage.distance_transform_edt(~s) * g.pixel_m).astype(np.float32) if s.any() else None
            df = ln.inventory(segs, maps, g, dist, seam_d, cov, scale_ranks)
            outs = ln.export(df, g, self.paths.d("04_eda", "candidate_lineaments"),
                             float(self.cfg["lineaments"]["simplify_m"])) if len(df) else []
            fs = self.results.get("faults", {}).get("summary", {})
            st, R = axial_mean(df.strike_deg, df.length_km) if len(df) else (np.nan, np.nan)
            far = df.mapped_fault_distance_m > 1000 if len(df) else pd.Series(dtype=bool)
            self.results["lineaments"] = {"n": int(len(df)), "threshold": thr, "total_km": float(df.length_km.sum()) if len(df) else 0,
                                          "n_multi": int((df.evidence_count >= 2).sum()) if len(df) else 0,
                                          "n_far": int(far.sum()), "n_multi_far": int(((df.evidence_count >= 2) & far).sum())
                                          if len(df) else 0, "strike_deg": st, "strike_R": R,
                                          "strike_diff_vs_faults": rep.strike_diff(st, fs.get("length_weighted_mean_strike_deg"))}
            self._done("lineaments", fp, outs, {"result": self.results["lineaments"]})
        try:
            return A.load_lineaments()
        except FileNotFoundError:
            return None

    # ------------------------------------------------------------------ 06: model-ready foundation
    def ml_ready(self) -> dict:
        from .feature_registry import AXIAL

        b = self.base()
        A = self.atlas()
        q = pd.read_csv(self.paths.d("04_eda", "tables") / "feature_quality.csv")
        keep = set(q[q.retain].feature_id)
        r = A.registry[A.registry.status.isin(["COMPLETE", "alias_of_delivered"])]
        axial = r[r.orientation_convention == AXIAL].feature_id.tolist()
        fp = stable_hash([sorted(keep), sorted(axial), sorted(zip(r.feature_id, r.fingerprint))])
        if not self._stage_ok("ml_ready", fp):
            fs = self.paths.d("06_ml_ready", "feature_stack")
            schema, norm, outs = {"grid": b["grid"].as_dict(), "families": {}}, {}, []
            for fam in sorted(r.family.unique()):
                ids = [f for f in r[r.family == fam].feature_id if f in keep or f in axial]
                if not ids:
                    continue
                stack = np.stack([A.load_feature(f) for f in ids]).astype(np.float32)
                outs.append(atomic_write_geotiff(fs / f"{fam}_stack.tif", stack, b["grid"].profile(), ids))
                for i, f in enumerate(ids):
                    v = stack[i][b["footprint"] & np.isfinite(stack[i])]
                    if f in axial or not v.size:
                        norm[f] = {"kind": "axial" if f in axial else "empty", "center": 0.0, "scale": 1.0}
                    else:
                        q05, q50, q95 = np.percentile(v[:: max(v.size // 2_000_000, 1)], [5, 50, 95])
                        norm[f] = {"kind": "scalar", "center": float(q50), "scale": float(q95 - q05) or 1.0,
                                   "q05": float(q05), "q95": float(q95)}
                schema["families"][fam] = {"file": f"06_ml_ready/feature_stack/{fam}_stack.tif", "bands": ids}
                del stack
            cm = self.paths.d("06_ml_ready", "coverage_masks")
            for n_ in ("valid_numerical", "label_footprint", "template_valid", "multimodal_coverage"):
                p = self.paths.d("02_geospatial", "masks") / f"{n_}.tif"
                if p.exists():
                    outs.append(atomic_write_geotiff(cm / f"{n_}.tif", read_geotiff(p, 1), b["grid"].profile(), [n_],
                                                     nodata=None))
            cov = self.dem_coverage()
            if cov is not None:
                outs.append(atomic_write_geotiff(cm / "dem_coverage.tif", cov, b["grid"].profile(), ["dem_coverage"]))
            schema["normalization"] = "06_ml_ready/normalization/robust_quantiles.json"
            schema["orientation_convention"] = AXIAL
            schema["note"] = ("fault labels are NOT part of the stack; masks are separate. Normalisation is label-free "
                              "(footprint quantiles). Retained = analytics 'retain' + every axial pair component.")
            outs += [atomic_write_json(self.paths.d("06_ml_ready", "normalization") / "robust_quantiles.json", norm),
                     atomic_write_json(self.paths.d("06_ml_ready", "schemas") / "feature_stack_schema.json", schema)]
            self._done("ml_ready", fp, outs, {"families": {k: len(v["bands"]) for k, v in schema["families"].items()}})
        return read_json(self.paths.d("06_ml_ready", "schemas") / "feature_stack_schema.json")

    # ------------------------------------------------------------------ 08: reports and artifacts
    def artifact_report(self) -> pd.DataFrame:
        rows = []
        for p in sorted(self.paths.root.rglob("*")):
            if p.is_file() and not p.name.endswith(".partial") and "00_raw_snapshot" not in p.parts:
                rel = p.relative_to(self.paths.root).as_posix()
                rows.append({"path": rel, "section": rel.split("/")[0], "bytes": p.stat().st_size,
                             "type": p.suffix.lstrip(".")})
        df = pd.DataFrame(rows)
        st = []
        for f in sorted(self.paths.stage_dir.glob("*.json")):
            r = read_json(f)
            if r and "stage" in r:
                stt, why = stage_status(self.paths.stage_dir, self.paths.root, r["stage"], r.get("fingerprint"))
                st.append({"stage": r["stage"], "status": stt, "detail": why, "fingerprint": r.get("fingerprint"),
                           "written_utc": r.get("written_utc"), "outputs": len(r.get("outputs", []))})
        stages = pd.DataFrame(st)
        atomic_write_csv(self.paths.reports / "artifact_inventory.csv", df)
        atomic_write_csv(self.paths.reports / "stage_status.csv", stages)
        return stages

    def write_reports(self) -> list:
        arts = self.artifact_report()
        return rep.write_all(self.paths, self.cfg, self.results, arts.to_dict("records"))


def out_prefix(family: str) -> str:
    return {"magnetic": "mag", "gravity": "grav"}[family]
