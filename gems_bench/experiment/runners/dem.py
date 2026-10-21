"""Kind ``dem_pilot`` (Experiment F1-F3): inventory -> balanced pilot -> explicit download -> 1 m morphology ->
100 m mosaic registered as ``dem:*`` features. F4 comparisons are ordinary lgbm5 / cnn5 configs with
``evaluation.restrict: dem_coverage``."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .base import Runner


class DemPilotRunner(Runner):
    kind = "dem_pilot"

    @property
    def out_dir(self) -> Path:
        return self.exp.paths.processed / "dem_1m_100m"

    def prepare(self):
        from ...dem import inventory as I
        from ...dem import pilot as P

        ctx, rd = self.ctx, self.rd
        js = self.exp.paths.root / self.cfg.get("dem.urls_json", "data/raw/dem_1m/tnm_urls.json")
        inv = I.study_area_intersection(I.load_inventory(js), ctx, float(self.cfg.get("dem.min_valid_frac", 0.05)))
        rd.save_table(inv, "dem_inventory")
        rd.save_json(I.storage_estimate(inv), "dem_storage_estimate")
        n_hdr = int(self.cfg.get("dem.inspect_headers", 0) or 0)
        tiles = I.choose_per_tile(inv[inv.in_study_area], self.cfg.get("dem.overlap_policy", "latest"))
        if n_hdr:
            h = I.inspect_headers(list(tiles.url), n_hdr)
            rd.save_table(h, "dem_headers")
            rd.save_table(I.verify_corners(tiles, h), "dem_corner_check")
        maps = {m: ctx.oof(m) for m in self.cfg.get("dem.disagreement_models", ["G1", "P4_B_gated"]) if ctx.has_oof(m)}
        short = P.missed_short_faults(ctx, ctx.oof("G1")) if ctx.has_oof("G1") else None
        ref = self.cfg.get("dem.reference_cnn", "DL00_CNN_B_dti")
        cnn_missed = P.missed_withheld(ctx, ref) if ctx.has_discovery(ref) else None
        desc = P.tile_descriptors(ctx, tiles, short_missed=short, cnn_missed=cnn_missed, maps=maps)
        rd.save_table(desc, "dem_tile_descriptors")
        sel = P.select_pilot(desc, n_total=int(self.cfg.get("dem.n_tiles", 24)), quotas=self.cfg.get("dem.quotas"),
                             seed=self.cfg.seed)
        sel = sel.merge(tiles[["tile_id", "file", "project_dir", "size_bytes"]], on="tile_id", how="left")
        rd.save_table(sel, "dem_pilot_tiles")
        return {"candidates": int(len(tiles)), "selected": int(len(sel))}

    def train(self):
        """Download (only if ``dem.download: true``) and process the pilot tiles; resumable per tile."""
        from ...dem import morphology as M
        from ...dem.download import download_selected

        rd, root = self.rd, self.exp.paths.root
        sel = rd.load_table("dem_pilot_tiles")
        if self.cfg.get("runtime.dem_download", False):        # runtime = non-semantic: approving keeps the id
            log = download_selected(sel, root, max_tiles=int(self.cfg.get("dem.max_tiles", 30)),
                                    max_gib=float(self.cfg.get("dem.max_gib", 15.0)), sync=self.exp.drive)
            rd.save_table(log, "dem_download_log")
        cfg = M.MorphConfig(**(self.cfg.get("dem.morphology", {}) or {}))
        prof = self.ctx.data.profile
        projects = {p: i for i, p in enumerate(sorted(sel.project_dir.dropna().unique()), start=1)}
        tdir = self.out_dir / "tiles"
        done = []
        for r in sel.itertuples():
            src = root / "data" / "raw" / "dem_1m" / str(r.project_dir) / str(r.file)
            dst = tdir / f"{r.tile_id}.tif"
            if dst.exists():
                done.append(dst)
                continue
            if not src.exists():
                rd.log(f"tile {r.tile_id}: raw GeoTIFF missing ({src}); skipped")
                continue
            out = M.process_tile(src, prof, cfg, projects.get(r.project_dir, 0))
            if out:
                M.save_tile(out, prof, dst, {"tile_id": r.tile_id, "project": r.project_dir, "source": str(src.name),
                                             "reason": r.reason, "config": M.config_record(cfg)})
                done.append(dst)
                self.exp.manifest.mark_unit(f"dem/{r.tile_id}")
                self.push_paths([dst, dst.with_suffix(".json")])
        if done:
            mos = M.mosaic(done, {**prof}, self.out_dir / "dem_features_100m.tif")
            rd.save_json({"projects": projects, "config": M.config_record(cfg), "tiles": [p.name for p in done]},
                         "dem_mosaic_provenance")
            self.push_paths([mos])
        return {"tiles_processed": len(done)}

    def evaluate(self):
        import matplotlib.pyplot as plt
        import numpy as np
        import rasterio

        from ...dem.benchmark import coverage_report
        from ...visualization.figures import coverage_map, raster_windows

        mos = self.out_dir / "dem_features_100m.tif"
        if not mos.exists():
            return {"mosaic": False}
        with rasterio.open(mos) as src:
            bands = {d: src.read(i + 1) for i, d in enumerate(src.descriptions)}
        cov = np.nan_to_num(bands["coverage"])
        ctx = self.ctx
        self.rd.save_table(coverage_report(ctx, cov >= float(self.cfg.get("evaluation.min_coverage", 0.9))), "dem_coverage_report")
        fig = coverage_map(cov, ctx.data.valid)
        self.rd.save_fig(fig, "dem_coverage_map")
        plt.close(fig)
        rr, cc = np.nonzero((cov > 0.9) & ctx.data.labels)
        if len(rr):
            idx = np.random.default_rng(self.cfg.seed).choice(len(rr), min(3, len(rr)), replace=False)
            win = [(int(rr[i]), int(cc[i]), 25) for i in idx]
            fig = raster_windows({"100 m det_elev slope": ctx.store.get("raw:det_elev_slope")
                                  if "raw:det_elev_slope" in ctx.store.specs else ctx.data.read_band("det_elev_slope"),
                                  "1 m scarp density": bands["scarp_density"], "1 m microrelief": bands["microrelief_rms"],
                                  "1 m orient. entropy": bands["orient_entropy"]}, win, ctx.data.labels,
                                 title="1 m morphology aggregated to 100 m vs native terrain (mapped faults in black)")
            self.rd.save_fig(fig, "dem_1m_vs_100m_examples")
            plt.close(fig)
        return {"mosaic": True, "covered_km2": float((cov >= 0.9).sum() * 0.01)}


__all__ = ["DemPilotRunner", "pd"]
