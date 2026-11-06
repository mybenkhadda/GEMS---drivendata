"""Production E6 runner: prepare/cache -> CV + discovery -> DTI -> FULL prediction -> GeoTIFF.

Uses the real Phase-5 training/checkpoint machinery. E6 results never reuse a
W2 submission. No ablation suite, no model comparisons, no raw data mirrors.
"""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd

from .cnn5 import Cnn5Runner


class E6Runner(Cnn5Runner):
    kind = "e6"

    @property
    def ctx(self):
        ctx = super().ctx
        if not getattr(self, "_e6_inputs_ready", False):
            from ...e6.features import stage_cached_dem, register_e6_dem
            if self.exp.paths.drive:
                ctx.store.drive_mirror = (self.exp.paths.root, self.exp.paths.drive)
            info = stage_cached_dem(self.exp, ctx, max_tiles=int(self.cfg.get("e6.max_cached_dem_tiles", 32)))
            available = register_e6_dem(ctx, Path(info["folder"]), info["fingerprint"],
                                        self.exp.paths.root)
            self._e6_inputs_ready = True
            self.rd.save_json({**info, "registered_channels": available}, "e6_input_cache")
            print(f"[E6] DEM cache {info['fingerprint']}: {info['n_source_tiles']} preexisting 10m tiles; "
                  f"{len(available)} cached 100m channels")
        return ctx

    def prepare(self):
        from ...training.cv import ensure_channels
        result = super().prepare()
        spec = self.spec()
        if spec.model != "geostructurenet_e6":
            raise ValueError("E6 milestone must use a GeoStructureNetE6 model, not an old model")
        assert spec.aux_dist > 0 and spec.aux_orient > 0, "E6 production requires auxiliary geometry heads"
        ensure_channels(self.ctx, spec)
        if self.exp.drive:
            # Persist only derived per-band E6 caches; raw USGS DEMs are not mirrored.
            self.exp.drive.push([self.exp.paths.rel(self.ctx.store.p5_dir)])
        self.rd.save_json({"model": "E6", "baseline": "original gated dual encoder",
                           "dem_modality_dropout": spec.e6_dropout,
                           "warm_start_dir": spec.e6_warm_dir,
                           "field_features": list(spec.gsn_groups.get("subsurface", [])),
                           "dem_features": list(spec.gsn_groups.get("dem", [])),
                           "mask": "dem:dem_coverage", "training_only_heads": ["dist", "orient"],
                           "normalisation": "robust for features; raw 0..1 for availability mask",
                           "external_data": "USGS 3DEP public-domain DEM (GEMS explicitly allows external data)"},
                          "e6_architecture")
        return result

    def evaluate(self):
        """Only E6 fold and discovery DTI; no W2 comparison or R&D ablation."""
        from ...metric import dti_components, pool
        ctx, rd = self.ctx, self.rd
        rows, geographic, discovery = [], [], []
        for k in ctx.fold_ids:
            raster = rd.raster_path(f"oof_fold{k}")
            if not raster.is_file():
                raise FileNotFoundError(f"E6 missing geographic fold {k}: {raster}")
            score = rd.read_raster(f"oof_fold{k}")
            mask = ctx.folds.val(k) & ctx.data.valid
            stats = dti_components(score, ctx.data.labels, eval_mask=mask)
            geographic.append(stats)
            rows.append({"protocol": "V3_GEOGRAPHIC", "unit": f"cv_fold{k}", **stats.as_dict()})
        for d in range(ctx.splits.n_splits):
            raster = rd.raster_path(f"disc{d}")
            if not raster.is_file():
                raise FileNotFoundError(f"E6 missing withheld-system prediction {d}: {raster}")
            score = rd.read_raster(f"disc{d}")
            mask = ctx.splits.eval_zone(d) & ctx.data.valid
            stats = dti_components(score, ctx.data.labels, eval_mask=mask)
            discovery.append(stats)
            rows.append({"protocol": "V1_DISCOVERY", "unit": f"disc{d}", **stats.as_dict()})
        rows.append({"protocol": "V3_GEOGRAPHIC", "unit": "POOLED", **pool(geographic).as_dict()})
        rows.append({"protocol": "V1_DISCOVERY", "unit": "POOLED", **pool(discovery).as_dict()})
        rd.save_table(pd.DataFrame(rows), "e6_dti")
        print("[E6] Geographic DTI", pool(geographic).dti,
              "| Discovery DTI", pool(discovery).dti)
        return {"geographic_dti": pool(geographic).dti,
                "discovery_dti": pool(discovery).dti,
                "n_fold": len(geographic), "n_split": len(discovery)}

    def export(self):
        """Resumable per-fold FULL prediction and exactly one validated NEW E6 GeoTIFF."""
        import torch
        from ...cnn import device_of
        from ...phase4_training import pooled_context, to_tensor_stack
        from ...training.checkpoint import CheckpointManager
        from ...training.trainer import load_unit_model
        from ...training.cv import prepare_stack5
        from ...inference.full_raster import predict_heads_mask
        from ...gems3.submission import eligible_mask
        from ...g07.inference import construct_final_prediction, verify_export_parity, write_submission_candidate

        ctx, rd = self.ctx, self.rd
        expected = list(ctx.fold_ids)
        checkpoints = [rd.checkpoints / f"cv_fold{k}_model.pt" for k in expected]
        missing = [str(p) for p in checkpoints if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"Cannot submit E6 with incomplete CV: {missing}")
        spec = self.spec()
        device = device_of()
        # Reuse one prepared stack for every fold; individual full maps
        # persist before the next fold, so a Colab disconnect loses at most one.
        stack = to_tensor_stack(prepare_stack5(ctx, spec, rd), device)
        context = pooled_context(stack, spec.context_factor) if spec.context else None
        fold_maps = []
        for k in expected:
            target = rd.raster_path(f"e6_full_fold{k}")
            if target.is_file():
                print(f"[E6 FULL] cached fold{k} score -> reuse")
                fold_maps.append(rd.read_raster(f"e6_full_fold{k}"))
                continue
            _, model = load_unit_model(CheckpointManager(rd.checkpoints, f"cv_fold{k}"), device)
            score = predict_heads_mask(model, stack, ctx.data.valid, spec, device, context)["seg"]
            rd.save_raster({"score": score.astype(np.float32)}, f"e6_full_fold{k}", ctx.data.grid)
            fold_maps.append(score)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            # Persist complete individual fold predictions + checkpoints;
            # never mirror the raw DEM dataset or run extra tests.
            self.exp.persist()
            print(f"[E6 FULL] fold{k} predicted and cached")
        final = np.nanmean(np.stack(fold_maps, axis=0), axis=0).astype(np.float32)
        final[~ctx.data.valid] = np.nan
        rd.save_raster({"probability": final}, "e6_full_probability", ctx.data.grid)

        eligibility = eligible_mask(ctx.data, 0, "eligible")
        frac = float(self.cfg.get("e6.submission_fraction", 0.01))
        if not 0 < frac <= 0.1:
            raise ValueError("E6 submission_fraction outside accepted 0..0.1 budget")
        selection, selection_report = construct_final_prediction(
            np.nan_to_num(final, nan=0), eligibility, frac, "w2_exact",
            seed=int(self.cfg.seed))
        parity = verify_export_parity(selection, eligibility, frac, ctx.data.labels)
        if parity["status"] != "PASS":
            raise RuntimeError(f"E6 selection parity failed: {parity}")
        path = rd.path / "submissions" / "submission_e6_geostructurenet.tif"
        checks = {"new_e6_model": True,
                  "three_complete_cv_checkpoints": len(checkpoints) == len(expected),
                  "completed_discovery_predictions": all(rd.raster_path(f"disc{d}").is_file()
                                                         for d in range(ctx.splits.n_splits)),
                  "full_region_prediction": bool(np.isfinite(final[ctx.data.valid]).all()),
                  "selection_parity": True,
                  "licensed_external_usgs_dem": bool(self.cfg.get("e6.usgs_dem_license_verified", True))}
        outcome = write_submission_candidate(
            selection, ctx.data, path,
            {"model": "E6_GeoStructureNet_MASKED_THREE_WAY",
             "experiment_id": self.exp.id, "config_hash": self.cfg.hash(),
             "git_commit": self.exp.manifest.data.get("git_commit"),
             "n_cv": len(expected), "dem_bank": "cached_10m_to_100m",
             "submission_fraction": frac, "policy": "E6_SCORE_EXACT_K",
             "selection": selection_report, "fold_checkpoints": [p.name for p in checkpoints],
             "parity": parity},
            checks)
        if outcome.get("status") != "READY_FOR_MANUAL_UPLOAD":
            raise RuntimeError(f"New E6 submission blocked: {outcome}")
        rd.save_json(outcome, "e6_submission_manifest")
        self.exp.persist()
        print(f"[E6] NEW E6 SUBMISSION: {outcome['path']}")
        print(f"[E6] SHA256: {outcome['sha256']}")
        return outcome
