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
            self._e6_fingerprint = info["fingerprint"]
            self._e6_inputs_ready = True
            self.rd.save_json({**info, "registered_channels": available}, "e6_input_cache")
            print(f"[E6] DEM cache {info['fingerprint']}: {info['n_source_tiles']} preexisting 10m tiles; "
                  f"{len(available)} cached 100m channels")
        return ctx

    def spec(self):
        """The Cnn5 spec + the identity of the DEM cache it trains on. ``e6_dem_fingerprint`` is semantic (part of
        ``spec_hash``): a rebuilt DEM bank never silently reuses checkpoints or predictions of another one."""
        from dataclasses import replace

        s = super().spec()                     # resolving the spec touches ctx, which stages the DEM cache first
        fp = getattr(self, "_e6_fingerprint", "")
        if not fp:
            raise RuntimeError("E6 DEM cache fingerprint unavailable: the DEM staging did not run")
        if s.e6_dem_fingerprint != fp:
            s = replace(s, e6_dem_fingerprint=fp)
            self._spec = s
        return s

    def prepare(self):
        from ...training.cv import ensure_channels
        result = super().prepare()
        spec = self.spec()
        if spec.model != "geostructurenet_e6":
            raise ValueError("E6 milestone must use a GeoStructureNetE6 model, not an old model")
        assert spec.aux_dist > 0 and spec.aux_orient > 0, "E6 production requires auxiliary geometry heads"
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
        """Only E6 fold and discovery DTI; no W2 comparison or R&D ablation.

        Two representations per unit: ``continuous`` (the probability map) and ``exact_k`` = the submission policy
        (exactly ``e6.submission_fraction`` of the unit's scored pixels, top score, label-free tie-breaking), so the
        reported number describes the file that is exported. Discovery units score the WITHHELD systems only; the
        retained (training-visible) faults inside the evaluation zone are excluded pixel-exactly, as the competition
        masks known faults - otherwise the model would be credited for reproducing its own training labels."""
        from ...g07.representation import exact_k_mask
        from ...metric import dti_components, pool

        ctx, rd = self.ctx, self.rd
        frac = float(self.cfg.get("e6.submission_fraction", 0.01))
        W = ctx.data.shape[1]
        rows = []
        acc = {(p, r): [] for p in ("V3_GEOGRAPHIC", "V1_DISCOVERY") for r in ("continuous", "exact_k")}

        def score_unit(protocol, unit, score, truth, mask, exclude):
            dom = mask & ~exclude
            if not np.isfinite(np.where(dom, score, np.nan)).any():
                raise ValueError(f"E6 {unit}: prediction has no finite value inside its scoring zone")
            q, rep = exact_k_mask(np.where(dom, score, np.nan), dom, frac, width_full=W, seed=int(self.cfg.seed))
            for rname, pred in (("continuous", score), ("exact_k", q.astype(np.float32))):
                st = dti_components(pred, truth, eval_mask=mask, exclude_mask=exclude, exclude_supports_tp=False)
                acc[(protocol, rname)].append(st)
                rows.append({"protocol": protocol, "unit": unit, "representation": rname,
                             "frac": frac if rname == "exact_k" else np.nan,
                             "selected_px": rep["actual_px"] if rname == "exact_k" else np.nan, **st.as_dict()})

        no_ex = np.zeros(ctx.data.shape, bool)
        for k in ctx.fold_ids:
            raster = rd.raster_path(f"oof_fold{k}")
            if not raster.is_file():
                raise FileNotFoundError(f"E6 missing geographic fold {k}: {raster}")
            score_unit("V3_GEOGRAPHIC", f"cv_fold{k}", rd.read_raster(f"oof_fold{k}"), ctx.data.labels,
                       ctx.folds.val(k) & ctx.data.valid, no_ex)
        for d in range(ctx.splits.n_splits):
            raster = rd.raster_path(f"disc{d}")
            if not raster.is_file():
                raise FileNotFoundError(f"E6 missing withheld-system prediction {d}: {raster}")
            withheld = ctx.data.labels & ctx.splits.withheld(d)
            retained = ctx.data.labels & ~ctx.splits.withheld(d)
            score_unit("V1_DISCOVERY", f"disc{d}", rd.read_raster(f"disc{d}"), withheld,
                       ctx.splits.eval_zone(d) & ctx.data.valid, retained)
        out = {"n_fold": len(list(ctx.fold_ids)), "n_split": ctx.splits.n_splits, "submission_fraction": frac}
        for (protocol, rname), comps in acc.items():
            pooled = pool(comps)
            rows.append({"protocol": protocol, "unit": "POOLED", "representation": rname,
                         "frac": frac if rname == "exact_k" else np.nan, **pooled.as_dict()})
            out[f"{protocol.lower()}_{rname}_dti"] = pooled.dti
        rd.save_table(pd.DataFrame(rows), "e6_dti")
        print("[E6] pooled DTI (offline proxy, not a leaderboard score):",
              {k: round(v, 4) for k, v in out.items() if k.endswith("_dti")})
        return out

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

        from ...artifacts.atomic import sha256_file
        from ...training.spec import spec_hash

        ctx, rd = self.ctx, self.rd
        expected = list(ctx.fold_ids)
        checkpoints = [rd.checkpoints / f"cv_fold{k}_model.pt" for k in expected]
        missing = [str(p) for p in checkpoints if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"Cannot submit E6 with incomplete CV: {missing}")
        spec = self.spec()
        sh = spec_hash(spec)
        stale = [f"cv_fold{k}" for k in expected if not CheckpointManager(rd.checkpoints, f"cv_fold{k}",
                                                                         spec_hash=sh).final_ok()]
        if stale:
            raise RuntimeError(f"E6 final models {stale} are unreadable or were trained by another spec / DEM cache "
                               f"({sh}); rerun EXP.train() before exporting")
        device = device_of()
        # Reuse one prepared stack for every fold; individual full maps
        # persist before the next fold, so a Colab disconnect loses at most one.
        stack = to_tensor_stack(prepare_stack5(ctx, spec, rd), device)
        context = pooled_context(stack, spec.context_factor) if spec.context else None
        fold_maps = []
        for k, ckpt_file in zip(expected, checkpoints):
            target = rd.raster_path(f"e6_full_fold{k}")
            prov_file = target.with_suffix(".json")
            ck_sha = sha256_file(ckpt_file)
            if target.is_file() and prov_file.is_file():
                prov = json.loads(prov_file.read_text(encoding="utf-8"))
                if prov.get("checkpoint_sha256") == ck_sha and prov.get("spec_hash") == sh:
                    print(f"[E6 FULL] cached fold{k} score (same checkpoint) -> reuse")
                    fold_maps.append(rd.read_raster(f"e6_full_fold{k}"))
                    continue
                print(f"[E6 FULL] fold{k} cached score belongs to another checkpoint -> recompute")
            spec_k, model = load_unit_model(CheckpointManager(rd.checkpoints, f"cv_fold{k}"), device)
            if list(spec_k.channels) != list(spec.channels) or spec_k.e6_dem_fingerprint != spec.e6_dem_fingerprint:
                raise RuntimeError(f"cv_fold{k}: checkpoint channel order / DEM cache differs from the inference stack")
            heads = predict_heads_mask(model, stack, ctx.data.valid, spec, device, context)
            if "seg" not in heads:
                raise RuntimeError(f"cv_fold{k}: no prediction over the valid footprint")
            score = heads["seg"]
            rd.save_raster({"score": score.astype(np.float32)}, f"e6_full_fold{k}", ctx.data.grid)
            prov_file.write_text(json.dumps({"checkpoint": ckpt_file.name, "checkpoint_sha256": ck_sha,
                                             "spec_hash": sh, "dem_fingerprint": spec.e6_dem_fingerprint}),
                                 encoding="utf-8")
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
        path.parent.mkdir(parents=True, exist_ok=True)
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
        # reopen the written file: it must equal the selection after the template's nodata rule
        readback = verify_export_parity(selection, eligibility, frac, ctx.data.labels, written=outcome["path"],
                                        data=ctx.data)
        outcome["readback"] = readback
        if readback["status"] != "PASS":
            raise RuntimeError(f"E6 submission readback failed: {readback}")
        outcome.update({"representation": "exact_k binary: top submission_fraction of the eligible domain by the "
                                          "fold-averaged E6 probability (predeclared, no threshold search)",
                        "probability_raster": str(rd.raster_path("e6_full_probability")),
                        "spec_hash": spec_hash(spec), "dem_fingerprint": spec.e6_dem_fingerprint,
                        "fold_checkpoint_sha256": {p.name: sha256_file(p) for p in checkpoints}})
        rd.save_json(outcome, "e6_submission_manifest")
        self.exp.persist()
        if self.exp.paths.drive is not None:
            # the Drive copy must exist and carry the same bytes as the local file
            rel = Path(outcome["path"]).resolve().relative_to(Path(self.exp.paths.root).resolve())
            drive_file = Path(self.exp.paths.drive) / rel
            if not drive_file.is_file() or sha256_file(drive_file) != outcome["sha256"]:
                raise RuntimeError(f"E6 submission not verified on Drive ({drive_file}); rerun EXP.persist()")
            outcome["drive_path"] = str(drive_file)
            rd.save_json(outcome, "e6_submission_manifest")
            self.exp.persist()
        print(f"[E6] NEW E6 SUBMISSION: {outcome['path']}")
        print(f"[E6] SHA256: {outcome['sha256']}")
        return outcome
