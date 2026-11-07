"""Section 07 - full-area inference of the EXISTING checkpoints (no training, no V1 ``predict``).

V1 ``GeoRegimeExperiment.predict(unit, roi)`` reuses a cached raster without the ROI in its cache key, so a
narrow-zone prediction could be returned for a full-area request. This module never calls it. Its cache
identity is ``stable_hash`` of: checkpoint sha256, V1 bank fingerprint, normalizer hash, ROI mask hash,
model-code hash (model.py + data.py crop/normalisation), architecture arguments, patch / stride / AMP, output
schema and ``FULL_AREA`` scope. A raster is reused only when its manifest identity equals the recomputed one
AND every output still has its recorded size and sha256.

Tiling and blending replicate V1 exactly (patch, stride, ``_tile_positions``, Hann window floored at 0.05,
weight = window * ROI). Inside a V1 zone the set of covering windows and their weights are identical, so the
re-inferred values must reproduce V1 up to GPU numerical noise - this is measured and recorded.

Resume: accumulators (6 channels + weight) live in RAM and are snapshotted atomically to
``cache/inference/<identity>/snapshot.npz`` together with the list of finished windows every
``inference.snapshot_every_windows`` windows; after a disconnect only windows after the last snapshot are
recomputed (accumulation is never applied twice).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from .cache import array_sha, atomic_write_geotiff, atomic_write_json, atomic_write_npz, read_json, read_raster
from .config import code_hash, stable_hash

CHANNELS = ("prob", "global", "expert0", "expert1", "route0", "specialist_weight")
CHANNEL_MEANING = {
    "prob": "final mixed fault confidence (model output 'prob')",
    "global": "sigmoid(global_logits): global head alone",
    "expert0": "sigmoid(experts[:,0]): specialist 0 (router target ~ magnetic-dominant prior)",
    "expert1": "sigmoid(experts[:,1]): specialist 1 (router target ~ gravity-dominant prior)",
    "route0": "routes[:,0]: router weight of specialist 0 (V1 'router_magnetic')",
    "specialist_weight": "sigmoid(specialist_strength): mixture strength of the specialists",
}
SCHEMA_VERSION = "georegime_recovery_v2/full_area/1"


def output_path(paths, unit: str, ch: str) -> Path:
    if ch == "prob":
        return paths.d("02_full_inference") / f"{unit}_full_prediction.tif"
    if ch in ("global", "expert0", "expert1"):
        return paths.d("02_full_inference", "experts") / f"{unit}_{ch}.tif"
    return paths.d("02_full_inference", "router") / f"{unit}_{ch}.tif"


def manifest_path(paths, unit: str) -> Path:
    return paths.d("02_full_inference", "inference_manifests") / f"{unit}_full_area.json"


def _tile_positions(length, patch, stride):
    from gems_bench.georegime.run import _tile_positions as v1

    return v1(length, patch, stride)


class FullAreaInference:
    def __init__(self, run):
        self.run = run
        self.P, self.bank, self.src = run.paths, run.bank, run.src_cfg
        self.roi = self.bank.valid
        self.roi_sha = array_sha(self.roi)
        root = run.code_root / "gems_bench" / "georegime"
        self.model_code = code_hash([root / "model.py", root / "data.py"])

    # ------------------------------------------------------------------ identity / cache
    def identity(self, unit: str) -> dict:
        c = self.run.ckpt[unit]
        g = self.bank.group_indices
        return {"scope": "FULL_AREA", "unit": unit, "checkpoint_sha256": c["sha256"], "bank": self.bank.key,
                "normalizer_sha": c["normalizer_sha"], "roi": "bank.valid (full eligible footprint)",
                "roi_sha": self.roi_sha, "model_code": self.model_code,
                "architecture": {"nmag": len(g["mag"]), "ngrav": len(g["grav"]), "nterrain": len(g["terrain"]),
                                 "width": int(self.src["width"])},
                "patch": int(self.src["patch"]), "stride": int(self.src["predict_stride"]),
                "amp": bool(self.src["use_amp"]), "channels": list(CHANNELS), "schema": SCHEMA_VERSION}

    def key(self, unit: str) -> str:
        return stable_hash(self.identity(unit))

    def cached(self, unit: str) -> tuple[bool, str]:
        m = read_json(manifest_path(self.P, unit))
        if m is None:
            return False, "MISSING"
        if m.get("status") != "COMPLETE":
            return False, f"INCOMPLETE ({m.get('status')})"
        if m.get("key") != self.key(unit):
            return False, "STALE (identity changed: checkpoint / bank / normalizer / ROI / code / schema)"
        for ch in CHANNELS:
            o = m["outputs"].get(ch)
            p = Path(o["path"]) if o else None
            if p is None or not p.is_file():
                return False, f"INCOMPLETE (missing {ch})"
            if p.stat().st_size != o["size"] or self.run.hashes.sha(p) != o["sha256"]:
                return False, f"STALE (content of {p.name} changed)"
        return True, "COMPLETE"

    # ------------------------------------------------------------------ inference
    def _load(self, unit, device):
        import torch

        from gems_bench.georegime.model import GeoRegimeNet

        state = torch.load(self.run.ckpt[unit]["path"], map_location="cpu", weights_only=False)
        if self.run.hashes.sha(self.run.ckpt[unit]["path"]) != self.run.ckpt[unit]["sha256"]:
            raise RuntimeError(f"{unit}: checkpoint changed since preflight")
        g = self.bank.group_indices
        model = GeoRegimeNet(len(g["mag"]), len(g["grav"]), len(g["terrain"]), int(self.src["width"]))
        model.load_state_dict(state["model"], strict=True)
        model.to(device).eval()
        return model, state["normalizer"]

    def run_unit(self, unit: str, log=print) -> dict:
        import torch

        P, bank = self.P, self.bank
        key = self.key(unit)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, norm = self._load(unit, device)
        patch, stride = int(self.src["patch"]), int(self.src["predict_stride"])
        h, w = bank.shape
        roi = self.roi
        win = np.maximum(np.outer(np.hanning(patch), np.hanning(patch)).astype(np.float32), 0.05)
        positions = [(r, c) for r in _tile_positions(h, patch, stride) for c in _tile_positions(w, patch, stride)
                     if roi[r:r + patch, c:c + patch].any()]
        snap = P.d("cache", "inference", key) / "snapshot.npz"
        acc = np.zeros((len(CHANNELS) + 1, h, w), np.float32)                 # last plane = weight sum
        done = np.zeros(len(positions), bool)
        if snap.is_file():
            with np.load(snap) as z:
                if z["acc"].shape == acc.shape and z["done"].shape == done.shape:
                    acc[:] = z["acc"]
                    done[:] = z["done"]
                    log(f"[infer {unit}] resumed from snapshot: {int(done.sum())}/{len(positions)} windows")
        todo = np.flatnonzero(~done)
        bs = int(self.run.cfg["inference"]["batch_size"])
        every = int(self.run.cfg["inference"]["snapshot_every_windows"])
        amp = bool(self.src["use_amp"]) and device.type == "cuda"
        t0, since = time.time(), 0
        log(f"[infer {unit}] FULL_AREA identity {key}: {len(todo)} of {len(positions)} windows to compute on {device}")
        with torch.no_grad():
            for b0 in range(0, len(todo), bs):
                idx = todo[b0:b0 + bs]
                crops = [bank.crop(positions[i][0], positions[i][1], patch, norm) for i in idx]
                x = {k: torch.from_numpy(np.stack([cr[k] for cr in crops])).float().to(device)
                     for k in ("mag", "grav", "terrain", "context")}
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                    out = model(x["mag"], x["grav"], x["terrain"], x["context"])
                planes = torch.stack([out["prob"][:, 0], torch.sigmoid(out["global_logits"][:, 0]),
                                      torch.sigmoid(out["experts"][:, 0]), torch.sigmoid(out["experts"][:, 1]),
                                      out["routes"][:, 0], out["specialist_weight"][:, 0]], 1).float().cpu().numpy()
                for j, i in enumerate(idx):
                    r, c = positions[i]
                    wt = win * roi[r:r + patch, c:c + patch]
                    acc[:-1, r:r + patch, c:c + patch] += planes[j] * wt
                    acc[-1, r:r + patch, c:c + patch] += wt
                    done[i] = True
                since += len(idx)
                if since >= every or b0 + bs >= len(todo):
                    atomic_write_npz(snap, acc=acc, done=done)
                    since = 0
                    log(f"[infer {unit}] {int(done.sum())}/{len(positions)} windows, snapshot saved "
                        f"({time.time() - t0:.0f} s)")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        wsum = acc[-1]
        ok = wsum > 0
        if (roi & ~ok).any():
            raise RuntimeError(f"{unit}: {int((roi & ~ok).sum())} eligible pixels received no prediction")
        outputs = {}
        for i, ch in enumerate(CHANNELS):
            arr = np.where(ok & roi, np.divide(acc[i], wsum, out=np.zeros_like(wsum), where=ok), np.nan)
            v = arr[roi]
            if not np.isfinite(v).all() or v.min() < 0 or v.max() > 1:
                raise RuntimeError(f"{unit}/{ch}: values outside [0,1] or non-finite inside the footprint")
            p = atomic_write_geotiff(output_path(P, unit, ch), arr, bank.profile, [f"{unit} {CHANNEL_MEANING[ch]}"])
            back = read_raster(p)
            if not np.array_equal(np.isfinite(back), roi):
                raise RuntimeError(f"{unit}/{ch}: readback footprint differs from the eligible mask")
            outputs[ch] = {"path": str(p), "size": p.stat().st_size, "sha256": self.run.hashes.sha(p)}
        repro = self.reproduction(unit, np.where(roi, acc[0] / np.where(ok, wsum, 1), np.nan))
        man = {"unit": unit, "key": key, "identity": self.identity(unit), "status": "COMPLETE",
               "n_windows": len(positions), "pixels": int(roi.sum()), "device": str(device),
               "seconds": round(time.time() - t0, 1), "outputs": outputs, "reproduction_vs_v1": repro,
               "channel_meaning": CHANNEL_MEANING}
        atomic_write_json(manifest_path(P, unit), man)
        try:
            snap.unlink()                                       # own cache only; the rasters are now complete
        except OSError:
            pass
        return man

    def reproduction(self, unit: str, prob: np.ndarray) -> dict:
        """Compare with the V1 raster on the pixels V1 predicted (its own ROI)."""
        p = self.P.v1_prediction(unit)
        if not p.is_file():
            return {"status": "V1_RASTER_ABSENT"}
        v1 = read_raster(p)
        m = np.isfinite(v1) & np.isfinite(prob)
        if not m.any():
            return {"status": "NO_OVERLAP"}
        d = np.abs(v1[m] - prob[m])
        tol = float(self.run.cfg["inference"]["reproduction_tol"])
        corr = float(np.corrcoef(v1[m][:: max(m.sum() // 2_000_000, 1)], prob[m][:: max(m.sum() // 2_000_000, 1)])[0, 1])
        return {"status": "REPRODUCED" if d.max() <= tol else "DIFFERS", "pixels_compared": int(m.sum()),
                "v1_pixels": int(np.isfinite(v1).sum()), "max_abs_diff": float(d.max()),
                "mean_abs_diff": float(d.mean()), "p999_abs_diff": float(np.quantile(d, 0.999)),
                "pearson": corr, "tolerance": tol}

    # ------------------------------------------------------------------ access
    def load(self, unit: str, ch: str = "prob") -> np.ndarray:
        ok, why = self.cached(unit)
        if not ok:
            raise FileNotFoundError(f"{unit} full-area {ch} not available: {why}")
        return read_raster(output_path(self.P, unit, ch))


def artifact_table(inf: FullAreaInference, units) -> pd.DataFrame:
    rows = []
    for u in units:
        ok, why = inf.cached(u)
        m = read_json(manifest_path(inf.P, u)) or {}
        rep = m.get("reproduction_vs_v1", {})
        rows.append({"unit": u, "status": why, "identity": inf.key(u) if u in inf.run.ckpt else None,
                     "pixels": m.get("pixels"), "path": str(output_path(inf.P, u, "prob")),
                     "n_windows": m.get("n_windows"), "seconds": m.get("seconds"),
                     "reproduction_status": rep.get("status"), "reproduction_max_abs_diff": rep.get("max_abs_diff"),
                     "reproduction_pixels": rep.get("pixels_compared")})
    return pd.DataFrame(rows)
