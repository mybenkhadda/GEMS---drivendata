"""Persisted training datasets: sample indices, scalers, class weights and sampling configuration.

:func:`build_training_dataset` turns (feature dataset version, split version, model config) into a small, durable
record under ``06_models/<namespace>/training_data/<key>/``:

* ``indices_<split>.npy``  - flat master-grid indices of the pixel samples (train / validation / stack / test),
* ``categories_<split>.npy`` - sampling category of each sample (positive / near / background / hard ...),
* ``scaler.json``          - robust (median / IQR) normaliser fitted on TRAIN samples only,
* ``class_weights.json``   - positive / unlabeled counts and weights (PU semantics: unlabeled != negative),
* ``sampling_config.json`` + ``manifest.json`` (dataset version + spec, split version + code hash, channels).

Features are never copied: rows are gathered on demand from the memory-mapped layers (``load_xy``). The key hashes
every input, so a changed split, channel list or sampling setting gives a new record and never reuses a stale one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import stable_hash
from .registry import atomic_write_json, atomic_write_npy, read_json
from .spatial_splits import CODES, SPLITS, TrainOnlyNormalizer


@dataclass
class TrainingData:
    dir: Path
    channels: list
    target: str
    indices: dict
    categories: dict
    scaler: dict
    class_weights: dict
    manifest: dict = field(default_factory=dict)

    def rows_cols(self, split: str, shape) -> tuple[np.ndarray, np.ndarray]:
        return np.unravel_index(self.indices[split], shape)

    def load_xy(self, dataset, split: str, shape, normalize: bool = False):
        r, c = self.rows_cols(split, shape)
        X = dataset.sample(self.channels, r, c) if len(r) else np.zeros((0, len(self.channels)), np.float32)
        if normalize:
            X = (X - np.asarray(self.scaler["center"], np.float32)) / np.asarray(self.scaler["scale"], np.float32)
        y = np.nan_to_num(np.asarray(dataset.get(self.target), np.float32)[r, c], nan=0.0)
        return X, y


def build_training_dataset(feature_dataset, split_manifest: dict, model_config: dict, *, split_code: np.ndarray,
                           fault: np.ndarray, out_root, cfg: dict, hard_geo=None, hard_terrain=None,
                           log=print) -> TrainingData:
    """``model_config``: {"experiment", "channels" (list) or "families", "target", "n_per_split" (optional)}."""
    from .training.sampling import category_masks, mixture_sample

    ds = feature_dataset
    channels = list(model_config.get("channels") or ds.predictors(model_config.get("families")))
    if not channels:
        raise ValueError(f"{model_config.get('experiment')}: no feature channel selected")
    target = model_config.get("target") or cfg["labels"].get("training_target", "Y_BUFFER_200M")
    sampling = {"mix": cfg["sampling"]["mix"], "balance_block_km": cfg["sampling"]["balance_block_km"],
                "n_train": int(model_config.get("n_per_split") or cfg["sampling"]["pixel_samples_per_split"]),
                "near_band_m": cfg["labels"]["near_band_m"], "seed": int(cfg["reproducibility"]["seed"]),
                "pu_unlabeled_weight": float(cfg["models"]["xgb"]["pu_unlabeled_weight"])}
    key = stable_hash({"dataset": getattr(ds, "version", "pool"), "spec": getattr(ds, "spec", {}),
                       "split": split_manifest.get("split_code_sha256"), "channels": channels, "target": target,
                       "sampling": sampling})
    d = Path(out_root) / "training_data" / f"{model_config.get('experiment', 'x')}_{key}"
    man = read_json(d / "manifest.json")
    if man and man.get("status") == "COMPLETE" and all((d / f"indices_{s}.npy").is_file() for s in SPLITS):
        log(f"[training-data] reused {d.name}")
        return _load(d, man)
    dist = np.asarray(ds.get("Y_DISTANCE"), np.float32)
    bpx = max(1, int(round(sampling["balance_block_km"] * 1000 / 100.0)))
    idx, cats = {}, {}
    for i, s in enumerate(SPLITS):
        region = split_code == CODES[s]
        cm = category_masks(region, fault, dist, sampling["near_band_m"], hard_geo, hard_terrain, None)
        n = sampling["n_train"] if s == "train" else max(sampling["n_train"] // 3, 1000)
        ix, code, _info = mixture_sample(cm, sampling["mix"], n, bpx, sampling["seed"] + i)
        idx[s], cats[s] = np.asarray(ix, np.int64), np.asarray(code)
    r, c = np.unravel_index(idx["train"], split_code.shape)
    if np.any(split_code[r, c] != CODES["train"]):
        raise RuntimeError("training indices outside the train split")
    sel = np.random.default_rng(sampling["seed"]).choice(len(r), min(len(r), 200_000), replace=False) if len(r) else []
    Xs = ds.sample(channels, r[sel], c[sel]) if len(r) else np.zeros((0, len(channels)), np.float32)
    norm = TrainOnlyNormalizer().fit(Xs, np.full(len(Xs), CODES["train"]))
    center = np.nan_to_num(norm.center).astype(float).tolist()
    scale = np.where(np.isfinite(norm.scale), norm.scale, 1.0).astype(float).tolist()
    y = np.nan_to_num(np.asarray(ds.get(target), np.float32)[r, c], nan=0.0) if len(r) else np.zeros(0)
    npos, nunl = int((y > 0.5).sum()), int((y <= 0.5).sum())
    cw = {"n_positive": npos, "n_unlabeled": nunl, "prevalence": npos / max(npos + nunl, 1),
          "pos_weight_balanced": nunl / max(npos, 1), "pu_unlabeled_weight": sampling["pu_unlabeled_weight"],
          "label_semantics": "unlabeled = background, not a verified negative"}
    for s in SPLITS:
        atomic_write_npy(d / f"indices_{s}.npy", idx[s])
        atomic_write_npy(d / f"categories_{s}.npy", cats[s])
    atomic_write_json(d / "scaler.json", {"center": center, "scale": scale, "channels": channels, "fitted_on": "train",
                                          "n_fit": int(len(Xs))})
    atomic_write_json(d / "class_weights.json", cw)
    atomic_write_json(d / "sampling_config.json", sampling)
    man = {"status": "COMPLETE", "key": key, "experiment": model_config.get("experiment"), "channels": channels,
           "target": target, "dataset_version": getattr(ds, "version", "pool"),
           "dataset_spec": getattr(ds, "spec", {}), "split_code_sha256": split_manifest.get("split_code_sha256"),
           "split_generator": split_manifest.get("generator"), "n_samples": {s: int(len(idx[s])) for s in SPLITS}}
    atomic_write_json(d / "manifest.json", man)
    log(f"[training-data] built {d.name}: {man['n_samples']} samples, {len(channels)} channel(s)")
    return _load(d, man)


def _load(d: Path, man: dict) -> TrainingData:
    sc = read_json(d / "scaler.json") or {}
    return TrainingData(d, man["channels"], man["target"],
                        {s: np.load(d / f"indices_{s}.npy") for s in SPLITS},
                        {s: np.load(d / f"categories_{s}.npy", allow_pickle=False) for s in SPLITS},
                        sc, read_json(d / "class_weights.json") or {}, man)
