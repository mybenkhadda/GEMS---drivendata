"""Experiment fidelity: SCREEN -> VALIDATE -> FULL (progressive experimentation at bounded GPU cost).

Fidelity is orthogonal to the pipeline *execution mode* (``run_mode``: which stages run). It says how much
evidence one research experiment produces:

``SCREEN``    one geographic fold, no pseudo-discovery, capped epochs / steps, scored on a fixed, deterministic
              subset of the held-out fold. Rapid triage only: proxy metrics, never on a leaderboard, never in
              the promotion gate, never in the model zoo.
``VALIDATE``  every geographic fold (full held-out regions), one pseudo-discovery split, capped budget.
              Intermediate evidence: proxy metrics, never in the promotion gate.
``FULL``      the experiment YAML exactly as committed (original budgets, three folds, three splits, the
              section-17 evaluation). Only FULL results can reach the registry / promotion gate.

Identity: a FULL config is the committed YAML unchanged, so its id / config hash are those of every run made
before fidelity existed (``b0_reference__d54e938a`` stays reusable). SCREEN / VALIDATE add a semantic
``fidelity:`` block, which changes the config hash, and ``ExperimentConfig.experiment_id`` names the folder
``<name>__<profile>__<hash8>``: a cheap run can never overwrite or resume a FULL run.

Leakage rules (enforced here, tested in tests/test_fidelity.py)
---------------------------------------------------------------
* Folds / splits are the immutable files of the session's validation design; fidelity only *selects* them.
* Training subsampling keeps whole spatial blocks of ``folds.inner_train(k)`` (``mask & blocks``): the result is
  a subset of the eligible training zone, so buffers and held-out regions are untouched. The patch sampler
  rejects any patch touching a non-allowed valid pixel, so no patch can reach outside the subset either.
* Evaluation subsampling keeps whole blocks of the held-out mask; the candidate and its control use the same
  seed / salt, hence identical masks (their sha256 is recorded and compared before any paired delta).
* Note: GPU cost is set by epochs x steps x batch, not by the training area. ``training_fraction`` < 1 saves no
  compute and only removes data diversity; it defaults to 1.0 in every profile (kept as an explicit knob).
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field, fields, replace

import numpy as np

PROFILES = ("SCREEN", "VALIDATE", "FULL")
RANK = {p: i for i, p in enumerate(PROFILES)}          # higher = more evidence
FIDELITY_VERSION = 1
LABEL = {"SCREEN": "SCREEN (rapid triage; proxy metrics)",
         "VALIDATE": "VALIDATE (intermediate validation; proxy metrics)",
         "FULL": "FULL (section-17 protocol)"}


class FidelityError(ValueError):
    pass


@dataclass(frozen=True)
class Fidelity:
    profile: str = "FULL"
    geographic_folds: tuple | None = None           # None = every fold of the design
    discovery_splits: tuple | None = None           # None = every split; () = none
    max_epochs: int | None = None                    # None = the experiment's own budget
    max_train_steps_per_epoch: int | None = None
    training_fraction: float = 1.0
    validation_fraction: float = 1.0
    full_resolution_inference: bool = True           # False: predict only the scored subset
    train_block_px: int = 512                        # >= 2 x patch (256 px) so patches fit inside kept blocks
    eval_block_px: int = 256
    subsample_seed: int = 2026
    checkpoint_policy: str = "best_and_latest"       # non-semantic (runtime.checkpoint)
    version: int = FIDELITY_VERSION

    # -- identity ---------------------------------------------------------------
    @property
    def is_full(self) -> bool:
        return self.profile == "FULL"

    def semantic_block(self) -> dict:
        """The ``fidelity:`` block written into the experiment config (empty for FULL: committed YAML unchanged)."""
        if self.is_full:
            return {}
        d = asdict(self)
        d.pop("checkpoint_policy")
        d["geographic_folds"] = None if self.geographic_folds is None else list(self.geographic_folds)
        d["discovery_splits"] = None if self.discovery_splits is None else list(self.discovery_splits)
        return d

    def protocol_hash(self, folds_dir: str = "", discovery_dir: str = "") -> str:
        """Hash of everything that must match before two runs are compared (budget, subsets, designs)."""
        s = json.dumps({**self.semantic_block(), "profile": self.profile, "folds_dir": folds_dir,
                        "discovery_dir": discovery_dir}, sort_keys=True, default=str)
        return hashlib.sha256(s.encode()).hexdigest()[:12]

    # -- selection ----------------------------------------------------------------
    def fold_ids(self, available) -> list[int]:
        av = [int(k) for k in available]
        if self.geographic_folds is None:
            return av
        bad = [k for k in self.geographic_folds if k not in av]
        if bad:
            raise FidelityError(f"geographic_folds {bad} not in the validation design (folds {av})")
        return [int(k) for k in self.geographic_folds]

    def split_ids(self, n_splits: int) -> list[int]:
        if self.discovery_splits is None:
            return list(range(int(n_splits)))
        bad = [d for d in self.discovery_splits if not 0 <= int(d) < int(n_splits)]
        if bad:
            raise FidelityError(f"discovery_splits {bad} outside 0..{int(n_splits) - 1}")
        return [int(d) for d in self.discovery_splits]

    def cap_spec(self, spec):
        """Budget caps on a P5Spec (FULL: unchanged). min_epochs never exceeds the capped epochs."""
        if self.is_full:
            return spec
        ep = spec.epochs if self.max_epochs is None else min(int(spec.epochs), int(self.max_epochs))
        st = spec.steps_per_epoch if self.max_train_steps_per_epoch is None else \
            min(int(spec.steps_per_epoch), int(self.max_train_steps_per_epoch))
        return replace(spec, epochs=ep, steps_per_epoch=st, min_epochs=min(int(spec.min_epochs), ep))

    # -- masks ------------------------------------------------------------------------
    def train_mask(self, allowed: np.ndarray, k: int) -> np.ndarray:
        return block_subsample(allowed, self.training_fraction, self.subsample_seed, self.train_block_px,
                               salt=3000 + int(k))

    def eval_mask(self, mask: np.ndarray, salt: int) -> np.ndarray:
        return block_subsample(mask, self.validation_fraction, self.subsample_seed, self.eval_block_px, salt=salt)

    def val_mask(self, val: np.ndarray, k: int) -> np.ndarray:
        return self.eval_mask(val, 1000 + int(k))

    def inner_mask(self, inner: np.ndarray, k: int) -> np.ndarray:
        return self.eval_mask(inner, 2000 + int(k))

    def zone_mask(self, zone: np.ndarray, d: int) -> np.ndarray:
        return self.eval_mask(zone, 4000 + int(d))

    # -- eligibility --------------------------------------------------------------------
    @property
    def leaderboard_eligible(self) -> bool:
        return self.is_full

    @property
    def promotion_eligible(self) -> bool:
        return self.is_full


def _parse_ids(v, name: str):
    if v is None:
        return None
    if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
        v = [v]
    if not isinstance(v, (list, tuple)) or not all(isinstance(x, (int, np.integer)) and not isinstance(x, bool) for x in v):
        raise FidelityError(f"{name} must be a list of integers or null, got {v!r}")
    return tuple(sorted(dict.fromkeys(int(x) for x in v)))


DEFAULTS: dict[str, dict] = {
    "SCREEN": {"geographic_folds": [0], "discovery_splits": [], "max_epochs": 5, "max_train_steps_per_epoch": 80,
               "training_fraction": 1.0, "validation_fraction": 0.25, "full_resolution_inference": False},
    "VALIDATE": {"geographic_folds": None, "discovery_splits": [0], "max_epochs": 10,
                 "max_train_steps_per_epoch": 150, "training_fraction": 1.0, "validation_fraction": 1.0,
                 "full_resolution_inference": True},
    "FULL": {},
}

KNOBS = {f.name for f in fields(Fidelity)} - {"profile", "version"}


def make_fidelity(profile: str = "FULL", overrides: dict | None = None) -> Fidelity:
    """Validated profile = defaults of ``profile`` + ``overrides`` (FULL accepts no budget overrides)."""
    p = str(profile or "FULL").upper()
    if p not in PROFILES:
        raise FidelityError(f"experiment profile {profile!r} not in {PROFILES}")
    ov = dict(overrides or {})
    unknown = sorted(set(ov) - KNOBS)
    if unknown:
        raise FidelityError(f"unknown fidelity keys {unknown}; valid: {sorted(KNOBS)}")
    if p == "FULL" and set(ov) - {"checkpoint_policy"}:
        raise FidelityError("FULL reuses the committed experiment budgets and evaluation exactly; budget overrides "
                            f"{sorted(set(ov) - {'checkpoint_policy'})} are only allowed for SCREEN / VALIDATE")
    kw = {**DEFAULTS[p], **ov}
    kw["geographic_folds"] = _parse_ids(kw.get("geographic_folds"), "geographic_folds")
    kw["discovery_splits"] = _parse_ids(kw.get("discovery_splits"), "discovery_splits")
    for k in ("max_epochs", "max_train_steps_per_epoch"):
        v = kw.get(k)
        if v is not None and (not isinstance(v, (int, np.integer)) or isinstance(v, bool) or v < 1):
            raise FidelityError(f"{k} must be a positive integer or null, got {v!r}")
    for k in ("training_fraction", "validation_fraction"):
        v = kw.get(k, 1.0)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0.0 < float(v) <= 1.0:
            raise FidelityError(f"{k} must be in (0, 1], got {v!r}")
        kw[k] = float(v)
    for k in ("train_block_px", "eval_block_px"):
        if k in kw and (not isinstance(kw[k], int) or kw[k] < 16):
            raise FidelityError(f"{k} must be an integer >= 16")
    if kw.get("checkpoint_policy", "best_and_latest") not in ("best_and_latest", "rotating"):
        raise FidelityError("checkpoint_policy must be 'best_and_latest' or 'rotating'")
    if p == "SCREEN" and kw["geographic_folds"] is not None and len(kw["geographic_folds"]) == 0:
        raise FidelityError("SCREEN needs at least one real geographic fold")
    return Fidelity(profile=p, **kw)


def fidelity_of(cfg) -> Fidelity:
    """Fidelity recorded in an experiment config (``ExperimentConfig`` or raw dict); no block = FULL."""
    raw = cfg.raw if hasattr(cfg, "raw") else (cfg or {})
    blk = dict(raw.get("fidelity") or {})
    if not blk:
        return Fidelity()
    p = blk.pop("profile", "FULL")
    blk.pop("version", None)
    return make_fidelity(p, blk)


def apply_fidelity(raw: dict, fid: Fidelity) -> dict:
    """Experiment config dict with the fidelity block (semantic) and the checkpoint policy (runtime) applied."""
    out = copy.deepcopy(raw)
    out.pop("fidelity", None)
    blk = fid.semantic_block()
    if blk:
        out["fidelity"] = blk
    out.setdefault("runtime", {}).setdefault("checkpoint", {})["policy"] = fid.checkpoint_policy
    return out


# =============================================================================== deterministic block subsampling
def _block_any(mask: np.ndarray, b: int) -> np.ndarray:
    H, W = mask.shape
    nby, nbx = -(-H // b), -(-W // b)
    pad = np.zeros((nby * b, nbx * b), bool)
    pad[:H, :W] = mask
    return pad.reshape(nby, b, nbx, b).any(axis=(1, 3))


def block_subsample(mask: np.ndarray, frac: float, seed: int, block_px: int, salt: int = 0) -> np.ndarray:
    """``mask`` restricted to a deterministic fraction of the ``block_px`` blocks it touches (always a subset).

    The same (mask, frac, seed, block_px, salt) always returns the same pixels, whatever the experiment; at
    least one block is kept so a non-empty mask never becomes empty."""
    mask = np.asarray(mask, bool)
    if frac >= 1.0 or not mask.any():
        return mask.copy()
    touched = _block_any(mask, block_px)
    ids = np.flatnonzero(touched.ravel())
    k = max(1, int(round(float(frac) * len(ids))))
    keep = np.sort(np.random.default_rng([int(seed), int(salt)]).permutation(ids)[:k])
    sel = np.zeros(touched.size, bool)
    sel[keep] = True
    sel = sel.reshape(touched.shape)
    H, W = mask.shape
    full = np.repeat(np.repeat(sel, block_px, axis=0), block_px, axis=1)[:H, :W]
    return mask & full


def mask_digest(mask: np.ndarray) -> str:
    """sha256 (16 hex) of a boolean mask: proves that a candidate and its control were scored on the same pixels."""
    m = np.ascontiguousarray(np.asarray(mask, bool))
    return hashlib.sha256(np.packbits(m).tobytes() + str(m.shape).encode()).hexdigest()[:16]


@dataclass
class CoverageRecord:
    """Effective budget of one training / evaluation unit (written to metrics/fidelity_coverage.csv)."""
    unit: str
    eligible_px: int
    used_px: int
    pixel_km2: float = 0.01
    extra: dict = field(default_factory=dict)

    def row(self) -> dict:
        return {"unit": self.unit, "eligible_px": self.eligible_px, "used_px": self.used_px,
                "used_frac": self.used_px / max(self.eligible_px, 1),
                "used_km2": self.used_px * self.pixel_km2, **self.extra}
