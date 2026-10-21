"""Evaluation protocol identity: one hash for (folds, discovery splits, budgets, policies, metric constants).

Two numbers are comparable only if they carry the same ``protocol_hash``. Every scoreboard row, registry
entry and promotion decision records it; ``assert_same_protocol`` refuses to mix rows from different
protocols (different masks, splits, budgets, score transformations or known-fault policies).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..metric import DTI_ALPHA, DTI_BETA, DTI_EPS, DTI_RADIUS_M
from ..p3.curves import DEFAULT_AREA_FRACS


def _file_fp(p: Path) -> str:
    p = Path(p)
    if not p.exists():
        return "missing"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        h.update(f.read(1 << 20))                     # first MiB + size: cheap, collision-safe enough here
    return f"{p.stat().st_size}:{h.hexdigest()[:16]}"


@dataclass(frozen=True)
class EvalProtocol:
    folds: str
    discovery: str
    folds_fp: str
    discovery_fp: str
    match_frac: float = 0.05
    policies: tuple = ("excl", "retained_fp")
    primary_policy: str = "excl"
    area_fracs: tuple = tuple(DEFAULT_AREA_FRACS)
    far_km: float = 1.0
    selection: str = "crossfit_binary_area"          # area budget chosen on the other folds / splits
    metric: dict = field(default_factory=lambda: {"alpha": DTI_ALPHA, "beta": DTI_BETA, "radius_m": DTI_RADIUS_M,
                                                  "eps": DTI_EPS, "impl": "gems_bench.metric.dti_components"})

    def hash(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, default=str).encode()).hexdigest()[:12]

    def as_dict(self) -> dict:
        return {**asdict(self), "protocol_hash": self.hash()}


def protocol_from_ctx(ctx, match_frac: float = 0.05, primary_policy: str = "excl", splits=None, folds=None) -> EvalProtocol:
    f = folds or ctx.folds
    s = splits or ctx.splits
    fd, sd = Path(f.directory), Path(s.directory)
    return EvalProtocol(folds=fd.name, discovery=sd.name,
                        folds_fp=_file_fp(fd / "fold_map.tif") + "|" + _file_fp(fd / "fold_params.json"),
                        discovery_fp=_file_fp(sd / "discovery_map.tif") + "|" + _file_fp(sd / "discovery_params.json"),
                        match_frac=match_frac, primary_policy=primary_policy)


def assert_same_protocol(*hashes: str) -> None:
    hs = {h for h in hashes if h}
    if len(hs) > 1:
        raise ValueError(f"results from different evaluation protocols cannot be compared: {sorted(hs)}")
