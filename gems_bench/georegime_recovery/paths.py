"""Isolated output namespace. Every write is checked against the protected source trees."""
from __future__ import annotations

from pathlib import Path

LAYOUT = (
    "00_inventory",
    "01_split_audit", "01_split_audit/masks",
    "02_full_inference", "02_full_inference/experts", "02_full_inference/router",
    "02_full_inference/inference_manifests",
    "03_validation", "03_validation/tp_fp_fn_spatial",
    "04_expert_analysis", "05_geological_evidence", "06_postprocessing",
    "07_figures/prediction_distribution", "07_figures/validation", "07_figures/spatial_errors",
    "07_figures/experts", "07_figures/geology", "07_figures/recovery",
    "08_tables", "09_reports", "10_submission",
    "cache", "cache/inference", "cache/hash", "cache/bank", "manifests",
)

# notebook plot_group name -> figure folder
FIGURE_DIR = {
    "prediction_distribution": "prediction_distribution", "known_fault_proximity": "prediction_distribution",
    "split_integrity": "validation", "original_dti": "validation", "full_area_coverage": "validation",
    "full_area_dti": "validation", "error_atlas": "spatial_errors", "expert_complementarity": "experts",
    "geological_disagreement": "geology", "structural_strata": "geology", "candidate_lineaments": "geology",
    "transform_tradeoffs": "recovery", "crossfit_uncertainty": "recovery",
}


class ProtectedPathError(PermissionError):
    pass


class RecoveryPaths:
    def __init__(self, cfg: dict):
        self.drive = Path(cfg["drive_root"])
        self.root = self.drive / cfg["output_root"]
        self.atlas = self.drive / cfg["atlas_root"]
        self.source_run = self.drive / cfg["source_run"]
        # GEMS_GEOREGIME_V1/runs/<key> -> GEMS_GEOREGIME_V1
        self.v1_root = self.source_run.parent.parent
        self.v1_cache = self.v1_root / "cache"
        self.scratch = Path(cfg["scratch_root"])
        self.protected = [self.v1_root.resolve(), self.atlas.resolve()]
        if any(self._inside(self.root.resolve(), p) for p in self.protected):
            raise ProtectedPathError(f"output root {self.root} lies inside a protected source tree")

    @staticmethod
    def _inside(path: Path, parent: Path) -> bool:
        try:
            path.relative_to(parent)
            return True
        except ValueError:
            return False

    def check_writable(self, path) -> Path:
        p = Path(path).resolve()
        for q in self.protected:
            if self._inside(p, q):
                raise ProtectedPathError(f"refusing to write inside protected source tree: {p}")
        return Path(path)

    def ensure(self) -> "RecoveryPaths":
        for d in LAYOUT:
            (self.root / d).mkdir(parents=True, exist_ok=True)
        self.scratch.mkdir(parents=True, exist_ok=True)
        return self

    def d(self, *parts) -> Path:
        p = self.root.joinpath(*parts)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def fig_dir(self, group: str) -> Path:
        return self.d("07_figures", FIGURE_DIR[group])

    # source paths (read-only)
    def src(self, *parts) -> Path:
        return self.source_run.joinpath(*parts)

    def checkpoint(self, unit: str) -> Path:
        return self.src("checkpoints", f"{unit}_latest.pt")

    def normalizer(self, unit: str) -> Path:
        return self.src("checkpoints", f"{unit}_normalizer.json")

    def v1_prediction(self, unit: str) -> Path:
        return self.src("predictions", f"{unit}_prediction.tif")

    def v1_router(self, unit: str) -> Path:
        return self.src("predictions", f"{unit}_router_magnetic.tif")
