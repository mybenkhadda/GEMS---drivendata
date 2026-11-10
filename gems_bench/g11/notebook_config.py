"""Validated adapter: master-notebook ``CONFIG`` dict -> G11 YAML overrides + an explicit stage plan.

Every notebook key maps to an EXISTING config key (``KEY_MAP``) or to the plan; unknown keys and incompatible values
raise :class:`NotebookConfigError` with the reason. Nothing here enables an automatic upload.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

RUN_MODES = ("AUDIT_ONLY", "INGEST_ONLY", "EDA_ONLY", "TRAIN_ONLY", "EVALUATE_ONLY", "SUBMISSION_ONLY", "FULL",
             "SMOKE_TEST")
FAILURE_POLICIES = ("STOP_REQUIRED_SKIP_OPTIONAL", "STOP_ON_ANY_FAILURE")
MODELS = {"xgboost": ["E0", "E1", "E2"], "segformer": ["E3"], "gated_fusion": ["E4"], "segment_ranker": []}
STEPS = ("audit", "ingest", "features", "eda", "splits", "train_pixel", "train_segformer", "train_gated", "fusion",
         "candidates", "evaluate", "submit", "report")
STEP_FLAG = {"audit": "run_audit", "ingest": "run_ingestion", "features": "run_features", "eda": "run_eda",
             "splits": "run_splits", "train_pixel": "run_training", "train_segformer": "run_training",
             "train_gated": "run_training", "fusion": "run_fusion", "candidates": "run_candidates",
             "evaluate": "run_evaluation", "submit": "run_submission", "report": None}
MODE_STEPS = {
    "AUDIT_ONLY": ["audit", "report"],
    "INGEST_ONLY": ["audit", "ingest", "report"],
    "EDA_ONLY": ["audit", "ingest", "features", "eda", "report"],
    "TRAIN_ONLY": ["audit", "ingest", "features", "splits", "train_pixel", "train_segformer", "train_gated", "report"],
    "EVALUATE_ONLY": ["audit", "splits", "fusion", "evaluate", "report"],
    "SUBMISSION_ONLY": ["audit", "candidates", "submit", "report"],
    "FULL": list(STEPS),
    "SMOKE_TEST": ["audit", "ingest", "features", "eda", "splits", "train_pixel", "train_segformer", "evaluate",
                   "report"],
}

NOTEBOOK_DEFAULTS = {
    "repo": "mybenkhadda/GEMS---drivendata", "ref": "g11-colab-production", "drive_root": "/content/drive/MyDrive/GEMS_2",
    "repo_root": "/content/GEMS", "local_root": "/content/g11_work", "run_mode": "FULL", "resume": True,
    "force_recompute": False, "use_gpu": True, "mixed_precision": True, "seed": 2026,
    "run_audit": True, "run_ingestion": True, "run_features": True, "run_eda": True, "run_splits": True,
    "run_training": True, "run_fusion": True, "run_candidates": True, "run_evaluation": True, "run_submission": True,
    "models": ["xgboost", "segformer", "gated_fusion", "segment_ranker"],
    "dem_native_resolution_m": 1.0, "dem_statistics_resolution_m": 2.0, "dem_training_resolution_m": 25.0,
    "geo_resolution_m": 100.0, "training_target_buffer_m": 200.0, "unlabeled_background_weight": 0.3,
    "strict_integrity": True, "failure_policy": "STOP_REQUIRED_SKIP_OPTIONAL", "save_intermediate": True,
    "generate_figures": True, "generate_html_report": True, "official_submission": True, "upload_submission": False,
    "smoke_window_km": 30.0, "output_subdir": "outputs/G11_MULTISCALE_FAULT_DISCOVERY",
}
BOOL_KEYS = {k for k, v in NOTEBOOK_DEFAULTS.items() if isinstance(v, bool)}


class NotebookConfigError(ValueError):
    pass


@dataclass
class Plan:
    mode: str
    steps: list
    experiments: list
    overrides: dict
    notebook: dict
    notes: list = field(default_factory=list)

    def enabled(self, step: str) -> bool:
        return step in self.steps

    def table(self):
        import pandas as pd

        return pd.DataFrame([{"step": s, "enabled": s in self.steps} for s in STEPS])


REQUIRED_STEPS = {"audit", "ingest", "features", "splits", "train_pixel", "evaluate", "submit"}


class StepRunner:
    """Run one notebook section: plan check, configuration echo, timing / memory tracking, failure policy.

    Required steps stop the notebook on failure; optional ones (EDA, deep models, fusion) are recorded as FAILED
    and skipped under ``STOP_REQUIRED_SKIP_OPTIONAL`` (everything stops under ``STOP_ON_ANY_FAILURE``).
    """

    def __init__(self, pipeline, plan: Plan, log=print):
        self.P, self.plan, self.log = pipeline, plan, log
        self.results: dict = {}

    def __call__(self, step: str, fn, *, show: dict | None = None, check=None):
        import time

        if not self.plan.enabled(step):
            self.log(f"[{step}] SKIPPED (not in plan for run_mode={self.plan.mode} / disabled in CONFIG)")
            self.results[step] = {"status": "SKIPPED"}
            return None
        if show:
            self.log(f"[{step}] configuration: {show}")
        t0 = time.time()
        try:
            with self.P.tracker.track(step, self.log):
                out = fn()
            if check is not None:
                msg = check(out)
                if msg:
                    raise AssertionError(f"integrity check failed: {msg}")
            self.results[step] = {"status": "COMPLETE", "seconds": round(time.time() - t0, 1)}
            return out
        except Exception as e:
            self.results[step] = {"status": "FAILED", "error": f"{type(e).__name__}: {e}"}
            stop_all = self.plan.notebook.get("failure_policy") == "STOP_ON_ANY_FAILURE"
            if stop_all or step in REQUIRED_STEPS:
                self.log(f"[{step}] FAILED (required) - fix the cause above, then re-run this cell; completed stages "
                         "are reused from their manifests")
                raise
            self.log(f"[{step}] FAILED (optional, skipped): {type(e).__name__}: {e}")
            return None


def build_plan(user: dict, base_cfg: dict | None = None) -> Plan:
    """Validate ``user`` (notebook CONFIG) and return the plan. ``base_cfg`` (the YAML) is used for checks."""
    unknown = sorted(set(user) - set(NOTEBOOK_DEFAULTS))
    if unknown:
        raise NotebookConfigError(f"unknown CONFIG key(s) {unknown}; allowed: {sorted(NOTEBOOK_DEFAULTS)}")
    nb = {**copy.deepcopy(NOTEBOOK_DEFAULTS), **copy.deepcopy(user)}
    for k in BOOL_KEYS:
        if not isinstance(nb[k], bool):
            raise NotebookConfigError(f"{k} must be True/False, got {nb[k]!r}")
    if nb["run_mode"] not in RUN_MODES:
        raise NotebookConfigError(f"run_mode {nb['run_mode']!r} not in {RUN_MODES}")
    if nb["failure_policy"] not in FAILURE_POLICIES:
        raise NotebookConfigError(f"failure_policy {nb['failure_policy']!r} not in {FAILURE_POLICIES}")
    if nb["upload_submission"]:
        raise NotebookConfigError("upload_submission=True is not supported: G11 never uploads to DrivenData "
                                  "(download the validated GeoTIFF and upload it manually)")
    bad = sorted(set(nb["models"]) - set(MODELS))
    if bad:
        raise NotebookConfigError(f"unknown model(s) {bad}; allowed: {sorted(MODELS)}")
    if nb["resume"] and nb["force_recompute"]:
        raise NotebookConfigError("resume=True and force_recompute=True are contradictory")
    for k in ("dem_native_resolution_m", "dem_statistics_resolution_m", "dem_training_resolution_m", "geo_resolution_m",
              "training_target_buffer_m", "smoke_window_km"):
        if not isinstance(nb[k], (int, float)) or nb[k] <= 0:
            raise NotebookConfigError(f"{k} must be a positive number, got {nb[k]!r}")
    if not 0 < nb["unlabeled_background_weight"] <= 1:
        raise NotebookConfigError("unlabeled_background_weight must be in (0, 1]")
    geo = float(nb["geo_resolution_m"])
    st = float(nb["dem_statistics_resolution_m"])
    if abs(round(geo / st) * st - geo) > 1e-6:
        raise NotebookConfigError(f"dem_statistics_resolution_m={st} must divide geo_resolution_m={geo} exactly")
    if nb["dem_native_resolution_m"] > st:
        raise NotebookConfigError("dem_statistics_resolution_m cannot be finer than the native DEM resolution")
    buf = float(nb["training_target_buffer_m"])
    buffers = (base_cfg or {}).get("labels", {}).get("buffers_m", [100, 200, 500, 1000])
    if buf not in [float(b) for b in buffers]:
        raise NotebookConfigError(f"training_target_buffer_m={buf} must be one of labels.buffers_m {buffers}")
    ov = {
        "data_root": nb["drive_root"], "output_root": nb["output_subdir"] if nb["run_mode"] != "SMOKE_TEST"
        else nb["output_subdir"].rstrip("/") + "_SMOKE", "scratch_root": nb["local_root"],
        "reuse_valid_cache": bool(nb["resume"] and not nb["force_recompute"]),
        "geophysical_grid": {"expected_res_m": geo},
        "dem": {"stats_resolution_m": st},
        "feature_engineering": {"cross_fault": {"resolution_m": float(nb["dem_native_resolution_m"])}},
        "models": {"terrain_encoder": {"dem_res_m": float(nb["dem_training_resolution_m"])},
                   "xgb": {"pu_unlabeled_weight": float(nb["unlabeled_background_weight"])}},
        "labels": {"training_target": f"Y_BUFFER_{int(buf)}M"},
        "training": {"amp": bool(nb["mixed_precision"])},
        "reproducibility": {"seed": int(nb["seed"])}, "spatial_splits": {"seed": int(nb["seed"])},
        "reporting": {"figures": nb["generate_figures"], "html": nb["generate_html_report"],
                      "save_intermediate": nb["save_intermediate"], "strict_integrity": nb["strict_integrity"],
                      "failure_policy": nb["failure_policy"]},
        "smoke": {"enabled": nb["run_mode"] == "SMOKE_TEST", "window_km": float(nb["smoke_window_km"])},
    }
    steps = [s for s in MODE_STEPS[nb["run_mode"]] if STEP_FLAG[s] is None or nb[STEP_FLAG[s]]]
    notes = []
    if "segformer" not in nb["models"]:
        steps = [s for s in steps if s != "train_segformer"]
    if "gated_fusion" not in nb["models"]:
        steps = [s for s in steps if s != "train_gated"]
    if "xgboost" not in nb["models"]:
        steps = [s for s in steps if s != "train_pixel"]
        notes.append("xgboost disabled: E1 importance (used to pick deep-model channels) will be missing")
    if "segment_ranker" not in nb["models"]:
        steps = [s for s in steps if s not in ("candidates", "submit")]
        notes.append("segment_ranker disabled: no candidate ranking and therefore no submission")
    if not nb["official_submission"]:
        steps = [s for s in steps if s != "submit"]
    exps = [e for m in nb["models"] for e in MODELS[m]]
    if nb["run_mode"] == "SMOKE_TEST":
        exps = [e for e in exps if e in ("E0", "E1", "E2", "E3")]
        ov["training"].update(epochs=2, patience=2)
        ov["sampling"] = {"pixel_samples_per_split": 60000, "patches_train": 400, "patches_validation": 150}
        ov["eda"] = {"n_bootstrap": 100, "atlas_sites": 6}
        notes.append("SMOKE_TEST: geographic subset, 2 epochs, reduced samples - results are NOT comparable to FULL")
    return Plan(nb["run_mode"], steps, exps, ov, nb, notes)
