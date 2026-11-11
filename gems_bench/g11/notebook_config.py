"""Validated adapter: master-notebook ``CONFIG`` dict -> G11 YAML overrides + an explicit stage plan.

Every notebook key maps to an EXISTING config key or to the plan; unknown keys and incompatible values raise
:class:`NotebookConfigError` with the reason. Nothing here enables an automatic upload, and nothing here can make
the pipeline recompute features silently: features are recomputed only for layers that are missing / invalid, or for
a NEW version with ``feature_policy="FORCE_NEW_VERSION"`` (never the default).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

RUN_MODES = ("FULL_RESUME", "RECOVER_ONLY", "PROCESS_FEATURES_ONLY", "FINALIZE_FEATURES_ONLY", "SPLITS_ONLY",
             "TRAIN_ONLY", "INFER_ONLY", "AUDIT_ONLY", "INGEST_ONLY", "EDA_ONLY", "EVALUATE_ONLY", "SUBMISSION_ONLY",
             "SMOKE_TEST", "FULL")
FAILURE_POLICIES = ("STOP_REQUIRED_SKIP_OPTIONAL", "STOP_ON_ANY_FAILURE")
FEATURE_POLICIES = ("REUSE_VERIFIED", "RESUME_MISSING", "FORCE_NEW_VERSION")
MODELS = {"xgboost": ["E0", "E1"], "segformer": ["E2"], "gated_fusion": ["E3"], "segment_ranker": []}
STEPS = ("audit", "ingest", "recover", "process_features", "finalize_features", "validate_features", "eda", "splits",
         "validate_splits", "training_data", "train_pixel", "train_segformer", "train_gated", "train_ranker",
         "inference", "candidates", "evaluate", "submit", "report")
STEP_FLAG = {"audit": "run_audit", "ingest": "run_ingestion", "recover": None, "process_features": "run_features",
             "finalize_features": None, "validate_features": None, "eda": "run_eda", "splits": "run_splits",
             "validate_splits": None, "training_data": "run_training", "train_pixel": "run_training",
             "train_segformer": "run_training", "train_gated": "run_training", "train_ranker": "run_candidates",
             "inference": "run_evaluation", "candidates": "run_candidates", "evaluate": "run_evaluation",
             "submit": "run_submission", "report": None}
_FEAT = ["recover", "process_features", "finalize_features", "validate_features"]
_TRAIN = ["training_data", "train_pixel", "train_segformer", "train_gated"]
MODE_STEPS = {
    "FULL_RESUME": list(STEPS),
    "RECOVER_ONLY": ["audit", "recover", "report"],
    "PROCESS_FEATURES_ONLY": ["audit", "ingest", "recover", "process_features", "report"],
    "FINALIZE_FEATURES_ONLY": ["audit", "ingest", "recover", "finalize_features", "validate_features", "report"],
    "SPLITS_ONLY": ["audit", "validate_features", "splits", "validate_splits", "report"],
    "TRAIN_ONLY": ["audit", "validate_features", "splits", "validate_splits", *_TRAIN, "report"],
    "INFER_ONLY": ["audit", "validate_features", "splits", "validate_splits", "inference", "train_ranker", "candidates",
                   "report"],
    "AUDIT_ONLY": ["audit", "report"],
    "INGEST_ONLY": ["audit", "ingest", "report"],
    "EDA_ONLY": ["audit", "validate_features", "eda", "report"],
    "EVALUATE_ONLY": ["audit", "validate_features", "splits", "validate_splits", "inference", "evaluate", "report"],
    "SUBMISSION_ONLY": ["audit", "candidates", "submit", "report"],
    "SMOKE_TEST": ["audit", "ingest", *_FEAT, "eda", "splits", "validate_splits", "training_data", "train_pixel",
                   "train_segformer", "inference", "report"],
}
MODE_STEPS["FULL"] = MODE_STEPS["FULL_RESUME"]                 # deprecated alias

NOTEBOOK_DEFAULTS = {
    "repo": "mybenkhadda/GEMS---drivendata", "ref": "g11-colab-production", "drive_root": "/content/drive/MyDrive/GEMS_2",
    "repo_root": "/content/GEMS", "local_root": "/content/g11_work", "run_mode": "FULL_RESUME", "resume": True,
    "force_recompute": False, "use_gpu": True, "mixed_precision": True, "seed": 2026,
    "feature_policy": "REUSE_VERIFIED", "feature_version": "g11_features_v1", "split_version": "v2",
    "recovery_verify": "AUTO", "save_every_chunk": True, "feature_time_budget_min": 0.0,
    "run_audit": True, "run_ingestion": True, "run_features": True, "run_eda": True, "run_splits": True,
    "run_training": True, "run_fusion": True, "run_candidates": True, "run_evaluation": True, "run_submission": True,
    "models": ["xgboost", "segformer", "gated_fusion", "segment_ranker"],
    "dem_native_resolution_m": 1.0, "dem_statistics_resolution_m": 10.0, "dem_training_resolution_m": 25.0,
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


REQUIRED_STEPS = {"audit", "ingest", "recover", "process_features", "finalize_features", "validate_features", "splits",
                  "validate_splits", "training_data", "train_pixel", "inference", "evaluate", "submit"}


class StepRunner:
    """Run one notebook section: plan check, configuration echo, timing / memory tracking, failure policy.

    Required steps stop the notebook on failure; optional ones (EDA, deep models, ranker) are recorded as FAILED and
    skipped under ``STOP_REQUIRED_SKIP_OPTIONAL`` (everything stops under ``STOP_ON_ANY_FAILURE``).
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
                         "and committed feature layers / DEM chunks are reused")
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
    if nb["feature_policy"] not in FEATURE_POLICIES:
        raise NotebookConfigError(f"feature_policy {nb['feature_policy']!r} not in {FEATURE_POLICIES}")
    if nb["recovery_verify"] not in ("AUTO", "FULL"):
        raise NotebookConfigError("recovery_verify must be AUTO or FULL")
    for k in ("feature_version", "split_version"):
        v = nb[k]
        if not isinstance(v, str) or not v or any(ch in v for ch in "/\\ #"):
            raise NotebookConfigError(f"{k} must be a plain name (no '/', '\\\\', ' ' or '#'), got {v!r}")
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
    if not isinstance(nb["feature_time_budget_min"], (int, float)) or nb["feature_time_budget_min"] < 0:
        raise NotebookConfigError("feature_time_budget_min must be >= 0 (0 = no limit)")
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
    smoke = nb["run_mode"] == "SMOKE_TEST"
    ov = {
        "data_root": nb["drive_root"], "output_root": nb["output_subdir"] if not smoke
        else nb["output_subdir"].rstrip("/") + "_SMOKE", "scratch_root": nb["local_root"],
        "reuse_valid_cache": bool(nb["resume"] and not nb["force_recompute"]),
        "geophysical_grid": {"expected_res_m": geo},
        "dem": {"stats_resolution_m": st, "save_every_chunk": bool(nb["save_every_chunk"])},
        "features": {"policy": nb["feature_policy"], "version": nb["feature_version"],
                     "recovery_verify": nb["recovery_verify"]},
        "feature_engineering": {"cross_fault": {"resolution_m": float(nb["dem_native_resolution_m"])}},
        "models": {"terrain_encoder": {"dem_res_m": float(nb["dem_training_resolution_m"])},
                   "xgb": {"pu_unlabeled_weight": float(nb["unlabeled_background_weight"])}},
        "labels": {"training_target": f"Y_BUFFER_{int(buf)}M"},
        "training": {"amp": bool(nb["mixed_precision"])},
        "reproducibility": {"seed": int(nb["seed"])}, "spatial_splits": {"seed": int(nb["seed"]),
                                                                         "version": nb["split_version"]},
        "reporting": {"figures": nb["generate_figures"], "html": nb["generate_html_report"],
                      "save_intermediate": nb["save_intermediate"], "strict_integrity": nb["strict_integrity"],
                      "failure_policy": nb["failure_policy"]},
        "smoke": {"enabled": smoke, "window_km": float(nb["smoke_window_km"])},
    }
    steps = [s for s in MODE_STEPS[nb["run_mode"]] if STEP_FLAG[s] is None or nb[STEP_FLAG[s]]]
    notes = []
    if nb["run_mode"] == "FULL":
        notes.append("run_mode FULL is a deprecated alias of FULL_RESUME")
    if nb["force_recompute"]:
        notes.append("force_recompute=True re-runs cached downstream stages; it never recomputes verified feature "
                     "layers (that requires feature_policy='FORCE_NEW_VERSION' with a new feature_version)")
    if nb["feature_policy"] == "FORCE_NEW_VERSION":
        notes.append(f"FORCE_NEW_VERSION: every layer of {nb['feature_version']} is recomputed (refused if that "
                     "version already exists) - expensive")
    if "segformer" not in nb["models"]:
        steps = [s for s in steps if s != "train_segformer"]
    if "gated_fusion" not in nb["models"]:
        steps = [s for s in steps if s != "train_gated"]
    if "xgboost" not in nb["models"]:
        steps = [s for s in steps if s != "train_pixel"]
        notes.append("xgboost disabled: E0 importance (used to pick deep-model channels) will be missing")
    if "segment_ranker" not in nb["models"]:
        steps = [s for s in steps if s not in ("train_ranker", "candidates", "submit")]
        notes.append("segment_ranker disabled: no candidate ranking (E4) and therefore no submission")
    if not nb["official_submission"]:
        steps = [s for s in steps if s != "submit"]
    exps = [e for m in nb["models"] for e in MODELS[m]]
    if smoke:
        exps = [e for e in exps if e in ("E0", "E1", "E2")]
        ov["training"].update(epochs=2, patience=2)
        ov["sampling"] = {"pixel_samples_per_split": 60000, "patches_train": 400, "patches_validation": 150}
        ov["eda"] = {"n_bootstrap": 100, "atlas_sites": 6}
        ov["spatial_splits"]["v2"] = {"n_regions": 8, "gates": {"min_blocks": 1, "require_patch_windows": False}}
        notes.append("SMOKE_TEST: geographic subset, 2 epochs, reduced samples, relaxed split block gates - results "
                     "are NOT comparable to FULL_RESUME")
    return Plan(nb["run_mode"], steps, exps, ov, nb, notes)
