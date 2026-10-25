"""Experiment fidelity, reuse, checkpoint retention, screening and dependency recovery (synthetic inputs only).

Covers: SCREEN / VALIDATE / FULL parsing, experiment-id isolation, reuse of completed FULL runs, partial-fold
resumption state, incompatible checkpoint rejection, safe checkpoint retention, NaN inner-validation handling,
leakage-free screening subsampling, matched evaluation masks, profile eligibility, B0_lineament recovery,
incremental Drive sync, E1 parity configuration and shortlist / promotion boundaries.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gems_bench.pipeline.fidelity import (Fidelity, FidelityError, apply_fidelity, block_subsample, fidelity_of,
                                          make_fidelity, mask_digest)

REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------- 1. profiles
def test_profile_defaults_and_validation():
    s = make_fidelity("screen")
    assert s.profile == "SCREEN" and s.geographic_folds == (0,) and s.discovery_splits == ()
    assert s.max_epochs == 5 and s.max_train_steps_per_epoch == 80 and s.validation_fraction == 0.25
    assert s.training_fraction == 1.0 and not s.full_resolution_inference
    v = make_fidelity("VALIDATE")
    assert v.geographic_folds is None and v.discovery_splits == (0,) and v.max_epochs == 10
    f = make_fidelity("FULL")
    assert f.is_full and f.semantic_block() == {} and f.max_epochs is None
    with pytest.raises(FidelityError):
        make_fidelity("FULL", {"max_epochs": 3})              # FULL keeps the committed budgets
    with pytest.raises(FidelityError):
        make_fidelity("SCREEN", {"max_epoch": 3})             # typo
    with pytest.raises(FidelityError):
        make_fidelity("SCREEN", {"validation_fraction": 0.0})
    with pytest.raises(FidelityError):
        make_fidelity("SCREEN", {"geographic_folds": []})     # a real fold is mandatory
    with pytest.raises(FidelityError):
        make_fidelity("TURBO")
    assert make_fidelity("SCREEN", {"geographic_folds": [2, 1, 2]}).geographic_folds == (1, 2)


def test_fold_and_split_selection_respect_the_design():
    s = make_fidelity("SCREEN", {"geographic_folds": [1]})
    assert s.fold_ids([0, 1, 2]) == [1] and s.split_ids(3) == []
    assert make_fidelity("FULL").fold_ids([0, 1, 2]) == [0, 1, 2] and make_fidelity("FULL").split_ids(3) == [0, 1, 2]
    with pytest.raises(FidelityError):
        make_fidelity("SCREEN", {"geographic_folds": [5]}).fold_ids([0, 1, 2])
    with pytest.raises(FidelityError):
        make_fidelity("VALIDATE", {"discovery_splits": [3]}).split_ids(3)


def test_cap_spec_only_below_full():
    from dataclasses import dataclass

    @dataclass
    class S:
        epochs: int = 40
        steps_per_epoch: int = 150
        min_epochs: int = 5

    s = make_fidelity("SCREEN", {"max_epochs": 3}).cap_spec(S())
    assert (s.epochs, s.steps_per_epoch, s.min_epochs) == (3, 80, 3)
    assert make_fidelity("FULL").cap_spec(S()) == S()


def test_pipeline_config_profile_is_a_separate_key():
    from gems_bench.pipeline.config import ConfigError, PipelineConfig

    c = PipelineConfig.load(flat={"run_mode": "ADVANCED_RESEARCH", "experiment_profile": "screen",
                                  "fidelity_overrides": {"screen": {"max_epochs": 4}}})
    assert c.mode == "ADVANCED_RESEARCH" and c.get("experiments.profile") == "SCREEN"
    assert c.get("experiments.fidelity_overrides") == {"SCREEN": {"max_epochs": 4}}
    assert PipelineConfig.load().get("experiments.profile") == "FULL"          # historical behaviour by default
    with pytest.raises(ConfigError):
        PipelineConfig.load(flat={"experiment_profile": "FAST"})
    with pytest.raises(ConfigError):
        PipelineConfig.load(flat={"fidelity_overrides": {"FULL": {"max_epochs": 2}}})
    with pytest.raises(ConfigError):
        PipelineConfig.load(flat={"screening_rules": {"promote_gain": 0.1}})
    with pytest.raises(ConfigError):
        PipelineConfig.load(flat={"checkpoint": {"policy": "everything"}})
    with pytest.raises(ConfigError):
        PipelineConfig.load(flat={"feature_stage": "sometimes"})


def test_notebook_config_validates():
    import ast

    nb = json.loads((REPO / "notebooks" / "colab" / "GEMS_MASTER_PIPELINE.ipynb").read_text(encoding="utf-8"))
    src = next("".join(c["source"]) for c in nb["cells"] if "".join(c["source"]).startswith("CONFIG = {"))
    tree = ast.parse(src)
    cfg = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and n.targets[0].id == "CONFIG")
    from gems_bench.pipeline.config import PipelineConfig

    c = PipelineConfig.load(flat=cfg)
    assert c.get("experiments.profile") in ("SCREEN", "VALIDATE", "FULL")


# ---------------------------------------------------------------------------- 2. identity
def _raw():
    return {"name": "b2_nnpu_p010", "kind": "cnn5", "seed": 2026, "model": {"base": "P4_B_gated"},
            "runtime": {"gpu": {}}}


def test_experiment_id_isolation():
    from gems_bench.experiment.config import ExperimentConfig

    base = ExperimentConfig.from_dict(_raw())
    full = ExperimentConfig.from_dict(apply_fidelity(_raw(), make_fidelity("FULL")))
    scr = ExperimentConfig.from_dict(apply_fidelity(_raw(), make_fidelity("SCREEN")))
    val = ExperimentConfig.from_dict(apply_fidelity(_raw(), make_fidelity("VALIDATE")))
    assert full.hash() == base.hash() and full.experiment_id() == base.experiment_id()   # legacy ids stay valid
    assert full.experiment_id() == f"b2_nnpu_p010__{base.hash()[:8]}"
    assert "__screen__" in scr.experiment_id() and "__validate__" in val.experiment_id()
    assert len({full.experiment_id(), scr.experiment_id(), val.experiment_id()}) == 3
    assert fidelity_of(scr).profile == "SCREEN" and fidelity_of(base).is_full
    other = ExperimentConfig.from_dict(apply_fidelity(_raw(), make_fidelity("SCREEN", {"max_epochs": 4})))
    assert other.experiment_id() != scr.experiment_id()                  # another screening budget = new run


def test_incompatible_profile_cannot_resume_manifest(tmp_path):
    from gems_bench.experiment.config import ExperimentConfig
    from gems_bench.experiment.manifest import ExperimentManifest, IncompatibleResume

    scr = ExperimentConfig.from_dict(apply_fidelity(_raw(), make_fidelity("SCREEN")))
    full = ExperimentConfig.from_dict(_raw())
    m = ExperimentManifest.load_or_create(tmp_path / "manifest.json", experiment_id=full.experiment_id(),
                                          config_hash=full.hash(), git_commit="abc")
    with pytest.raises(IncompatibleResume):
        m.check_compatible(scr.hash(), "abc", "config")


# ---------------------------------------------------------------------------- 3. leakage-free subsampling
def _roles():
    """30 x 40 grid: val block, 2-px buffer around it, train elsewhere (0 = train, 1 = val, 2 = buffer)."""
    r = np.zeros((30, 40), np.int8)
    r[8:20, 10:24] = 2
    r[10:18, 12:22] = 1
    return r


def test_block_subsample_subset_deterministic_nonempty():
    m = np.ones((50, 70), bool)
    m[:, :5] = False
    a = block_subsample(m, 0.25, 7, 16, salt=1)
    b = block_subsample(m, 0.25, 7, 16, salt=1)
    assert np.array_equal(a, b) and not (a & ~m).any() and a.any()
    assert not np.array_equal(a, block_subsample(m, 0.25, 7, 16, salt=2))
    assert np.array_equal(block_subsample(m, 1.0, 7, 16), m)
    tiny = np.zeros((50, 70), bool)
    tiny[3, 3] = True
    assert block_subsample(tiny, 0.01, 1, 16).sum() == 1             # never empties a non-empty mask


def test_screen_training_mask_never_reaches_validation_or_buffer():
    r = _roles()
    train, val, buf = r == 0, r == 1, r == 2
    f = make_fidelity("SCREEN", {"training_fraction": 0.5, "train_block_px": 8})
    t = f.train_mask(train, 0)
    assert not (t & val).any() and not (t & buf).any() and not (t & ~train).any() and t.any()
    s = f.val_mask(val, 0)
    assert not (s & ~val).any() and s.any()
    assert np.array_equal(make_fidelity("FULL").train_mask(train, 0), train)          # FULL never subsamples
    assert np.array_equal(make_fidelity("SCREEN").train_mask(train, 0), train)        # nor does the SCREEN default


def test_matched_masks_between_candidate_and_control():
    val = _roles() == 1
    cand, ctrl = make_fidelity("SCREEN"), make_fidelity("SCREEN")
    assert mask_digest(cand.val_mask(val, 0)) == mask_digest(ctrl.val_mask(val, 0))
    assert cand.protocol_hash("f", "d") == ctrl.protocol_hash("f", "d")
    seeded = make_fidelity("SCREEN", {"subsample_seed": 1, "eval_block_px": 4})
    assert seeded.protocol_hash("f", "d") != cand.protocol_hash("f", "d")


def test_profile_eligibility():
    assert make_fidelity("FULL").leaderboard_eligible and make_fidelity("FULL").promotion_eligible
    for p in ("SCREEN", "VALIDATE"):
        assert not make_fidelity(p).leaderboard_eligible and not make_fidelity(p).promotion_eligible


def test_contract_stubs_say_not_evaluated_below_full(tmp_path):
    from gems_bench.artifacts.store import RunDir

    rd = RunDir(tmp_path / "e")
    rd.finalize_contract("cnn5", "SCREEN")
    t = rd.load_table("geographic_dti")
    assert bool(t.not_evaluated.iloc[0]) and t.fidelity_profile.iloc[0] == "SCREEN"


# ---------------------------------------------------------------------------- 4. inventory / reuse
def _exp(root: Path, eid: str, name: str, *, profile="FULL", stages=("done",) * 4, units=(), finals=(),
         config_hash="h", scoreboard=None):
    d = root / "experiments" / eid
    (d / "checkpoints").mkdir(parents=True)
    (d / "metrics").mkdir()
    cfg = {"name": name, "kind": "cnn5"}
    if profile != "FULL":
        cfg["fidelity"] = make_fidelity(profile).semantic_block()
    import yaml

    (d / "config.yaml").write_text(yaml.safe_dump(cfg))
    st = dict(zip(("prepare", "train", "evaluate", "export"), stages))
    (d / "manifest.json").write_text(json.dumps({
        "experiment_id": eid, "name": name, "kind": "cnn5", "config_hash": config_hash,
        "stages": {k: {"status": v} for k, v in st.items()},
        "units": {u: {"status": "done"} for u in units}, "maps": {}}))
    for u in finals:
        (d / "checkpoints" / f"{u}_model.pt").write_bytes(b"x")
    if scoreboard is not None:
        scoreboard.to_csv(d / "metrics" / "scoreboard.csv", index=False)
    return d


def test_inventory_states(tmp_path):
    from gems_bench.pipeline.inventory import experiment_inventory, remaining_units

    allu = ["cv_fold0", "cv_fold1", "cv_fold2", "disc0", "disc1", "disc2"]
    _exp(tmp_path, "b2_nnpu_p010__e0173a9a", "b2_nnpu_p010", units=allu, finals=allu,
         scoreboard=pd.DataFrame({"model": ["P5_B2_nnpu_p010"], "geo_dti_area": [0.2]}))
    # E1: fold 0 trained, fold 1 marked done in the manifest but its final model is missing -> not counted
    _exp(tmp_path, "e1_gsn_parity__b8b8c3db", "e1_gsn_parity", stages=("done", "interrupted", "pending", "pending"),
         units=["cv_fold0", "cv_fold1"], finals=["cv_fold0"])
    _exp(tmp_path, "b1_bg_weight__screen__1234abcd", "b1_bg_weight", profile="SCREEN",
         stages=("done", "running", "pending", "pending"))
    inv = experiment_inventory(root=tmp_path).set_index("experiment_id")
    b2 = inv.loc["b2_nnpu_p010__e0173a9a"]
    assert b2.profile == "FULL" and b2.complete and b2.reusable and b2.metrics_available
    assert b2.next_action.startswith("REPORT")
    e1 = inv.loc["e1_gsn_parity__b8b8c3db"]
    assert e1.folds_done == "0" and not e1.complete and e1.next_action.startswith("RESUME")
    assert "[1, 2]" in e1.next_action and e1.role == "parity_check"
    b1 = inv.loc["b1_bg_weight__screen__1234abcd"]
    assert b1.profile == "SCREEN" and b1.folds_expected == "0" and b1.discovery_expected == ""
    rem = remaining_units(experiment_inventory(root=tmp_path)).set_index("experiment_id")
    assert rem.loc["e1_gsn_parity__b8b8c3db"].units_remaining == 5


class _PC:
    """Minimal pipeline context for experiment_config / plan_experiments (configs from this checkout)."""

    def __init__(self, **flat):
        from gems_bench.pipeline.config import PipelineConfig

        self.cfg = PipelineConfig.load(flat=flat)
        self.fold_rel = "outputs/benchmark/folds/geo_k3_b20km_buf13km_s2026"
        self.discovery_rel = "outputs/benchmark/folds/discovery_d3_link1km_s2026"

    def config_file(self, *parts):
        return REPO.joinpath("configs", *parts)

    def runtime_overrides(self):
        return {}


def _inv_row(pc, stem, profile, complete=True):
    from gems_bench.pipeline.experiments import experiment_config

    c = experiment_config(pc, stem, profile=profile)
    return {"experiment": stem, "experiment_id": c.experiment_id(), "profile": profile, "complete": complete,
            "config_compatible": True, "next_action": "x", "configuration_hash": c.hash()}


def test_plan_reuses_completed_full_and_adds_matched_control():
    pytest.importorskip("yaml")
    from gems_bench.pipeline.experiments import plan_experiments

    pc = _PC(experiment_profile="SCREEN")
    inv = pd.DataFrame([_inv_row(pc, "b2_nnpu_p010", "FULL"), _inv_row(pc, "c1_geometry_heads", "FULL"),
                        _inv_row(pc, "b0_reference", "FULL")])
    plan = plan_experiments(pc, ["b1_bg_weight", "b2_nnpu_p010", "c1_geometry_heads"], inventory=inv).set_index("experiment")
    assert plan.loc["b2_nnpu_p010"].action == "REUSE_FULL" and plan.loc["c1_geometry_heads"].action == "REUSE_FULL"
    assert plan.loc["b1_bg_weight"].action == "RUN" and "__screen__" in plan.loc["b1_bg_weight"].experiment_id
    assert plan.loc["b0_reference"].action == "RUN"           # matched control at SCREEN, added automatically
    only = plan_experiments(pc, ["b0_reference", "b2_nnpu_p010"], inventory=inv).set_index("experiment")
    assert only.loc["b0_reference"].action == "NOT_NEEDED"    # no candidate needs a SCREEN control
    full = plan_experiments(_PC(experiment_profile="FULL"), ["b2_nnpu_p010"], inventory=inv)
    assert full.action.iloc[0] == "DONE"


def test_plan_resumes_incomplete_same_profile_run():
    pytest.importorskip("yaml")
    from gems_bench.pipeline.experiments import plan_experiments

    pc = _PC(experiment_profile="FULL")
    inv = pd.DataFrame([_inv_row(pc, "e1_gsn_parity", "FULL", complete=False)])
    assert plan_experiments(pc, ["e1_gsn_parity"], inventory=inv).action.iloc[0] == "RESUME"


# ---------------------------------------------------------------------------- 5. checkpoints
def test_best_and_latest_retention_without_sync(tmp_path):
    torch = pytest.importorskip("torch")
    from gems_bench.training.checkpoint import CheckpointManager

    ck = CheckpointManager(tmp_path, "cv_fold0", spec_hash="h", policy="best_and_latest")
    for ep in (1, 2, 3):
        ck.save({"epoch": ep, "best_state": {"w": torch.ones(1) * ep}, "best_ep": min(ep, 2), "best": 0.1 * ep}, ep)
    assert ck.latest_path.exists() and not ck.prev_path.exists() and not ck.resume_files()
    assert ck.latest()["epoch"] == 3
    assert torch.load(ck.best_path, weights_only=False)["best_epoch"] == 2
    with pytest.raises(RuntimeError):                                     # another spec: refuse, never load
        CheckpointManager(tmp_path, "cv_fold0", spec_hash="other", policy="best_and_latest").latest()


class _Sync:
    def __init__(self, root, fail=False):
        self.local_root, self.fail, self.pushed = Path(root), fail, []

    def push(self, rels, verify=None):
        if self.fail:
            raise OSError("drive down")
        self.pushed += list(rels)

    def is_synced(self, p):
        return not self.fail


def test_previous_state_kept_until_drive_copy_verified(tmp_path):
    pytest.importorskip("torch")
    from gems_bench.training.checkpoint import CheckpointManager

    bad = CheckpointManager(tmp_path / "a", "u", spec_hash="h", policy="best_and_latest",
                            sync=_Sync(tmp_path, fail=True), sync_every_min=0)
    bad.save({"epoch": 1}, 1)
    bad.save({"epoch": 2}, 2)
    assert bad.prev_path.exists()                                        # push failed: older state retained
    ok_sync = _Sync(tmp_path)
    ok = CheckpointManager(tmp_path / "b", "u", spec_hash="h", policy="best_and_latest", sync=ok_sync,
                           sync_every_min=0)
    ok.save({"epoch": 1}, 1)
    ok.save({"epoch": 2}, 2)
    assert not ok.prev_path.exists() and any(r.endswith("u_latest.pt") for r in ok_sync.pushed)


def test_corrupt_latest_falls_back_and_legacy_files_resume(tmp_path):
    pytest.importorskip("torch")
    from gems_bench.training.checkpoint import CheckpointManager

    ck = CheckpointManager(tmp_path, "u", spec_hash="h", policy="best_and_latest", sync=_Sync(tmp_path, fail=True),
                           sync_every_min=0)
    ck.save({"epoch": 1}, 1)
    ck.save({"epoch": 2}, 2)
    ck.latest_path.write_bytes(b"truncated")                              # sha256 no longer matches the pointer
    assert ck.latest()["epoch"] == 1
    legacy = CheckpointManager(tmp_path / "legacy", "cv_fold0", spec_hash="h", keep=2)        # rotating run
    legacy.save({"epoch": 5}, 5)
    resumed = CheckpointManager(tmp_path / "legacy", "cv_fold0", spec_hash="h", policy="best_and_latest")
    assert resumed.latest()["epoch"] == 5                                 # an older run resumes under the new policy


def test_clear_resume_waits_for_verified_final(tmp_path):
    pytest.importorskip("torch")
    from gems_bench.training.checkpoint import CheckpointManager

    ck = CheckpointManager(tmp_path, "u", spec_hash="h", policy="best_and_latest", sync=_Sync(tmp_path, fail=True))
    ck.save({"epoch": 1}, 1)
    ck.save_final({"state_dict": {}})
    ck.clear_resume()
    assert ck.latest_path.exists()                                       # final not on Drive yet
    assert ck.final_ok()


def test_nan_inner_score_is_never_an_improvement():
    pytest.importorskip("torch")
    from gems_bench.training.trainer import improved

    assert improved(0.2, 0.1) and not improved(0.1, 0.1)
    assert not improved(float("nan"), -1.0) and not improved(float("inf"), -1.0) and not improved(None, -1.0)


# ---------------------------------------------------------------------------- 6. B0_lineament recovery
class _Ctx:
    def __init__(self, present):
        self.present, self._maps = present, {}

    def has_oof(self, name):
        return self.present


class _BenchCfg:
    n_discovery_folds = 3


class _DepPC:
    def __init__(self, ctx, auto=True):
        self._ctx, self.auto, self.local, self.pushed, self.staged = ctx, auto, Path("."), [], []
        self.bench_cfg = _BenchCfg()

        class C:
            def get(_s, k, d=None):
                return auto if k == "experiments.auto_provision_maps" else d

        self.cfg = C()

    def ctx5(self):
        return self._ctx

    def stage_in(self, rels):
        self.staged += rels

    def push(self, rels):
        self.pushed += rels

    def grid(self, root=None):
        return None


def _patch_status(monkeypatch, statuses, trained):
    from gems_bench.pipeline import baselines as B
    from gems_bench.pipeline import stages

    seq = iter(statuses)
    monkeypatch.setattr(B, "baseline_status", lambda *a, **k: {"status": next(seq), "origin": "", "detail": "d"})
    monkeypatch.setattr(stages, "folds_sha", lambda pc: "f")
    monkeypatch.setattr(stages, "splits_sha", lambda pc: "s")

    def train(pc, name, **k):
        trained.append(name)
        pc._ctx.present = True
        return {"rels": ["outputs/benchmark/experiments/b0_lineament_fused_s2026"]}

    monkeypatch.setattr(B, "train_baseline", train)


def test_missing_b0_lineament_is_built_with_its_recipe(monkeypatch):
    from gems_bench.pipeline import baselines as B
    from gems_bench.pipeline.experiments import ensure_label_free_map

    trained = []
    _patch_status(monkeypatch, [B.NOT_AVAILABLE, B.AVAILABLE], trained)
    pc = _DepPC(_Ctx(False))
    out = ensure_label_free_map(pc, "B0_lineament", requested_by="b1_bg_weight")
    assert trained == ["B0_lineament"] and out["action"].startswith("built")
    assert pc.pushed and pc.staged                                        # looked on Drive first, persisted after


def test_existing_b0_lineament_is_reused_and_never_rebuilt(monkeypatch):
    from gems_bench.pipeline import baselines as B
    from gems_bench.pipeline.experiments import ensure_label_free_map

    trained = []
    _patch_status(monkeypatch, [B.AVAILABLE], trained)
    answers = iter([False, True])            # not in the session context; resolvable once staged from Drive
    ctx = _Ctx(False)
    ctx.has_oof = lambda name: next(answers)
    pc = _DepPC(ctx)
    assert ensure_label_free_map(pc, "B0_lineament")["action"].startswith("reused")
    assert trained == [] and pc.staged
    assert any("b0_lineament_fused_s2026" in r for r in pc.staged)


def test_b0_lineament_guards(monkeypatch):
    from gems_bench.pipeline import baselines as B
    from gems_bench.pipeline.context import PipelineStop
    from gems_bench.pipeline.experiments import CATALOGUE, ensure_label_free_map

    assert CATALOGUE["b1_bg_weight"].requires_maps == ("B0_lineament",)
    with pytest.raises(PipelineStop):                       # a supervised model's map is never substituted
        ensure_label_free_map(None, "G1")
    trained = []
    _patch_status(monkeypatch, [B.INCOMPATIBLE], trained)
    with pytest.raises(PipelineStop):                       # other fold files: kept, never overwritten
        ensure_label_free_map(_DepPC(_Ctx(False)), "B0_lineament")
    _patch_status(monkeypatch, [B.NOT_AVAILABLE], trained)
    with pytest.raises(PipelineStop):                       # provisioning switched off: explicit recovery message
        ensure_label_free_map(_DepPC(_Ctx(False), auto=False), "B0_lineament")
    assert trained == []


# ---------------------------------------------------------------------------- 7. Drive sync
def test_incremental_scoped_push_and_ledger_flush(tmp_path):
    from gems_bench.artifacts.sync import LEDGER, DriveSync

    local, drive = tmp_path / "l", tmp_path / "d"
    e = local / "experiments" / "x"
    for sub, f in (("checkpoints", "u_latest.pt"), ("predictions", "oof.tif"), ("logs", "run.log")):
        (e / sub).mkdir(parents=True, exist_ok=True)
        (e / sub / f).write_bytes(b"abc")
    drive.mkdir()
    s = DriveSync(local, drive, ledger_mirror_s=3600)
    r = s.push_stage(["experiments/x"], "training")
    assert set(r[r.action == "copy"].rel) == {"experiments/x/checkpoints/u_latest.pt", "experiments/x/logs/run.log"}
    assert not (drive / "experiments/x/predictions/oof.tif").exists()
    assert json.loads((drive / LEDGER).read_text())["out"]                    # flushed at the stage boundary
    r2 = s.push(["experiments/x"], include=("*/checkpoints/*",))
    assert (r2.action == "skip").all()                                       # unchanged files never re-copied
    (e / "predictions" / "new.tif").write_bytes(b"z")
    s.push(["experiments/x"])                                                # throttled: Drive ledger not rewritten
    assert "experiments/x/predictions/new.tif" not in json.loads((drive / LEDGER).read_text())["out"]
    s.flush()
    assert "experiments/x/predictions/new.tif" in json.loads((drive / LEDGER).read_text())["out"]


def test_lazy_feature_store_stages_on_first_read(tmp_path):
    from gems_bench.features import FeatureStore

    local, drive = tmp_path / "l", tmp_path / "d"
    (drive / "cache").mkdir(parents=True)
    np.save(drive / "cache" / "a.npy", np.arange(4, dtype=np.float32))
    st = FeatureStore.__new__(FeatureStore)                                  # no NativeData needed for paths
    st.dir, st._mm, st.drive_mirror = local / "cache", {}, None
    st.dir.mkdir(parents=True)
    assert not st.has("a")
    st.drive_mirror = (local, drive)
    assert st.has("a") and not (local / "cache" / "a.npy").exists()          # available, not yet copied
    assert float(np.asarray(st.get("a")).sum()) == 6.0 and (local / "cache" / "a.npy").exists()


# ---------------------------------------------------------------------------- 8. E1 parity
def test_e1_is_routed_to_geostructurenet_in_parity_configuration():
    pytest.importorskip("yaml")
    from gems_bench.experiment.config import load_yaml_tree
    from gems_bench.pipeline.baselines import legacy_spec
    from gems_bench.pipeline.experiments import CATALOGUE

    raw = load_yaml_tree(REPO / "configs" / "experiments" / "e1_gsn_parity.yaml")
    m = raw["model"]
    assert m["type"] == "geostructurenet" and m["fusion"] == "gated" and m["long_range"] == "none"
    ref = legacy_spec("P4_B_gated")
    assert (m["overrides"]["enc_width"], m["overrides"]["dec_width"]) == (ref["enc_width"], ref["dec_width"])
    assert ref["fusion"] == "gated" and not ref["context"]
    assert CATALOGUE["e1_gsn_parity"].role == "parity_check" and CATALOGUE["b0_reference"].role == "control"


def test_e1_parity_model_is_numerically_identical_at_init():
    torch = pytest.importorskip("torch")
    from gems_bench.geological_encoders import GeoUNet
    from gems_bench.models.geostructurenet import GeoStructureNet

    groups = [[0, 1], [2, 3, 4]]
    torch.manual_seed(0)
    a = GeoUNet(groups, 5, enc_width=24, dec_width=32, fusion="gated")
    torch.manual_seed(0)
    b = GeoStructureNet(groups, 5, enc_width=24, dec_width=32, fusion="gated", long_range="none")
    pa, pb = list(a.parameters()), list(b.parameters())
    assert len(pa) == len(pb) and all(x.shape == y.shape and torch.equal(x, y) for x, y in zip(pa, pb))
    a.eval(), b.eval()
    x = torch.randn(1, 5, 64, 64)
    assert torch.allclose(a(x)["seg"], b(x)["seg"])


# ---------------------------------------------------------------------------- 9. shortlist boundaries
def _mc(eid, exp, prof="SCREEN", hi=0.0, mean_p=0.05):
    return {"experiment": exp, "experiment_id": eid, "profile": prof, "metric_kind": "proxy_geo", "status": "ok",
            "frac_p_gt_099": hi, "mean_p": mean_p}


def _dl(eid, gains, rec=0.0):
    rows = []
    for k, g in enumerate(gains):
        rows.append({"experiment_id": eid, "status": "ok", "zone": "geo", "unit": f"cv_fold{k}",
                     "metric": "proxy_dti_at_match", "delta": g})
        rows.append({"experiment_id": eid, "status": "ok", "zone": "geo", "unit": f"cv_fold{k}",
                     "metric": "proxy_recall300_at_match", "delta": rec})
    return rows


def _decide(mc, dl, **rules):
    from gems_bench.pipeline.screening import shortlist

    return shortlist(rules=rules or None, comparison=pd.DataFrame(mc), deltas=pd.DataFrame(dl)).set_index("experiment_id")


def test_shortlist_boundaries():
    d = _decide([_mc("a", "b2_nnpu_p010")], _dl("a", [0.005]))
    assert d.loc["a"].decision == "PROMOTE_TO_VALIDATE"                     # exactly the promotion threshold
    assert _decide([_mc("a", "b2_nnpu_p010")], _dl("a", [0.0049])).loc["a"].decision == "REVIEW"
    assert _decide([_mc("a", "b2_nnpu_p010")], _dl("a", [-0.010])).loc["a"].decision == "REJECT"
    assert _decide([_mc("a", "b2_nnpu_p010")], _dl("a", [0.02], rec=-0.05)).loc["a"].decision == "REVIEW"
    assert _decide([_mc("a", "b2_nnpu_p010", hi=0.9)], _dl("a", [0.05])).loc["a"].decision == "REJECT"   # collapse
    assert _decide([_mc("a", "b2_nnpu_p010", mean_p=1e-5)], _dl("a", [0.05])).loc["a"].decision == "REJECT"
    v = _decide([_mc("v", "b2_nnpu_p010", "VALIDATE")], _dl("v", [0.02, 0.02, -0.01]))
    assert v.loc["v"].decision == "PROMOTE_TO_FULL"                         # 2 of 3 folds beat the control
    v = _decide([_mc("v", "b2_nnpu_p010", "VALIDATE")], _dl("v", [0.05, -0.01, -0.01]))
    assert v.loc["v"].decision == "REVIEW"
    p = _decide([_mc("p", "e1_gsn_parity")], _dl("p", [0.0]))
    assert p.loc["p"].decision == "PARITY_CONFIRMED"
    n = _decide([_mc("n", "b2_nnpu_p010")], [])
    assert n.loc["n"].decision == "INCOMPARABLE"
    ctrl = _decide([_mc("c", "b0_reference")], _dl("c", [0.1]))
    assert "c" not in ctrl.index                                           # the control is never shortlisted


def test_full_runs_are_never_shortlisted_and_missing_metrics_stay_not_evaluated(tmp_path):
    from gems_bench.pipeline.screening import NE, metrics_comparison, paired_deltas, shortlist

    sb = pd.DataFrame({"model": ["P5_B2_nnpu_p010", "P5_B0_gated"], "geo_dti_area": [0.21, 0.20],
                       "protocol_hash": ["p", "p"]})
    d = _exp(tmp_path, "b2_nnpu_p010__e0173a9a", "b2_nnpu_p010", scoreboard=sb)
    m = json.loads((d / "manifest.json").read_text())
    m["maps"] = {"P5_B2_nnpu_p010": {"oof": "predictions/oof.tif", "disc": {}}}
    (d / "manifest.json").write_text(json.dumps(m))
    mc = metrics_comparison(root=tmp_path)
    cand = mc[mc.model == "P5_B2_nnpu_p010"].iloc[0]
    assert cand.is_candidate and cand.disc_dti_excl == NE and cand.geo_dti_area == 0.21
    pdl = paired_deltas(root=tmp_path, comparison=mc)
    g = pdl[pdl.metric == "geo_dti_area"].iloc[0]
    assert abs(g.delta - 0.01) < 1e-9
    assert pdl[pdl.metric == "disc_dti_excl"].iloc[0].delta == NE
    assert len(shortlist(root=tmp_path, comparison=mc, deltas=pdl)) == 0
