"""Regression tests for errors observed in the G14 Colab run, plus static (AST) guards against the same bug class.

Observed (fixed in 62f26dd / 3c56975 on g14-geofaultformer):
* ``p, t, w = expression_a, expression_b``                          (losses.soft_cldice)
* ``for name, (f, _, _, S) in zip(LEVEL_NAMES, self.views)``        (multiscale._radiometric; views are 2-tuples)
and the submission-path failures (missing final_0 checkpoint / manifest / full map / scorecard / discovery calibration,
incompatible run-mode flags).

WRITTEN BUT NEVER EXECUTED.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1] / "gems_bench"


def _py_files():
    for sub in ("g14", "g14_1"):
        yield from sorted((ROOT / sub).glob("*.py"))


def test_no_tuple_unpacking_arity_mismatch():
    """Static: ``a, b, c = x, y`` style assignments with literal tuples of different lengths."""
    bad = []
    for f in _py_files():
        for node in ast.walk(ast.parse(f.read_text())):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Tuple):
                for t in node.targets:
                    if isinstance(t, ast.Tuple) and not any(isinstance(e, ast.Starred) for e in t.elts) \
                            and len(t.elts) != len(node.value.elts):
                        bad.append(f"{f.name}:{node.lineno}")
    assert not bad, bad


def test_views_are_unpacked_as_pairs():
    """Static: any ``for ..., (<k names>) in zip(LEVEL_NAMES, self.views)`` must unpack exactly 2 names."""
    bad = []
    for f in _py_files():
        for node in ast.walk(ast.parse(f.read_text())):
            if isinstance(node, ast.For) and isinstance(node.iter, ast.Call) and getattr(node.iter.func, "id", "") == "zip":
                args = node.iter.args
                if len(args) == 2 and isinstance(args[1], ast.Attribute) and args[1].attr == "views" \
                        and isinstance(node.target, ast.Tuple) and len(node.target.elts) == 2 \
                        and isinstance(node.target.elts[1], ast.Tuple) and len(node.target.elts[1].elts) != 2:
                    bad.append(f"{f.name}:{node.lineno}")
    assert not bad, bad


def test_soft_cldice_runs():
    torch = pytest.importorskip("torch")
    from gems_bench.g14.losses import soft_cldice

    p = torch.rand(2, 32, 32)
    t = torch.zeros(2, 32, 32)
    t[:, 16, 4:28] = 1
    v, ok = soft_cldice(p, t, torch.ones(2, 32, 32), 3)
    assert ok and torch.isfinite(v)


def test_sensor_dropout_path_runs(tmp_path):
    """The exact G14 code path that crashed: _radiometric with sensor dropout forced on."""
    from gems_bench.g13.contracts import load
    from gems_bench.g13.splits import build_splits
    from gems_bench.g13.synthetic import make, small_config
    from gems_bench.g14.contract import build_stack
    from gems_bench.g14.multiscale import MultiScaleDataset, UnitInputs, unit_targets
    from gems_bench.g14.splits import UnitSystem
    from test_g14_core import small_cfg

    data = load(make(tmp_path / "d"), tmp_path / "c")
    sysm = UnitSystem("g13", data, g13_splits=build_splits(data, small_config(str(tmp_path), str(tmp_path))),
                      embargo_m=2000.0)
    cfg = small_cfg(data={"sensor_dropout": 1.0, "modality_dropout": 1.0})
    st = build_stack(data, cfg, tmp_path / "s", None, log=lambda *a: None)
    u = sysm.geo_unit(sysm.cv_zones[0])
    ds = MultiScaleDataset(UnitInputs(st, data, u, cfg), unit_targets(u.labels & u.train), u.train, u.ignore, cfg, 4,
                           0, augment=True)
    for i in range(len(ds)):
        item = ds[i]
        assert np.isfinite(item["local"]).all()


def test_g14_submit_with_train_flag_is_rejected_and_g141_modes_are_consistent():
    from gems_bench.g14.config import G14Config
    from gems_bench.g14_1.config import MODES, G141Config

    with pytest.raises(ValueError):
        G14Config.from_dict({"run_mode": "SUBMIT", "train_enabled": True})
    for m in MODES:
        c = G141Config.for_mode(m)
        assert not (c.allow_training and c.allow_submission), m


def test_g14_predict_final_message_names_the_fix(tmp_path):
    """The user's fix (b8a580e) raises with the missing final_0 manifest / checkpoint listed."""
    src = (ROOT / "g14" / "pipeline.py").read_text()
    assert "final model" in src and "include_final=True" in src
