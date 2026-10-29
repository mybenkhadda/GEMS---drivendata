"""Preflight raster-metadata contract and environment checks on tiny synthetic GeoTIFFs (no competition data).
NOT executed at authoring time; needs rasterio + affine."""
from __future__ import annotations

import numpy as np
import pytest

from test_geological_eda import _runner  # noqa: F401

rasterio = pytest.importorskip("rasterio")


def _write(path, arr, dtype, nodata=None, transform=None):
    from affine import Affine

    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.atleast_3d(arr)
    if arr.ndim == 3 and arr.shape[0] not in (1, 19):
        arr = arr[None]
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[1], width=arr.shape[2], count=arr.shape[0],
                       dtype=dtype, crs="EPSG:32611", nodata=nodata,
                       transform=transform or Affine(100.0, 0, 243350.0, 0, -100.0, 4508550.0)) as d:
        d.write(arr.astype(dtype))


def _inputs(root, h=4, w=3, bands=19, faults_value=1, shift=False):
    from affine import Affine

    f = np.random.default_rng(0).normal(size=(bands, h, w)).astype("float32")
    _write(root / "data/raw/geodawn/gems-geodawn-numerical-features.tif", f, "float32")
    lab = np.zeros((1, h, w), "int8")
    lab[0, 1, 1] = faults_value
    t = Affine(100.0, 0, 243450.0 if shift else 243350.0, 0, -100.0, 4508550.0)
    _write(root / "data/raw/faults/existing_faults.tif", lab, "int8", nodata=-1, transform=t)
    _write(root / "data/raw/example_submission.tif", np.zeros((1, h, w), "float32"), "float32", nodata=np.nan)


@pytest.fixture
def tiny_grid(monkeypatch):
    from gems_eda.discovery import artifacts as AR

    monkeypatch.setitem(AR.GRID, "height", 4)
    monkeypatch.setitem(AR.GRID, "width", 3)
    monkeypatch.setitem(AR.GRID, "fault_px", 1)


def _pf(tmp_path):
    from gems_eda.discovery.preflight import Preflight

    E = _runner(tmp_path / "work")
    E.c["drive_root"] = str(tmp_path / "drive")
    return Preflight(E)


def test_raw_metadata_contract_passes(tmp_path, tiny_grid):
    _inputs(tmp_path / "drive")
    I = _pf(tmp_path).inputs().set_index("input")
    for k in ("features", "faults", "submission"):
        assert I.ok[k], (k, I.note[k])
    assert I.check_same_grid_as_features["faults"] and I.positive_px["faults"] == 1
    assert I.sample_finite_share["features"] > 0


def test_shape_mismatch_is_reported_not_resampled(tmp_path):
    _inputs(tmp_path / "drive")                             # 4 x 3 vs the recorded 3730 x 3292
    I = _pf(tmp_path).inputs().set_index("input")
    assert not I.ok["features"] and "shape" in I.note["features"]


def test_misaligned_fault_raster_fails(tmp_path, tiny_grid):
    _inputs(tmp_path / "drive", shift=True)
    I = _pf(tmp_path).inputs().set_index("input")
    assert not I.ok["faults"] and "same_grid_as_features" in I.note["faults"]


def test_non_binary_labels_fail(tmp_path, tiny_grid):
    _inputs(tmp_path / "drive", faults_value=5)
    I = _pf(tmp_path).inputs().set_index("input")
    assert not I.ok["faults"] and "labels_binary" in I.note["faults"]


def test_missing_input_and_optional_dem(tmp_path, tiny_grid):
    I = _pf(tmp_path).inputs().set_index("input")
    assert not I.ok["features"] and not I.exists["features"]
    assert I.ok["dem_urls"]                                 # optional input absent: not a failure


def test_env_report_and_require_ready(tmp_path):
    from gems_eda.discovery.preflight import PreflightBlocked, PreflightReport

    P = _pf(tmp_path)
    env = P.env()
    assert {"python", "eda_code_rev", "drive_root_mounted", "ram_total_gb"} <= set(env.check)
    R = PreflightReport(env, None, None, None, None, None, ["input features: missing or empty"])
    with pytest.raises(PreflightBlocked):
        R.require_ready()


def test_verify_checkout_detects_stale_install(tmp_path):
    from gems_eda.discovery.run import verify_checkout

    with pytest.raises(RuntimeError, match="stale install"):
        verify_checkout(str(tmp_path))                     # gems_eda is not imported from this directory
