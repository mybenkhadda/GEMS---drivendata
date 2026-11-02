"""G06 full-region inference: output contract, train / inference parity, research tagging, submission gate."""
import numpy as np
import pandas as pd
import pytest

rasterio = pytest.importorskip("rasterio")
pytest.importorskip("lightgbm")

from gems_bench.g06 import inference as INF  # noqa: E402
from gems_bench.g06 import models_pixel as MP  # noqa: E402
from gems_bench.g06.contracts import GridSpec  # noqa: E402
from gems_bench.g06.raster_io import verify_raster_alignment  # noqa: E402

GRID = GridSpec(48, 40, "EPSG:32611", (100.0, 0.0, 5e5, 0.0, -100.0, 4e6), 100.0)


class Store:
    schema_hash = "h"

    def __init__(self):
        rng = np.random.default_rng(0)
        self.arrays = {k: rng.normal(size=GRID.shape).astype(np.float32) for k in ("a", "b")}
        self.arrays["a"][:, :4] = np.nan
        self.names = list(self.arrays)
        self.grid = GRID
        self.fp = np.ones(GRID.shape, bool)
        self.fp[:3] = False

    def footprint(self):
        return self.fp

    def load(self, names, window=None):
        r0, r1, c0, c1 = window or (0, GRID.height, 0, GRID.width)
        return np.stack([self.arrays[n][r0:r1, c0:c1] for n in names])

    def sample(self, names, rows, cols):
        return np.stack([self.arrays[n][rows, cols] for n in names], 1)


def model(store):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(500, 2)).astype(np.float32)
    y = (X[:, 0] > 0.5).astype(int)
    est, imp, _ = MP.fit_pixel_model("lightgbm", X, y, np.ones(500, np.float32),
                                     {"n_estimators": 20, "num_leaves": 7, "n_jobs": 1}, 0)
    return MP.PixelModel("lightgbm", est, ["a", "b"], store.schema_hash, imp)


def test_full_raster_contract_and_parity(tmp_path):
    st = Store()
    m = model(st)
    out = tmp_path / "research.tif"
    man = INF.predict_full_raster(m, st, out, tile_px=16, progress=None)
    assert verify_raster_alignment(out, GRID, ["probability"], 32611) == []
    with rasterio.open(out) as ds:
        a = ds.read(1)
        assert ds.tags()["ROLE"] == INF.RESEARCH_ROLE
    assert np.isnan(a[:3]).all() and np.isfinite(a[3:]).all()                 # full footprint, nothing outside
    assert ((a[3:] >= 0) & (a[3:] <= 1)).all() and man["n_px"] == int(st.fp.sum())
    rr, cc = np.nonzero(st.fp)
    meta = pd.DataFrame({"row_id": np.arange(rr.size), "row": rr, "col": cc})
    X = pd.DataFrame(st.sample(["a", "b"], rr, cc), columns=["a", "b"]).assign(row_id=meta.row_id)
    par = INF.verify_inference_parity(m, st, out, X, meta, n=300)
    assert par["status"] == "PASS", par
    X2 = X.copy()
    X2["b"] = X2["b"] + 1.0                                                   # training rows from other features
    assert INF.verify_inference_parity(m, st, out, X2, meta, n=300)["status"] == "FAIL"


def test_submission_gate():
    PD = pd.DataFrame({"model": ["M", "S"], "status": ["PROMOTE_CANDIDATE", "PROMISING_SCREENING_ONLY"]})
    ok, why = INF.submission_gate(PD, "M", {"EXPORT_SUBMISSION": True}, {"status": "PASS"})
    assert ok and not why
    assert not INF.submission_gate(PD, "M", {"EXPORT_SUBMISSION": False}, {"status": "PASS"})[0]
    assert not INF.submission_gate(PD, "S", {"EXPORT_SUBMISSION": True}, {"status": "PASS"})[0]
    assert not INF.submission_gate(PD, "M", {"EXPORT_SUBMISSION": True}, {"status": "FAIL"})[0]
    assert not INF.submission_gate(pd.DataFrame(), "M", {"EXPORT_SUBMISSION": True}, {"status": "PASS"})[0]


def test_export_candidate_is_never_uploaded(tmp_path):
    prof = {"driver": "GTiff", "height": GRID.height, "width": GRID.width, "count": 1, "dtype": "float32",
            "crs": GRID.crs, "transform": GRID.affine()}
    score = np.random.default_rng(2).random(GRID.shape).astype(np.float32)
    labels = np.zeros(GRID.shape, bool)
    labels[10, :] = True
    res = INF.export_submission_candidate(score, prof, np.ones(GRID.shape, bool), np.ones(GRID.shape, bool), labels,
                                          {"select": "top", "post": []}, 0.05, tmp_path / "c.tif", {"MODEL": "M"})
    assert res["status"] == "READY_FOR_MANUAL_UPLOAD" and res["uploaded"] is False
    with rasterio.open(tmp_path / "c.tif") as ds:
        q = ds.read(1)
    assert not (q[labels] > 0).any()                                          # known faults never selected
    elig = ~labels
    assert int((q > 0).sum()) == int(np.ceil(0.05 * elig.sum()))
