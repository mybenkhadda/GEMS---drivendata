"""G05.1 reconstruction helpers: zone rank, research GeoTIFF tags, no upload path, self-test table."""
import numpy as np
import pytest

from gems_bench.g05_1 import export as KX
from gems_bench.g05_1 import selection as SL
from gems_bench.g05_1 import selftest as ST


def test_zone_rank():
    ST.t_zone_rank()


def test_research_tif_roundtrip(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    prof = {"driver": "GTiff", "height": 4, "width": 5, "count": 1, "dtype": "uint8", "crs": "EPSG:32611",
            "transform": from_origin(500000, 4000000, 100, 100), "nodata": 255}
    a = np.full((4, 5), np.nan, np.float32)
    a[1:3, 1:4] = np.linspace(0, 1, 6).reshape(2, 3)
    p = tmp_path / "research_score_X.tif"
    KX.write_research_tif(p, a, prof, {"ROLE": "RESEARCH_NOT_A_SUBMISSION"})
    with rasterio.open(p) as s:
        b = s.read(1)
        assert s.tags()["ROLE"] == "RESEARCH_NOT_A_SUBMISSION" and s.dtypes[0] == "float32"
    assert np.allclose(np.nan_to_num(a, nan=-1), np.nan_to_num(b, nan=-1))


def test_validated_candidate_refuses_without_promotion():
    st, info = KX.validated_candidate(None, "G1", {"status": SL.PROMISING}, ("top_area", 0.01))
    assert st == "SKIPPED_BY_VALID_DECISION" and info["status"] == KX.NONE


def test_no_upload_function_exists():
    from gems_bench.g05_1 import export as ex
    from gems_bench.g05_1 import runner as rn

    for mod in (ex, rn):
        assert not [n for n in dir(mod) if "upload" in n.lower() and callable(getattr(mod, n))]
    for cls in (rn.G051,):
        assert not [n for n in dir(cls) if "upload" in n.lower()]


def test_selftests_all_pass():
    T = ST.run_selftests()
    bad = T[T.status != "PASS"]
    assert not len(bad), bad.to_dict("records")
