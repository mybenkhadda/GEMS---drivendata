import numpy as np
import pytest

from gems_bench.g11.grid import (GridError, GridSpec, align_raster, inside, load_master_grid, pixel_to_world,
                                 validate_alignment, world_to_pixel)
from gems_bench.g11.tests.conftest import H, PX, W, X0, Y0


def test_master_grid_metadata(grid):
    assert grid.shape == (H, W) and grid.epsg == 32611 and grid.res == (PX, PX)
    assert grid.bounds == (X0, Y0 - H * PX, X0 + W * PX, Y0)
    assert not grid.valid[:, :3].any() and grid.valid[:, 3:].all()


def test_coordinate_round_trip(grid):
    rr, cc = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    x, y = pixel_to_world(rr, cc, grid)
    r2, c2 = world_to_pixel(x, y, grid)
    assert np.array_equal(r2, rr) and np.array_equal(c2, cc)
    assert pixel_to_world(0, 0, grid) == (X0 + 50, Y0 - 50)
    assert not inside(*world_to_pixel(X0 - 1, Y0 + 1, grid), grid)


def test_expected_grid_mismatch_raises(synth):
    with pytest.raises(GridError):
        load_master_grid(synth["raw"] / "gems-geodawn-numerical-features.tif", expected={"expected_shape": [10, 10]})


def test_missing_crs_raises(tmp_path):
    import rasterio
    from rasterio.transform import from_origin

    p = tmp_path / "nocrs.tif"
    with rasterio.open(p, "w", driver="GTiff", height=4, width=4, count=1, dtype="float32",
                       transform=from_origin(0, 4, 1, 1)) as ds:
        ds.write(np.ones((1, 4, 4), np.float32))
    with pytest.raises(GridError):
        load_master_grid(p)


def test_align_nearest_and_bilinear(grid):
    coarse = GridSpec(grid.crs, (200.0, 0, X0, 0, -200.0, Y0), (H // 2, W // 2))
    a = np.arange(coarse.shape[0] * coarse.shape[1], dtype=np.float32).reshape(coarse.shape)
    nn = align_raster((a, coarse), grid, kind="categorical")
    assert nn.shape == grid.shape
    assert np.all(nn[0:2, 0:2] == a[0, 0])                     # each coarse cell maps to 2x2 fine cells
    bl = align_raster((a, coarse), grid, kind="continuous")
    assert np.isfinite(bl).mean() > 0.9
    with pytest.raises(GridError):
        align_raster((a, coarse), grid, kind="unknown")


def test_validate_alignment(grid):
    rep = validate_alignment(grid, grid)
    assert rep.aligned and rep.transform_max_diff == 0
    shifted = GridSpec(grid.crs, (PX, 0, X0 + 50, 0, -PX, Y0), grid.shape)
    rep2 = validate_alignment(shifted, grid)
    assert not rep2.aligned and any("sub-pixel" in n for n in rep2.notes)
