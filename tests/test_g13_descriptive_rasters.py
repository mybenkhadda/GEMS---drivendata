"""Regression tests for official GeoDAWN descriptive band metadata (G13 inventory)."""

from __future__ import annotations

import numpy as np
import rasterio
from rasterio.transform import from_origin

from gems_bench.g13 import contracts as C


def _write_raster(path, data, *, descriptions=(), nodata=None, dtype="float32"):
    arr = np.asarray(data, dtype=dtype)
    if arr.ndim == 2:
        arr = arr[np.newaxis]
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=arr.shape[1],
        width=arr.shape[2],
        count=arr.shape[0],
        dtype=dtype,
        crs="EPSG:32611",
        transform=from_origin(500000, 4200000, 100, 100),
        nodata=nodata,
    ) as dst:
        dst.write(arr)
        for index, description in enumerate(descriptions, 1):
            dst.set_band_description(index, description)


def test_canonicalizes_descriptive_and_short_band_names():
    assert C.canonical_band_name("rtp - reduced to pole magnetic data", 2) == "rtp"
    assert C.canonical_band_name("iso_grav_anom - isostatic gravity anomaly", 13) == "iso_grav_anom"
    assert C.canonical_band_name("det_elev - detrended elevation", 12) == "det_elev"
    assert C.canonical_band_name("tmi_hg", 3) == "tmi_hg"
    assert C.canonical_band_name(None, 3) == "band_3"


def test_inventory_accepts_descriptive_19_band_raster_and_integer_labels(tmp_path):
    names = list(C.MODALITY)
    assert len(names) >= 10
    h, w = 16, 20
    raw = np.stack([np.full((h, w), i + 0.5, dtype=np.float32) for i in range(len(names))])
    raw[:, 0, 0] = -9999
    raw_path = tmp_path / C.OFFICIAL_NAMES["raw"]
    known_path = tmp_path / C.OFFICIAL_NAMES["known"]
    template_path = tmp_path / C.OFFICIAL_NAMES["template"]
    _write_raster(
        raw_path, raw,
        descriptions=[f"{name} - documented explanatory sentence" for name in names],
        nodata=-9999,
    )
    known = np.zeros((h, w), dtype=np.int16)
    known[4:12, 10] = 1
    known[0, 0] = -1
    _write_raster(known_path, known, dtype="int16", nodata=-1)
    template = np.zeros((h, w), dtype=np.float32)
    template[0, 0] = -9999
    _write_raster(template_path, template, nodata=-9999)

    assert C._probe(known_path)["sample_integer_like"]
    raw_info = C._probe(raw_path)
    assert C._role_ok("raw", raw_info)
    assert set(names) == set(raw_info["descriptions"])
    paths = C.discover_inputs(tmp_path)
    data = C.load(paths, tmp_path / "cache")
    assert data.names == names
    assert data.has("rtp") and data.has("iso_grav_anom")
    assert data.band("rtp").shape == (h, w)
    assert data.labels.sum() == 8

    inv = C.band_inventory(data)
    assert len(inv) == len(names)
    assert set(inv["name"]) == set(names)
    assert not (inv["meaning"] == "undocumented").all()
