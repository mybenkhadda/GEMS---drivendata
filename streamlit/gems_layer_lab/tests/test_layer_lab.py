from pathlib import Path
import numpy as np
import pytest
import rasterio
from rasterio.windows import Window

from lab.formulas import compile_formulas, evaluate_formulas, FormulaError, suggested_halo
from lab.rasters import catalog, aoi_window, padded, read_layers, read_faults, export_tiff, parse_extra_entries
from lab.stats import correlation_matrices, local_corr, grid_correlations, fault_exploration
from lab.export_full import main as export_main
from demo.create_demo import generate_demo


def test_formulas_and_reuse():
    a = np.array([[1., 2.], [3., 4.]], np.float32)
    b = np.ones((2, 2), np.float32)
    formulas = compile_formulas("ratio_ab = ratio(a,b,0.01)\nscale = ratio_ab * 2\nmask = where(a > 2, scale, 0)", ["a", "b"])
    result = evaluate_formulas(formulas, {"a": a, "b": b})
    np.testing.assert_allclose(result["scale"], a * 2)
    np.testing.assert_allclose(result["mask"], [[0,0],[6,8]])


@pytest.mark.parametrize("bad",[
    "x = __import__('os').system('whoami')", "x = a.__class__",
    "x = a[0]", "import os", "for i in range(10):\n    pass",
    "x = unknown + 1", "a = a + 1", "x = sum(a)",
    "x = gaussian(a,999)",
])
def test_parser_rejects_dangerous_and_invalid(bad):
    with pytest.raises(FormulaError):
        compile_formulas(bad,["a"])


def test_gaussian_nan_aware():
    a=np.ones((50,50),np.float32)
    a[10:15,10:15]=np.nan
    f=compile_formulas("z = gaussian(a,3)\nedge = highpass(a,3)",["a"])
    out=evaluate_formulas(f,{"a":a})
    assert np.isfinite(out["z"][12,12])
    assert abs(out["z"][20,20]-1)<1e-4
    assert suggested_halo(f)>=12


def test_local_stats():
    rng=np.random.default_rng(2)
    a=rng.normal(size=(45,47)).astype(np.float32)
    b=a*2+1
    corr,counts=local_corr(a,b,window_px=5)
    assert np.nanmedian(corr)>0.999
    assert counts.max()>=25
    grid,df=grid_correlations(a,b,grid=4)
    assert len(df)==16
    assert np.nanmean(grid)>0.999
    pear,spear,n,sampled,total=correlation_matrices({"a":a,"b":b},["a","b"],max_pixels=500,block_px=3)
    assert total<=500
    assert pear.loc["a","b"]>.999
    assert spear.loc["a","b"]>.999
    assert (n.to_numpy()==total).all()


def test_rasters_and_georef(tmp_path):
    a,b=generate_demo(tmp_path)
    sources,info=catalog(str(a))
    assert len(sources)==19
    assert info["crs"]=="EPSG:32611"
    assert sources["rtp"].band==2
    w=aoi_window(info["height"],info["width"],100,100,10,100)
    ext,crop=padded(w,info["height"],info["width"],24)
    loaded=read_layers(sources,["rtp","det_elev"],ext)
    assert loaded["rtp"].shape==(int(ext.height),int(ext.width))
    faults=read_faults(str(b),str(a),w)
    assert faults.dtype==bool and faults.shape==(int(w.height),int(w.width))
    tif=export_tiff(loaded["rtp"][crop],w,str(a),"test")
    out=tmp_path/"roi.tif";out.write_bytes(tif)
    with rasterio.open(out) as src:
        assert src.width==100 and src.height==100
        with rasterio.open(a) as full:
            from rasterio.windows import transform
            assert src.transform==transform(w,full.transform)
            assert src.crs==full.crs
    assert parse_extra_entries("test=/tmp/data.tif\nmask=/tmp/arrays.npz::labels")==[
        ("test","/tmp/data.tif",None),("mask","/tmp/arrays.npz","labels")]


def test_fault_distribution(tmp_path):
    a,b=generate_demo(tmp_path)
    sources,info=catalog(str(a))
    w=Window(0,0,420,400)
    loaded=read_layers(sources,["det_elev_slope"],w)
    faults=read_faults(str(b),str(a),w)
    stats,samples=fault_exploration(loaded["det_elev_slope"],faults,pixel_m=100,min_background_km=1)
    assert "error" not in stats
    assert 0<=stats["rank_separation"]<=1
    assert len(samples)>0


def test_full_export(tmp_path):
    features,faults=generate_demo(tmp_path)
    formulas=tmp_path/"formulas.json"
    formulas.write_text('{"formulas":"smooth = gaussian(rtp,4)\\ncombo = normdiff(smooth, iso_grav_anom, .01)"}')
    output=tmp_path/"export"
    export_main(["--features",str(features),"--formulas",str(formulas),"--out",str(output),
                 "--layers","smooth","combo","--tile","128"])
    with rasterio.open(features) as base,rasterio.open(output/"smooth.tif") as derived:
        assert derived.count==1 and base.height==derived.height
        assert base.crs==derived.crs and base.transform==derived.transform
        full=base.read(2,masked=True).filled(np.nan)
        from lab.formulas import _gaussian
        expect=_gaussian(full,4)
        got=derived.read(1)
        np.testing.assert_allclose(got[40:-40,40:-40],expect[40:-40,40:-40],atol=1e-5)
    assert (output/"export_manifest.json").is_file()
    with pytest.raises(SystemExit):
        export_main(["--features",str(features),"--formulas",str(formulas),"--out",str(output),
                     "--layers","smooth"])


def test_dynamic_sigma_not_allowed():
    with pytest.raises(FormulaError):
        compile_formulas("bad = gaussian(a, 2+3)", ["a"])
