import numpy as np
import pandas as pd
import pytest

from gems_bench.g11.training.calibration import CalibrationLeakage, Calibrator, reliability


def test_calibration_only_on_stack():
    rng = np.random.default_rng(0)
    p = rng.random(2000)
    y = (rng.random(2000) < p ** 2).astype(float)
    with pytest.raises(CalibrationLeakage):
        Calibrator("isotonic").fit(p, y, "test")
    c = Calibrator("isotonic").fit(p, y, "stack")
    _, e0 = reliability(p, y)
    _, e1 = reliability(c.transform(p), y)
    assert e1 < e0
    assert np.isnan(Calibrator("platt").fit(p, y, "stack").transform(np.array([np.nan]))[0])


def test_xgb_baseline_trains_and_predicts():
    pytest.importorskip("xgboost")
    from gems_bench.g11.models.xgb_baseline import train_pixel_model

    rng = np.random.default_rng(0)

    class D:
        def __init__(self, n):
            self.X = rng.normal(size=(n, 3)).astype(np.float32)
            self.y = (self.X[:, 0] + 0.3 * rng.normal(size=n) > 1).astype(np.float32)
            self.weight = np.where(self.y > 0, 1.0, 0.3).astype(np.float32)
            self.feature_names = ["a", "b", "c"]

    m = train_pixel_model(D(3000), D(1000), {"n_estimators": 60, "max_depth": 3, "learning_rate": 0.2})
    from gems_bench.g11.evaluation.pixel_metrics import pr_auc

    te = D(1000)
    assert pr_auc(te.y, m.predict(te.X)) > 0.6
    assert m.importance().feature.iloc[0] == "a"


def test_segment_ranker_binary_and_rank():
    pytest.importorskip("lightgbm")
    from gems_bench.g11.models.segment_ranker import train_ranker

    rng = np.random.default_rng(0)

    def frame(n):
        f = rng.normal(size=(n, 2))
        return pd.DataFrame({"f0": f[:, 0], "f1": f[:, 1], "label": (f[:, 0] > 0.5).astype(int),
                             "group": rng.integers(0, 10, n)})

    for obj in ("binary", "lambdarank"):
        r = train_ranker(frame(600), frame(200), ["f0", "f1"], obj, {"n_estimators": 50})
        t = frame(300)
        s = r.score(t)
        assert np.corrcoef(s, t.f0)[0, 1] > 0.5


torch = pytest.importorskip("torch")


def test_segformer_shapes_and_weights(tmp_path):
    from gems_bench.g11.models.segformer import SegFormer, load_compatible_weights

    m = SegFormer(5, dims=(8, 16, 24, 32), heads=(1, 2, 2, 4), decoder_dim=16)
    y = m(torch.randn(2, 5, 64, 64))
    assert y.shape == (2, 1, 64, 64) and torch.isfinite(y).all()
    torch.save(m.state_dict(), tmp_path / "w.pt")
    m2 = SegFormer(5, dims=(8, 16, 24, 32), heads=(1, 2, 2, 4), decoder_dim=16)
    rep = load_compatible_weights(m2, str(tmp_path / "w.pt"))
    assert rep["coverage"] == 1.0
    m3 = SegFormer(7, dims=(8, 16, 24, 32), heads=(1, 2, 2, 4), decoder_dim=16)
    assert load_compatible_weights(m3, m.state_dict())["skipped_shape"]       # channel mismatch reported


def test_gated_fusion_missing_dem_equals_geo_only():
    from gems_bench.g11.models.gated_fusion import MODES, GatedFusionNet

    torch.manual_seed(0)
    x, dem, dv = torch.randn(2, 4, 32, 32), torch.randn(2, 1, 64, 64), torch.ones(2, 1, 64, 64)
    no = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    for mode in MODES:
        net = GatedFusionNet(4, mode=mode, geo_dim=16, dem_dim=8).eval()
        out = net(x, dem, dv, torch.tensor([[1.0, 1.0], [1.0, 1.0]]))
        assert out["logits"].shape == (2, 1, 32, 32) and torch.isfinite(out["logits"]).all()
        if mode in ("gated", "cross_attention"):
            ref = net(x)["logits"]
            assert torch.allclose(net(x, dem, dv, no)["logits"], ref, atol=1e-6)


def test_batch_size_consistency_eval():
    from gems_bench.g11.models.gated_fusion import GatedFusionNet

    net = GatedFusionNet(3, mode="concat", geo_dim=16, dem_dim=8).eval()
    x, dem, dv = torch.randn(3, 3, 16, 16), torch.randn(3, 1, 32, 32), torch.ones(3, 1, 32, 32)
    mod = torch.ones(3, 2)
    full = net(x, dem, dv, mod)["logits"]
    one = net(x[:1], dem[:1], dv[:1], mod[:1])["logits"]
    assert torch.allclose(full[:1], one, atol=1e-5)


def test_trainer_runs_and_checkpoint_parity(tmp_path, cfg):
    from gems_bench.g11.models.gated_fusion import GatedFusionNet
    from gems_bench.g11.training.checkpointing import reload_parity
    from gems_bench.g11.training.losses import build_loss
    from gems_bench.g11.training.trainer import Trainer, pixel_eval_fn, seed_everything

    seed_everything(0)
    g = torch.Generator().manual_seed(0)
    xs = torch.randn(8, 2, 16, 16, generator=g)
    ys = (xs[:, :1] > 0.8).float()
    batches = [{"x": xs[i:i + 4], "y": ys[i:i + 4], "valid": torch.ones_like(ys[i:i + 4])} for i in (0, 4)]
    cfg["training"].update(epochs=3, patience=5, amp=False, grad_accum=1, lr=1e-2)
    model = GatedFusionNet(2, mode="geo_only", geo_dim=8, dem_dim=4, aux=False)
    tr = Trainer(model, build_loss("weighted_bce"), cfg, tmp_path, device="cpu", evaluate_fn=pixel_eval_fn)
    hist = tr.fit(batches, batches, meta={"split": "train/validation"})
    assert len(hist) == 3 and np.isfinite(hist.train_loss).all()
    assert hist.train_loss.iloc[-1] < hist.train_loss.iloc[0]
    assert (tmp_path / "best.pt").is_file() and (tmp_path / "history.csv").is_file()
    tr2 = Trainer(GatedFusionNet(2, mode="geo_only", geo_dim=8, dem_dim=4, aux=False), build_loss("weighted_bce"), cfg,
                  tmp_path, device="cpu", evaluate_fn=pixel_eval_fn)
    assert len(tr2.fit(batches, batches)) == 3                          # resumed: no extra epochs
    diff = reload_parity(lambda: GatedFusionNet(2, mode="geo_only", geo_dim=8, dem_dim=4, aux=False),
                         tmp_path / "best.pt", {"x": xs[:2]})
    assert diff == 0.0


def test_multiresolution_dataset_reads_dem_and_flags_absence(tmp_path, grid, synth, cfg):
    import pandas as pd

    from gems_bench.g11.datasets.patch_dataset import MultiresolutionPatchDataset
    from gems_bench.g11.dem_catalog import build_dem_catalog
    from gems_bench.g11.dem_reader import DEMReader
    from gems_bench.g11.features import FeatureStore
    from gems_bench.g11.spatial_splits import CODES

    st = FeatureStore(tmp_path, grid)
    st.put("a", np.where(grid.valid, 1.0, np.nan), "geo")
    code = np.where(grid.valid, CODES["train"], 0).astype(np.uint8)
    rd = DEMReader(build_dem_catalog(synth["dem_dir"], grid.crs, cfg), cfg, allow_unknown_vertical=True)
    # window over the DEM tiles (cols 40-60, rows 50-70) and one far away from them
    win = pd.DataFrame([{"row0": 52, "col0": 42, "size": 10, "split": "train"},
                        {"row0": 0, "col0": 80, "size": 10, "split": "train"}])
    ds = MultiresolutionPatchDataset(st, ["a"], np.zeros(grid.shape, np.float32), win, code, "train",
                                     dem_reader=rd, grid=grid, dem_res_m=10.0, dem_px=64)
    a, b = ds[0], ds[1]
    assert a["modality"][1] == 1.0 and a["dem"].shape == (1, 64, 64) and a["dem_valid"].mean() > 0.9
    assert a["provenance"]["dem_tiles"]
    assert b["modality"][1] == 0.0 and float(np.abs(b["dem"]).max()) == 0.0      # no DEM -> explicit absence


def test_patch_dataset_and_collate(tmp_path, grid, synth):
    from gems_bench.g11 import fault_labels as FL
    from gems_bench.g11 import spatial_splits as SS
    from gems_bench.g11.datasets.patch_dataset import GeoPatchDataset, collate, patch_windows
    from gems_bench.g11.features import FeatureStore

    st = FeatureStore(tmp_path, grid)
    rng = np.random.default_rng(0)
    for n in ("a", "b"):
        st.put(n, np.where(grid.valid, rng.normal(size=grid.shape), np.nan), "geo")
    mask = FL.label_raster(synth["raw"] / "existing_faults.tif", grid)
    res = SS.make_splits(grid.valid, mask, grid, synth["cfg"])
    rr, cc = np.nonzero(res.mask("train"))
    win = patch_windows(rr[::50], cc[::50], 8, grid.shape, res.code, "train")
    assert len(win) and not SS.check_patch_leakage(win, res)
    ds = GeoPatchDataset(st, ["a", "b"], mask.astype(np.float32), win, res.code, "train")
    b = collate([ds[0], ds[1]])
    assert b["x"].shape == (2, 2, 8, 8) and b["valid"].shape == (2, 1, 8, 8)
    assert float(b["valid"].max()) == 1.0
