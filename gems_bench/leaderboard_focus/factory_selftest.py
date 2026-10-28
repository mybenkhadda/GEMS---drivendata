"""Runtime self-tests of the submission factory (synthetic arrays + temporary files; no Drive, no GPU).

    T = run_selftests(tmp_dir)       # DataFrame: test, critical, status PASS / FAIL, detail

They exercise the SAME functions the factory uses on real data (gems3.submission.finalise / eligible_mask /
validate, submission.write_submission, p6.final_submission.zone_partition / stitch / submission_path,
parity.zone_overlaps, forensics.budget_table, validation.array_sha, factory.decide / rank / readback_checks,
Factory.candidate isolation). A failed critical test disables every export of the run.
"""
from __future__ import annotations

import json
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


def fake_data(h: int = 256, w: int = 320) -> SimpleNamespace:
    """A small grid with the official contract: EPSG 32611, 100 m, template NaN columns, a data-footprint gap, one
    mapped fault (>= one 256 px tile each way: the official writer uses 256 x 256 tiles)."""
    import rasterio
    from rasterio.transform import from_origin

    tv = np.ones((h, w), bool)
    tv[:, :3] = False                                   # outside the competition bounds (template NaN)
    valid = tv.copy()
    valid[:2, :] = False                                # inside the bounds, outside the data footprint
    labels = np.zeros((h, w), bool)
    labels[20, 10:20] = True
    prof = {"driver": "GTiff", "height": h, "width": w, "count": 1, "dtype": "float32",
            "crs": rasterio.crs.CRS.from_epsg(32611), "transform": from_origin(500000.0, 4000000.0, 100.0, 100.0),
            "nodata": np.nan}
    return SimpleNamespace(valid=valid, template_valid=tv, labels=labels, template_profile=prof, shape=(h, w),
                           template_matches=lambda: True)


def _binary(data, frac: float = 0.05, seed: int = 0) -> np.ndarray:
    from ..gems3 import submission as SUB

    el = SUB.eligible_mask(data, 0, "eligible")
    rng = np.random.default_rng(seed)
    return (el & (rng.random(data.shape) < frac)).astype(np.float32)


EXPECTED = {"epsg": 32611, "res_m": 100.0, "dtype": "float32", "count": 1}


# =============================================================================== tests (each returns a detail str)
def t_raster_roundtrip(d: Path) -> str:
    from ..artifacts.atomic import sha256_file
    from ..gems3 import submission as SUB
    from ..submission import write_submission
    from . import submission as SU
    from .factory import readback_checks
    from .validation import array_sha

    data = fake_data()
    arr = SUB.finalise(_binary(data), data, "template")
    p = write_submission(arr, data, d / "rt.tif")
    SUB.validate(p, data, "template")                   # raises AssertionError on any official-format failure
    facts = SU.raster_facts(p)
    rb = readback_checks(facts, data.template_profile, EXPECTED, p.stat().st_size / 2 ** 20, 50.0, array_sha(arr),
                         array_sha(facts["array"]), "fp", "fp", sha256_file(p))
    bad = [c["check"] for c in rb if not c["ok"]]
    assert not bad, f"read-back checks failed: {bad}"
    return f"{len(rb)} read-back checks PASS, {int((arr > 0).sum())} positive px"


def t_shape_transform(d: Path) -> str:
    import rasterio
    from rasterio.transform import from_origin

    from ..gems3 import submission as SUB
    from . import submission as SU
    from .factory import readback_checks
    from .validation import array_sha

    data = fake_data()
    arr = SUB.finalise(_binary(data), data, "template")
    prof = dict(data.template_profile, transform=from_origin(500100.0, 4000000.0, 100.0, 100.0))
    p = d / "shifted.tif"
    with rasterio.open(p, "w", **prof) as dst:
        dst.write(arr, 1)
    try:
        SUB.validate(p, data, "template")
        raise AssertionError("a shifted raster passed gems3.submission.validate")
    except AssertionError as e:
        if "passed" in str(e):
            raise
    facts = SU.raster_facts(p)
    rb = {c["check"]: c["ok"] for c in readback_checks(facts, data.template_profile, EXPECTED, 0.1, 50.0,
                                                       array_sha(arr), array_sha(facts["array"]), "a", "a", "x")}
    assert rb["transform (bounds) == template"] is False, "shifted transform not detected by the read-back checks"
    return "shifted raster rejected by gems3 validate and by the read-back transform check"


def t_eligible_mask(d: Path) -> str:
    from ..gems3 import submission as SUB

    data = fake_data()
    el = SUB.eligible_mask(data, 0, "eligible")
    assert not (el & data.labels).any(), "mapped fault inside the eligible domain"
    assert not (el & ~data.template_valid).any(), "template-NaN pixel inside the eligible domain"
    assert not (el & ~data.valid).any(), "pixel outside the data footprint inside the eligible domain"
    assert el.sum() == (data.valid & data.template_valid & ~data.labels).sum()
    return f"{int(el.sum())} eligible px (faults, template NaN, footprint gap excluded)"


def t_valid_range(d: Path) -> str:
    from ..gems3 import submission as SUB

    data = fake_data()
    q = np.zeros(data.shape, np.float32)
    q[10, 10], q[11, 11], q[12, 12] = 2.0, -1.0, np.nan
    a = SUB.finalise(q, data, "template")
    fin = np.isfinite(a)
    assert not fin[~data.template_valid].any(), "finite value where the template is NaN"
    assert fin[data.template_valid].all(), "NaN where the template is finite"
    assert float(np.nanmin(a)) >= 0 and float(np.nanmax(a)) <= 1, "values outside [0, 1]"
    assert (a[data.template_valid & ~data.valid] == 0).all(), "footprint gap not written 0"
    return "clipped to [0, 1]; NaN exactly where the template is NaN; footprint gap = 0"


def t_zone_stitching(d: Path) -> str:
    from ..p6.final_submission import stitch, zone_partition

    data = fake_data()
    fm = np.full(data.shape, -1, np.int16)
    w = data.shape[1]
    fm[:, :100], fm[:, 100:200], fm[:, 200:w - 2] = 0, 1, 2     # last 2 columns: valid but in no zone
    z, cov = zone_partition(fm, data.valid, [0, 1, 2])
    exp_unc = int((data.valid & (fm < 0)).sum())
    assert cov["uncovered_px"] == exp_unc > 0, f"uncovered {cov['uncovered_px']} != {exp_unc}"
    parts = []
    for k in (0, 1, 2):
        sl = (slice(0, data.shape[0]), slice(0, data.shape[1]))
        parts.append((sl, z == k, np.full(data.shape, k + 1, np.float32)))
    S = stitch(data.shape, parts)
    for k in (0, 1, 2):
        assert (S[z == k] == k + 1).all(), f"zone {k} not stitched from its own map"
    assert (S[z < 0] == 0).all(), "pixels outside every zone not 0"
    return f"3 zones stitched from their own maps; {exp_unc} uncovered px detected"


def t_duplicate_zones(d: Path) -> str:
    from . import parity as PA

    data = fake_data()
    a, b = np.zeros(data.shape, bool), np.zeros(data.shape, bool)
    a[:, :150], b[:, 150:] = True, True
    ok = PA.zone_overlaps({0: a, 1: b}, data.valid)
    b2 = b.copy()
    b2[:, 140:150] = True
    bad = PA.zone_overlaps({0: a, 1: b2}, data.valid)
    assert ok["duplicated_px"] == 0 and bad["duplicated_px"] > 0, f"{ok} / {bad}"
    return f"disjoint: 0 duplicated px; overlap detected: {bad['duplicated_px']} px"


def t_budget(d: Path) -> str:
    from .forensics import budget_table

    Z = pd.DataFrame({"zone": [0, 1, 2, 3], "eligible_px": [1000] * 4, "positive_px": [5, 15, 4, 0],
                      "frac_of_eligible": [0.005, 0.015, 0.004, 0.0], "positive_on_known_px": [0, 0, 2, 0]})
    B = budget_table(Z, 0.01, 0.02).set_index("zone")
    got = B.ok.to_dict()
    assert got == {0: True, 1: False, 2: False, 3: False}, f"budget verdicts {got} (thin ok / over / leak / empty)"
    return "thin area ok; over-allocation, known-fault leak and empty map rejected"


def t_checksum(d: Path) -> str:
    from ..artifacts.atomic import sha256_file
    from .validation import array_sha

    p = d / "c.bin"
    p.write_bytes(b"gems" * 1000)
    assert sha256_file(p) == sha256_file(p)
    a = np.zeros((5, 5), np.float32)
    a[0, 0] = np.nan
    b = a.copy()
    b.view(np.uint32)[0, 0] = 0x7FC00001                # another NaN payload: canonicalised
    assert array_sha(a) == array_sha(b), "NaN payload changes the array sha"
    c = a.copy()
    c[1, 1] = 1.0
    assert array_sha(a) != array_sha(c), "a changed pixel keeps the same sha"
    return "file sha256 stable; array sha NaN-canonical and pixel-sensitive"


def t_csv_loader(d: Path) -> str:
    from ..p6.full_benchmark import read_csv_checked
    from .session import Workspace

    ws = Workspace(d / "ws")
    assert not len(ws.load_table("absent.csv"))
    (d / "ws").mkdir(parents=True, exist_ok=True)
    (d / "ws" / "empty.csv").write_bytes(b"")
    (d / "ws" / "header.csv").write_text("task,zone\n", encoding="utf-8")
    assert not len(ws.load_table("empty.csv")) and not len(ws.load_table("header.csv"))
    st = {n: read_csv_checked(d / "ws" / f"{n}.csv", ["task"])[1]["status"] for n in ("empty", "header", "absent")}
    assert len(set(st.values())) == 3 and read_csv_checked(d / "ws" / "header.csv", ["task"])[0] is None, st
    return f"empty / header-only / absent -> {st}"


def t_candidate_isolation(d: Path) -> str:
    from .factory import FAILED, UNAVAILABLE, CandidateUnavailable, Factory

    f = Factory({"local_root": str(d / "iso"), "drive_root": str(d / "drive"), "sync_to_drive": False}, "selftest")
    f.block = {"w2_champion": [], "w2_crest": [], "w2_skeleton": [], "b0_b1_fallback": ["artifact x: MISSING"]}

    def recover(cid):
        if cid == "w2_champion":
            raise RuntimeError("synthetic programming error")
        if cid == "w2_crest":
            raise CandidateUnavailable("synthetic: not reconstructable from frozen artifacts")
        raise KeyError("synthetic")

    f.recover = recover
    got = {c: f.candidate(c)["status"] for c in f.c["candidates"]}
    want = {"w2_champion": FAILED, "w2_crest": UNAVAILABLE, "w2_skeleton": FAILED, "b0_b1_fallback": "BLOCKED"}
    assert got == want, f"{got} != {want}"
    on_disk = sorted(p.stem for p in (d / "iso" / "outputs/leaderboard_focus/submission_factory/manifests/candidates")
                     .glob("*.json"))
    assert on_disk == sorted(want), f"persisted manifests {on_disk}"
    assert len(f.errors) == 4 and all(e["explanation"] for e in f.errors)
    return f"every candidate recorded independently: {got}"


def t_version_identity(d: Path) -> str:
    from ..p6.final_submission import submission_path
    from ..pipeline.m6 import fingerprint

    r1 = {"policy": {"rep": "skeleton_1px", "frac": 0.01}, "kind": "stack"}
    r2 = {"kind": "stack", "policy": {"frac": 0.01, "rep": "skeleton_1px"}}
    assert fingerprint("recipe", r1) == fingerprint("recipe", r2), "key order changes the recipe id"
    assert fingerprint("recipe", r1) != fingerprint("recipe", {**r1, "policy": {"rep": "skeleton_1px", "frac": 0.02}})
    sub = d / "subs"
    sub.mkdir()
    p, reuse = submission_path(sub, "submission_x_abc", "fpA")
    assert p.name == "submission_x_abc.tif" and not reuse
    p.write_bytes(b"x")
    p.with_suffix(".json").write_text(json.dumps({"recipe_fp": "fpA"}), encoding="utf-8")
    assert submission_path(sub, "submission_x_abc", "fpA") == (p, True), "same recipe not reused"
    q, reuse = submission_path(sub, "submission_x_abc", "fpBBBBBBBB")
    assert q.name == "submission_x_abc__fpBBBBBB.tif" and not reuse, "a different recipe would overwrite"
    q.write_bytes(b"y")
    q.with_suffix(".json").write_text(json.dumps({"recipe_fp": "other"}), encoding="utf-8")
    try:
        submission_path(sub, "submission_x_abc", "fpBBBBBBBB")
        raise AssertionError("no FileExistsError when both names hold other recipes")
    except FileExistsError:
        pass
    return "recipe id order-insensitive and value-sensitive; an existing file of another recipe is never overwritten"


def t_status_logic(d: Path) -> str:
    from .factory import DUPLICATE, FAIL, HELD, INVALID, PASS, READY, VALIDATED, WARNING, decide

    cases = [((PASS, PASS, PASS, False, "report_only", True, None), READY),
             ((PASS, WARNING, PASS, True, "report_only", True, None), READY),
             ((PASS, PASS, PASS, False, "strict", True, None), HELD),
             ((PASS, FAIL, PASS, True, "report_only", True, None), INVALID),
             ((FAIL, PASS, PASS, True, "report_only", True, None), INVALID),
             ((PASS, PASS, FAIL, True, "report_only", True, None), INVALID),
             ((PASS, PASS, "NOT_RUN", True, "report_only", False, None), VALIDATED),
             ((PASS, PASS, "NOT_RUN", True, "report_only", False, "w2_champion"), DUPLICATE)]
    bad = [(a, decide(*a), w) for a, w in cases if decide(*a) != w]
    assert not bad, f"{bad}"
    return "geo target never blocks in report_only; technical / parity / format failures always block"


def t_ranking(d: Path) -> str:
    from .factory import READY, rank

    rows = [{"candidate": "w2_champion", "status": READY, "p6g_eligible": True, "disc_dti": 0.242, "disc_worst": 0.19,
             "geo_dti": 0.105},
            {"candidate": "w2_skeleton", "status": "DUPLICATE", "p6g_eligible": True, "disc_dti": 0.242,
             "disc_worst": 0.19, "geo_dti": 0.105},
            {"candidate": "w2_crest", "status": READY, "p6g_eligible": True, "disc_dti": 0.238, "disc_worst": 0.19,
             "geo_dti": 0.114},
            {"candidate": "b0_b1_fallback", "status": READY, "p6g_eligible": False, "disc_dti": 0.162,
             "disc_worst": 0.12, "geo_dti": 0.158}]
    R = {r["candidate"]: r["upload_priority"] for r in rank(rows)}
    want = {"w2_champion": "PRIMARY", "w2_skeleton": "DO_NOT_UPLOAD", "w2_crest": "OPTIONAL",
            "b0_b1_fallback": "ALTERNATIVE"}
    assert R == want, f"{R} != {want}"
    return f"{R}"


def t_config(d: Path) -> str:
    from .factory import FactoryConfigError, factory_config

    for bad in ({"train_models": True}, {"device": "cuda"}, {"candidates": ["nope"]}, {"run_mode": "TRAIN"},
                {"manual_upload_only": False}):
        try:
            factory_config(bad)
            raise AssertionError(f"accepted {bad}")
        except FactoryConfigError:
            pass
    assert factory_config({"run_mode": "resume", "reuse_cached": False})["reuse_cached"] is True
    return "training / GPU / unknown candidates / auto-upload refused; RESUME forces cache reuse"


TESTS = (("raster_roundtrip", True, t_raster_roundtrip), ("shape_transform", True, t_shape_transform),
         ("eligible_mask", True, t_eligible_mask), ("valid_output_range", True, t_valid_range),
         ("zone_stitching", True, t_zone_stitching), ("duplicate_zones", True, t_duplicate_zones),
         ("budget_representation", True, t_budget), ("checksum", True, t_checksum),
         ("csv_loader", True, t_csv_loader), ("candidate_isolation", True, t_candidate_isolation),
         ("version_recipe_identity", True, t_version_identity), ("status_logic", True, t_status_logic),
         ("ranking", True, t_ranking), ("config_guards", True, t_config))


def run_selftests(tmp: Path) -> pd.DataFrame:
    rows = []
    for name, critical, fn in TESTS:
        d = Path(tmp) / name
        d.mkdir(parents=True, exist_ok=True)
        try:
            rows.append({"test": name, "critical": critical, "status": "PASS", "detail": fn(d)})
        except Exception as e:
            rows.append({"test": name, "critical": critical, "status": "FAIL",
                         "detail": f"{type(e).__name__}: {e} | {traceback.format_exc()[-600:]}"})
    return pd.DataFrame(rows)
