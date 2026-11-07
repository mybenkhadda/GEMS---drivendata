"""Section 16 - optional, guarded export of a NEW continuous GeoTIFF (disabled by default; no upload).

Export happens only if EVERY check passes: user opt-in, verified provenance, all required full-area rasters
valid, a cross-fitted decision that passed all gates with a non-identity label-free policy, the V1 full-model
raster present with the hash recorded at preflight. The transform is applied to that ORIGINAL full-model
prediction; the file is written with V1's fixed writer ``_write_submission_masked`` (finite float32 in [0, 1]
everywhere + internal validity mask = official footprint), reopened and revalidated against the template,
and its SHA-256 is saved. Otherwise the result is ``NO RECOVERY PROMOTION - DO NOT SUBMIT``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .cache import atomic_write_json, atomic_write_text, read_raster, sha256_file
from .crossfit import NO_PROMOTION
from .transforms import apply


def export(run, decision) -> dict:
    P, cfg = run.paths, run.cfg
    checks = []

    def chk(name, ok, detail=""):
        checks.append({"check": name, "passed": bool(ok), "detail": str(detail)})
        return bool(ok)

    st = decision.stats if decision is not None else {}
    chk("user opt-in (enable_submission_export)", cfg["enable_submission_export"])
    chk("checkpoint and split provenance verified", getattr(run, "split_verified", False))
    req = cfg["units"]["discovery"] + cfg["units"]["geographic"]
    inf_ok = all(run.inference.cached(u)[0] for u in req) if run.inference is not None else False
    chk("all required full-area rasters valid", inf_ok)
    chk("cross-fitted decision passed every gate", bool(st.get("promote")), st.get("decision", "no decision"))
    pol = next((p for p in cfg["transforms"]["policies"] if p["name"] == st.get("final_policy")), None)
    chk("final policy is a predeclared non-identity policy", pol is not None and pol["kind"] != "identity",
        st.get("final_policy"))
    full = cfg["units"]["full"]
    src = P.v1_prediction(full)
    pre_sha = run.ckpt_pred_sha.get(full)
    src_ok = src.is_file() and pre_sha is not None and run.hashes.sha(src) == pre_sha
    chk("V1 full-model prediction present and unchanged since preflight", src_ok, str(src))
    out = P.d("10_submission") / "submission_georegime_recovery_v2.tif"
    result = {"path": None, "status": NO_PROMOTION, "decision": st.get("decision")}
    if all(c["passed"] for c in checks):
        from gems_bench.georegime.run import _write_submission_masked

        p = read_raster(src)
        S = run.support_raster() if pol["kind"] == "support" else None
        q = apply(pol, p, S)
        P.check_writable(out)
        _write_submission_masked(out, q, run.bank.profile, run.bank.template_valid)
        import rasterio

        tpl = P.atlas / "00_raw_snapshot/official_template/example_submission.tif"
        with rasterio.open(out) as ds, rasterio.open(tpl) as t:
            v = ds.read(1)
            mask = ds.dataset_mask() > 0
            chk("readback: one float32 band", ds.count == 1 and ds.dtypes[0] == "float32")
            chk("readback: template CRS / transform / shape",
                ds.crs == t.crs and ds.transform == t.transform and ds.shape == t.shape)
            chk("readback: stored values finite in [0, 1]", np.isfinite(v).all() and ((v >= 0) & (v <= 1)).all())
            chk("readback: internal mask == official footprint", np.array_equal(mask, run.bank.template_valid))
            chk("continuous scores retained (> 2 distinct values)", np.unique(v[mask][:: 97]).size > 2)
            chk("not a renamed V1 raster", not np.allclose(np.nan_to_num(p)[mask], v[mask]))
        sha = sha256_file(out)
        if all(c["passed"] for c in checks):
            atomic_write_text(P.d("10_submission") / "submission_sha256.txt", f"{sha}  {out.name}\n")
            result.update(path=str(out), status="READY_FOR_MANUAL_REVIEW (no automatic upload)", sha256=sha,
                          policy=pol, source_prediction=str(src), source_sha256=pre_sha)
        else:
            bad = out.with_name(out.stem + ".FAILED_VALIDATION.tif")
            out.replace(bad)
            result.update(path=None, status=f"{NO_PROMOTION} (written file failed validation: {bad.name})")
    table = pd.DataFrame(checks)
    atomic_write_json(P.d("10_submission") / "submission_validation.json", {**result, "checks": checks})
    return {"validation_table": table, "stats": result}
