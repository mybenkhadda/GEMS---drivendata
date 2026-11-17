"""G13 synthetic smoke run (no competition data): builds the synthetic example and runs a mode end to end.

    python scripts/g13_smoke.py --mode FULL_RESEARCH --root /tmp/g13_smoke
"""
from __future__ import annotations

import argparse
import time
import warnings

from gems_bench.g13.engine import Engine
from gems_bench.g13.synthetic import make, small_config

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="FULL_RESEARCH")
    ap.add_argument("--root", default="/tmp/g13_smoke")
    a = ap.parse_args()
    t0 = time.time()
    make(f"{a.root}/data")
    eng = Engine(small_config(f"{a.root}/work", f"{a.root}/data", a.mode, out_dir=f"{a.root}/drive"))
    st = eng.run()
    print(st[["stage", "complete", "detail", "seconds"]].to_string())
    print(f"total {time.time() - t0:.0f}s; report: {a.root}/work/reports/report.md")
    assert st.complete.all() or set(st[~st.complete].stage) <= {"deep"}, st
