# Installation

## Colab (the only environment that sees the real data)
Every notebook's first code cell does the setup:
1. Mounts Drive.
2. Reuses an existing checkout among `/content/GEMS`, `/content/GEMS---drivendata` and `/content/GEMS_G11`, or clones `mybenkhadda/GEMS---drivendata` into `/content/GEMS`.
3. Fetches `REF` (default `main`) and checks it out detached.
4. Refuses to continue if `gems_bench/g11/pipeline.py` is missing.
5. Runs `pip install -e .[colab] xgboost lightgbm`.
6. Prints the commit.

Colab already provides torch. Use a GPU runtime for G11.2 (E3 / E4).

Push the G11 package to `main` before the first Colab run, or set `REF` to a commit that contains it.

## Local (synthetic tests only)
```bash
python -m venv .venv && .venv/bin/pip install numpy scipy pandas pyarrow pyyaml rasterio shapely geopandas pyogrio \
    scikit-image scikit-learn matplotlib xgboost lightgbm pytest
pip install torch --index-url https://download.pytorch.org/whl/cpu      # optional: model / loss / training tests
PYTHONPATH=. python -m pytest gems_bench/g11/tests -q
```
Tests that need torch, xgboost or lightgbm are skipped (`importorskip`) when the package is missing.
