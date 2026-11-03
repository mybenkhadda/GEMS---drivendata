# G07 experiment matrix

| ID | experiment | question | stage / RUN flag | decision use |
|---|---|---|---|---|
| G07-A0 | frozen W2 exact-k reference (`W2` rows) | can we reproduce the baseline with exact budgets? | EVALUATE | reference for every gain |
| G07-A1 | DEM geomorphometry classifier (LightGBM; A1RF = Random Forest) | does the DEM carry independent fault evidence? | TRAIN_SURFACE_CLASSIC | only after DEM GO |
| G07-A2 | A1 + scarps / lineaments / persistence / terrain-geophysics agreement | do scarps and terrain continuity help? | TRAIN_SURFACE_CLASSIC | A2 vs A1 ablation |
| G07-B0 | G06-P4 features, G06 labels | reference geology-only model | TRAIN_SUBSURFACE | baseline for B1 / B2 |
| G07-B1 | G06-P4 + improved supervision | was label design limiting geology-only ML? | TRAIN_SUBSURFACE | nested fits feed D1 |
| G07-B2 | G06-P4 + proximity regression | does distance-aware learning improve localisation? | TRAIN_SUBSURFACE | |
| G07-C1 | compact terrain U-Net | does spatial DEM context help? | TRAIN_SURFACE_UNET | only if A1 / A2 add systems |
| G07-C2 | surface + subsurface + W2 + geophysics candidate union | more systems at matched budget? | GENERATE_CANDIDATES | `evaluation/recall/candidate_recall.csv` |
| G07-D1 / D1W | candidate LightGBM (geology-only / W2-aware) | true structures vs false edges? | TRAIN_CANDIDATE_RANKER | D1 strict-capable, D1W screening |
| G07-D2 | W2-preserving portfolio (85 % W2 exact-k base + whole candidates by utility + W2 fill) | improve DTI without losing W2 support? | TRAIN_CANDIDATE_RANKER + EVALUATE | the only export path besides W2 |
| G07-E1 | continuous confidence corridor | soft localisation vs binary top-area? | TRAIN_CANDIDATE_RANKER + EVALUATE | diagnostic (not exported) |

Run A0-A2 and B0-B2 before any expensive model. No unlimited hyper-parameter search.
