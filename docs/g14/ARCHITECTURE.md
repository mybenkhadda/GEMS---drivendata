# G14 GeoFaultFormer: architecture

Status: **UNTESTED**. Code in `gems_bench/g14/model.py`; nothing has been executed.

## 1. Inputs: three aligned views per sample

| view | resolution | size | extent | role |
|---|---|---|---|---|
| local | 100 m | 128 px | 12.8 km | trace localisation; output grid |
| regional | 200 m | 256 px | 51.2 km | fault-system continuity, terrain / potential-field trends |
| global | 800 m | 256 px | 204.8 km | tectonic regime, basin / range context |

All three are centred on the same geographic point. Centres sit on an 8 px lattice, so the coarse windows align
exactly (`multiscale.lattice_centres`, `MultiScaleDataset.view_tops`). The local window is the central 1/4 of the
regional extent and the central 1/16 of the global extent. Each view carries a per-modality availability mask; values
outside coverage are 0 after normalisation and their masks are 0.

Channels (see DATA_CONTRACT.md) come from three sources: raw GeoDAWN bands, the G13 feature store when present, and
G14 composites. Permitted known-fault context (distance, density, tip distance, strike) is added per unit.

## 2. Network

```
local (C,128,128) --ModalityFusion--> LocalEncoder (UNet | ResNet | ConvNeXt), strides 1,2,4,8,16
regional (C,256,256) --ModalityFusion--> RegionalEncoder (Swin | MiT | none)
global (C,256,256) --ModalityFusion--> GlobalTokenEncoder (conv stem -> K tokens) | none

regional features --align_center(frac=1/4)--> ContextInjection (gated add, gate bias -2) at local stages {8,16}
global tokens --TokenCrossAttention--> local bottleneck;  FiLM(global pooled) on decoder stages (zero init)
U-Net decoder -> heads: fault logit (bias -3), orientation (cos2θ, sin2θ, unit norm),
                 connectivity (endpoint, junction), optional log-variance; deep-supervision aux logits
```

* **ModalityFusion** (`concat | gated | cross_attention`): one small conv encoder per modality, then mask-aware
  combination. In `gated` and `cross_attention` modes the softmax over modalities is masked, so an unavailable modality
  gets exactly zero weight and its (garbage) values cannot affect the output. Weights are returned for diagnostics
  (`viz.modality_weight_summary`).
* **Context injection** is the core idea. The regional feature map is resampled with an affine `grid_sample` so the
  central fraction matching the local footprint lines up pixel for pixel with the local feature map at that stage.
  It is then added through a learned sigmoid gate initialised near closed (bias −2), so training starts from the
  local-only model and opens the gate only if context helps.
* **Global tokens** inform the bottleneck through cross-attention and modulate decoder stages via zero-initialised
  FiLM. Both start as identity.
* **Uncertainty**: `mc_dropout` (dropout kept on at inference, N passes) or `heteroscedastic` (log-variance head;
  logits sampled in the loss).

All encoders are native PyTorch implementations (no timm download). `build_model` optionally loads a local pretrained
state dict with `strict=False` and reports missing keys. No weights are downloaded.

## 3. Why this design (from the audit)

* W2 (disc 0.242) beat every tabular model, using CNN maps plus known-fault geometry. Faults are line structures
  whose evidence is spread over kilometres (12.8 km correlation lengths), so a pixel classifier is the wrong shape.
* Coarse context is cheap at 200 / 800 m, and the gates let the ladder test whether it helps rather than assuming so.
* Mask-aware fusion handles the strongly uneven coverage (conductivity, seismicity, depth gaps).

## 4. Inference

`inference.predict_full`: lattice-snapped tiles with a halo, Hann blending, optional rot180 TTA (vector / axial
channels transformed correctly), optional MC dropout. Output is a probability map on the 100 m grid, NaN outside the
footprint. Held-out maps come from the unit model that never saw that zone (`heldout_maps`).

## 5. Ladder (EXPERIMENT_MATRIX.md)

T0 references → T1 local CNN → T2 + regional context → T3 + global tokens → T4 + mask-aware fusion →
T5 + topology losses → T6 graph recovery → T7 ensemble. Each step changes one thing.
