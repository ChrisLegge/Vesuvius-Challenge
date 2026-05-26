# Project Summary

## Problem

The Vesuvius Challenge Surface Detection task asks for binary segmentation of
papyrus sheet surfaces from 3D X-ray CT volumes of carbonised Herculaneum
scrolls. Volumes are acquired at ~8 um/voxel resolution. Scroll layers are
tightly packed, thin (2-5 voxels), and physically close together, creating
narrow gaps between adjacent surfaces.

The scoring metric combines three components:

```
S = 0.30 * T  +  0.35 * D_tau  +  0.35 * V
```

- `D_tau`: SurfaceDice at tolerance tau=2 (boundary proximity in physical units)
- `V = 1 / (1 + 0.3 * (VOI_split + VOI_merge))`: information-theoretic
  volume overlap, sensitive to over- and under-segmentation
- `T`: Betti-matching topological score

VOI and T are event-driven: a single bridge between two surfaces changes
connected-component structure and causes a large discontinuous jump in both V
and T. This makes mean score an unreliable optimisation target. The tail of
the score distribution (worst-case volumes) is the operative concern.

## Approach

### Constrained optimisation framing

Training is framed as a constrained problem rather than unconstrained loss
minimisation:

```
max  E[S]  -  lambda * CVaR_0.10(1 - S)
subject to:  E[bridge_proxy(P)] <= eps_b
             E[fragmentation_proxy(P)] <= eps_s
```

Dual variables for the topology constraints are updated by projected ascent,
so loss weighting adapts dynamically if a model begins producing bridges or
excessive fragmentation.

### Phase-gated loss schedule

Loss components are introduced progressively to avoid gradient conflict between
surface-proximity and topology objectives:

| Phase | Active components |
|---|---|
| Early | BCE, Dice, SDF regression |
| Mid | + Medial surface recall, Gap-negative, Centreline Dice |
| Late | + Tversky, Topology guide, increased SDF weight |

Gap-negative loss penalises predictions in GT-labelled separation zones,
targeting VOI_merge directly.

### Specialist ensemble

Three models trained with different seeds, folds, and loss emphasis:

| Model | Role | Training emphasis |
|---|---|---|
| A — Generalist | Stable baseline | Conservative loss mix, best calibration |
| B — Anti-merge | Bridge suppression | Strong gap-negative, neck-risk sampling |
| C — Surface specialist | Boundary precision | SDF head, boundary classification head |

Inference fuses logits by weighted mean, applies temperature scaling (T=0.85),
hysteresis thresholding (t_low=0.35, t_high=0.65), and topology-safe
post-processing: bridge-neck cutting via skeleton thickness, dust removal
by connected-component size, and small cavity filling.

### Adaptive inference

A six-level degradation ladder reduces patch overlap, window size, and active
model count as the per-volume time budget is consumed (hard cap: 240 s/volume).

## Results

| Model | Best proxy score | SurfaceDice | VOI score | TopoScore | Epochs | Time |
|---|---|---|---|---|---|---|
| A — Generalist | 0.5816 | 0.4842 | 0.9205 | 0.3000 | 34 | 8.77 h |
| B — Anti-merge | 0.5110 | — | — | — | 35 | 8.83 h |
| C — Surface | 0.4902 | — | — | — | 39 | 8.67 h |

Best validation epoch for Model A: epoch 23. SWA applied from epoch 29.

## Known Limitations

- Proxy metrics are computed on held-out validation volumes but may not match
  Kaggle evaluation conventions exactly (ignore-mask handling, spacing units).
- Bridge cutting uses a 2D per-slice skeleton approximation rather than full
  3D thinning, which may miss bridges only visible in the Z axis.
- Threshold calibration was fitted on the same validation fold used for model
  selection, introducing potential overfitting of hysteresis parameters.
- No ablation isolating the contribution of each specialist to ensemble gain.
