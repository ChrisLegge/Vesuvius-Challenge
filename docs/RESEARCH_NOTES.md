# Research Notes

## Problem Framing

The challenge is a 3D surface segmentation problem. The target is a thin, topologically sensitive surface embedded in CT volumes. Small local mistakes can produce large metric changes:

- a false bridge can merge surfaces
- a gap can split a surface into disconnected pieces
- a fuzzy boundary can reduce surface-distance agreement

The project therefore treats topology as a first-class modeling concern rather than a post-hoc visual cleanup issue.

## Metric-Aware Design

The internal research direction tracks three families of behavior:

| Area | Practical Target |
|---|---|
| Surface alignment | Better boundary localization and Surface Dice |
| Connectivity | Fewer split components and fewer missing thin regions |
| Anti-merge behavior | Fewer accidental bridges across close surfaces |

The best model is not necessarily the one with the lowest training loss. A useful model must also be stable under thresholding and robust across hard volumes.

## Specialist Ensemble Strategy

The ensemble is organized around failure modes:

- **Generalist**: conservative baseline model with stable behavior
- **Anti-merge**: emphasizes prevention of bridge artifacts
- **Surface specialist**: emphasizes boundary quality and surface proximity

This is more defensible than training several near-identical models because each member has a distinct reason to exist.

## Inference Strategy

The inference notebook uses:

- sliding-window 3D prediction
- test-time augmentation
- temperature scaling
- hysteresis thresholding
- topology-aware post-processing
- runtime degradation levels

The degradation ladder is important for Kaggle notebooks because GPU/runtime conditions can vary. The system can reduce overlap, disable TTA, or reduce model count rather than simply timing out.
