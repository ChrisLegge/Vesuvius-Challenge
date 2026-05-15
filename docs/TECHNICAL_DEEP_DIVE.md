# Technical Deep Dive

## Why Dice Alone Is Not Enough

For thin 3D surfaces, mean segmentation overlap can hide catastrophic structural errors. A prediction can have reasonable local overlap while still creating a bridge between two nearby surfaces or splitting one continuous sheet into pieces.

That is why this project emphasizes topology-aware behavior:

- component stability
- bridge avoidance
- split avoidance
- threshold robustness
- surface-distance quality

## Specialist Ensemble Rationale

The ensemble is not just multiple seeds of the same idea. Each member has a role:

- **Generalist**: stable baseline behavior
- **Anti-merge**: reduces accidental connectors between nearby regions
- **Surface specialist**: improves boundary-local behavior

This is useful because ensemble value comes from complementary errors, not just averaging similar predictions.

## Hysteresis Thresholding

Simple thresholding treats every voxel independently. Hysteresis thresholding uses two thresholds:

- high threshold: confident seed regions
- low threshold: candidate support regions

Weak regions are kept only when connected to confident seeds. This reduces isolated noise while preserving plausible thin structures.

## Runtime Degradation

Kaggle notebooks have strict runtime and memory limits. The production notebook therefore includes a degradation ladder:

- full TTA and ensemble when time allows
- reduced overlap when needed
- fewer models under pressure
- smaller windows as a last resort

This is a practical reliability pattern: the system returns a reasonable output rather than failing late.
