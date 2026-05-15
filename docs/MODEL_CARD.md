# Model Card

## Intended Use

This model family is intended for the Kaggle Vesuvius Challenge Surface Detection competition and related research exploration in 3D surface segmentation.

It is not intended for medical diagnosis, safety-critical archaeological interpretation, or production use without independent validation.

## Model Family

| Model | Role | Seed | Fold | Patch Size | Best Validation Proxy |
|---|---:|---:|---:|---:|---:|
| Model A | Generalist | 42 | 0 | 192³ | 0.5816 |
| Model B | Anti-merge | 1337 | 1 | 160³ | 0.5110 |
| Model C | Surface specialist | 2024 | 2 | 192³ | 0.4902 |

## Training Characteristics

Common design choices:

- 3D patch-based segmentation
- foreground and boundary-biased sampling
- topology and surface-distance inspired losses
- learning-rate floor for late-stage polishing
- gradient clipping
- exponential moving average for selected runs

## Inference Characteristics

The production inference path uses:

- 3-model ensemble
- ensemble weights `[0.40, 0.35, 0.25]` for the selected production configuration
- temperature scaling
- hysteresis thresholds
- dust removal and topology-aware post-processing
- test-time augmentation where the time budget allows

## Limitations

- Validation proxies are approximations of the full competition metric.
- Results depend on Kaggle input paths and GPU availability.
- Large checkpoints are not tracked in this repository.
- The model family should be judged by reproducible notebooks and metadata, not by this repository alone.
