# Vesuvius Challenge Surface Detection

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Kaggle](https://img.shields.io/badge/Kaggle-Vesuvius%20Challenge-20BEFF)
![License](https://img.shields.io/badge/License-MIT-green)

Deep learning research repository for the Kaggle **Vesuvius Challenge - Surface Detection** competition.

This project explores 3D surface segmentation for scroll-volume CT data, with an emphasis on topology-aware validation, specialist model ensembles, and robust inference under Kaggle runtime constraints.

## Research Focus

The core hypothesis is that strong leaderboard performance in surface detection is not only a better U-Net. The system has to reduce topology failures:

- **Surface quality**: maximize Surface Dice around thin sheet boundaries.
- **Merge control**: prevent bridges between nearby surfaces.
- **Split control**: preserve continuity in thin, fragmented regions.
- **Threshold stability**: prefer models whose predictions stay stable under small threshold changes.
- **Runtime resilience**: degrade gracefully when inference time is limited.

## Repository Map

| Path | Purpose |
|---|---|
| `notebooks/` | Kaggle training and inference notebooks, curated into a readable sequence |
| `docs/` | Research methodology, architecture notes, experiment log, and data policy |
| `configs/` | Lightweight configuration snapshots for model roles and inference choices |
| `outputs/metadata/` | Small JSON metadata from trained models; large weights are intentionally excluded |
| `reports/` | Audit reports and experiment summaries |
| `scripts/` | Notebook builders, fix scripts, and validation utilities from the project history |
| `src/vesuvius/` | Lightweight tested utilities extracted from the notebook workflow |
| `tests/` | Local tests for config, metadata, postprocessing, and repository hygiene |
| `versions/` | Earlier notebook iterations retained from the original project history |

## Standout Technical Artifacts

| Document | Why It Matters |
|---|---|
| [Architecture](docs/ARCHITECTURE.md) | System view of the training and inference pipeline |
| [Experiment Log](docs/EXPERIMENTS.md) | Research decisions, hypotheses, outcomes, and keep/drop logic |
| [Technical Deep Dive](docs/TECHNICAL_DEEP_DIVE.md) | Explains topology-aware modeling beyond plain Dice optimization |
| [Performance Engineering](docs/PERFORMANCE_ENGINEERING.md) | GPU memory, runtime, TTA, overlap, and degradation tradeoffs |
| [Reproducibility](docs/REPRODUCIBILITY.md) | Data, environment, metadata, and local verification notes |
| [Project Brief](docs/PROJECT_BRIEF.md) | Recruiter-friendly summary of impact and skills demonstrated |

## Current System

The current project state uses a specialist ensemble design:

| Model | Role | Seed | Fold | Patch Size | Best Proxy Score |
|---|---:|---:|---:|---:|---:|
| Model A | Generalist | 42 | 0 | 192 x 192 x 192 | 0.5816 |
| Model B | Anti-merge | 1337 | 1 | 160 x 160 x 160 | 0.5110 |
| Model C | Surface specialist | 2024 | 2 | 192 x 192 x 192 | 0.4902 |

Inference uses temperature scaling, hysteresis thresholds, topology-aware post-processing, test-time augmentation, and a time-budget degradation ladder.

## What Is Not Committed

Large competition assets, model checkpoints, and generated datasets are intentionally excluded from Git:

- Kaggle competition zip files
- extracted CT volumes and label TIFFs
- `.pt`, `.pth`, `.ckpt`, `.onnx`, and similar model weights
- local cache folders and generated notebook checkpoints

This keeps the repository professional, cloneable, and safe for LinkedIn/GitHub presentation. See [docs/DATA_AND_WEIGHTS.md](docs/DATA_AND_WEIGHTS.md) for reproduction notes.

## Kaggle Challenge

Competition page: https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
pytest
```

The notebooks are written for Kaggle GPU environments. Local execution is mainly useful for code review, documentation, and lightweight validation.

## Project Status

This repository is presented as a research portfolio project. It documents the modeling direction, experiments, validation logic, and final Kaggle-facing notebooks without committing private datasets or heavyweight artifacts.
