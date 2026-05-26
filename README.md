# Vesuvius Challenge Surface Detection

![CI](https://github.com/ChrisLegge/Vesuvius-Challenge/actions/workflows/ci.yml/badge.svg)
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

## Documentation

| Document | Contents |
|---|---|
| [Paper Draft](docs/PAPER_DRAFT.md) | Structured abstract: motivation, problem formulation, method, ablation results, known limitations, what would be needed to publish |
| [Project Summary](docs/PROJECT_BRIEF.md) | Problem statement, scoring metric, constrained optimisation framing, results, known limitations |
| [Architecture](docs/ARCHITECTURE.md) | Training and inference pipeline design |
| [Experiment Log](docs/EXPERIMENTS.md) | Research decisions, hypotheses tested, outcomes, and what was dropped |
| [Technical Deep Dive](docs/TECHNICAL_DEEP_DIVE.md) | Topology-aware modelling: VOI, Betti matching, phase-gated losses |
| [Performance Engineering](docs/PERFORMANCE_ENGINEERING.md) | GPU memory, patch scheduling, TTA cost, degradation ladder tradeoffs |
| [Reproducibility](docs/REPRODUCIBILITY.md) | Data paths, environment, known sources of nondeterminism |

## Current System

The current project state uses a specialist ensemble design:

| Model | Role | Seed | Fold | Patch Size | Best Proxy Score |
|---|---:|---:|---:|---:|---:|
| Model A | Generalist | 42 | 0 | 192 x 192 x 192 | 0.5816 |
| Model B | Anti-merge | 1337 | 1 | 160 x 160 x 160 | 0.5110 |
| Model C | Surface specialist | 2024 | 2 | 192 x 192 x 192 | 0.4902 |

Inference uses temperature scaling, hysteresis thresholds, topology-aware post-processing, test-time augmentation, and a time-budget degradation ladder.

## Training Evidence

All figures below are generated from committed training history by `analysis/plot_training_curves.py` — no competition data or GPU required to reproduce them.

**Loss trajectories across all three specialists** — phase bands show early/mid/late gating. Dotted vertical line marks the best validation epoch per model.

![Loss trajectories](results/figures/fig1_loss_trajectories.png)

**Phase-gated loss component activation** — shows which components (gap-negative, centreline Dice, topology guide) are inactive in early phase and progressively introduced in mid and late. Each specialist has a different activation pattern reflecting its role.

![Phase-gated components](results/figures/fig2_phase_gated_components.png)

**Learning rate schedules** — cosine schedule with a non-trivial floor. Late-phase LR held above 3e-4 to keep surface polishing active.

![LR schedules](results/figures/fig3_lr_schedules.png)

**Model comparison** — score decomposition (composite, SurfaceDice, VOI, TopoScore), training time, and patch size per specialist.

![Model comparison](results/figures/fig4_model_comparison.png)

## Topology Post-Processing

The two structural failure modes the system targets, visualised on synthetic probability maps generated to match typical scroll CT characteristics. Figures produced by `analysis/plot_topology_postprocess.py`.

**Bridge case (VOI_merge / Topo k1 failure)** — two topologically distinct surfaces connected by a thin low-confidence bridge. Naive thresholding at 0.50 merges them into one component. Hysteresis thresholding reduces the bridge; bridge-neck cutting via distance-transform thickness removes it.

![Bridge case](results/figures/fig5_topology_bridge_case.png)

**Split case (VOI_split / Topo k0 failure)** — a single continuous surface fragmented by a CT attenuation shadow. Naive thresholding produces two components where one should exist. Hysteresis recovers continuity by propagating from high-confidence seed regions.

![Split case](results/figures/fig6_topology_split_case.png)

**Full pipeline overview** — both failure modes through each processing stage, with connected-component counts annotated at each step.

![Pipeline overview](results/figures/fig7_postprocess_pipeline.png)

## Metric Sensitivity Analysis

Three figures examining why the composite metric S = 0.30·T + 0.35·D_tau + 0.35·V cannot be reduced to Dice optimisation. Generated by `analysis/plot_score_sensitivity.py` using synthetic data — no competition data or GPU required.

**Threshold sensitivity — S(t) curves** — how the composite score varies with binarisation threshold for three calibration regimes. A well-calibrated model maintains a wide plateau; an overconfident model collapses to a sharp spike. Temperature scaling (T=0.85) provides intermediate behaviour.

![Threshold sensitivity](results/figures/fig8_threshold_sensitivity.png)

**Bridge event curve** — composite score vs number of bridges inserted between two distinct surfaces. VOI and TopoScore respond discontinuously: a single bridge can drop S by 0.15+ while Dice barely moves. This is the core motivation for topology-aware post-processing.

![Bridge event curve](results/figures/fig9_bridge_event_curve.png)

**Dice vs composite scatter** — predictions with identical Dice coefficients can span a 0.2+ range of composite scores depending on topology error type. Bridge errors (red triangles) and fragmentation errors (blue squares) diverge substantially from the identity line.

![Dice vs composite](results/figures/fig10_dice_vs_composite.png)

## 3D Surface Visualisation

Synthetic 3D CT scroll volumes showing the two failure modes (bridge, split) at every stage of the post-processing pipeline, from raw probability maps through to topologically corrected output. Generated by `analysis/plot_surface_3d.py`.

**Orthogonal cross-sections** — three axis-aligned slices through the synthetic probability volume with GT surface boundaries overlaid. Mirrors the view used during model debugging on real scroll CT data.

![Cross-sections](results/figures/fig11_cross_sections.png)

**3D surface meshes** — marching-cubes extraction on GT, clean prediction, and bridge prediction. Component count annotated per mesh; the bridge case merges two topologically distinct surfaces into one component.

![Surface meshes](results/figures/fig12_surface_meshes.png)

**Post-processing pipeline — 2D** — XY cross-section through a bridge case at four processing stages: GT, naive threshold (t=0.50), hysteresis thresholding [0.35, 0.65], and bridge-neck cutting. Connected-component count annotated at each stage.

![Post-processing pipeline 2D](results/figures/fig14_postprocess_pipeline_2d.png)

**SurfaceDice tolerance map** — per-voxel false-colour overlay showing which predicted surface voxels fall within tau=2.0 of the GT surface (precision, green) and which do not (red). Mirrors how the metric is computed in `src/vesuvius/metrics.py`.

![SurfaceDice contribution](results/figures/fig15_surfdice_contribution.png)

## Calibration Analysis

Probability calibration determines whether threshold selection is principled or arbitrary. A miscalibrated model forces the user to search for the right threshold empirically; a well-calibrated model has a stable operating point near 0.5. The T=0.85 temperature scaling in the inference pipeline is motivated by this analysis. Generated by `analysis/plot_calibration.py`.

**Reliability diagrams** — for each calibration regime, the fraction of positive voxels in each probability bin is plotted against the mean predicted probability. Perfect calibration lies on the diagonal. The shaded gap between curve and diagonal is integrated to form ECE; histogram bars show the density of predictions per bin.

![Reliability diagrams](results/figures/fig16_reliability_diagrams.png)

**ECE vs temperature T** — Expected Calibration Error swept over the temperature scaling parameter. The minimum identifies T* for each model; the pipeline value T=0.85 is marked for reference. The overconfident model requires stronger correction (T* further from 1.0).

![ECE vs temperature](results/figures/fig17_ece_vs_temperature.png)

**Sharpness vs ECE scatter** — each point is a (sharpness, ECE) pair for a different (model sharpness, temperature) configuration. Overconfident models occupy the high-sharpness, high-ECE region. Temperature scaling moves points toward the well-calibrated region.

![Sharpness vs ECE](results/figures/fig18_sharpness_vs_ece.png)

**Calibration impact on composite score** — best achievable composite score S (over threshold grid) and ECE vs temperature T for both calibration regimes. Shows that the composite-score plateau is wider for the well-calibrated model and that T=0.85 is near-optimal for the overconfident model.

![Calibration vs composite](results/figures/fig19_calibration_vs_composite.png)

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

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e ".[dev]"
pytest                           # 20 tests, no GPU or data required
python analysis/plot_training_curves.py      # regenerate training figures
python analysis/plot_topology_postprocess.py # regenerate topology figures
python analysis/plot_score_sensitivity.py    # regenerate metric sensitivity figures
python analysis/plot_surface_3d.py          # regenerate 3D surface visualisation figures
python analysis/plot_calibration.py         # regenerate calibration analysis figures
```

The training notebooks run on Kaggle GPU environments. Everything else — tests, analysis scripts, and figure generation — runs locally without competition data.

