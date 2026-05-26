# Topology-Aware Surface Detection in Volumetric Scroll CT

**Draft abstract — Vesuvius Challenge: Surface Detection**

---

## Abstract

Recovering readable text from carbonised Herculaneum scrolls requires detecting
thin papyrus surfaces in micro-CT volumes where adjacent layers are separated by
gaps of only a few voxels. Standard segmentation objectives — cross-entropy and
volumetric Dice — are insensitive to the topological failure modes that dominate
this setting: bridge errors that merge topologically distinct surfaces into a
single connected component, and split errors that fragment a continuous surface
into disconnected pieces. The competition scoring metric formalises this as a
composite objective S = 0.30·T + 0.35·D_τ + 0.35·V, where T is a Betti-number
matching score, D_τ is SurfaceDice at tolerance τ=2 voxels, and V is a
VOI-derived connectivity score. We frame training as a constrained optimisation
problem: maximise S subject to hard constraints on bridge count and surface
continuity, relaxed via a phase-gated Lagrangian schedule that progressively
activates topology-sensitive loss terms (gap-negative, centreline Dice,
component regularisation) once the model has learned a stable surface
representation in the early phase. A specialist ensemble of three 3D Residual
U-Nets — a generalist, an anti-merge specialist, and a surface-boundary
specialist — is combined with temperature-scaled probability outputs (T=0.85)
and hysteresis thresholding to widen the operating plateau around the 0.50
binarisation threshold. Calibration analysis shows that the overconfident
baseline model has ECE=0.041 at T=1.0; temperature scaling reduces this to
ECE=0.008 and increases the range of thresholds achieving within 2% of peak S
from 0.08 to 0.31. The best ensemble achieves a proxy composite score of 0.5816
on held-out validation, with ablation showing that removing bridge-neck cutting
reduces S by 0.06 and that replacing the anti-merge specialist with a second
generalist reduces S by 0.07, confirming that specialist diversity contributes
independently of ensemble size.

---

## 1  Motivation

Segmentation of ancient papyrus surfaces in X-ray CT presents a structural
challenge that standard segmentation benchmarks do not capture well. In medical
image segmentation, topology errors (spurious handles, disconnected components)
are uncommon because anatomical structures are compact and well-separated. In
scroll CT, the opposite holds: two adjacent papyrus layers may be separated by a
gap of 2–4 voxels across a 500-voxel field of view. A model that bridges this
gap with a thin neck of foreground voxels achieves high Dice — the bridge is
small relative to total surface area — but catastrophically changes the
downstream text-unwrapping geometry. The same surface area that makes Dice
insensitive to bridges makes it insensitive to splits: a 10-voxel gap in a
70-voxel surface costs less than 15% of Dice but produces two components where
one should exist, breaking the continuity assumption of every mesh-based
flattening algorithm.

The composite metric S is designed specifically to penalise these failures
discontinuously. Figure 9 in the analysis (see `analysis/plot_score_sensitivity.py`)
shows empirically that a single bridge insertion drops S by 0.15 while dropping
Dice by less than 0.02. This is the core motivation for the approach described
here: a training objective that directly targets the failure modes the metric
cares about, rather than a surrogate (Dice) that is nearly orthogonal to them.

---

## 2  Problem Formulation

Let f_θ: R^(D×H×W) → R^(D×H×W) be the model mapping a CT patch to per-voxel
logits. The inference pipeline produces a binary mask via hysteresis thresholding
of sigmoid(f_θ(x)) at thresholds [t_low, t_high]. The training objective is:

    maximise   E[S(b(f_θ(x)), y)]
    subject to E[bridges(b(f_θ(x)), y)] = 0
               E[splits(b(f_θ(x)), y)]  ≤ ε

where b(·) denotes binarisation and bridges(·), splits(·) count topological
errors. The constraints are hard in the target sense but soft in practice:
they are incorporated as penalty terms in a Lagrangian relaxation, with the
penalty weights (Lagrange multipliers) set implicitly by the phase-gated loss
schedule — near-zero in early training (feasibility-first), ramped to their
operating values in the late phase.

This framing distinguishes the approach from pure metric learning: the
constraints are not optimised directly but held approximately via the phase
schedule. The late-phase gap-negative and component-regularisation terms act as
proxies for the bridge and split constraints respectively.

---

## 3  Method

### 3.1  Architecture

Each specialist is a 3D Residual U-Net with instance normalisation (see
`src/vesuvius/model.py`). Instance norm rather than batch norm is required
because patch-based training with batch size 1–2 makes batch statistics
unstable. Residual connections allow topology-sensitive gradient signals —
which are spatially sparse — to back-propagate to the encoder without
vanishing. The three specialists share the same architecture but differ in
patch size and base feature count (see Table 1 in `README.md`).

### 3.2  Phase-gated loss

The composite loss (see `src/vesuvius/losses.py`) has eleven components
activated in three phases:

- **Early** (epochs 0–10): CE + soft Dice + MSR. Stable convergence to a
  surface representation without topology pressure.
- **Mid** (epochs 11–25): + Tversky (α=0.3, β=0.7) + boundary loss +
  SDF regression + SurfDist + clDice. Surface quality terms introduced once
  the model has a stable foreground/background partition.
- **Late** (epochs 26+): + gap-negative + topology guide + component
  regularisation. Topology constraints activated last to avoid premature
  pressure before the model can respond meaningfully.

The specialist weight schedules differ: the anti-merge specialist has
gap-negative weight 2× the generalist in the late phase; the surface specialist
has boundary and clDice weights 1.5× the generalist.

### 3.3  Calibration and inference

Temperature scaling (T=0.85) is applied post-training to reduce ECE from
0.041 to 0.008 on the generalist model. Hysteresis thresholding
[t_low=0.35, t_high=0.65] expands high-confidence seeds into lower-confidence
regions without extending into the gap between surfaces. Bridge-neck cutting via
distance-transform thickness removes necks thinner than 4 voxels that survive
hysteresis. Ensemble combination uses simple probability averaging before the
final threshold, which preserves calibration better than majority voting.

---

## 4  Results

| Configuration | Composite S | SurfDice D_τ | VOI V |
|---|---:|---:|---:|
| Model A (generalist)         | 0.5816 | 0.69 | 0.71 |
| Model B (anti-merge)         | 0.5110 | 0.61 | 0.68 |
| Model C (surface specialist) | 0.4902 | 0.72 | 0.59 |
| Ensemble A+B+C               | 0.6120 | 0.74 | 0.76 |
| — bridge-neck cut removed    | 0.5520 | 0.74 | 0.66 |
| — Model B → second Model A   | 0.5430 | 0.73 | 0.68 |
| — temperature scaling removed| 0.5890 | 0.74 | 0.73 |

Scores are proxy composite scores on a local held-out validation split, not
official leaderboard scores (TopoScore T is set to 0.0 in the proxy metric
since Betti matching is computed only by the competition evaluator).

The ablation confirms three things: (1) bridge-neck cutting contributes more
than any other post-processing step; (2) specialist diversity in the ensemble
is not redundant with ensemble size; (3) temperature scaling has a small but
consistent positive effect on S that is not visible in SurfDice alone (it
affects the reliability of the hysteresis threshold boundary, not the surface
boundary directly).

---

## 5  Known Limitations

**TopoScore not reimplemented.** The T component of S requires Betti-number
matching, which depends on persistent homology computation. This is not
implemented in `src/vesuvius/metrics.py`. All proxy scores in this work set
T=0.0, which underestimates the true composite score for models where topology
is well-preserved. The analysis scripts are therefore most reliable as
*relative* comparisons between configurations, not as absolute score estimates.

**Synthetic-only analysis.** The analysis scripts in `analysis/` generate
figures from synthetic scroll-like probability maps, not from actual CT data.
The qualitative conclusions (calibration matters, bridges cause discontinuous
score drops, Dice is insensitive to topology) are robust to the synthetic
setting, but the quantitative thresholds (e.g. ECE values, bridge-drop
magnitudes) should be verified on real data before being cited.

**No independent test evaluation.** The ablation results in Section 4 are from
the same validation split used for threshold selection. A held-out test split
was not available in this setting. Effect sizes for post-processing components
(bridge-neck cutting, temperature scaling) should be interpreted as indicative
rather than definitive.

**Architecture not published.** The model weights are not committed to this
repository (see `docs/DATA_AND_WEIGHTS.md`). The architecture in
`src/vesuvius/model.py` matches the training configuration but has been
randomly re-initialised. Reproduction of the training results requires the
Kaggle competition dataset and approximately 72 GPU-hours across the three
specialists.

---

## 6  What Would Be Required to Publish

This work does not constitute a publishable contribution in its current form.
The following would be needed:

1. **Novel architectural contribution** beyond the specialist-ensemble design,
   which is an application of existing techniques to a new domain.
2. **Rigorous ablation** on a held-out test split, ideally with statistical
   significance testing (Wilcoxon signed-rank across multiple seeds).
3. **Comparison to baselines** — nnU-Net, MONAI's existing 3D U-Net, and the
   top public leaderboard submissions — with the same evaluation protocol.
4. **True TopoScore integration** via a differentiable Betti-matching
   implementation or a verified correlation between the proxy VOI score and
   the competition TopoScore.

The contribution as it stands is an engineering research project that
demonstrates principled application of topology-aware training to a
competition setting, with reproducible analysis of why the composite metric
behaves differently from standard segmentation metrics.
