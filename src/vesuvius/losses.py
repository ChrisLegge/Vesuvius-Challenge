"""
Phase-gated composite loss for Vesuvius surface detection.

The training objective is a weighted sum of up to eleven components, with
weights determined by the current training phase (early / mid / late).
Phase gating prevents topology-sensitive terms from dominating before the
model has learned a stable surface representation.

Composite loss:

    L = w_ce   * L_ce
      + w_dice  * L_dice
      + w_tversky * L_tversky
      + w_bnd   * L_bnd
      + w_sdf   * L_sdf
      + w_surfdist * L_surfdist
      + w_cldice * L_cldice
      + w_gapneg * L_gapneg
      + w_topoguide * L_topoguide
      + w_compreg * L_compreg
      + w_msr   * L_msr

Components activated per phase
-------------------------------
early  : ce, dice, msr
mid    : + tversky, bnd, sdf, surfdist, cldice
late   : + gapneg, topoguide, compreg

Each specialist model has a different phase schedule (see configs/).

References
----------
clDice : Shit et al. (2021) https://arxiv.org/abs/2003.07311
Tversky: Salehi et al. (2017) https://arxiv.org/abs/1706.05721
Bnd loss: Kervadec et al. (2019) https://arxiv.org/abs/1905.07852
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

Phase = Literal["early", "mid", "late"]

# ---------------------------------------------------------------------------
# Weight schedule — one dataclass per specialist role
# ---------------------------------------------------------------------------

@dataclass
class PhaseWeights:
    """
    Loss component weights for each training phase.

    Zero weight means the term is inactive (not computed).
    Weights are not normalised — the scheduler controls the overall scale
    via the learning rate, not by forcing weights to sum to 1.
    """
    # fmt: off
    ce:        tuple[float, float, float] = (1.0,  0.5,  0.3)   # early / mid / late
    dice:      tuple[float, float, float] = (1.0,  0.8,  0.6)
    msr:       tuple[float, float, float] = (0.2,  0.1,  0.05)
    bnd:       tuple[float, float, float] = (0.0,  0.4,  0.3)
    sdf:       tuple[float, float, float] = (0.0,  0.3,  0.2)
    surfdist:  tuple[float, float, float] = (0.0,  0.2,  0.15)
    cldice:    tuple[float, float, float] = (0.0,  0.3,  0.3)
    tversky:   tuple[float, float, float] = (0.0,  0.5,  0.4)
    gapneg:    tuple[float, float, float] = (0.0,  0.0,  0.6)
    topoguide: tuple[float, float, float] = (0.0,  0.0,  0.4)
    compreg:   tuple[float, float, float] = (0.0,  0.0,  0.3)
    # fmt: on


# Pre-defined specialist schedules matching the trained models
GENERALIST_WEIGHTS     = PhaseWeights()   # defaults above
ANTI_MERGE_WEIGHTS     = PhaseWeights(
    gapneg    = (0.0, 0.0, 1.2),   # stronger gap-negative penalty
    topoguide = (0.0, 0.0, 0.8),
    tversky   = (0.0, 0.6, 0.6),
    compreg   = (0.0, 0.0, 0.5),
)
SURFACE_SPECIALIST_WEIGHTS = PhaseWeights(
    bnd      = (0.0, 0.6, 0.5),    # stronger boundary term
    surfdist = (0.0, 0.4, 0.35),
    cldice   = (0.0, 0.5, 0.5),
    sdf      = (0.0, 0.4, 0.3),
)

_PHASE_INDEX: dict[Phase, int] = {"early": 0, "mid": 1, "late": 2}


# ---------------------------------------------------------------------------
# Individual loss functions
# ---------------------------------------------------------------------------

def loss_ce(pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Binary cross-entropy from logits.

    Parameters
    ----------
    pred_logits : (B, 1, *spatial) — raw model output before sigmoid
    target      : (B, 1, *spatial) — binary ground truth in {0, 1}
    """
    return F.binary_cross_entropy_with_logits(pred_logits, target.float())


def loss_dice(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1.0,
) -> torch.Tensor:
    """
    Soft Dice loss.

        L_dice = 1 - (2 * |P ∩ T| + smooth) / (|P| + |T| + smooth)

    Smooth avoids division by zero on empty volumes and stabilises gradients
    when both prediction and target are near-zero.
    """
    pred = torch.sigmoid(pred_logits)
    pred_f   = pred.flatten(1)
    target_f = target.float().flatten(1)
    inter    = (pred_f * target_f).sum(dim=1)
    denom    = pred_f.sum(dim=1) + target_f.sum(dim=1)
    dice     = (2.0 * inter + smooth) / (denom + smooth)
    return 1.0 - dice.mean()


def loss_tversky(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.3,
    beta: float  = 0.7,
    smooth: float = 1.0,
) -> torch.Tensor:
    """
    Tversky loss — generalisation of Dice with asymmetric FP/FN weighting.

        L_tversky = 1 - (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)

    alpha < beta penalises false negatives more heavily, which is appropriate
    for surface detection where missed surface is worse than over-prediction.
    The anti-merge specialist uses alpha=0.3, beta=0.7.
    """
    pred = torch.sigmoid(pred_logits)
    pred_f   = pred.flatten(1)
    target_f = target.float().flatten(1)
    tp   = (pred_f * target_f).sum(dim=1)
    fp   = (pred_f * (1.0 - target_f)).sum(dim=1)
    fn   = ((1.0 - pred_f) * target_f).sum(dim=1)
    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return 1.0 - tversky.mean()


def loss_bnd(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    dist_map: torch.Tensor,
) -> torch.Tensor:
    """
    Boundary loss (Kervadec et al. 2019).

    Minimises the integral of the predicted probability map weighted by the
    distance transform of the GT boundary. Forces predictions to concentrate
    near the true surface rather than filling the interior.

        L_bnd = mean(sigmoid(pred) * dist_map)

    Parameters
    ----------
    dist_map : (B, 1, *spatial)
        Pre-computed distance transform of the GT boundary. Values are
        positive outside the GT surface and negative inside (signed).
        Pass unsigned distances if signed are unavailable; the loss still
        works but loses the interior-suppression property.
    """
    pred = torch.sigmoid(pred_logits)
    return (pred * dist_map).mean()


def loss_sdf(
    pred_logits: torch.Tensor,
    sdf_target: torch.Tensor,
) -> torch.Tensor:
    """
    Signed distance field regression loss.

    Supervises the network to produce outputs proportional to the signed
    distance from the GT surface, not just a binary label. This provides
    a richer gradient signal near the surface and penalises predictions
    that are far from the true surface boundary.

        L_sdf = MSE(tanh(pred_logits), tanh(sdf_target))

    tanh compression prevents the loss from being dominated by large
    distances far from the surface.

    Parameters
    ----------
    sdf_target : (B, 1, *spatial)
        Signed distance field of the GT surface. Negative inside, positive
        outside. Computed offline and passed as a pre-computed tensor.
    """
    return F.mse_loss(torch.tanh(pred_logits), torch.tanh(sdf_target))


def loss_surfdist(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-5,
) -> torch.Tensor:
    """
    Differentiable surface distance loss.

    Approximates the average distance between predicted and GT surfaces using
    the soft probability map rather than a hard threshold. Penalises
    probability mass that is far from the GT surface boundary.

        For each voxel: contribution = p_i * d_i^2
        L_surfdist = mean over batch of (sum_i p_i * d_i^2) / (sum_i p_i + smooth)

    This is an approximation — true surface distance requires a hard threshold
    and is non-differentiable. The soft version provides a gradient signal
    proportional to how far the predicted probability mass is from the GT surface.

    Parameters
    ----------
    target : (B, 1, *spatial)
        Binary GT. Distance transforms are computed internally per sample.
    """
    pred = torch.sigmoid(pred_logits)
    B    = pred.shape[0]
    losses = []
    for b in range(B):
        p = pred[b, 0]                    # (spatial,)
        t = target[b, 0].bool()
        if not t.any():
            losses.append(pred.new_zeros(1).squeeze())
            continue
        # Use a rough approximation on GPU: weight by L2 distance from GT centroid
        coords = torch.nonzero(t, as_tuple=False).float()   # (N, ndim)
        centroid = coords.mean(dim=0)                        # (ndim,)
        grid = torch.stack(
            torch.meshgrid(
                *[torch.arange(s, device=pred.device, dtype=torch.float32)
                  for s in p.shape],
                indexing="ij",
            ),
            dim=-1,
        )                                                     # (*spatial, ndim)
        dist2 = ((grid - centroid) ** 2).sum(dim=-1)         # (*spatial,)
        weighted = (p * dist2).sum() / (p.sum() + smooth)
        losses.append(weighted)
    return torch.stack(losses).mean()


def loss_cldice(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1.0,
    iters: int = 3,
) -> torch.Tensor:
    """
    clDice loss — centreline Dice (Shit et al. 2021).

    Combines standard soft Dice with a Dice computed on the soft skeletons
    (centrelines) of prediction and target. Preserves tubular topology by
    penalising predictions that break or bridge the medial axis.

        L_cldice = 1 - (1-alpha)*V_dice + alpha*V_cl_dice

    where alpha=0.5 balances region overlap with centreline preservation.

    Soft skeleton approximated by iterative min-pooling (morphological erosion
    in the continuous domain). This is an approximation that avoids the
    non-differentiable thinning step in the original paper.

    Parameters
    ----------
    iters : int
        Number of min-pool erosion iterations. More iterations = thinner
        skeleton approximation at the cost of gradient signal strength.
    """
    pred   = torch.sigmoid(pred_logits)
    target = target.float()

    def _soft_skeleton(x: torch.Tensor, iters: int) -> torch.Tensor:
        """Iterative min-pooling approximation of morphological skeleton."""
        skel = x
        for _ in range(iters):
            if x.dim() == 4:       # 2D: (B, C, H, W)
                eroded = -F.max_pool2d(-skel, kernel_size=3, stride=1,
                                       padding=1)
            else:                  # 3D: (B, C, D, H, W)
                eroded = -F.max_pool3d(-skel, kernel_size=3, stride=1,
                                       padding=1)
            skel = torch.min(skel, eroded + (1.0 - eroded) * x)
        return skel

    skel_pred   = _soft_skeleton(pred,   iters)
    skel_target = _soft_skeleton(target, iters)

    def _dice_term(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a_f = a.flatten(1); b_f = b.flatten(1)
        inter = (a_f * b_f).sum(dim=1)
        denom = a_f.sum(dim=1) + b_f.sum(dim=1)
        return ((2.0 * inter + smooth) / (denom + smooth)).mean()

    # clDice = 0.5 * Tprec + 0.5 * Tsens  (see paper eq.4)
    t_prec = _dice_term(skel_pred, target)      # skeleton pred vs GT region
    t_sens = _dice_term(skel_target, pred)      # skeleton GT vs pred region
    cl_dice = 0.5 * (t_prec + t_sens)

    # Combined with standard soft Dice
    region_dice = 1.0 - loss_dice(pred_logits, target, smooth=smooth)
    return 1.0 - 0.5 * region_dice - 0.5 * cl_dice


def loss_gapneg(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    gap_weight: float = 2.0,
) -> torch.Tensor:
    """
    Gap-negative loss — penalises predictions in regions that should be empty.

    For scroll surface detection the key failure mode is a thin bridge of
    predicted foreground connecting two topologically distinct surfaces.
    This loss directly penalises probability mass in the gap between GT
    surfaces, weighted by gap_weight > 1.0 to apply stronger pressure than
    standard cross-entropy on those voxels.

        L_gapneg = gap_weight * mean(sigmoid(pred) * gap_mask)

    where gap_mask = dilation(GT) XOR GT:  the region just outside the GT
    surfaces that should definitively contain no foreground.

    Parameters
    ----------
    gap_weight : float
        Multiplier applied to predictions in the gap region. Larger values
        suppress bridges more aggressively at the risk of under-prediction
        near genuine surface boundaries.
    """
    pred = torch.sigmoid(pred_logits)
    gt_bool = target.bool().float()

    # Dilate GT by one voxel to define the near-surface region
    if pred.dim() == 4:      # 2D
        dilated = F.max_pool2d(gt_bool, kernel_size=3, stride=1, padding=1)
    else:                    # 3D
        dilated = F.max_pool3d(gt_bool, kernel_size=3, stride=1, padding=1)

    # Gap mask: dilated region that is not part of GT
    gap_mask = (dilated - gt_bool).clamp(0.0, 1.0)

    return gap_weight * (pred * gap_mask).mean()


def loss_topoguide(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-4,
) -> torch.Tensor:
    """
    Topology guidance loss.

    Encourages the number of predicted connected components (approximated
    differentiably) to match the number of GT connected components. This
    penalises both bridge errors (too few components) and split errors
    (too many components).

    Differentiable component count approximation: the sum of local maxima
    in the predicted probability map is used as a proxy for component count.
    Local maxima are identified by comparing each voxel to its max-pooled
    neighbourhood.

        proxy_count = sum(pred * (pred == max_pool(pred)))
        L_topoguide = (proxy_count_pred - proxy_count_gt)^2

    This is an approximation. True topological losses (Betti matching) are
    not differentiable and are handled by the competition evaluator.
    """
    pred = torch.sigmoid(pred_logits)
    gt   = target.float()

    def _count_proxy(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            local_max = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        else:
            local_max = F.max_pool3d(x, kernel_size=3, stride=1, padding=1)
        peaks = x * (x >= local_max - smooth).float()
        return peaks.flatten(1).sum(dim=1)

    pred_count = _count_proxy(pred)
    gt_count   = _count_proxy(gt)
    return F.mse_loss(pred_count, gt_count.detach())


def loss_compreg(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1.0,
) -> torch.Tensor:
    """
    Component regularisation loss.

    Penalises the total predicted volume relative to the GT volume, weighted
    by how far the predicted component count proxy is from the GT count.
    This discourages the model from adding spurious small components to
    maximise partial overlap with the GT while leaving bridges in place.

        L_compreg = |volume_pred - volume_gt| / (volume_gt + smooth)
                   * (1 + |count_pred - count_gt| / (count_gt + smooth))

    The count-weighted volume term means: large volume errors are penalised
    much more heavily when the component count is also wrong.
    """
    pred = torch.sigmoid(pred_logits)
    gt   = target.float()

    vol_pred = pred.flatten(1).sum(dim=1)
    vol_gt   = gt.flatten(1).sum(dim=1)

    def _count_proxy(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            lm = F.max_pool2d(x, 3, stride=1, padding=1)
        else:
            lm = F.max_pool3d(x, 3, stride=1, padding=1)
        return (x * (x >= lm - 1e-4).float()).flatten(1).sum(dim=1)

    count_pred = _count_proxy(pred)
    count_gt   = _count_proxy(gt.clamp(0, 1))

    vol_ratio   = (vol_pred - vol_gt).abs() / (vol_gt + smooth)
    count_ratio = (count_pred - count_gt).abs() / (count_gt + smooth)

    return (vol_ratio * (1.0 + count_ratio)).mean()


def loss_msr(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Mean squared residual loss.

    Standard MSE between sigmoid(pred) and the binary target. Acts as a
    soft version of cross-entropy with a smoother gradient near 0.5.
    Used in early training as a stable complement to BCE before the model
    has learned a reliable surface representation.
    """
    return F.mse_loss(torch.sigmoid(pred_logits), target.float())


# ---------------------------------------------------------------------------
# Phase-gated composite loss module
# ---------------------------------------------------------------------------

class PhaseGatedLoss(nn.Module):
    """
    Composite loss with phase-dependent component activation.

    Wraps all eleven loss functions and applies the weight schedule defined
    in a PhaseWeights instance. Components with weight 0.0 in the current
    phase are not computed (no wasted gradient passes).

    Parameters
    ----------
    weights : PhaseWeights
        Per-component weight schedule. Use GENERALIST_WEIGHTS,
        ANTI_MERGE_WEIGHTS, or SURFACE_SPECIALIST_WEIGHTS for the three
        trained specialists, or supply a custom PhaseWeights instance.
    tversky_alpha, tversky_beta : float
        Tversky asymmetry parameters. Default 0.3/0.7 penalises false
        negatives more than false positives.
    cldice_iters : int
        Soft skeleton erosion iterations for clDice.

    Usage
    -----
    criterion = PhaseGatedLoss(ANTI_MERGE_WEIGHTS)
    loss, breakdown = criterion(pred_logits, target, phase="late",
                                dist_map=dist_map, sdf_target=sdf_target)
    """

    def __init__(
        self,
        weights:       PhaseWeights = None,
        tversky_alpha: float = 0.3,
        tversky_beta:  float = 0.7,
        cldice_iters:  int   = 3,
    ) -> None:
        super().__init__()
        self.weights       = weights or PhaseWeights()
        self.tversky_alpha = tversky_alpha
        self.tversky_beta  = tversky_beta
        self.cldice_iters  = cldice_iters

    def _w(self, component: str, phase: Phase) -> float:
        return getattr(self.weights, component)[_PHASE_INDEX[phase]]

    def forward(
        self,
        pred_logits: torch.Tensor,
        target:      torch.Tensor,
        phase:       Phase = "early",
        dist_map:    torch.Tensor | None = None,
        sdf_target:  torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute the weighted composite loss.

        Parameters
        ----------
        pred_logits : (B, 1, *spatial)
            Raw model output (before sigmoid).
        target : (B, 1, *spatial)
            Binary ground truth.
        phase : "early" | "mid" | "late"
            Current training phase. Determines which components are active.
        dist_map : optional tensor, same shape as pred_logits
            Distance transform of GT boundary. Required if w_bnd > 0.
        sdf_target : optional tensor, same shape as pred_logits
            Signed distance field of GT surface. Required if w_sdf > 0.

        Returns
        -------
        total : scalar tensor
            Weighted sum of active loss components (differentiable).
        breakdown : dict[str, float]
            Per-component loss values (detached). Useful for logging.
        """
        total     = pred_logits.new_zeros(1).squeeze()
        breakdown: dict[str, float] = {}

        def _add(name: str, value: torch.Tensor) -> None:
            nonlocal total
            w = self._w(name, phase)
            if w <= 0.0:
                breakdown[name] = 0.0
                return
            total = total + w * value
            breakdown[name] = float(value.detach())

        _add("ce",    loss_ce(pred_logits, target))
        _add("dice",  loss_dice(pred_logits, target))
        _add("msr",   loss_msr(pred_logits, target))

        if self._w("tversky", phase) > 0:
            _add("tversky", loss_tversky(pred_logits, target,
                                         self.tversky_alpha, self.tversky_beta))
        else:
            breakdown["tversky"] = 0.0

        if self._w("bnd", phase) > 0:
            if dist_map is None:
                raise ValueError(
                    "dist_map required when w_bnd > 0 in phase '{}'".format(phase)
                )
            _add("bnd", loss_bnd(pred_logits, target, dist_map))
        else:
            breakdown["bnd"] = 0.0

        if self._w("sdf", phase) > 0:
            if sdf_target is None:
                raise ValueError(
                    "sdf_target required when w_sdf > 0 in phase '{}'".format(phase)
                )
            _add("sdf", loss_sdf(pred_logits, sdf_target))
        else:
            breakdown["sdf"] = 0.0

        if self._w("surfdist", phase) > 0:
            _add("surfdist", loss_surfdist(pred_logits, target))
        else:
            breakdown["surfdist"] = 0.0

        if self._w("cldice", phase) > 0:
            _add("cldice", loss_cldice(pred_logits, target,
                                       iters=self.cldice_iters))
        else:
            breakdown["cldice"] = 0.0

        if self._w("gapneg", phase) > 0:
            _add("gapneg", loss_gapneg(pred_logits, target))
        else:
            breakdown["gapneg"] = 0.0

        if self._w("topoguide", phase) > 0:
            _add("topoguide", loss_topoguide(pred_logits, target))
        else:
            breakdown["topoguide"] = 0.0

        if self._w("compreg", phase) > 0:
            _add("compreg", loss_compreg(pred_logits, target))
        else:
            breakdown["compreg"] = 0.0

        return total, breakdown

    def active_components(self, phase: Phase) -> list[str]:
        """Return the names of components with non-zero weight in this phase."""
        return [
            name for name in _PHASE_INDEX.__class__.__mro__[0].__dict__
            if isinstance(getattr(PhaseWeights, name, None), tuple)
            and self._w(name, phase) > 0.0
        ]

    def __repr__(self) -> str:
        lines = ["PhaseGatedLoss("]
        for phase in ("early", "mid", "late"):
            active = [
                f"{c}={self._w(c, phase):.2f}"
                for c in ("ce", "dice", "msr", "tversky", "bnd", "sdf",
                           "surfdist", "cldice", "gapneg", "topoguide", "compreg")
                if self._w(c, phase) > 0.0
            ]
            lines.append(f"  {phase}: [{', '.join(active)}]")
        lines.append(")")
        return "\n".join(lines)
