"""
Evaluation metric implementations for Vesuvius surface detection.

Implements the two non-trivial components of the competition scoring metric:

    S = 0.30 * T  +  0.35 * surface_dice_tau  +  0.35 * voi_score

SurfaceDice@tau and VOI split/merge are the operative metrics for training
and threshold selection. TopoScore (Betti matching) is computed by the
competition evaluator and is not reimplemented here.

Both functions are intentionally 2D and dependency-light so they can be
unit-tested without GPU, MONAI, or competition data.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt, label as nd_label


# ---------------------------------------------------------------------------
# SurfaceDice @ tolerance tau
# ---------------------------------------------------------------------------

def extract_surface(binary: np.ndarray) -> np.ndarray:
    """
    Return a boolean mask of surface voxels: foreground voxels that have at
    least one background neighbour under 4-connectivity (2D) or 6-connectivity
    (3D).

    The competition uses 6-connectivity for surface extraction in 3D. This
    implementation is dimension-agnostic.
    """
    if binary.sum() == 0:
        return np.zeros_like(binary, dtype=bool)

    from scipy.ndimage import binary_erosion, generate_binary_structure
    struct = generate_binary_structure(binary.ndim, 1)   # face-connected only
    interior = binary_erosion(binary, structure=struct, border_value=0)
    return binary.astype(bool) & ~interior


def surface_dice_tau(
    pred: np.ndarray,
    gt: np.ndarray,
    tau: float = 2.0,
    spacing: tuple[float, ...] | None = None,
) -> float:
    """
    SurfaceDice at tolerance tau in physical units.

    Measures the fraction of predicted surface within tau of the GT surface
    (precision) and the fraction of GT surface within tau of the predicted
    surface (recall), then returns their mean.

    Edge cases:
    - Both empty  -> 1.0  (trivially correct)
    - One empty   -> 0.0  (complete miss)

    Parameters
    ----------
    pred, gt : np.ndarray
        Binary arrays of equal shape (dtype need not be bool).
    tau : float
        Tolerance in physical units. Default 2.0 matches the competition.
    spacing : tuple of float, optional
        Voxel spacing per axis (e.g. (1.0, 1.0) for isotropic 2D).
        Defaults to isotropic unit spacing.

    Returns
    -------
    float in [0, 1]
    """
    pred = pred.astype(bool)
    gt   = gt.astype(bool)

    if not pred.any() and not gt.any():
        return 1.0
    if not pred.any() or not gt.any():
        return 0.0

    if spacing is None:
        spacing = (1.0,) * pred.ndim

    surf_pred = extract_surface(pred)
    surf_gt   = extract_surface(gt)

    # Distance from every voxel to the GT surface, in physical units
    dist_to_gt   = distance_transform_edt(~surf_gt,   sampling=spacing)
    # Distance from every voxel to the predicted surface, in physical units
    dist_to_pred = distance_transform_edt(~surf_pred, sampling=spacing)

    n_pred = surf_pred.sum()
    n_gt   = surf_gt.sum()

    precision = (dist_to_gt[surf_pred]   <= tau).sum() / n_pred
    recall    = (dist_to_pred[surf_gt]   <= tau).sum() / n_gt

    return float(0.5 * (precision + recall))


# ---------------------------------------------------------------------------
# Volume of Information — split and merge
# ---------------------------------------------------------------------------

def voi_split_merge(
    pred: np.ndarray,
    gt: np.ndarray,
    connectivity: int = 1,
) -> tuple[float, float]:
    """
    VOI split and VOI merge via connected-component conditional entropy.

    VOI_split = H(GT | Pred) — measures over-segmentation: how uncertain
        is the GT label given the predicted label?
    VOI_merge = H(Pred | GT) — measures under-segmentation: how uncertain
        is the predicted label given the GT label?

    Lower is better for both. The competition uses:
        V = 1 / (1 + 0.3 * (VOI_split + VOI_merge))

    Parameters
    ----------
    pred, gt : np.ndarray
        Binary arrays of equal shape.
    connectivity : int
        1 = face-connected (6-connectivity in 3D / 4-connectivity in 2D)
        2 = full-connected  (26-connectivity in 3D / 8-connectivity in 2D)
        The competition uses 26-connectivity; connectivity=2 matches that.

    Returns
    -------
    (voi_split, voi_merge) : tuple of float
    """
    pred = pred.astype(bool)
    gt   = gt.astype(bool)

    from scipy.ndimage import generate_binary_structure
    struct = generate_binary_structure(pred.ndim, connectivity)

    pred_labeled, _ = nd_label(pred, structure=struct)
    gt_labeled,   _ = nd_label(gt,   structure=struct)

    # Work on the union of foreground voxels
    mask = pred.astype(bool) | gt.astype(bool)
    if not mask.any():
        return 0.0, 0.0

    pl = pred_labeled[mask].astype(np.int64)
    gl = gt_labeled[mask].astype(np.int64)
    n  = mask.sum()

    # Joint count matrix (sparse via np.unique on paired labels)
    pairs, counts = np.unique(
        np.stack([gl, pl], axis=1), axis=0, return_counts=True
    )
    p_ij = counts / n

    # Marginals
    gt_ids, gt_counts = np.unique(gl, return_counts=True)
    pr_ids, pr_counts = np.unique(pl, return_counts=True)
    p_gt = dict(zip(gt_ids.tolist(), (gt_counts / n).tolist()))
    p_pr = dict(zip(pr_ids.tolist(), (pr_counts / n).tolist()))

    eps = 1e-12
    voi_split = 0.0   # H(GT | Pred)
    voi_merge = 0.0   # H(Pred | GT)

    for (gi, pi), pij in zip(pairs.tolist(), p_ij.tolist()):
        if pij < eps:
            continue
        voi_split += pij * np.log2(p_pr[pi] / (pij + eps) + eps)
        voi_merge += pij * np.log2(p_gt[gi] / (pij + eps) + eps)

    return float(max(voi_split, 0.0)), float(max(voi_merge, 0.0))


def voi_score(pred: np.ndarray, gt: np.ndarray,
              alpha: float = 0.3, connectivity: int = 1) -> float:
    """
    Scalar VOI score used in the composite metric.

        V = 1 / (1 + alpha * (VOI_split + VOI_merge))
    """
    split, merge = voi_split_merge(pred, gt, connectivity=connectivity)
    return 1.0 / (1.0 + alpha * (split + merge))


def composite_score(
    pred: np.ndarray,
    gt: np.ndarray,
    topo_score: float = 0.0,
    tau: float = 2.0,
    spacing: tuple[float, ...] | None = None,
    alpha: float = 0.3,
    connectivity: int = 1,
) -> dict[str, float]:
    """
    Compute all metric components and the composite score.

        S = 0.30 * T  +  0.35 * D_tau  +  0.35 * V

    TopoScore T must be supplied externally (competition evaluator).
    Defaults to 0.0 if not available.

    Returns a dict with keys: surfdice, voi_split, voi_merge, voi, topo,
    composite.
    """
    sd    = surface_dice_tau(pred, gt, tau=tau, spacing=spacing)
    v     = voi_score(pred, gt, alpha=alpha, connectivity=connectivity)
    split, merge = voi_split_merge(pred, gt, connectivity=connectivity)
    s     = 0.30 * topo_score + 0.35 * sd + 0.35 * v

    return {
        "surfdice":  sd,
        "voi_split": split,
        "voi_merge": merge,
        "voi":       v,
        "topo":      topo_score,
        "composite": s,
    }
