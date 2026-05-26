"""
3D surface visualisation for the Vesuvius segmentation pipeline.

Generates synthetic CT scroll volumes with two curved papyrus surfaces,
then shows both the raw probability volume and the effect of post-processing
(hysteresis thresholding + bridge-neck cutting) in 3D.

Five figures:

  fig11 - Volume cross-sections: three orthogonal slices through the synthetic
          probability volume, with GT surfaces overlaid in colour. Shows how
          the surfaces appear in the raw CT representation used as model input.

  fig12 - 3D surface mesh (GT): marching-cubes surface extraction on the clean
          GT binary volume. Shows the two curved scroll surfaces as triangulated
          meshes with depth shading.

  fig13 - 3D surface mesh (with bridge): same extraction but with a thin
          low-confidence bridge connecting the surfaces — the topology failure
          the pipeline is designed to remove.

  fig14 - Post-processing comparison: side-by-side 2D cross-section showing
          raw threshold, hysteresis result, and bridge-removed result with
          connected-component count annotated at each stage.

  fig15 - Metric decomposition volume: per-voxel contribution to SurfaceDice
          precision and recall, shown as a false-colour slice. Voxels within
          tau of the GT surface are highlighted; distant voxels are shown in red.

No GPU, no competition data required.

Usage:
    python analysis/plot_surface_3d.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.ndimage import (
    binary_erosion,
    distance_transform_edt,
    generate_binary_structure,
    label as nd_label,
)

REPO_ROOT  = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

OUTPUT_DIR = REPO_ROOT / "results" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ---------------------------------------------------------------------------
# Synthetic volume helpers
# ---------------------------------------------------------------------------

def _make_scroll_volume(
    D: int = 48, H: int = 64, W: int = 64,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (prob_volume, gt_volume) of shape (D, H, W).

    Two curved surfaces at roughly z=H//4 and z=3*H//4, each with a gentle
    sinusoidal warp in the W direction and additive noise. Probabilities decay
    with distance from the surface centre.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    gt = np.zeros((D, H, W), dtype=np.uint8)

    # Surface 1: row band centred at H//4 with slight curvature
    cx = np.linspace(-np.pi, np.pi, W)
    row1 = H // 4 + (3 * np.sin(cx)).astype(int)    # sinusoidal in W
    row2 = 3 * H // 4 + (3 * np.cos(cx)).astype(int)

    thickness = 3
    for w in range(W):
        r1 = int(np.clip(row1[w], 2, H - 3))
        r2 = int(np.clip(row2[w], 2, H - 3))
        gt[:, r1 - 1: r1 + thickness, w] = 1
        gt[:, r2 - 1: r2 + thickness, w] = 1

    # Probability map: distance falloff from GT surface centres
    from scipy.ndimage import distance_transform_edt
    dist = distance_transform_edt(~gt.astype(bool))
    probs = np.exp(-dist / 3.5)
    probs = np.clip(probs + rng.normal(0, 0.025, probs.shape), 0.0, 1.0)

    return probs.astype(np.float32), gt


def _add_bridge_3d(
    probs: np.ndarray,
    col: int | None = None,
    confidence: float = 0.70,
) -> np.ndarray:
    """Insert a thin vertical bridge connecting the two surfaces in the volume."""
    D, H, W = probs.shape
    if col is None:
        col = W // 2
    out = probs.copy()
    row1_centre = H // 4
    row2_centre = 3 * H // 4
    out[:, row1_centre + 2: row2_centre - 1, col - 1: col + 2] = confidence
    return out


def _hysteresis_threshold(probs: np.ndarray, t_low: float = 0.35,
                           t_high: float = 0.65) -> np.ndarray:
    """Simple 3D hysteresis thresholding."""
    seeds = probs >= t_high
    candidates = probs >= t_low
    struct = generate_binary_structure(3, 1)
    # Expand seeds within candidates (iterative dilation)
    from scipy.ndimage import binary_dilation
    prev = seeds
    while True:
        expanded = binary_dilation(prev, structure=struct) & candidates
        if (expanded == prev).all():
            break
        prev = expanded
    return prev.astype(np.uint8)


def _cut_bridge_necks(binary: np.ndarray, min_thickness: int = 4) -> np.ndarray:
    """
    Remove thin necks by eroding to the skeleton, then keeping only
    components that survive erosion at min_thickness radius.
    """
    struct = generate_binary_structure(3, 1)
    eroded = binary.copy().astype(bool)
    for _ in range(min_thickness):
        eroded = binary_erosion(eroded, structure=struct)
    labeled, n = nd_label(eroded, structure=struct)
    if n == 0:
        return binary
    # Keep only foreground touching a surviving component
    keep = np.zeros_like(binary, dtype=bool)
    dist = distance_transform_edt(~eroded)
    keep = dist <= min_thickness
    return (binary.astype(bool) & keep).astype(np.uint8)


# ---------------------------------------------------------------------------
# Figure 11 — Orthogonal cross-sections through the probability volume
# ---------------------------------------------------------------------------

def fig11_cross_sections() -> None:
    probs, gt = _make_scroll_volume()
    D, H, W = probs.shape

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle(
        "Synthetic scroll CT — orthogonal cross-sections with GT surface overlay",
        fontsize=10,
    )

    slices = [
        ("XY plane  (D={}//2)".format(D), probs[D // 2],   gt[D // 2],   "XY"),
        ("XZ plane  (H={}//4)".format(H), probs[:, H // 4, :], gt[:, H // 4, :], "XZ"),
        ("YZ plane  (W={}//2)".format(W), probs[:, :, W // 2], gt[:, :, W // 2], "YZ"),
    ]

    for ax, (title, prob_slice, gt_slice, _) in zip(axes, slices):
        im = ax.imshow(prob_slice, cmap="gray", vmin=0, vmax=1, origin="lower",
                       interpolation="nearest")
        # Overlay GT surface boundary in yellow
        gt_surf = (gt_slice > 0)
        overlay = np.zeros((*gt_slice.shape, 4), dtype=np.float32)
        overlay[gt_surf, 0] = 1.0   # R
        overlay[gt_surf, 1] = 0.85  # G
        overlay[gt_surf, 3] = 0.6   # A
        ax.imshow(overlay, origin="lower", interpolation="nearest")
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("W axis")
        ax.set_ylabel("D / H axis")
        plt.colorbar(im, ax=ax, fraction=0.035, pad=0.03)

    fig.tight_layout()
    out = OUTPUT_DIR / "fig11_cross_sections.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Figure 12 & 13 — 3D surface meshes (GT clean vs bridge prediction)
# ---------------------------------------------------------------------------

def _render_surface_3d(
    ax,
    binary: np.ndarray,
    title: str,
    colour: str = "#4393c3",
    alpha: float = 0.55,
) -> None:
    """Render marching-cubes surface mesh on a 3D axes."""
    from skimage.measure import marching_cubes
    verts, faces, normals, _ = marching_cubes(
        binary.astype(np.float32), level=0.5, spacing=(1.0, 1.0, 1.0)
    )
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    mesh = [[verts[j] for j in face] for face in faces]
    poly = Poly3DCollection(mesh, alpha=alpha, linewidth=0)
    poly.set_facecolor(colour)
    ax.add_collection3d(poly)

    # Axis limits
    D, H, W = binary.shape
    ax.set_xlim(0, D); ax.set_ylim(0, H); ax.set_zlim(0, W)
    ax.set_xlabel("D"); ax.set_ylabel("H"); ax.set_zlabel("W")
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=6)


def fig12_13_surface_meshes() -> None:
    probs_clean, gt = _make_scroll_volume()
    probs_bridge = _add_bridge_3d(probs_clean)

    pred_clean  = (probs_clean  >= 0.50).astype(np.uint8)
    pred_bridge = (probs_bridge >= 0.50).astype(np.uint8)

    _, n_clean  = nd_label(pred_clean,  structure=generate_binary_structure(3, 1))
    _, n_bridge = nd_label(pred_bridge, structure=generate_binary_structure(3, 1))

    fig = plt.figure(figsize=(13, 5))
    fig.suptitle(
        "3D surface meshes — marching cubes on binarised prediction",
        fontsize=10,
    )

    for idx, (binary, title, col, n_comp) in enumerate([
        (gt,           "Ground truth\n(2 components)",          "#2166ac", 2),
        (pred_clean,   f"Clean prediction\n({n_clean} component{'s' if n_clean != 1 else ''})",
                                                                "#4dac26", n_clean),
        (pred_bridge,  f"Bridge prediction\n({n_bridge} component{'s' if n_bridge != 1 else ''})",
                                                                "#d6604d", n_bridge),
    ]):
        ax = fig.add_subplot(1, 3, idx + 1, projection="3d")
        try:
            _render_surface_3d(ax, binary, title, colour=col)
        except Exception:
            ax.text(0.5, 0.5, 0.5, "No surface\n(empty mask)",
                    ha="center", va="center", fontsize=8)
            ax.set_title(title, fontsize=8)

    fig.tight_layout()
    out = OUTPUT_DIR / "fig12_surface_meshes.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Figure 14 — Post-processing pipeline: 2D cross-section at each stage
# ---------------------------------------------------------------------------

def fig14_postprocess_pipeline_2d() -> None:
    probs, gt = _make_scroll_volume()
    probs_bridge = _add_bridge_3d(probs)

    # Take the central XY slice for illustration
    D = probs.shape[0]
    slice_idx = D // 2
    p2d  = probs_bridge[slice_idx]
    gt2d = gt[slice_idx]

    naive     = (p2d >= 0.50).astype(np.uint8)
    hyst      = _hysteresis_threshold(probs_bridge, t_low=0.35, t_high=0.65)[slice_idx]
    cleaned   = _cut_bridge_necks(
        _hysteresis_threshold(probs_bridge, t_low=0.35, t_high=0.65), min_thickness=3
    )[slice_idx]

    def _n_comp(mask):
        _, n = nd_label(mask.astype(bool),
                        structure=generate_binary_structure(2, 1))
        return n

    stages = [
        ("GT surface",           gt2d,   "#2166ac"),
        ("Naive threshold t=0.50\n({} components)".format(_n_comp(naive)),
                                 naive,  "#d6604d"),
        ("Hysteresis [0.35, 0.65]\n({} components)".format(_n_comp(hyst)),
                                 hyst,   "#f4a582"),
        ("+ Bridge-neck cut\n({} components)".format(_n_comp(cleaned)),
                                 cleaned, "#4dac26"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    fig.suptitle(
        "Post-processing pipeline — XY slice through bridge case",
        fontsize=10,
    )
    for ax, (title, mask, col) in zip(axes, stages):
        ax.imshow(p2d, cmap="gray", vmin=0, vmax=1, origin="lower",
                  interpolation="nearest")
        overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
        rgb = tuple(int(col.lstrip("#")[i:i+2], 16) / 255 for i in (0, 2, 4))
        overlay[mask > 0, 0] = rgb[0]
        overlay[mask > 0, 1] = rgb[1]
        overlay[mask > 0, 2] = rgb[2]
        overlay[mask > 0, 3] = 0.65
        ax.imshow(overlay, origin="lower", interpolation="nearest")
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("W"); ax.set_ylabel("H")
        ax.grid(False)

    fig.tight_layout()
    out = OUTPUT_DIR / "fig14_postprocess_pipeline_2d.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Figure 15 — Per-voxel SurfaceDice contribution map
# ---------------------------------------------------------------------------

def fig15_surfdice_contribution() -> None:
    """
    False-colour map showing which predicted surface voxels fall within
    tau of the GT surface (precision) and which GT surface voxels are within
    tau of the prediction (recall). Voxels outside tolerance are shown in red.
    """
    from scipy.ndimage import binary_erosion
    probs, gt = _make_scroll_volume()
    probs_bridge = _add_bridge_3d(probs)
    D = probs.shape[0]
    sl = D // 2

    pred2d = (probs_bridge[sl] >= 0.50).astype(bool)
    gt2d   = gt[sl].astype(bool)

    tau = 2.0

    def surface_2d(b):
        struct = generate_binary_structure(2, 1)
        interior = binary_erosion(b, structure=struct, border_value=0)
        return b & ~interior

    surf_pred = surface_2d(pred2d)
    surf_gt   = surface_2d(gt2d)

    dist_to_gt   = distance_transform_edt(~surf_gt)
    dist_to_pred = distance_transform_edt(~surf_pred)

    prec_within = surf_pred & (dist_to_gt   <= tau)
    prec_outside = surf_pred & (dist_to_gt  >  tau)
    rec_within  = surf_gt   & (dist_to_pred <= tau)
    rec_outside = surf_gt   & (dist_to_pred >  tau)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"SurfaceDice@tau={tau} — per-voxel tolerance map  (XY slice, bridge prediction)",
        fontsize=10,
    )

    prob_slice = probs_bridge[sl]

    titles = ["Precision: predicted surface voxels",
              "Recall: GT surface voxels"]
    within_masks  = [prec_within,  rec_within]
    outside_masks = [prec_outside, rec_outside]

    for ax, title, w_mask, o_mask in zip(axes, titles, within_masks, outside_masks):
        ax.imshow(prob_slice, cmap="gray", vmin=0, vmax=1, origin="lower",
                  interpolation="nearest")

        # Within tau → green
        ov_g = np.zeros((*prob_slice.shape, 4), dtype=np.float32)
        ov_g[w_mask, 1] = 0.9
        ov_g[w_mask, 3] = 0.8
        ax.imshow(ov_g, origin="lower", interpolation="nearest")

        # Outside tau → red
        ov_r = np.zeros((*prob_slice.shape, 4), dtype=np.float32)
        ov_r[o_mask, 0] = 0.9
        ov_r[o_mask, 3] = 0.8
        ax.imshow(ov_r, origin="lower", interpolation="nearest")

        n_within  = w_mask.sum()
        n_outside = o_mask.sum()
        frac = n_within / (n_within + n_outside + 1e-9)
        ax.set_title(
            f"{title}\n"
            f"within tau: {n_within}  |  outside: {n_outside}  |  "
            f"fraction: {frac:.3f}",
            fontsize=8,
        )
        ax.set_xlabel("W"); ax.set_ylabel("H")
        ax.grid(False)

    # Colour legend
    for ax in axes:
        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(facecolor="#00cc44", alpha=0.8, label=f"Within tau={tau}"),
            Patch(facecolor="#cc0000", alpha=0.8, label=f"Outside tau={tau}"),
        ], fontsize=7, loc="upper right")

    fig.tight_layout()
    out = OUTPUT_DIR / "fig15_surfdice_contribution.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating 3D surface visualisation figures...")
    fig11_cross_sections()
    fig12_13_surface_meshes()
    fig14_postprocess_pipeline_2d()
    fig15_surfdice_contribution()
    print(f"\nAll figures written to {OUTPUT_DIR.relative_to(REPO_ROOT)}/")
