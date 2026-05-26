"""
Topology post-processing visualisation for Vesuvius surface detection.

Generates synthetic probability maps that replicate the two structural failure
modes this system targets — bridges (VOI_merge / Topo k1 failures) and splits
(VOI_split / Topo k0 failures) — then shows the effect of each post-processing
stage on binary predictions.

No GPU, no competition data, no Kaggle environment required.

Usage:
    python analysis/plot_topology_postprocess.py

Output:
    results/figures/fig5_topology_bridge_case.png
    results/figures/fig6_topology_split_case.png
    results/figures/fig7_postprocess_pipeline.png
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from scipy.ndimage import (
    binary_fill_holes,
    distance_transform_edt,
    label as nd_label,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT  = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "results" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Probability colourmap: dark blue (0) -> white (0.5) -> dark red (1)
PROB_CMAP = LinearSegmentedColormap.from_list(
    "prob", ["#053061", "#f7f7f7", "#67001f"], N=256
)
# Binary colourmap: background grey, foreground teal
BIN_CMAP = LinearSegmentedColormap.from_list(
    "bin", ["#f0f0f0", "#1a7abd"], N=2
)
OVERLAY_CMAP = LinearSegmentedColormap.from_list(
    "overlay", ["#f0f0f0", "#c0392b"], N=2
)

# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------

def _gaussian_blob(grid_y, grid_x, cy, cx, sy, sx, amplitude=1.0):
    return amplitude * np.exp(
        -0.5 * ((grid_y - cy) / sy) ** 2
        - 0.5 * ((grid_x - cx) / sx) ** 2
    )


def make_bridge_case(H=96, W=96):
    """
    Two parallel horizontal surfaces separated by a narrow gap.
    A thin low-confidence bridge connects them near the centre.
    This is the VOI_merge failure mode: naive thresholding merges
    two surfaces that should be topologically distinct.
    """
    y, x = np.mgrid[0:H, 0:W]

    # Surface 1: band around row 28
    s1 = np.clip(
        1.2 * np.exp(-0.5 * ((y - 28) / 3.5) ** 2), 0, 1
    )
    # Surface 2: band around row 58
    s2 = np.clip(
        1.2 * np.exp(-0.5 * ((y - 58) / 3.5) ** 2), 0, 1
    )
    # Bridge: faint vertical connector at x=50, linking rows 31-55
    bridge_mask = (x > 44) & (x < 56) & (y > 30) & (y < 56)
    bridge = 0.52 * bridge_mask.astype(float)  # just above naive threshold 0.5
    bridge *= np.exp(-0.5 * ((x - 50) / 4.0) ** 2)  # taper width

    probs = np.clip(s1 + s2 + bridge, 0, 1)

    # Add realistic noise
    rng = np.random.default_rng(0)
    probs = np.clip(probs + rng.normal(0, 0.03, probs.shape), 0, 1)

    return probs


def make_split_case(H=96, W=96):
    """
    A single continuous horizontal surface with a low-confidence gap
    at the centre-left — a shadow artefact typical in scroll CT.
    Naive thresholding breaks the surface into two fragments.
    This is the VOI_split / Topo k0 failure mode.
    """
    y, x = np.mgrid[0:H, 0:W]

    # Continuous surface at row 45, high confidence on both sides
    surface = np.clip(
        1.2 * np.exp(-0.5 * ((y - 45) / 3.5) ** 2), 0, 1
    )
    # Attenuation region: low signal between x=30-50 (CT shadow)
    attenuation = 1.0 - 0.72 * np.exp(-0.5 * ((x - 40) / 8.0) ** 2)
    probs = surface * attenuation

    rng = np.random.default_rng(1)
    probs = np.clip(probs + rng.normal(0, 0.025, probs.shape), 0, 1)

    return probs


# ---------------------------------------------------------------------------
# Post-processing algorithms
# ---------------------------------------------------------------------------

def threshold_simple(probs: np.ndarray, t: float = 0.5) -> np.ndarray:
    return (probs >= t).astype(np.uint8)


def hysteresis_threshold(probs: np.ndarray,
                         t_low: float = 0.35,
                         t_high: float = 0.65) -> np.ndarray:
    """
    Keep weak-mask components only if they contain at least one
    strong-seed voxel. 8-connectivity (2D).
    Mirrors the algorithm in src/vesuvius/postprocess.py.
    """
    H, W = probs.shape
    strong = probs >= t_high
    weak   = probs >= t_low
    output = np.zeros_like(probs, dtype=np.uint8)
    visited = np.zeros_like(probs, dtype=bool)

    for r in range(H):
        for c in range(W):
            if visited[r, c] or not weak[r, c]:
                continue
            queue     = deque([(r, c)])
            component = []
            touches   = False
            visited[r, c] = True
            while queue:
                cr, cc = queue.popleft()
                component.append((cr, cc))
                if strong[cr, cc]:
                    touches = True
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nr, nc = cr + dr, cc + dc
                        if (0 <= nr < H and 0 <= nc < W
                                and not visited[nr, nc]
                                and weak[nr, nc]):
                            visited[nr, nc] = True
                            queue.append((nr, nc))
            if touches:
                for cr, cc in component:
                    output[cr, cc] = 1
    return output


def remove_dust(binary: np.ndarray, min_size: int = 40) -> np.ndarray:
    """Remove connected components smaller than min_size pixels."""
    labeled, n = nd_label(binary)
    out = np.zeros_like(binary)
    for i in range(1, n + 1):
        if (labeled == i).sum() >= min_size:
            out[labeled == i] = 1
    return out


def cut_bridge_necks(binary: np.ndarray,
                     neck_thickness: float = 3.0) -> np.ndarray:
    """
    Identify thin-neck voxels via distance transform and remove them
    if cutting increases component count without creating dust.

    Simplified 2D version of the 3D bridge-cutting step used in inference.
    The full 3D version uses skeleton thinning + local thickness; this
    approximation uses the distance transform directly.
    """
    if binary.sum() == 0:
        return binary.copy()

    dist = distance_transform_edt(binary)
    thin_mask = binary.astype(bool) & (dist <= neck_thickness)

    candidate = binary.copy()
    candidate[thin_mask] = 0
    candidate = remove_dust(candidate, min_size=20)

    _, n_before = nd_label(binary)
    _, n_after  = nd_label(candidate)

    # Accept if: more components (bridge cut) and no explosion (< 10x)
    if n_after > n_before and n_after < n_before * 10:
        return candidate.astype(np.uint8)
    return binary.copy()


# ---------------------------------------------------------------------------
# Helper: annotate component count
# ---------------------------------------------------------------------------

def annotate_components(ax, binary, color="black"):
    _, n = nd_label(binary)
    ax.text(
        0.03, 0.97, f"components: {n}",
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=7.5, color=color,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, lw=0),
    )


def add_panel_label(ax, label, color="white"):
    ax.text(
        0.03, 0.03, label,
        transform=ax.transAxes,
        va="bottom", ha="left",
        fontsize=8, fontweight="bold", color=color,
    )


# ---------------------------------------------------------------------------
# Figure 5 — Bridge case
# ---------------------------------------------------------------------------

def fig5_bridge_case() -> None:
    probs = make_bridge_case()
    naive = threshold_simple(probs, t=0.50)
    hyst  = hysteresis_threshold(probs, t_low=0.35, t_high=0.65)
    cut   = cut_bridge_necks(hyst, neck_thickness=3.0)
    cut   = remove_dust(cut, min_size=40)

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    fig.suptitle(
        "Bridge failure mode — VOI_merge / Topo k1\n"
        "Two distinct surfaces connected by a thin low-confidence bridge",
        fontsize=9,
    )

    titles = [
        "Raw probability map",
        "Naive threshold  t=0.50",
        "Hysteresis  t_low=0.35  t_high=0.65",
        "After bridge-neck cutting",
    ]
    imgs = [probs, naive, hyst, cut]
    cmaps = [PROB_CMAP, BIN_CMAP, BIN_CMAP, BIN_CMAP]

    for ax, img, title, cmap in zip(axes, imgs, titles, cmaps):
        ax.imshow(img, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        if img is not probs:
            annotate_components(ax, img)

    # Mark bridge region on naive result
    axes[1].axhline(30, color="#e74c3c", linewidth=0.8, linestyle="--", alpha=0.7)
    axes[1].axhline(56, color="#e74c3c", linewidth=0.8, linestyle="--", alpha=0.7)
    axes[1].text(50, 43, "bridge", color="#e74c3c", fontsize=7,
                 ha="center", va="center")

    # Colourbar for probability panel
    cb = plt.colorbar(axes[0].images[0], ax=axes[0], fraction=0.046, pad=0.04)
    cb.set_label("p(surface)", fontsize=7)
    cb.ax.tick_params(labelsize=7)

    for i, ax in enumerate(axes):
        add_panel_label(ax, f"({chr(65+i)})")

    fig.tight_layout()
    out = OUTPUT_DIR / "fig5_topology_bridge_case.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Figure 6 — Split case
# ---------------------------------------------------------------------------

def fig6_split_case() -> None:
    probs = make_split_case()
    naive = threshold_simple(probs, t=0.50)
    hyst  = hysteresis_threshold(probs, t_low=0.35, t_high=0.65)
    filled = binary_fill_holes(hyst).astype(np.uint8)
    filled = remove_dust(filled, min_size=40)

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    fig.suptitle(
        "Split failure mode — VOI_split / Topo k0\n"
        "Single continuous surface fragmented by a low-signal CT attenuation region",
        fontsize=9,
    )

    titles = [
        "Raw probability map",
        "Naive threshold  t=0.50",
        "Hysteresis  t_low=0.35  t_high=0.65",
        "After gap fill / dust removal",
    ]
    imgs = [probs, naive, hyst, filled]
    cmaps = [PROB_CMAP, BIN_CMAP, BIN_CMAP, BIN_CMAP]

    for ax, img, title, cmap in zip(axes, imgs, titles, cmaps):
        ax.imshow(img, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        if img is not probs:
            annotate_components(ax, img)

    # Annotate the attenuation gap
    for ax in axes[:3]:
        ax.axvline(30, color="#e74c3c", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.axvline(52, color="#e74c3c", linewidth=0.8, linestyle="--", alpha=0.6)
    axes[0].text(41, 8, "attenuation\nregion", color="#e74c3c", fontsize=6.5,
                 ha="center", va="top")

    cb = plt.colorbar(axes[0].images[0], ax=axes[0], fraction=0.046, pad=0.04)
    cb.set_label("p(surface)", fontsize=7)
    cb.ax.tick_params(labelsize=7)

    for i, ax in enumerate(axes):
        add_panel_label(ax, f"({chr(65+i)})")

    fig.tight_layout()
    out = OUTPUT_DIR / "fig6_topology_split_case.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Figure 7 — Full pipeline overview: 3x2 grid showing both failure modes
#             side by side through each processing stage
# ---------------------------------------------------------------------------

def fig7_pipeline_overview() -> None:
    bridge_probs = make_bridge_case()
    split_probs  = make_split_case()

    b_naive = threshold_simple(bridge_probs, 0.50)
    b_hyst  = hysteresis_threshold(bridge_probs, 0.35, 0.65)
    b_final = remove_dust(cut_bridge_necks(b_hyst, 3.0), 40)

    s_naive = threshold_simple(split_probs, 0.50)
    s_hyst  = hysteresis_threshold(split_probs, 0.35, 0.65)
    s_final = remove_dust(binary_fill_holes(s_hyst).astype(np.uint8), 40)

    fig = plt.figure(figsize=(13, 7))
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.08)

    row_labels = ["Bridge case\n(VOI_merge / Topo k1)", "Split case\n(VOI_split / Topo k0)"]
    col_titles = [
        "Raw probability",
        "Naive  t=0.50",
        "Hysteresis\nt_low=0.35  t_high=0.65",
        "Final output",
    ]

    bridge_row = [bridge_probs, b_naive, b_hyst, b_final]
    split_row  = [split_probs,  s_naive, s_hyst, s_final]
    cmaps_row  = [PROB_CMAP, BIN_CMAP, BIN_CMAP, BIN_CMAP]

    for col_idx, (bp, sp, cmap) in enumerate(
            zip(bridge_row, split_row, cmaps_row)):

        ax_top = fig.add_subplot(gs[0, col_idx])
        ax_bot = fig.add_subplot(gs[1, col_idx])

        ax_top.imshow(bp, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        ax_bot.imshow(sp, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")

        for ax, img in [(ax_top, bp), (ax_bot, sp)]:
            ax.set_xticks([])
            ax.set_yticks([])
            if cmap is not PROB_CMAP:
                annotate_components(ax, img)

        if col_idx == 0:
            ax_top.set_title(col_titles[0], fontsize=8)
            ax_bot.set_title(col_titles[0], fontsize=8)
            ax_top.set_ylabel(row_labels[0], fontsize=8)
            ax_bot.set_ylabel(row_labels[1], fontsize=8)
        else:
            ax_top.set_title(col_titles[col_idx], fontsize=8)

    fig.suptitle(
        "Post-processing pipeline — bridge and split failure modes\n"
        "Component counts show topology correction at each stage",
        fontsize=10, y=1.01,
    )

    out = OUTPUT_DIR / "fig7_postprocess_pipeline.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating topology post-processing figures...")
    fig5_bridge_case()
    fig6_split_case()
    fig7_pipeline_overview()
    print(f"\nAll figures written to {OUTPUT_DIR.relative_to(REPO_ROOT)}/")
