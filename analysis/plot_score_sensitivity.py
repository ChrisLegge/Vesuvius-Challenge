"""
Score sensitivity analysis for the Vesuvius composite metric.

Answers the question: why not just optimise Dice?

Three figures show the properties of S = 0.30*T + 0.35*D_tau + 0.35*V
that make it fundamentally different from standard segmentation metrics:

  fig8  - Threshold sensitivity: S(t) curves showing how score varies
          with binarisation threshold. Well-calibrated models have a wide
          plateau; poorly calibrated models have a sharp spike.

  fig9  - Bridge event curve: S vs number of bridges. The discontinuous
          nature of VOI and TopoScore means one bridge can drop S by 0.15+
          while SurfaceDice barely moves.

  fig10 - Dice vs composite scatter: two models with the same Dice score
          can have very different composite scores depending on topology.

No GPU, no competition data required.

Usage:
    python analysis/plot_score_sensitivity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.ndimage import binary_fill_holes, label as nd_label

REPO_ROOT  = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from vesuvius.metrics import surface_dice_tau, voi_score, composite_score
from vesuvius.postprocess import hysteresis_components

OUTPUT_DIR = REPO_ROOT / "results" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
})

# ---------------------------------------------------------------------------
# Synthetic helpers
# ---------------------------------------------------------------------------

def _two_surface_gt(H=80, W=80):
    """Ground truth: two clean horizontal surfaces."""
    gt = np.zeros((H, W), dtype=np.uint8)
    gt[15:22, 5:75] = 1
    gt[58:65, 5:75] = 1
    return gt


def _prob_map_from_gt(gt, sharpness=6.0, rng=None):
    """
    Smooth probability map centred on the GT surfaces.
    sharpness controls how peaked the probability is.
    """
    from scipy.ndimage import distance_transform_edt
    if rng is None:
        rng = np.random.default_rng(0)
    dist = distance_transform_edt(~gt.astype(bool))
    probs = np.exp(-dist / sharpness)
    probs = np.clip(probs + rng.normal(0, 0.02, probs.shape), 0, 1)
    return probs


def _add_bridge(probs, row_start=22, row_end=58, col=40, width=3,
                confidence=0.72):
    """Insert a thin bridge between the two surfaces in the probability map."""
    out = probs.copy()
    half = width // 2
    out[row_start:row_end, col - half: col + half + 1] = confidence
    return out


def _threshold_to_binary(probs, t):
    return (probs >= t).astype(np.uint8)


def _dice(pred, gt):
    inter = (pred.astype(bool) & gt.astype(bool)).sum()
    denom = pred.sum() + gt.sum()
    return 2 * inter / denom if denom > 0 else 1.0


# ---------------------------------------------------------------------------
# Figure 8 — Threshold sensitivity curves S(t)
# ---------------------------------------------------------------------------

def fig8_threshold_sensitivity() -> None:
    gt    = _two_surface_gt()
    rng   = np.random.default_rng(42)

    # Well-calibrated: sharp, centred near the surface, flat region around 0.5
    probs_good = _prob_map_from_gt(gt, sharpness=4.0, rng=rng)

    # Poorly calibrated: overconfident — probabilities pushed to extremes
    probs_poor = np.clip(
        _prob_map_from_gt(gt, sharpness=8.0, rng=np.random.default_rng(7)),
        0, 1
    )
    probs_poor = np.where(probs_poor > 0.3, np.clip(probs_poor * 1.6, 0, 1),
                          probs_poor * 0.4)

    # Temperature-scaled (T=0.85 applied to logits): intermediate
    logits_raw = np.log(np.clip(probs_good, 1e-6, 1-1e-6) /
                        np.clip(1 - probs_good, 1e-6, 1-1e-6))
    probs_temp = 1 / (1 + np.exp(-logits_raw / 0.85))

    thresholds = np.linspace(0.15, 0.85, 60)
    models = {
        "Well-calibrated (T=1.0)":     probs_good,
        "Temperature-scaled (T=0.85)": probs_temp,
        "Overconfident (T=1.0)":       probs_poor,
    }
    colours = ["#2166ac", "#4dac26", "#d6604d"]
    linestyles = ["-", "--", ":"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle(
        "Threshold sensitivity — S(t) curves for different calibration regimes",
        fontsize=10,
    )

    metrics = ["composite", "surfdice", "voi"]
    ylabels = ["Composite  S", "SurfaceDice  D_tau", "VOI score  V"]
    for ax, metric, ylabel in zip(axes, metrics, ylabels):
        for (label, probs), col, ls in zip(models.items(), colours, linestyles):
            scores = []
            for t in thresholds:
                pred = _threshold_to_binary(probs, t)
                r = composite_score(pred, gt)
                scores.append(r[metric])
            ax.plot(thresholds, scores, color=col, linestyle=ls,
                    linewidth=1.6, label=label)

        ax.set_xlabel("Threshold  t")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0.15, 0.85)
        ax.set_ylim(0, 1.05)
        if metric == "composite":
            ax.legend(fontsize=7.5)
            ax.axvspan(0.35, 0.65, color="#ffffcc", alpha=0.5, zorder=0,
                       label="hysteresis range")
            ax.text(0.50, 0.06, "hysteresis\nrange", ha="center",
                    fontsize=7, color="#888800")

    fig.tight_layout()
    out = OUTPUT_DIR / "fig8_threshold_sensitivity.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Figure 9 — Bridge event curve: S vs number of bridges
# ---------------------------------------------------------------------------

def fig9_bridge_event_curve() -> None:
    gt      = _two_surface_gt()
    probs   = _prob_map_from_gt(gt, sharpness=4.0, rng=np.random.default_rng(0))

    bridge_cols   = [20, 35, 50, 65]
    bridge_widths = [3, 2, 3, 2]

    n_bridges_list = list(range(0, len(bridge_cols) + 1))
    results = {k: [] for k in ("composite", "surfdice", "voi", "dice")}

    for n in n_bridges_list:
        p = probs.copy()
        for i in range(n):
            p = _add_bridge(p, col=bridge_cols[i], width=bridge_widths[i],
                            confidence=0.72)
        pred = _threshold_to_binary(p, t=0.50)
        r    = composite_score(pred, gt)
        for k in ("composite", "surfdice", "voi"):
            results[k].append(r[k])
        results["dice"].append(_dice(pred, gt))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(
        "Bridge event curve — effect of additional bridges on metric components",
        fontsize=10,
    )

    ax = axes[0]
    ax.plot(n_bridges_list, results["composite"], "o-", color="#2166ac",
            linewidth=2, markersize=6, label="Composite  S")
    ax.plot(n_bridges_list, results["surfdice"],  "s--", color="#4dac26",
            linewidth=1.5, markersize=5, label="SurfaceDice  D_tau")
    ax.plot(n_bridges_list, results["voi"],       "^--", color="#d6604d",
            linewidth=1.5, markersize=5, label="VOI score  V")
    ax.set_xlabel("Number of bridges added to prediction")
    ax.set_ylabel("Score (0-1)")
    ax.set_xticks(n_bridges_list)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.set_title("Topology-sensitive metrics vs bridge count", fontsize=9)

    ax = axes[1]
    ax.plot(n_bridges_list, results["dice"],      "o-", color="#762a83",
            linewidth=2, markersize=6, label="Dice coefficient")
    ax.plot(n_bridges_list, results["composite"], "s--", color="#2166ac",
            linewidth=1.5, markersize=5, label="Composite  S")

    # Annotate the drop from 0 to 1 bridge
    delta_s    = results["composite"][0] - results["composite"][1]
    delta_dice = results["dice"][0] - results["dice"][1]
    ax.annotate(
        f"S drops {delta_s:.3f}",
        xy=(1, results["composite"][1]),
        xytext=(1.3, results["composite"][1] - 0.08),
        fontsize=7.5, color="#2166ac",
        arrowprops=dict(arrowstyle="-|>", color="#2166ac", lw=0.9),
    )
    ax.annotate(
        f"Dice drops {delta_dice:.3f}",
        xy=(1, results["dice"][1]),
        xytext=(1.3, results["dice"][1] + 0.05),
        fontsize=7.5, color="#762a83",
        arrowprops=dict(arrowstyle="-|>", color="#762a83", lw=0.9),
    )
    ax.set_xlabel("Number of bridges added to prediction")
    ax.set_ylabel("Score (0-1)")
    ax.set_xticks(n_bridges_list)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.set_title("Dice vs composite: divergence under topology failure", fontsize=9)

    fig.tight_layout()
    out = OUTPUT_DIR / "fig9_bridge_event_curve.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Figure 10 — Dice vs composite scatter: same Dice, different S
# ---------------------------------------------------------------------------

def fig10_dice_vs_composite() -> None:
    """
    Generate a family of predictions that vary topology (bridges, splits,
    dust, holes) while keeping Dice in a similar range. Shows that Dice
    alone is insufficient to characterise segmentation quality.
    """
    gt  = _two_surface_gt()
    rng = np.random.default_rng(99)

    records = []

    # Base predictions at varying thresholds (covers a Dice range)
    for sharpness in [2.5, 3.5, 5.0, 7.0]:
        probs = _prob_map_from_gt(gt, sharpness=sharpness, rng=rng)
        for t in np.linspace(0.25, 0.75, 18):
            pred = _threshold_to_binary(probs, t)
            r    = composite_score(pred, gt)
            records.append({"dice": _dice(pred, gt),
                            "composite": r["composite"],
                            "tag": "threshold variation"})

    # Add bridges at best-Dice threshold
    probs_base = _prob_map_from_gt(gt, sharpness=4.0,
                                    rng=np.random.default_rng(0))
    best_t     = max(np.linspace(0.3, 0.7, 30),
                     key=lambda t: _dice(_threshold_to_binary(probs_base, t), gt))
    for n_b in range(1, 5):
        p = probs_base.copy()
        for i in range(n_b):
            p = _add_bridge(p, col=20 + i * 15, width=2, confidence=0.68)
        pred = _threshold_to_binary(p, best_t)
        r    = composite_score(pred, gt)
        records.append({"dice": _dice(pred, gt), "composite": r["composite"],
                        "tag": "bridge"})

    # Add fragmentation: erase random segments
    probs_base2 = _prob_map_from_gt(gt, sharpness=4.0,
                                     rng=np.random.default_rng(5))
    for i in range(6):
        p  = probs_base2.copy()
        c  = rng.integers(10, 70)
        w  = rng.integers(3, 12)
        p[14:23, c:c+w] *= 0.2   # suppress surface 1 patch
        pred = _threshold_to_binary(p, 0.45)
        r    = composite_score(pred, gt)
        records.append({"dice": _dice(pred, gt), "composite": r["composite"],
                        "tag": "split"})

    tags    = ["threshold variation", "bridge", "split"]
    colours = {"threshold variation": "#aaaaaa",
               "bridge": "#d6604d",
               "split":  "#4393c3"}
    markers = {"threshold variation": ".", "bridge": "^", "split": "s"}
    sizes   = {"threshold variation": 20,  "bridge": 60,  "split": 50}

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle(
        "Dice coefficient vs composite score S\n"
        "Same Dice range, different topology error types",
        fontsize=10,
    )

    for tag in tags:
        pts = [r for r in records if r["tag"] == tag]
        ax.scatter([p["dice"] for p in pts],
                   [p["composite"] for p in pts],
                   c=colours[tag], marker=markers[tag],
                   s=sizes[tag], label=tag, alpha=0.75, zorder=3)

    # Identity line: if composite == dice
    lims = [0.3, 1.0]
    ax.plot(lims, lims, "--", color="#cccccc", linewidth=1, zorder=1,
            label="Dice = composite (reference)")

    ax.set_xlabel("Dice coefficient")
    ax.set_ylabel("Composite score  S")
    ax.set_xlim(0.3, 1.02)
    ax.set_ylim(0.3, 1.02)
    ax.legend(fontsize=8, loc="lower right")

    # Annotate the key insight
    bridge_pts = [r for r in records if r["tag"] == "bridge"]
    if bridge_pts:
        p = max(bridge_pts, key=lambda r: r["dice"] - r["composite"])
        ax.annotate(
            "High Dice, low S\n(bridge present)",
            xy=(p["dice"], p["composite"]),
            xytext=(p["dice"] - 0.15, p["composite"] + 0.08),
            fontsize=8, color="#d6604d",
            arrowprops=dict(arrowstyle="-|>", color="#d6604d", lw=0.9),
        )

    fig.tight_layout()
    out = OUTPUT_DIR / "fig10_dice_vs_composite.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating score sensitivity figures...")
    fig8_threshold_sensitivity()
    fig9_bridge_event_curve()
    fig10_dice_vs_composite()
    print(f"\nAll figures written to {OUTPUT_DIR.relative_to(REPO_ROOT)}/")
