"""
Calibration analysis for the Vesuvius probability outputs.

Answers: are the model's predicted probabilities trustworthy as probabilities?

A well-calibrated model satisfies E[Y | p(x)=p] = p — the average label
frequency in any probability bin equals the stated confidence. Poor calibration
means binarisation threshold choice is fragile and temperature scaling is
necessary.

Four figures:

  fig16 - Reliability diagrams: fraction-of-positives vs mean predicted
          probability, per calibration regime. Perfect calibration lies on the
          diagonal. Histogram bars show the fraction of predictions in each bin.

  fig17 - ECE vs temperature: Expected Calibration Error as a function of
          temperature scaling parameter T. Shows the optimal T and how much
          miscalibration is present before correction.

  fig18 - Sharpness vs calibration scatter: sharpness (mean predicted
          probability for positives) vs ECE across a sweep of model
          configurations. Illustrates the overconfidence-miscalibration
          relationship.

  fig19 - Calibration impact on composite score: composite S vs temperature T,
          showing the calibration-induced plateau widening from fig8 in
          quantitative terms (ECE on secondary axis).

No GPU, no competition data required.

Usage:
    python analysis/plot_calibration.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.ndimage import distance_transform_edt

REPO_ROOT  = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from vesuvius.metrics import composite_score

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
# Synthetic helpers — same GT/probs as sensitivity analysis for consistency
# ---------------------------------------------------------------------------

def _two_surface_gt(H: int = 80, W: int = 80) -> np.ndarray:
    gt = np.zeros((H, W), dtype=np.uint8)
    gt[15:22, 5:75] = 1
    gt[58:65, 5:75] = 1
    return gt


def _prob_map(gt: np.ndarray, sharpness: float = 6.0,
              rng: np.random.Generator | None = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng(0)
    dist = distance_transform_edt(~gt.astype(bool))
    probs = np.exp(-dist / sharpness)
    probs = np.clip(probs + rng.normal(0, 0.02, probs.shape), 0.0, 1.0)
    return probs.astype(np.float32)


def _apply_temperature(probs: np.ndarray, T: float) -> np.ndarray:
    """Platt / temperature scaling: divide logits by T before sigmoid."""
    eps = 1e-6
    logits = np.log(np.clip(probs, eps, 1 - eps) /
                    np.clip(1 - probs, eps, 1 - eps))
    return (1.0 / (1.0 + np.exp(-logits / T))).astype(np.float32)


def _make_overconfident(probs: np.ndarray) -> np.ndarray:
    """Push probabilities to extremes (simulates an overconfident model)."""
    return np.where(probs > 0.3,
                    np.clip(probs * 1.6, 0, 1),
                    probs * 0.4).astype(np.float32)


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------

def reliability_diagram_data(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """
    Compute reliability diagram quantities.

    Returns a dict with:
      bin_centres    : mid-point of each probability bin
      frac_positives : observed positive rate in each bin
      mean_pred      : mean predicted probability in each bin
      bin_counts     : number of samples in each bin
      ece            : Expected Calibration Error
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centres     = 0.5 * (bins[:-1] + bins[1:])
    frac_positives  = np.zeros(n_bins)
    mean_pred       = np.zeros(n_bins)
    bin_counts      = np.zeros(n_bins, dtype=int)

    p_flat = probs.ravel()
    y_flat = labels.ravel().astype(float)
    n_total = len(p_flat)

    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        mask = (p_flat >= lo) & (p_flat < hi)
        if i == n_bins - 1:         # include right edge in last bin
            mask = (p_flat >= lo) & (p_flat <= hi)
        n = mask.sum()
        bin_counts[i] = n
        if n > 0:
            frac_positives[i] = y_flat[mask].mean()
            mean_pred[i]      = p_flat[mask].mean()

    ece = float(
        np.sum(bin_counts * np.abs(frac_positives - mean_pred)) / n_total
    )

    return {
        "bin_centres":    bin_centres,
        "frac_positives": frac_positives,
        "mean_pred":      mean_pred,
        "bin_counts":     bin_counts,
        "ece":            ece,
    }


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray,
                                n_bins: int = 10) -> float:
    return reliability_diagram_data(probs, labels, n_bins)["ece"]


# ---------------------------------------------------------------------------
# Figure 16 — Reliability diagrams
# ---------------------------------------------------------------------------

def fig16_reliability_diagrams() -> None:
    gt   = _two_surface_gt()
    rng  = np.random.default_rng(42)

    probs_good = _prob_map(gt, sharpness=4.0, rng=rng)
    probs_poor = _make_overconfident(
        _prob_map(gt, sharpness=8.0, rng=np.random.default_rng(7))
    )
    probs_temp = _apply_temperature(probs_good, T=0.85)

    models = [
        ("Well-calibrated (T=1.0)",     probs_good, "#2166ac"),
        ("Temperature-scaled (T=0.85)", probs_temp, "#4dac26"),
        ("Overconfident (T=1.0)",       probs_poor, "#d6604d"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.suptitle(
        "Reliability diagrams — fraction of positives vs mean predicted probability",
        fontsize=10,
    )

    for ax, (label, probs, col) in zip(axes, models):
        rd = reliability_diagram_data(probs, gt, n_bins=10)

        # Histogram of predictions in each bin (background, normalised)
        ax2 = ax.twinx()
        ax2.bar(
            rd["bin_centres"],
            rd["bin_counts"] / rd["bin_counts"].sum(),
            width=0.09, color=col, alpha=0.15, zorder=1,
        )
        ax2.set_ylim(0, 0.5)
        ax2.set_ylabel("Fraction of predictions", fontsize=7, color="#999999")
        ax2.tick_params(axis="y", labelsize=6, colors="#999999")
        ax2.spines["top"].set_visible(False)

        # Calibration curve
        ax.plot([0, 1], [0, 1], "--", color="#aaaaaa", linewidth=1,
                label="Perfect calibration", zorder=2)
        ax.plot(rd["mean_pred"], rd["frac_positives"],
                "o-", color=col, linewidth=2, markersize=5,
                label=f"ECE = {rd['ece']:.4f}", zorder=3)

        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_title(label, fontsize=9)
        ax.legend(fontsize=7.5, loc="upper left")

        # Shade the gap between curve and diagonal
        ax.fill_between(
            rd["mean_pred"], rd["mean_pred"], rd["frac_positives"],
            alpha=0.12, color=col, zorder=2,
        )

    fig.tight_layout()
    out = OUTPUT_DIR / "fig16_reliability_diagrams.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Figure 17 — ECE vs temperature scaling parameter T
# ---------------------------------------------------------------------------

def fig17_ece_vs_temperature() -> None:
    gt   = _two_surface_gt()
    rng  = np.random.default_rng(42)

    probs_good = _prob_map(gt, sharpness=4.0, rng=rng)
    probs_poor = _make_overconfident(
        _prob_map(gt, sharpness=8.0, rng=np.random.default_rng(7))
    )

    temps = np.linspace(0.40, 2.50, 60)
    models = {
        "Well-calibrated":  (probs_good, "#2166ac"),
        "Overconfident":    (probs_poor, "#d6604d"),
    }

    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.suptitle(
        "Expected Calibration Error vs temperature scaling parameter T",
        fontsize=10,
    )

    for label, (probs, col) in models.items():
        ece_vals = [expected_calibration_error(_apply_temperature(probs, T), gt)
                    for T in temps]
        ax.plot(temps, ece_vals, "-", color=col, linewidth=2, label=label)

        # Mark optimal T
        best_idx = int(np.argmin(ece_vals))
        ax.scatter([temps[best_idx]], [ece_vals[best_idx]],
                   color=col, s=60, zorder=5)
        ax.annotate(
            f"T* = {temps[best_idx]:.2f}\nECE = {ece_vals[best_idx]:.4f}",
            xy=(temps[best_idx], ece_vals[best_idx]),
            xytext=(temps[best_idx] + 0.15, ece_vals[best_idx] + 0.003),
            fontsize=7.5, color=col,
            arrowprops=dict(arrowstyle="-|>", color=col, lw=0.8),
        )

    ax.axvline(1.0, color="#999999", linewidth=1, linestyle=":", label="T=1 (no scaling)")
    ax.axvline(0.85, color="#4dac26", linewidth=1, linestyle="--",
               label="T=0.85 (used in pipeline)")
    ax.set_xlabel("Temperature T")
    ax.set_ylabel("Expected Calibration Error (ECE)  ↓")
    ax.set_xlim(temps[0], temps[-1])
    ax.legend(fontsize=8)

    fig.tight_layout()
    out = OUTPUT_DIR / "fig17_ece_vs_temperature.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Figure 18 — Sharpness vs ECE scatter across model configurations
# ---------------------------------------------------------------------------

def fig18_sharpness_vs_ece() -> None:
    """
    Sweep sharpness and temperature to produce a family of (ECE, sharpness)
    points, labelled by calibration regime.
    """
    gt  = _two_surface_gt()
    rng = np.random.default_rng(0)

    records = []

    for sharpness in np.linspace(1.5, 10.0, 14):
        base = _prob_map(gt, sharpness=float(sharpness), rng=rng)
        # Baseline (T=1)
        ece  = expected_calibration_error(base, gt)
        sharp = float(base[gt.astype(bool)].mean())
        records.append({"ece": ece, "sharpness": sharp,
                        "tag": "baseline T=1", "T": 1.0})
        # Temperature-scaled
        for T in [0.7, 0.85, 1.2, 1.6]:
            scaled = _apply_temperature(base, T)
            ece_t  = expected_calibration_error(scaled, gt)
            sharp_t = float(scaled[gt.astype(bool)].mean())
            records.append({"ece": ece_t, "sharpness": sharp_t,
                            "tag": f"T={T}", "T": T})
        # Overconfident variant
        poor = _make_overconfident(base)
        records.append({"ece": expected_calibration_error(poor, gt),
                        "sharpness": float(poor[gt.astype(bool)].mean()),
                        "tag": "overconfident", "T": 1.0})

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.suptitle(
        "Sharpness vs Expected Calibration Error\n"
        "across sharpness × temperature configurations",
        fontsize=10,
    )

    tag_styles = {
        "baseline T=1":  ("#2166ac", "o", 45),
        "T=0.7":         ("#762a83", "s", 35),
        "T=0.85":        ("#4dac26", "s", 45),
        "T=1.2":         ("#f4a582", "s", 35),
        "T=1.6":         ("#d6604d", "s", 35),
        "overconfident": ("#d6604d", "^", 55),
    }

    for tag, (col, marker, size) in tag_styles.items():
        pts = [r for r in records if r["tag"] == tag]
        if pts:
            ax.scatter(
                [p["sharpness"] for p in pts],
                [p["ece"]       for p in pts],
                c=col, marker=marker, s=size, label=tag, alpha=0.8, zorder=3,
            )

    ax.set_xlabel("Sharpness (mean predicted probability on positive voxels)  ↑")
    ax.set_ylabel("Expected Calibration Error (ECE)  ↓")
    ax.legend(fontsize=7.5, ncol=2)

    # Annotate ideal region
    ax.axhspan(0, 0.02, color="#ffffcc", alpha=0.5, zorder=0)
    ax.text(0.52, 0.005, "Well-calibrated region  (ECE < 0.02)",
            fontsize=7, color="#888800")

    fig.tight_layout()
    out = OUTPUT_DIR / "fig18_sharpness_vs_ece.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Figure 19 — Calibration impact on composite score: S vs T with ECE overlay
# ---------------------------------------------------------------------------

def fig19_calibration_vs_composite() -> None:
    """
    Sweep temperature T. At each T, pick the best binarisation threshold
    (grid search over t in [0.15, 0.85]) and record S and ECE.
    Shows that improving calibration widens the plateau in composite score.
    """
    gt   = _two_surface_gt()
    rng  = np.random.default_rng(42)

    probs_good = _prob_map(gt, sharpness=4.0, rng=rng)
    probs_poor = _make_overconfident(
        _prob_map(gt, sharpness=8.0, rng=np.random.default_rng(7))
    )

    temps = np.linspace(0.40, 2.50, 50)
    thresh_grid = np.linspace(0.20, 0.80, 30)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    fig.suptitle(
        "Temperature scaling: composite score (best threshold) and ECE vs T",
        fontsize=10,
    )
    ax2 = ax1.twinx()

    for label, probs, col_s, col_e in [
        ("Well-calibrated", probs_good, "#2166ac", "#7fbfff"),
        ("Overconfident",   probs_poor, "#d6604d", "#ffb3a0"),
    ]:
        s_vals, ece_vals = [], []
        for T in temps:
            scaled = _apply_temperature(probs, T)
            best_s = max(
                composite_score((scaled >= t).astype(np.uint8), gt)["composite"]
                for t in thresh_grid
            )
            s_vals.append(best_s)
            ece_vals.append(expected_calibration_error(scaled, gt))

        ax1.plot(temps, s_vals,   "-",  color=col_s, linewidth=2,
                 label=f"{label} — S (best t)")
        ax2.plot(temps, ece_vals, "--", color=col_e, linewidth=1.5,
                 label=f"{label} — ECE")

    ax1.axvline(0.85, color="#4dac26", linewidth=1, linestyle=":",
                label="T=0.85 (pipeline)")
    ax1.set_xlabel("Temperature T")
    ax1.set_ylabel("Best composite score  S  ↑", color="#333333")
    ax2.set_ylabel("ECE  ↓", color="#888888")
    ax2.tick_params(axis="y", colors="#888888")
    ax1.set_xlim(temps[0], temps[-1])

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=7.5,
               loc="lower center", ncol=2)

    fig.tight_layout()
    out = OUTPUT_DIR / "fig19_calibration_vs_composite.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating calibration analysis figures...")
    fig16_reliability_diagrams()
    fig17_ece_vs_temperature()
    fig18_sharpness_vs_ece()
    fig19_calibration_vs_composite()
    print(f"\nAll figures written to {OUTPUT_DIR.relative_to(REPO_ROOT)}/")
