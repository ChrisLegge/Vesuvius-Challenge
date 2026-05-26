"""
Training curve analysis for the Vesuvius surface detection specialist ensemble.

Reads per-epoch training history from outputs/metadata/ and produces four
figures saved to results/figures/. No GPU, no competition data, no Kaggle
environment required.

Usage:
    python analysis/plot_training_curves.py

Output:
    results/figures/fig1_loss_trajectories.png
    results/figures/fig2_phase_gated_components.png
    results/figures/fig3_lr_schedules.png
    results/figures/fig4_model_comparison.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
METADATA_DIR = REPO_ROOT / "outputs" / "metadata"
OUTPUT_DIR = REPO_ROOT / "results" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Style — plain, no AI-default colour palette
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "legend.framealpha": 0.7,
    "legend.fontsize": 8,
})

MODEL_COLOURS = {
    "model_a": "#2166ac",   # blue   — generalist
    "model_b": "#d6604d",   # red    — anti-merge specialist
    "model_c": "#4dac26",   # green  — surface specialist
}

MODEL_LABELS = {
    "model_a": "Model A  generalist  (seed 42,   fold 0)",
    "model_b": "Model B  anti-merge  (seed 1337, fold 1)",
    "model_c": "Model C  surface     (seed 2024, fold 2)",
}

PHASE_COLOURS = {
    "early": "#f7f7f7",
    "mid":   "#e8f0fa",
    "late":  "#fde8e8",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_history(model: str) -> list[dict]:
    path = METADATA_DIR / f"{model}_history.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_meta(model: str) -> dict:
    path = METADATA_DIR / f"{model}_meta.json"
    return json.loads(path.read_text(encoding="utf-8"))


def phase_spans(history: list[dict]) -> list[tuple[str, int, int]]:
    """Return list of (phase_name, start_epoch, end_epoch) spans."""
    spans = []
    cur_phase = history[0]["phase"]
    cur_start = 0
    for i, row in enumerate(history[1:], 1):
        if row["phase"] != cur_phase:
            spans.append((cur_phase, cur_start, i - 1))
            cur_phase = row["phase"]
            cur_start = i
    spans.append((cur_phase, cur_start, len(history) - 1))
    return spans


def shade_phases(ax: plt.Axes, history: list[dict]) -> None:
    """Draw phase background bands on an axes."""
    for phase, start, end in phase_spans(history):
        ax.axvspan(start, end + 0.5, color=PHASE_COLOURS[phase], alpha=0.9, zorder=0)


# ---------------------------------------------------------------------------
# Figure 1 — Total loss trajectories, all three models
# ---------------------------------------------------------------------------

def fig1_loss_trajectories() -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=False)
    fig.suptitle("Training loss trajectories — specialist ensemble", fontsize=11, y=0.98)

    for ax, model in zip(axes, ["model_a", "model_b", "model_c"]):
        history = load_history(model)
        meta    = load_meta(model)
        epochs  = [r["epoch"] for r in history]
        losses  = [r["loss"]  for r in history]

        shade_phases(ax, history)
        ax.plot(epochs, losses, color=MODEL_COLOURS[model], linewidth=1.6, zorder=3)

        # Mark best epoch from meta
        best_ep = meta["best_val_epoch"]
        if best_ep < len(history):
            ax.axvline(best_ep, color=MODEL_COLOURS[model], linewidth=1.0,
                       linestyle=":", alpha=0.8, zorder=2)
            ax.annotate(
                f"best val\n{meta['best_val_score']:.4f}",
                xy=(best_ep, losses[best_ep]),
                xytext=(best_ep + 1.2, losses[best_ep] * 1.08),
                fontsize=7,
                color=MODEL_COLOURS[model],
                arrowprops=dict(arrowstyle="-", color=MODEL_COLOURS[model], lw=0.8),
            )

        role   = meta["model_role"].replace("_", " ")
        n_ep   = meta["total_epochs"]
        t_h    = meta["total_time_h"]
        ax.set_title(
            f"{model.replace('_', ' ').upper()}  —  {role}  |  "
            f"{n_ep} epochs  {t_h:.1f}h",
            fontsize=9, loc="left", pad=4,
        )
        ax.set_ylabel("Total loss", fontsize=8)
        ax.set_xlim(-0.5, len(history) - 0.5)

        # Phase legend (first subplot only)
        if model == "model_a":
            handles = [
                mpatches.Patch(color=PHASE_COLOURS[p], label=f"{p} phase", alpha=0.9)
                for p in ("early", "mid", "late")
            ]
            ax.legend(handles=handles, loc="upper right", ncol=3)

    axes[-1].set_xlabel("Epoch")
    fig.tight_layout()
    out = OUTPUT_DIR / "fig1_loss_trajectories.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Figure 2 — Phase-gated loss component activation (Model A)
# ---------------------------------------------------------------------------

COMPONENTS = [
    ("ce",        "Cross-entropy",        "#4393c3"),
    ("dice",      "Dice",                 "#2166ac"),
    ("sdf",       "SDF regression",       "#92c5de"),
    ("msr",       "Medial surface recall","#f4a582"),
    ("gapneg",    "Gap-negative",         "#d6604d"),
    ("cldice",    "Centreline Dice",      "#b2182b"),
    ("topo",      "Topology guide",       "#762a83"),
    ("tversky",   "Tversky",              "#4dac26"),
]


def fig2_phase_gated_components() -> None:
    """
    Show how loss components activate across training phases for each model.
    Each row = one model, each column = one training phase.
    Bar height = mean contribution of that component in that phase.
    """
    fig, axes = plt.subplots(3, 3, figsize=(12, 8), sharey=False)
    fig.suptitle(
        "Phase-gated loss component activation — mean contribution per phase",
        fontsize=11, y=0.99,
    )

    phase_order = ["early", "mid", "late"]

    for row_idx, model in enumerate(["model_a", "model_b", "model_c"]):
        history = load_history(model)
        meta    = load_meta(model)
        role    = meta["model_role"].replace("_", " ")

        # Group epochs by phase
        phase_epochs: dict[str, list[dict]] = {"early": [], "mid": [], "late": []}
        for row in history:
            phase_epochs[row["phase"]].append(row)

        for col_idx, phase in enumerate(phase_order):
            ax = axes[row_idx][col_idx]
            rows = phase_epochs[phase]
            if not rows:
                ax.set_visible(False)
                continue

            labels, means, colours = [], [], []
            for key, label, colour in COMPONENTS:
                vals = [r.get(key, 0.0) for r in rows]
                mean_val = float(np.mean(vals))
                if mean_val > 1e-5:
                    labels.append(label)
                    means.append(mean_val)
                    colours.append(colour)

            if not means:
                ax.text(0.5, 0.5, "inactive", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color="grey")
                ax.set_xticks([])
            else:
                bars = ax.barh(range(len(labels)), means, color=colours,
                               height=0.65, zorder=2)
                ax.set_yticks(range(len(labels)))
                ax.set_yticklabels(labels, fontsize=7)
                ax.set_xlabel("Mean loss contribution", fontsize=7)

            if col_idx == 0:
                ax.set_ylabel(
                    f"{model.replace('_', ' ').upper()}\n{role}",
                    fontsize=8, labelpad=6,
                )
            if row_idx == 0:
                ax.set_title(f"{phase.capitalize()} phase", fontsize=9)

    fig.tight_layout()
    out = OUTPUT_DIR / "fig2_phase_gated_components.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Figure 3 — Learning rate schedules, all models
# ---------------------------------------------------------------------------

def fig3_lr_schedules() -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_title("Learning rate schedules — all three specialists", fontsize=10)

    for model in ["model_a", "model_b", "model_c"]:
        history = load_history(model)
        epochs  = [r["epoch"] for r in history]
        lrs     = [r["lr"]    for r in history]
        ax.plot(epochs, lrs, color=MODEL_COLOURS[model], linewidth=1.6,
                label=MODEL_LABELS[model])

    # Phase legend for model A (representative)
    history_a = load_history("model_a")
    shade_phases(ax, history_a)
    phase_handles = [
        mpatches.Patch(color=PHASE_COLOURS[p], label=f"{p} phase", alpha=0.9)
        for p in ("early", "mid", "late")
    ]
    model_handles, model_lbls = ax.get_legend_handles_labels()
    ax.legend(
        handles=model_handles + phase_handles,
        labels=model_lbls + [p.get_label() for p in phase_handles],
        loc="upper right", fontsize=7, ncol=2,
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate")
    ax.set_yscale("log")
    fig.tight_layout()
    out = OUTPUT_DIR / "fig3_lr_schedules.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Figure 4 — Model comparison: best scores, training time, patch size
# ---------------------------------------------------------------------------

def fig4_model_comparison() -> None:
    models   = ["model_a", "model_b", "model_c"]
    metas    = [load_meta(m) for m in models]
    labels   = ["Generalist\n(Model A)", "Anti-merge\n(Model B)", "Surface\n(Model C)"]
    colours  = [MODEL_COLOURS[m] for m in models]

    val_scores   = [m["best_val_score"]   for m in metas]
    surfdice     = [m["best_surfdice"]    for m in metas]
    voi_scores   = [m["best_voi"]         for m in metas]
    topo_scores  = [m["best_topo"]        for m in metas]
    train_times  = [m["total_time_h"]     for m in metas]
    patch_sizes  = [m["patch_size"][0]    for m in metas]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("Specialist model comparison — final training run", fontsize=11)

    # Panel 1: score decomposition per model
    ax = axes[0]
    x  = np.arange(len(labels))
    w  = 0.2
    ax.bar(x - 1.5*w, val_scores,  width=w, color=colours, alpha=1.0,  label="Composite (best val)")
    ax.bar(x - 0.5*w, surfdice,    width=w, color=colours, alpha=0.75, label="SurfaceDice")
    ax.bar(x + 0.5*w, voi_scores,  width=w, color=colours, alpha=0.55, label="VOI score")
    ax.bar(x + 1.5*w, topo_scores, width=w, color=colours, alpha=0.35, label="TopoScore")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Score (0–1)")
    ax.set_title("Score decomposition at best epoch", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, loc="lower right")

    # Panel 2: training time
    ax = axes[1]
    bars = ax.bar(labels, train_times, color=colours, zorder=2)
    for bar, val in zip(bars, train_times):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.05,
                f"{val:.1f}h", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Training time (hours)")
    ax.set_title("GPU training time per specialist", fontsize=9)
    ax.set_ylim(0, max(train_times) * 1.2)

    # Panel 3: epochs and patch size
    ax   = axes[2]
    ep   = [m["total_epochs"] for m in metas]
    x    = np.arange(len(labels))
    ax2  = ax.twinx()
    bars = ax.bar(x, ep, color=colours, alpha=0.8, zorder=2, label="Epochs")
    ax2.plot(x, patch_sizes, "o--", color="#555555", linewidth=1.4,
             markersize=6, label="Patch size (voxels)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Total epochs")
    ax2.set_ylabel("Patch size (voxels per side)")
    ax.set_title("Epochs and patch size per specialist", fontsize=9)
    lines1, lbls1 = ax.get_legend_handles_labels()
    lines2, lbls2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lbls1 + lbls2, fontsize=7, loc="upper right")

    fig.tight_layout()
    out = OUTPUT_DIR / "fig4_model_comparison.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating training curve figures...")
    fig1_loss_trajectories()
    fig2_phase_gated_components()
    fig3_lr_schedules()
    fig4_model_comparison()
    print(f"\nAll figures written to {OUTPUT_DIR.relative_to(REPO_ROOT)}/")
