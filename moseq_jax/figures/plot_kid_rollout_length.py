"""Combined KID panel: MIMIC vs C2A across rollout lengths.

Produces two plots:
1. Original VAE — KID against training distribution
2. Generalization VAE — KID against generalization distribution

Each plot: x-axis = rollout length, dark blue = MIMIC, light blue = Code2Act.

Usage:
    cd moseq_jax
    python -m figures.plot_kid_rollout_length
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

MOSEQ_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = MOSEQ_DIR / "figures" / "outputs" / "kid_rollout_length"

LENGTHS = [250, 500, 1000, 2000]
RESULT_PATHS = {
    length: MOSEQ_DIR / f"outputs/moseq_generalization_kid_{length}/results.json"
    for length in LENGTHS
}

MIMIC_COLOR = "#0C2D6B"   # dark blue
C2A_COLOR = "#56B4E9"     # light blue


def _setup_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.titlesize": 9,
        "axes.titleweight": "bold",
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": False,
    })


def _add_border(fig):
    fig.patch.set_facecolor("white")
    rect = mpatches.FancyBboxPatch(
        (0.005, 0.005), 0.99, 0.99,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        transform=fig.transFigure,
        facecolor="white", edgecolor="#cccccc", linewidth=0.8, zorder=-1,
    )
    fig.patches.append(rect)
    return rect


def _plot_single(
    ax,
    available_lengths: list[int],
    mimic_means: list[float],
    mimic_stds: list[float],
    c2a_means: list[float],
    c2a_stds: list[float],
    title: str,
    noise_floor_idx: int | None = None,
):
    n = len(available_lengths)
    x = np.arange(n)
    bar_width = 0.35

    bars_mimic = ax.bar(
        x - bar_width / 2, mimic_means, bar_width,
        yerr=mimic_stds, color=MIMIC_COLOR, alpha=0.85, capsize=4,
        error_kw={"linewidth": 0.8, "capthick": 0.8},
        edgecolor="white", linewidth=0.5, label="Mimic-MJX",
    )
    bars_c2a = ax.bar(
        x + bar_width / 2, c2a_means, bar_width,
        yerr=c2a_stds, color=C2A_COLOR, alpha=0.85, capsize=4,
        error_kw={"linewidth": 0.8, "capthick": 0.8},
        edgecolor="white", linewidth=0.5, label="Code2Act",
    )

    for bars, means, stds in [(bars_mimic, mimic_means, mimic_stds),
                               (bars_c2a, c2a_means, c2a_stds)]:
        for bar, mean, std in zip(bars, means, stds):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                mean + std + 0.01,
                f"{mean:.2f}",
                ha="center", va="bottom", fontsize=5.5, fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in available_lengths])
    ax.set_xlabel("Testing Rollout Length (frames)")
    ax.set_ylabel("KID")
    ax.set_title(title)
    ax.set_ylim(bottom=0)
    ax.axhline(0, color="#e0e0e0", linewidth=0.4, zorder=0)
    ax.legend(frameon=False, fontsize=6, loc="upper left")

    if noise_floor_idx is not None:
        xi = noise_floor_idx
        ax.annotate(
            "(noise floor)", xy=(xi, 0), xytext=(xi, -0.18),
            ha="center", va="top", fontsize=5.5, fontstyle="italic",
            color="#666666", annotation_clip=False,
        )


def main():
    _setup_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load all results
    available_lengths = []
    orig = {"mimic_means": [], "mimic_stds": [], "c2a_means": [], "c2a_stds": []}
    gen = {"mimic_means": [], "mimic_stds": [], "c2a_means": [], "c2a_stds": []}
    trn = {"mimic_means": [], "mimic_stds": [], "c2a_means": [], "c2a_stds": []}

    for length in LENGTHS:
        path = RESULT_PATHS[length]
        if not path.exists():
            print(f"WARNING: {path} not found, skipping {length}")
            continue
        with open(path) as f:
            data = json.load(f)

        available_lengths.append(length)

        # Tst rollout + training VAE
        agg_orig = data.get("tst_orig", data.get("original_vae", {})).get("aggregated", data.get("aggregated", {}))
        orig["mimic_means"].append(agg_orig["mimic_mjx"]["kid_mean"])
        orig["mimic_stds"].append(agg_orig["mimic_mjx"]["kid_std"])
        orig["c2a_means"].append(agg_orig["code2act"]["kid_mean"])
        orig["c2a_stds"].append(agg_orig["code2act"]["kid_std"])

        # Tst rollout + testing VAE
        agg_gen = data.get("tst_gen", data.get("generalization_vae", {})).get("aggregated", None)
        if agg_gen is not None:
            gen["mimic_means"].append(agg_gen["mimic_mjx"]["kid_mean"])
            gen["mimic_stds"].append(agg_gen["mimic_mjx"]["kid_std"])
            gen["c2a_means"].append(agg_gen["code2act"]["kid_mean"])
            gen["c2a_stds"].append(agg_gen["code2act"]["kid_std"])

        # Trn rollout + training VAE
        agg_trn = data.get("trn_orig", {}).get("aggregated", None)
        if agg_trn is not None:
            trn["mimic_means"].append(agg_trn["mimic_mjx"]["kid_mean"])
            trn["mimic_stds"].append(agg_trn["mimic_mjx"]["kid_std"])
            trn["c2a_means"].append(agg_trn["code2act"]["kid_mean"])
            trn["c2a_stds"].append(agg_trn["code2act"]["kid_std"])

    if not available_lengths:
        print("ERROR: No results found.")
        sys.exit(1)

    has_gen = len(gen["mimic_means"]) == len(available_lengths)
    has_trn = len(trn["mimic_means"]) == len(available_lengths)

    # --- Plot 1: Original VAE only (same as before) ---
    fig1, ax1 = plt.subplots(figsize=(4.5, 3.3))
    _plot_single(ax1, available_lengths,
                 orig["mimic_means"], orig["mimic_stds"],
                 orig["c2a_means"], orig["c2a_stds"],
                 "Tst. Rollout w/ Training Data Trained VAE")
    fig1.tight_layout(rect=[0.02, 0.02, 0.98, 0.93])
    rect1 = _add_border(fig1)
    for ext in ("png", "pdf"):
        fig1.savefig(str(OUTPUT_DIR / f"kid_rollout_length.{ext}"))
    rect1.set_facecolor("none")
    fig1.savefig(str(OUTPUT_DIR / "kid_rollout_length.svg"), transparent=True)
    plt.close(fig1)
    print(f"Saved: kid_rollout_length.{{png,pdf,svg}}")

    # --- Plot 2: Triple panel ---
    if has_gen and has_trn:
        fig2, (ax_trn, ax_orig, ax_gen) = plt.subplots(1, 3, figsize=(13.5, 3.8))
        _plot_single(ax_trn, available_lengths,
                     trn["mimic_means"], trn["mimic_stds"],
                     trn["c2a_means"], trn["c2a_stds"],
                     "Trn. Rollout w/ Training Data Trained VAE",
                     noise_floor_idx=0)
        _plot_single(ax_orig, available_lengths,
                     orig["mimic_means"], orig["mimic_stds"],
                     orig["c2a_means"], orig["c2a_stds"],
                     "Tst. Rollout w/ Training Data Trained VAE")
        _plot_single(ax_gen, available_lengths,
                     gen["mimic_means"], gen["mimic_stds"],
                     gen["c2a_means"], gen["c2a_stds"],
                     "Tst. Rollout w/ Testing Data Trained VAE",
                     noise_floor_idx=0)
        # Same y-axis: 1.6 for middle and right, match for left
        for ax in (ax_trn, ax_orig, ax_gen):
            ax.set_ylim(0, 1.6)
        fig2.subplots_adjust(left=0.05, right=0.98, top=0.88, bottom=0.15, wspace=0.15)
        rect2 = _add_border(fig2)
        for ext in ("png", "pdf"):
            fig2.savefig(str(OUTPUT_DIR / f"kid_rollout_length_triple.{ext}"))
        rect2.set_facecolor("none")
        fig2.savefig(str(OUTPUT_DIR / "kid_rollout_length_triple.svg"), transparent=True)
        plt.close(fig2)
        print(f"Saved: kid_rollout_length_triple.{{png,pdf,svg}}")
    elif has_gen:
        fig2, (ax_orig, ax_gen) = plt.subplots(1, 2, figsize=(9, 3.3), sharey=False)
        _plot_single(ax_orig, available_lengths,
                     orig["mimic_means"], orig["mimic_stds"],
                     orig["c2a_means"], orig["c2a_stds"],
                     "Tst. Rollout w/ Training Data Trained VAE")
        _plot_single(ax_gen, available_lengths,
                     gen["mimic_means"], gen["mimic_stds"],
                     gen["c2a_means"], gen["c2a_stds"],
                     "Tst. Rollout w/ Testing Data Trained VAE",
                     noise_floor_idx=0)
        fig2.tight_layout(rect=[0.01, 0.02, 0.99, 0.93])
        rect2 = _add_border(fig2)
        for ext in ("png", "pdf"):
            fig2.savefig(str(OUTPUT_DIR / f"kid_rollout_length_dual.{ext}"))
        rect2.set_facecolor("none")
        fig2.savefig(str(OUTPUT_DIR / "kid_rollout_length_dual.svg"), transparent=True)
        plt.close(fig2)
        print(f"Saved: kid_rollout_length_dual.{{png,pdf,svg}}")


if __name__ == "__main__":
    main()
