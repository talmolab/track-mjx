"""Nature-style KID bar plot for poster/paper.

Vertical bar chart with all datasets: split baseline, mimic-mjx, code2act, arhmm.
KID on y-axis, error bars = std across VAE seeds.

Usage:
    cd moseq_jax/figures
    python plot_kid_barplot.py
"""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"
RESULTS_PATH = SCRIPT_DIR.parent / "outputs" / "moseq_inception_distance" / "results.json"

# Display names and colors (Nature colorblind-safe)
DISPLAY_NAMES = {
    "split_baseline": "Split Baseline\n(noise floor)",
    "mimic_mjx": "Mimic-MJX\n(oracle)",
    "decoder_original_codes": "Code2Act\n(real codes)",
    "transition_matrix": "Transition\nMatrix",
    "arhmm_level2": "ARHMM L2",
    "hmm_dynamax": "HMM",
    "uniform_random": "Random\nCodes",
}

COLORS = {
    "split_baseline": "#999999",       # gray
    "mimic_mjx": "#009E73",            # green
    "decoder_original_codes": "#0072B2",  # blue
    "transition_matrix": "#56B4E9",    # light blue
    "arhmm_level2": "#D55E00",         # orange
    "hmm_dynamax": "#CC79A7",          # pink
    "uniform_random": "#999999",       # gray
}

# Order for display (best to worst KID, no split baseline)
DISPLAY_ORDER = [
    "decoder_original_codes",
    "transition_matrix", "arhmm_level2", "uniform_random",
]


def _setup_nature_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.titlesize": 9,
        "axes.titleweight": "bold",
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6,
        "lines.linewidth": 1.2,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
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


def _add_rounded_border(fig, axes_list):
    fig.patch.set_facecolor("white")
    rect = mpatches.FancyBboxPatch(
        (0.01, 0.01), 0.98, 0.98,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        transform=fig.transFigure,
        facecolor="white",
        edgecolor="#cccccc",
        linewidth=0.8,
        zorder=-1,
    )
    fig.patches.append(rect)
    for ax in axes_list:
        ax.spines["left"].set_visible(True)
        ax.spines["bottom"].set_visible(True)
    return rect


def main():
    _setup_nature_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_PATH) as f:
        results = json.load(f)

    aggregated = results["aggregated"]

    # Filter to datasets that exist in results
    datasets = [d for d in DISPLAY_ORDER if d in aggregated]
    names = [DISPLAY_NAMES.get(d, d) for d in datasets]
    colors = [COLORS.get(d, "#999999") for d in datasets]
    kid_means = [aggregated[d]["kid_mean"] for d in datasets]
    kid_stds = [aggregated[d]["kid_std"] for d in datasets]

    x = np.arange(len(datasets))
    fig, ax = plt.subplots(figsize=(5.0, 3.5))

    bars = ax.bar(
        x, kid_means, yerr=kid_stds,
        color=colors, alpha=0.85, capsize=4, width=0.6,
        error_kw={"linewidth": 0.8, "capthick": 0.8},
        edgecolor="white", linewidth=0.5,
    )

    # Add value labels at top of error bars
    for bar, mean, std in zip(bars, kid_means, kid_stds):
        label = f"{abs(mean):.2f}" if abs(mean) < 0.005 else f"{mean:.2f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean + std + 0.01,
            label,
            ha="center", va="bottom", fontsize=6.5, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(names, ha="center")
    ax.set_ylabel("KID (Kernel Inception Distance)")
    ax.set_title("Distribution Quality: Real vs Generated Behavior")
    ax.set_ylim(bottom=0, top=max(kid_means) + max(kid_stds) + 0.04)
    ax.axhline(0, color="#e0e0e0", linewidth=0.4, zorder=0)

    fig.tight_layout()
    border_rect = _add_rounded_border(fig, [ax])

    for ext in ("png", "pdf"):
        fig.savefig(OUTPUT_DIR / f"kid_barplot.{ext}")
    border_rect.set_facecolor("none")
    fig.savefig(OUTPUT_DIR / "kid_barplot.svg", transparent=True)
    border_rect.set_facecolor("white")
    plt.close(fig)

    for ext in ("png", "pdf", "svg"):
        print(f"Saved: {OUTPUT_DIR / f'kid_barplot.{ext}'}")


if __name__ == "__main__":
    main()
