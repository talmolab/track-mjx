"""Nature-style round-trip code consistency bar chart.

Reads roundtrip_summary.json and plots grouped bars: 3 conditions
(Reference, Mimic-MJX, Code2Act) × 2 datasets (500-frame, 1000-frame).

Usage:
    cd moseq_jax/figures
    python plot_roundtrip_accuracy.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR / "outputs"

# Wong Nature Methods 2011 palette
CONDITION_COLORS = {
    "reference": "#009E73",  # green
    "mimic_mjx": "#D55E00",  # orange
    "code2act": "#0072B2",   # blue
}
CONDITION_LABELS = {
    "reference": "Reference (ceiling)",
    "mimic_mjx": "Mimic-MJX",
    "code2act": "Code2Act",
}
CONDITIONS = ["reference", "mimic_mjx", "code2act"]
DATASET_LABELS = {
    "inference": "250-Frame\nTest Set",
    "generalization": "1000-Frame\nTest Set",
}


def _setup_nature_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.linewidth": 0.6,
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
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": False,
    })


def main() -> None:
    _setup_nature_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(DATA_DIR / "roundtrip_summary.json") as f:
        results = json.load(f)

    datasets = list(results.keys())
    n_ds = len(datasets)
    n_cond = len(CONDITIONS)
    bar_width = 0.22
    x = np.arange(n_ds)

    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    for ci, cond in enumerate(CONDITIONS):
        vals = [results[ds].get(cond, 0) for ds in datasets]
        offset = (ci - (n_cond - 1) / 2) * bar_width
        bars = ax.bar(
            x + offset, vals, bar_width,
            label=CONDITION_LABELS[cond],
            color=CONDITION_COLORS[cond],
            edgecolor="none",
        )
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"{v:.1%}", ha="center", va="bottom", fontsize=5.5,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[ds] for ds in datasets])
    ax.set_ylabel("Frame-Level Accuracy")
    ax.set_ylim(0, 1.15)

    for y in (0.25, 0.5, 0.75, 1.0):
        ax.axhline(y, color="#e0e0e0", linewidth=0.3, zorder=0)

    leg = ax.legend(
        frameon=True,
        framealpha=0.9,
        edgecolor="none",
        borderpad=0.4,
        fancybox=True,
    )
    leg.get_frame().set_linewidth(0)
    leg.get_frame().set_boxstyle("round,pad=0.3,rounding_size=0.2")

    fig.tight_layout()

    # Rounded border
    fig.patch.set_visible(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    rect = mpatches.FancyBboxPatch(
        (0.02, 0.02), 0.96, 0.96,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        transform=fig.transFigure,
        facecolor="white",
        edgecolor="#cccccc",
        linewidth=0.8,
        zorder=-1,
    )
    fig.patches.append(rect)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)

    out_pdf = OUTPUT_DIR / "roundtrip_accuracy.pdf"
    out_png = OUTPUT_DIR / "roundtrip_accuracy.png"
    out_svg = OUTPUT_DIR / "roundtrip_accuracy.svg"
    fig.savefig(out_pdf)
    fig.savefig(out_png)
    rect.set_facecolor("none")
    fig.savefig(out_svg, transparent=True)
    plt.close(fig)
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")
    print(f"Saved: {out_svg}")


if __name__ == "__main__":
    main()
