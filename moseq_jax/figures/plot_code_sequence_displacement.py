"""Nature-style code-sequence displacement plots (Δz and Δxy, per behavior).

For each behavior category (groom / walk / rear), shows side-by-side:
  Left:  z displacement from initial position over episode timesteps
  Right: cumulative XY path length over episode timesteps
comparing high-z vs low-z killer code sequences.

Usage:
    cd moseq_jax/figures
    python plot_code_sequence_displacement.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR / "outputs"

# ── Behaviors and conditions ─────────────────────────────────────────────────
BEHAVIORS = ["groom", "walk", "rear"]
HEIGHTS = ["high", "low"]

# ── Nature colorblind-safe palette (Wong, Nature Methods 2011) ───────────────
HEIGHT_COLORS = {
    "high": "#D55E00",   # Nature orange
    "low": "#0072B2",    # Nature blue
}

HEIGHT_LABELS = {
    "high": "Rear Pose Instantiation",
    "low": "Walk Pose Instantiation",
}

BEHAVIOR_TITLES = {
    "groom": "Groom",
    "walk": "Walk",
    "rear": "Rear",
}

BEHAVIOR_COLORS = {
    "groom": "#0072B2",   # blue
    "walk": "#D55E00",    # orange
    "rear": "#009E73",    # green
}


def _setup_nature_style() -> None:
    """Configure matplotlib rcParams for Nature-quality figures."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.titleweight": "bold",
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "lines.linewidth": 2.0,
        "axes.linewidth": 0.8,
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


def _compute_displacements(
    qpos_list: list[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Compute raw displacements from initial position per trajectory.

    Args:
        qpos_list: List of ``[T_i, nq]`` arrays.

    Returns:
        ``(dz_list, dx_list, dy_list)`` — each a list of 1-D arrays.
    """
    dz_list = []
    dx_list = []
    dy_list = []
    for qpos in qpos_list:
        dz_list.append(qpos[:, 2] - qpos[0, 2])
        dx_list.append(qpos[:, 0] - qpos[0, 0])
        dy_list.append(qpos[:, 1] - qpos[0, 1])
    return dz_list, dx_list, dy_list


def _plot_curves_on_ax(
    ax: plt.Axes,
    curves: list[np.ndarray],
    color: str,
    label: str,
    linestyle: str = "-",
) -> None:
    """Plot mean (thick) with SEM band."""
    # Truncate to shortest for mean computation
    min_len = min(len(c) for c in curves)
    mat = np.array([c[:min_len] for c in curves])
    mean = mat.mean(axis=0)
    sem = mat.std(axis=0) / np.sqrt(mat.shape[0])
    t = np.arange(min_len)

    # Mean + SEM
    ax.plot(t, mean, color=color, linewidth=2.2, linestyle=linestyle, label=label, zorder=3)
    ax.fill_between(
        t, mean - sem, mean + sem,
        color=color, alpha=0.18, linewidth=0, zorder=2,
    )


def load_killer_data() -> dict[str, dict[str, list[np.ndarray]]]:
    """Load qpos arrays for each behavior × height.

    Returns:
        ``{behavior: {height: [qpos_0, qpos_1, ...]}}``
    """
    data = {}
    for beh in BEHAVIORS:
        data[beh] = {}
        for h in HEIGHTS:
            fp = DATA_DIR / f"killer_{beh}_{h}.npz"
            d = np.load(fp, allow_pickle=True)
            qpos_arr = d["qpos"]
            max_len = max(qpos_arr[i].shape[0] for i in range(len(qpos_arr)))
            data[beh][h] = [
                np.asarray(qpos_arr[i], dtype=np.float64)
                for i in range(len(qpos_arr))
                if qpos_arr[i].shape[0] >= max_len
            ]
    return data


def _add_rounded_border(fig: plt.Figure, axes_list: list[plt.Axes]) -> mpatches.FancyBboxPatch:
    """Add rounded figure border and restore axis spines. Returns the patch."""
    fig.patch.set_visible(False)
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


def plot_single_height(
    data: dict[str, dict[str, list[np.ndarray]]],
    height: str,
) -> plt.Figure:
    """1-row × 2-col figure for one height: behaviors as colors, cols = Δz | Δxy."""
    title = HEIGHT_LABELS[height]
    fig, (ax_dz, ax_xy) = plt.subplots(1, 2, figsize=(5.0, 2.4))

    for beh in BEHAVIORS:
        qpos_list = data[beh][height]
        dz_list, dx_list, dy_list = _compute_displacements(qpos_list)
        color = BEHAVIOR_COLORS[beh]
        label = BEHAVIOR_TITLES[beh]
        _plot_curves_on_ax(ax_dz, dz_list, color, label, linestyle="-")
        _plot_curves_on_ax(ax_xy, dx_list, color, f"{label} (Δx)", linestyle="-")
        _plot_curves_on_ax(ax_xy, dy_list, color, f"{label} (Δy)", linestyle="--")

    ax_dz.set_ylabel("Δz (m)")
    ax_xy.set_ylabel("Displacement (m)")
    ax_dz.set_title(f"{title} — Δz")
    ax_xy.set_title(f"{title} — Δxy")
    ax_dz.set_xlabel("Episode Timestep")
    ax_xy.set_xlabel("Episode Timestep")
    ax_dz.axhline(0, color="#e0e0e0", linewidth=0.4, zorder=0)
    ax_xy.axhline(0, color="#e0e0e0", linewidth=0.4, zorder=0)
    ax_dz.set_xlim(left=0)
    ax_xy.set_xlim(left=0)

    # Legend: behavior colors + line style for Δx/Δy
    from matplotlib.lines import Line2D
    handles, labels = ax_dz.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="gray", linestyle="-", linewidth=1.2))
    labels.append("Δx")
    handles.append(Line2D([0], [0], color="gray", linestyle="--", linewidth=1.2))
    labels.append("Δy")
    fig.legend(
        handles, labels,
        loc="upper center",
        ncol=5,
        frameon=True,
        framealpha=0.9,
        edgecolor="none",
        fontsize=7,
        bbox_to_anchor=(0.5, 1.0),
        fancybox=True,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    border_rect = _add_rounded_border(fig, [ax_dz, ax_xy])
    return fig, border_rect


def main() -> None:
    _setup_nature_style()
    data = load_killer_data()

    for height in HEIGHTS:
        fig, border_rect = plot_single_height(data, height)
        out_pdf = OUTPUT_DIR / f"code_sequence_displacement_{height}.pdf"
        out_png = OUTPUT_DIR / f"code_sequence_displacement_{height}.png"
        out_svg = OUTPUT_DIR / f"code_sequence_displacement_{height}.svg"
        fig.savefig(out_pdf)
        fig.savefig(out_png)
        border_rect.set_facecolor("none")
        fig.savefig(out_svg, transparent=True)
        border_rect.set_facecolor("white")
        plt.close(fig)
        print(f"Saved: {out_pdf}")
        print(f"Saved: {out_png}")
        print(f"Saved: {out_svg}")


if __name__ == "__main__":
    main()
