"""Nature-style kinematic signature bar charts (XY speed, Z height, joint vel).

For each start height (low / high), shows a 1×3 panel figure:
  Panel 1: Mean XY speed per behavior (walk >> groom ≈ rear)
  Panel 2: Mean root Z height per behavior (rear >> walk ≈ groom)
  Panel 3: Mean joint angular velocity per behavior (walk > rear > groom)

Each bar includes individual-clip scatter and error bars.

Usage:
    cd moseq_jax/figures
    python plot_kinematic_signatures.py
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
BEHAVIORS = ["walk", "groom", "rear"]
HEIGHTS = ["low", "high"]
CTRL_DT = 0.01  # 100 Hz control rate

# ── Nature colorblind-safe palette ───────────────────────────────────────────
BEHAVIOR_COLORS = {
    "walk": "#D55E00",    # orange
    "groom": "#0072B2",   # blue
    "rear": "#009E73",    # green
}

BEHAVIOR_LABELS = {
    "walk": "Walk",
    "groom": "Groom",
    "rear": "Rear",
}

HEIGHT_LABELS = {
    "low": "Instructing the Body Starting from Walk Pose",
    "high": "Instructing the Body Starting from Rear Pose",
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


def _add_rounded_border(fig: plt.Figure, axes_list: list[plt.Axes]) -> mpatches.FancyBboxPatch:
    """Add rounded figure border and restore axis spines."""
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


def load_killer_data() -> dict[str, dict[str, list[np.ndarray]]]:
    """Load qpos arrays for each behavior x height.

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


def _strip_plot(ax, x_pos, values, color, jitter_seed=0):
    """Add jittered individual dots to a bar chart."""
    jitter = np.random.default_rng(jitter_seed).uniform(
        -0.15, 0.15, size=len(values)
    )
    ax.scatter(
        x_pos + jitter, values, color=color,
        s=10, alpha=0.45, edgecolors="none", zorder=3,
    )


def plot_kinematic_signatures(
    data: dict[str, dict[str, list[np.ndarray]]],
    height: str,
) -> tuple[plt.Figure, mpatches.FancyBboxPatch]:
    """1-row x 3-col bar chart for one height condition.

    Panels: XY speed | Root Z height | Joint angular velocity.
    """
    title = HEIGHT_LABELS[height]
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.5))
    x = np.arange(len(BEHAVIORS))
    colors = [BEHAVIOR_COLORS[b] for b in BEHAVIORS]
    labels = [BEHAVIOR_LABELS[b] for b in BEHAVIORS]

    # --- Compute per-behavior kinematic features ---
    xy_speeds = {}
    z_heights = {}
    joint_vels = {}

    for beh in BEHAVIORS:
        trajs = data[beh][height]
        speeds, heights_z, vels = [], [], []
        for qpos in trajs:
            # XY speed
            xy = qpos[:, :2]
            dists = np.linalg.norm(np.diff(xy, axis=0), axis=1)
            speeds.append(np.mean(dists) / CTRL_DT)
            # Root Z height
            heights_z.append(np.mean(qpos[:, 2]))
            # Joint angular velocity (joints start at index 7)
            joints = qpos[:, 7:]
            angular_vel = np.abs(np.diff(joints, axis=0)) / CTRL_DT
            vels.append(np.mean(angular_vel))
        xy_speeds[beh] = np.array(speeds)
        z_heights[beh] = np.array(heights_z)
        joint_vels[beh] = np.array(vels)

    # --- Panel 1: XY speed ---
    means = [xy_speeds[b].mean() for b in BEHAVIORS]
    sems = [xy_speeds[b].std() / np.sqrt(len(xy_speeds[b])) for b in BEHAVIORS]
    axes[0].bar(x, means, yerr=sems, color=colors, alpha=0.8, capsize=3, width=0.6,
                error_kw={"linewidth": 0.8})
    for i, beh in enumerate(BEHAVIORS):
        _strip_plot(axes[0], x[i], xy_speeds[beh], colors[i], jitter_seed=i)
    axes[0].set_ylabel("XY speed (m/s)")
    axes[0].set_title("Locomotion")

    # --- Panel 2: Z height ---
    means = [z_heights[b].mean() for b in BEHAVIORS]
    sems = [z_heights[b].std() / np.sqrt(len(z_heights[b])) for b in BEHAVIORS]
    axes[1].bar(x, means, yerr=sems, color=colors, alpha=0.8, capsize=3, width=0.6,
                error_kw={"linewidth": 0.8})
    for i, beh in enumerate(BEHAVIORS):
        _strip_plot(axes[1], x[i], z_heights[beh], colors[i], jitter_seed=i + 10)
    axes[1].set_ylabel("Root Z height (m)")
    axes[1].set_title("Posture")

    # --- Panel 3: Joint angular velocity ---
    means = [joint_vels[b].mean() for b in BEHAVIORS]
    sems = [joint_vels[b].std() / np.sqrt(len(joint_vels[b])) for b in BEHAVIORS]
    axes[2].bar(x, means, yerr=sems, color=colors, alpha=0.8, capsize=3, width=0.6,
                error_kw={"linewidth": 0.8})
    for i, beh in enumerate(BEHAVIORS):
        _strip_plot(axes[2], x[i], joint_vels[beh], colors[i], jitter_seed=i + 20)
    axes[2].set_ylabel("Joint angular vel (rad/s)")
    axes[2].set_title("Joint Motion")

    # Shared formatting
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels)

    fig.suptitle(title, fontsize=9, fontweight="bold", y=0.97)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    border_rect = _add_rounded_border(fig, list(axes))
    return fig, border_rect


def main() -> None:
    _setup_nature_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_killer_data()

    for height in HEIGHTS:
        fig, border_rect = plot_kinematic_signatures(data, height)
        stem = f"kinematic_signatures_{height}"
        out_pdf = OUTPUT_DIR / f"{stem}.pdf"
        out_png = OUTPUT_DIR / f"{stem}.png"
        out_svg = OUTPUT_DIR / f"{stem}.svg"
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
