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
OUTPUT_DIR = SCRIPT_DIR / "outputs" / "kinematic_signatures"

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
    "groom": "Immobility",
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

    Panels: XY speed | Root Z height | Joint angular velocity (fore vs hind).
    """
    title = HEIGHT_LABELS[height]
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.5))
    x = np.arange(len(BEHAVIORS))
    colors = [BEHAVIOR_COLORS[b] for b in BEHAVIORS]
    labels = [BEHAVIOR_LABELS[b] for b in BEHAVIORS]

    # --- Compute per-behavior kinematic features ---
    xy_speeds = {}
    z_heights = {}

    # Pure limb-swing joints only (6 per group, matched):
    #   Fore: shoulder, elbow, wrist (L+R)
    #   Hind: hip_extend, knee, ankle (L+R)
    fore_idx = [28, 30, 31, 35, 37, 38]
    hind_idx = [10, 11, 12, 16, 17, 18]
    fore_vels, hind_vels = {}, {}

    for beh in BEHAVIORS:
        trajs = data[beh][height]
        speeds, heights_z = [], []
        fv, hv = [], []
        for qpos in trajs:
            qpos = np.asarray(qpos, dtype=np.float64)
            # XY speed
            xy = qpos[:, :2]
            dists = np.linalg.norm(np.diff(xy, axis=0), axis=1)
            speeds.append(np.mean(dists) / CTRL_DT)
            # Root Z height
            heights_z.append(np.mean(qpos[:, 2]))
            # Fore / hind angular velocity
            fv.append(np.mean(np.abs(np.diff(qpos[:, fore_idx], axis=0)) / CTRL_DT))
            hv.append(np.mean(np.abs(np.diff(qpos[:, hind_idx], axis=0)) / CTRL_DT))
        xy_speeds[beh] = np.array(speeds)
        z_heights[beh] = np.array(heights_z)
        fore_vels[beh] = np.array(fv)
        hind_vels[beh] = np.array(hv)

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

    # --- Panel 3: Joint motion (forelimbs vs hindlimbs) ---
    w = 0.28
    for i, beh in enumerate(BEHAVIORS):
        fm = fore_vels[beh].mean()
        fs = fore_vels[beh].std() / np.sqrt(len(fore_vels[beh]))
        hm = hind_vels[beh].mean()
        hs = hind_vels[beh].std() / np.sqrt(len(hind_vels[beh]))
        axes[2].bar(x[i] - w / 2, fm, w, yerr=fs, color=colors[i], alpha=0.50,
                    capsize=2, error_kw={"linewidth": 0.8}, edgecolor=colors[i], linewidth=0.8)
        axes[2].bar(x[i] + w / 2, hm, w, yerr=hs, color=colors[i], alpha=0.90,
                    capsize=2, error_kw={"linewidth": 0.8})
        _strip_plot(axes[2], x[i] - w / 2, fore_vels[beh], colors[i], jitter_seed=i + 20)
        _strip_plot(axes[2], x[i] + w / 2, hind_vels[beh], colors[i], jitter_seed=i + 30)

    from matplotlib.patches import Patch
    axes[2].legend(
        handles=[Patch(facecolor="gray", alpha=0.50, label="Fore"),
                 Patch(facecolor="gray", alpha=0.90, label="Hind")],
        frameon=False, fontsize=5.5, loc="upper right",
    )
    axes[2].set_ylabel("Angular vel (rad/s)")
    axes[2].set_title("Joint Motion")

    # Shared formatting
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels)

    fig.suptitle(title, fontsize=9, fontweight="bold", y=0.97)
    fig.tight_layout(rect=[0, 0.06, 1, 0.90])

    # Bottom legend: which joints are Fore vs Hind
    fig.text(0.5, 0.01,
             "Fore: shoulder, elbow, wrist (L+R)    |    Hind: hip, knee, ankle (L+R)",
             ha="center", va="bottom", fontsize=5.5, color="#555555",
             fontstyle="italic")
    border_rect = _add_rounded_border(fig, list(axes))
    return fig, border_rect


# Joint index groups (qpos indices, joints start at 7)
# Rodent joint order: vertebra_1_extend(7), hip_L_supinate(8), hip_L_abduct(9),
# hip_L_extend(10), knee_L(11), ankle_L(12), toe_L(13), hip_R_supinate(14),
# hip_R_abduct(15), hip_R_extend(16), knee_R(17), ankle_R(18), toe_R(19),
# vertebra_C11_extend(20), vertebra_cervical_1_bend(21), vertebra_axis_twist(22),
# atlas(23), mandible(24), scapula_L_supinate(25), scapula_L_abduct(26),
# scapula_L_extend(27), shoulder_L(28), shoulder_sup_L(29), elbow_L(30), wrist_L(31),
# scapula_R_supinate(32), scapula_R_abduct(33), scapula_R_extend(34),
# shoulder_R(35), shoulder_sup_R(36), elbow_R(37), wrist_R(38), finger_R(39)

BODY_PART_INDICES = {
    "Forelimbs": [28, 30, 31, 35, 37, 38],   # shoulder+elbow+wrist (L+R)
    "Hindlimbs": [10, 11, 12, 16, 17, 18],   # hip_extend+knee+ankle (L+R)
    "Spine": [7, 20, 21, 22, 23],             # vertebra + atlas
    "Head": [24],                              # mandible
}

BODY_PART_COLORS_MAP = {
    "Forelimbs": "#0072B2",
    "Hindlimbs": "#D55E00",
    "Spine": "#009E73",
    "Head": "#CC79A7",
}


def plot_joint_motion_by_bodypart(
    data: dict[str, dict[str, list[np.ndarray]]],
    height: str,
) -> tuple[plt.Figure, mpatches.FancyBboxPatch]:
    """Bar chart: joint angular velocity broken down by body part per behavior.

    Layout: one panel per body part, bars = behaviors (walk/groom/rear).
    """
    title = HEIGHT_LABELS[height]
    body_parts = ["Forelimbs", "Hindlimbs", "Spine", "Head"]
    n_parts = len(body_parts)
    fig, axes = plt.subplots(1, n_parts, figsize=(7.5, 2.5))
    x = np.arange(len(BEHAVIORS))
    beh_colors = [BEHAVIOR_COLORS[b] for b in BEHAVIORS]
    labels = [BEHAVIOR_LABELS[b] for b in BEHAVIORS]

    # Ensure float arrays
    panels_f = {
        beh: [np.asarray(q, dtype=np.float64) for q in data[beh][height]]
        for beh in BEHAVIORS
    }

    for pi, part in enumerate(body_parts):
        ax = axes[pi]
        indices = BODY_PART_INDICES[part]

        part_vels = {}
        for beh in BEHAVIORS:
            vels = []
            for qpos in panels_f[beh]:
                joint_subset = qpos[:, indices]
                angular_vel = np.abs(np.diff(joint_subset, axis=0)) / CTRL_DT
                vels.append(np.mean(angular_vel))
            part_vels[beh] = np.array(vels)

        means = [part_vels[b].mean() for b in BEHAVIORS]
        sems = [part_vels[b].std() / np.sqrt(len(part_vels[b])) for b in BEHAVIORS]
        ax.bar(x, means, yerr=sems, color=beh_colors, alpha=0.8, capsize=3,
               width=0.6, error_kw={"linewidth": 0.8})
        for i, beh in enumerate(BEHAVIORS):
            _strip_plot(ax, x[i], part_vels[beh], beh_colors[i], jitter_seed=pi * 10 + i)

        ax.set_title(part)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        if pi == 0:
            ax.set_ylabel("Joint angular vel (rad/s)")

    fig.suptitle(title, fontsize=9, fontweight="bold", y=0.97)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    border_rect = _add_rounded_border(fig, list(axes))
    return fig, border_rect


def main() -> None:
    _setup_nature_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_killer_data()

    for height in HEIGHTS:
        # Original 3-panel figure
        fig, border_rect = plot_kinematic_signatures(data, height)
        stem = f"kinematic_signatures_{height}"
        for ext in ("pdf", "png"):
            fig.savefig(OUTPUT_DIR / f"{stem}.{ext}")
        border_rect.set_facecolor("none")
        fig.savefig(OUTPUT_DIR / f"{stem}.svg", transparent=True)
        plt.close(fig)
        print(f"Saved: {stem}")

        # Body-part breakdown figure
        fig2, border_rect2 = plot_joint_motion_by_bodypart(data, height)
        stem2 = f"joint_motion_bodypart_{height}"
        for ext in ("pdf", "png"):
            fig2.savefig(OUTPUT_DIR / f"{stem2}.{ext}")
        border_rect2.set_facecolor("none")
        fig2.savefig(OUTPUT_DIR / f"{stem2}.svg", transparent=True)
        plt.close(fig2)
        print(f"Saved: {stem2}")


if __name__ == "__main__":
    main()
