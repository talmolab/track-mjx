"""Nature-style PSD overlay: Code2Act vs Mimic-MJX gait dynamics.

Two panels: Hip (L+R) and Knee (L+R). Conditions by color, sides by
line style (solid=R, dashed=L). Plus dominant frequency bar chart.

Usage:
    cd moseq_jax/figures
    python plot_psd_overlay.py
"""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
from scipy.signal import welch

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "outputs" / "moseq_gait_dynamics"
OUTPUT_DIR = SCRIPT_DIR / "outputs"

CONDITION_COLORS = {
    "mimic_mjx": "#D55E00",
    "code2act": "#0072B2",
}
CONDITION_LABELS = {
    "mimic_mjx": "Mimic-MJX",
    "code2act": "Code2Act",
}

PANELS = {
    "Hip": {"R": "hip_R_extend", "L": "hip_L_extend"},
    "Knee": {"R": "knee_R", "L": "knee_L"},
}
SIDE_LINESTYLE = {"R": "-", "L": "--"}
ALL_JOINTS = ["hip_R_extend", "knee_R", "hip_L_extend", "knee_L"]
JOINT_DISPLAY = {
    "hip_R_extend": "Hip R",
    "knee_R": "Knee R",
    "hip_L_extend": "Hip L",
    "knee_L": "Knee L",
}

FS = 100
NPERSEG = 128  # match experiment's Welch parameters
FMIN = 1.0
FMAX = 10.0


def _setup_nature_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.titlesize": 9,
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


def _add_rounded_border(fig, axes_list):
    fig.patch.set_facecolor("white")
    rect = mpatches.FancyBboxPatch(
        (0.01, 0.01), 0.98, 0.98,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        transform=fig.transFigure,
        facecolor="white", edgecolor="#cccccc", linewidth=0.8, zorder=-1,
    )
    fig.patches.append(rect)
    for ax in axes_list:
        ax.spines["left"].set_visible(True)
        ax.spines["bottom"].set_visible(True)
    return rect


def get_joint_addrs():
    """Get qpos addresses for joints using MuJoCo model."""
    import mujoco
    from vnl_playground.tasks.rodent import consts as rodent_consts

    arena_spec = mujoco.MjSpec.from_file(str(rodent_consts.ARENA_XML_PATH))
    walker_spec = mujoco.MjSpec.from_file(str(rodent_consts.RODENT_XML_PATH))
    frame = arena_spec.worldbody.add_frame(pos=(0, 0, 0), quat=(1, 0, 0, 0))
    frame.attach_body(walker_spec.body("walker"), "", suffix="-rodent")
    model = arena_spec.compile()

    addrs = {}
    for jn in ALL_JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{jn}-rodent")
        if jid >= 0:
            addrs[jn] = model.jnt_qposadr[jid]
    return addrs


def compute_psd(angles, fs=FS, nperseg=NPERSEG):
    """Compute PSD using Welch's method with zero-padding for smooth curves.

    nperseg=128 matches the experiment's spectral resolution.
    nfft=1024 zero-pads for visual interpolation (smoother plot, same info).
    """
    freqs, psd = welch(angles, fs=fs, nperseg=nperseg, nfft=1024)
    return freqs, psd


def dominant_frequency(freqs, psd, fmin=FMIN, fmax=FMAX):
    """Find peak frequency in [fmin, fmax]."""
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not mask.any():
        return 0.0
    return float(freqs[mask][np.argmax(psd[mask])])


def main():
    _setup_nature_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(DATA_DIR / "summary.json") as f:
        summary = json.load(f)
    clip_indices = summary["walk_clip_indices"]
    joint_addrs = get_joint_addrs()

    # Compute PSD from saved rollout data
    psd_data = {cond: {jn: [] for jn in ALL_JOINTS} for cond in ["mimic_mjx", "code2act"]}
    dom_freqs = {cond: {jn: [] for jn in ALL_JOINTS} for cond in ["mimic_mjx", "code2act"]}

    for clip_idx in clip_indices:
        for cond in ["mimic_mjx", "code2act"]:
            npz_path = DATA_DIR / f"rollout_{cond}_clip{clip_idx}.npz"
            if not npz_path.exists():
                print(f"  Missing: {npz_path}")
                continue
            qpos = np.load(npz_path)["qpos"]
            for jn in ALL_JOINTS:
                if jn not in joint_addrs:
                    continue
                angles = qpos[:, joint_addrs[jn]]
                freqs, psd = compute_psd(angles)
                psd_data[cond][jn].append((freqs, psd))
                dom_freqs[cond][jn].append(dominant_frequency(freqs, psd))

    freq_res = FS / NPERSEG
    print(f"Frequency resolution: {freq_res:.3f} Hz (nperseg={NPERSEG}, fs={FS})")

    # ── Figure 1: PSD overlay — one image per side (R / L) ──────────
    SIDES = {
        "right": [("Hip R", "hip_R_extend"), ("Knee R", "knee_R")],
        "left": [("Hip L", "hip_L_extend"), ("Knee L", "knee_L")],
    }

    legend_handles = [
        Line2D([0], [0], color=CONDITION_COLORS["mimic_mjx"], lw=1.4,
               linestyle="-", label="Mimic-MJX"),
        Line2D([0], [0], color=CONDITION_COLORS["code2act"], lw=1.4,
               linestyle="--", label="Code2Act"),
    ]

    for side_name, panels in SIDES.items():
        fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.5))

        for pi, (label, jn) in enumerate(panels):
            ax = axes[pi]
            for cond in ["mimic_mjx", "code2act"]:
                color = CONDITION_COLORS[cond]
                psds = psd_data[cond][jn]
                if not psds:
                    continue
                freqs = psds[0][0]
                psd_matrix = np.array([p[1] for p in psds])
                mean_psd = psd_matrix.mean(axis=0)
                std_psd = psd_matrix.std(axis=0)

                ls = "-" if cond == "mimic_mjx" else "--"
                ax.plot(freqs, mean_psd, color=color, linestyle=ls, linewidth=1.4)
                ax.fill_between(freqs, mean_psd - std_psd, mean_psd + std_psd,
                                color=color, alpha=0.10)

                mask = (freqs >= FMIN) & (freqs <= FMAX)
                if mask.any():
                    dom_f = freqs[mask][np.argmax(mean_psd[mask])]
                    dom_v = mean_psd[mask][np.argmax(mean_psd[mask])]
                    ax.plot(dom_f, dom_v, "o", color=color, markersize=3.5,
                            zorder=5, markeredgecolor="white", markeredgewidth=0.4)

            ax.set_xlim(0, 6)
            ax.set_ylim(bottom=0)
            ax.set_title(label)
            ax.set_xlabel("Frequency (Hz)")
            if pi == 0:
                ax.set_ylabel("PSD")
                ax.legend(handles=legend_handles, frameon=False, fontsize=6,
                          loc="upper right")

        fig.tight_layout()
        border_rect = _add_rounded_border(fig, list(axes))

        for ext in ("png", "pdf"):
            fig.savefig(OUTPUT_DIR / f"psd_overlay_{side_name}.{ext}")
        border_rect.set_facecolor("none")
        fig.savefig(OUTPUT_DIR / f"psd_overlay_{side_name}.svg", transparent=True)
        plt.close()
        for ext in ("png", "pdf", "svg"):
            print(f"Saved: {OUTPUT_DIR / f'psd_overlay_{side_name}.{ext}'}")

    # ── Figure 2: Dominant frequency bar chart ────────────────────────
    fig2, ax2 = plt.subplots(figsize=(5, 3))
    x = np.arange(len(ALL_JOINTS))
    width = 0.35

    for i, cond in enumerate(["mimic_mjx", "code2act"]):
        means = [np.mean(dom_freqs[cond][jn]) for jn in ALL_JOINTS]
        stds = [np.std(dom_freqs[cond][jn]) for jn in ALL_JOINTS]
        offset = (i - 0.5) * width
        bars = ax2.bar(
            x + offset, means, width, yerr=stds,
            color=CONDITION_COLORS[cond], alpha=0.85,
            capsize=3, label=CONDITION_LABELS[cond],
            error_kw={"linewidth": 0.8},
        )
        for bar, m in zip(bars, means):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                     f"{m:.2f}", ha="center", va="bottom", fontsize=6, fontweight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels([JOINT_DISPLAY[j] for j in ALL_JOINTS])
    ax2.set_ylabel("Dominant Frequency (Hz)")
    ax2.set_title("Dominant Gait Frequency Comparison", fontweight="bold")
    ax2.legend(frameon=False, fontsize=7)
    ax2.set_ylim(bottom=0)

    fig2.tight_layout()
    border_rect2 = _add_rounded_border(fig2, [ax2])

    for ext in ("png", "pdf"):
        fig2.savefig(OUTPUT_DIR / f"dominant_frequencies.{ext}")
    border_rect2.set_facecolor("none")
    fig2.savefig(OUTPUT_DIR / "dominant_frequencies.svg", transparent=True)
    plt.close()
    for ext in ("png", "pdf", "svg"):
        print(f"Saved: {OUTPUT_DIR / f'dominant_frequencies.{ext}'}")

    # Print summary
    print(f"\nDominant frequencies (resolution={freq_res:.3f} Hz):")
    for jn in ALL_JOINTS:
        m_mim = np.mean(dom_freqs["mimic_mjx"][jn])
        m_c2a = np.mean(dom_freqs["code2act"][jn])
        print(f"  {JOINT_DISPLAY[jn]:<8}: Mimic={m_mim:.2f} Hz, Code2Act={m_c2a:.2f} Hz")


if __name__ == "__main__":
    main()
