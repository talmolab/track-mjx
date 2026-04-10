"""Experiment 10: Gait Dynamics PSD — Code2Act vs Mimic-MJX.

Compares power spectral density of joint angles during walking to show
that Code2Act preserves the temporal dynamics of locomotion.

Usage:
    cd moseq_jax
    python -m experiments.run_gait_dynamics
"""

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import json
import logging
import sys
from pathlib import Path

import hydra
import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from omegaconf import DictConfig
from scipy.signal import welch

MOSEQ_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = MOSEQ_DIR.parent
for _p in (str(MOSEQ_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from track_mjx.config import utils
from vnl_playground.tasks.rodent.imitation import ReferenceClips
from moseq_env_wrapper import MoSeqImitation

from experiments.shared.checkpoint_utils import (
    load_moseq_checkpoint,
    load_mimic_checkpoint,
    make_inference_fn,
    make_mimic_inference_fn,
    run_rollout,
)
from experiments.shared.plotting import set_nature_style, NATURE_COLORS

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

CONDITION_COLORS = {
    "mimic_mjx": "#D55E00",   # orange
    "code2act": "#0072B2",    # blue
}
CONDITION_LABELS = {
    "mimic_mjx": "Mimic-MJX (oracle)",
    "code2act": "Code2Act",
}

JOINT_DISPLAY = {
    "hip_R_extend": "Hip R",
    "knee_R": "Knee R",
    "hip_L_extend": "Hip L",
    "knee_L": "Knee L",
}


# ---------------------------------------------------------------------------
# Walking clip selection
# ---------------------------------------------------------------------------


def select_walking_clips(
    ref_clips,
    splits: dict,
    k: int = 10,
    min_speed: float = 0.03,
    ctrl_dt: float = 0.01,
    mocap_hz: float = 50.0,
) -> list[int]:
    """Select K clips that are purely walking (XY always moving).

    Args:
        ref_clips: ReferenceClips with all balanced clips.
        splits: Balanced split dict.
        k: Number of clips to select.
        min_speed: Minimum per-frame XY speed (m/s) over entire clip.
        ctrl_dt: Control timestep.
        mocap_hz: Mocap sampling rate.

    Returns:
        List of clip indices (into balanced_clips) for the K best walkers.
    """
    all_categories = (
        splits["balanced"]["train_categories"]
        + splits["balanced"]["test_categories"]
    )
    dt_mocap = 1.0 / mocap_hz

    walk_candidates = []
    qpos_all = np.array(ref_clips.qpos)  # (N_balanced, 250, 74)

    for i, cat in enumerate(all_categories):
        if cat != "walk":
            continue
        xy = qpos_all[i, :, :2]
        frame_speeds = np.linalg.norm(np.diff(xy, axis=0), axis=1) / dt_mocap
        min_s = frame_speeds.min()
        mean_s = frame_speeds.mean()
        if min_s >= min_speed:
            walk_candidates.append((i, mean_s, min_s))

    walk_candidates.sort(key=lambda x: -x[1])  # sort by mean speed descending
    selected = [c[0] for c in walk_candidates[:k]]

    log.info(
        f"Walking clip selection: {len(walk_candidates)} pure walkers "
        f"(min_speed >= {min_speed} m/s) out of "
        f"{sum(1 for c in all_categories if c == 'walk')} walk clips"
    )
    for i, (idx, mean_s, min_s) in enumerate(walk_candidates[:k]):
        log.info(f"  #{i}: clip {idx}, mean_speed={mean_s:.4f}, min_speed={min_s:.4f}")

    return selected


# ---------------------------------------------------------------------------
# Joint angle extraction
# ---------------------------------------------------------------------------


def extract_joint_angles(
    env, qpos_traj: np.ndarray, joint_names: list[str],
) -> dict[str, np.ndarray]:
    """Extract joint angles from qpos trajectory.

    Args:
        env: MoSeqImitation environment (for mj_model access).
        qpos_traj: (T, nq) qpos array.
        joint_names: List of joint names (without model suffix).

    Returns:
        {joint_name: (T,) array of angles in radians}
    """
    mj_model = env.mj_model
    suffix = "-rodent"
    result = {}
    for jn in joint_names:
        full_name = f"{jn}{suffix}"
        joint_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, full_name)
        if joint_id == -1:
            log.warning(f"Joint '{full_name}' not found, skipping")
            continue
        addr = mj_model.jnt_qposadr[joint_id]
        result[jn] = qpos_traj[:, addr]
    return result


# ---------------------------------------------------------------------------
# PSD computation
# ---------------------------------------------------------------------------


def compute_psd(
    angles: np.ndarray, fs: float = 100.0, nperseg: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute power spectral density using Welch's method."""
    freqs, psd = welch(angles, fs=fs, nperseg=nperseg)
    return freqs, psd


def dominant_frequency(
    freqs: np.ndarray, psd: np.ndarray, fmin: float = 1.0, fmax: float = 10.0,
) -> float:
    """Find dominant frequency in [fmin, fmax] range."""
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not mask.any():
        return 0.0
    return float(freqs[mask][np.argmax(psd[mask])])


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_psd_overlay(
    psd_data: dict[str, dict[str, list[tuple[np.ndarray, np.ndarray]]]],
    joint_names: list[str],
    fmin: float,
    fmax: float,
    output_path: str,
) -> None:
    """Plot PSD overlay: Mimic vs Code2Act, per joint, mean + std band.

    Args:
        psd_data: {condition: {joint: [(freqs, psd), ...]}}
    """
    set_nature_style()
    n_joints = len(joint_names)
    ncols = 2
    nrows = (n_joints + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5, 2.5 * nrows))
    axes = axes.flatten()

    for ji, jn in enumerate(joint_names):
        ax = axes[ji]
        for cond in ["mimic_mjx", "code2act"]:
            psds = psd_data[cond][jn]
            freqs = psds[0][0]
            psd_matrix = np.array([p[1] for p in psds])
            mean_psd = psd_matrix.mean(axis=0)
            std_psd = psd_matrix.std(axis=0)

            color = CONDITION_COLORS[cond]
            label = CONDITION_LABELS[cond]
            ls = "-" if cond == "mimic_mjx" else "--"

            ax.plot(freqs, mean_psd, color=color, linestyle=ls, linewidth=1.2, label=label)
            ax.fill_between(freqs, mean_psd - std_psd, mean_psd + std_psd,
                            color=color, alpha=0.15)

            # Mark dominant frequency
            dom_f = dominant_frequency(freqs, mean_psd, fmin, fmax)
            dom_idx = np.argmin(np.abs(freqs - dom_f))
            ax.axvline(dom_f, color=color, linestyle=":", alpha=0.5, linewidth=0.8)

        ax.set_xlim(0, fmax + 2)
        ax.set_title(JOINT_DISPLAY.get(jn, jn), fontsize=8, fontweight="bold")
        ax.set_xlabel("Frequency (Hz)", fontsize=7)
        ax.set_ylabel("PSD", fontsize=7)
        if ji == 0:
            ax.legend(frameon=False, fontsize=6)

    for ji in range(n_joints, len(axes)):
        axes[ji].set_visible(False)

    fig.suptitle("Gait Dynamics: Power Spectral Density", fontsize=9, fontweight="bold")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(f"{output_path}.{ext}", dpi=300, bbox_inches="tight")
    plt.close()
    log.info(f"PSD overlay saved to: {output_path}.png")


def plot_dominant_frequencies(
    dom_freqs: dict[str, dict[str, list[float]]],
    joint_names: list[str],
    output_path: str,
) -> None:
    """Grouped bar chart of dominant frequencies per joint."""
    set_nature_style()
    x = np.arange(len(joint_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(5, 3))

    for i, cond in enumerate(["mimic_mjx", "code2act"]):
        means = [np.mean(dom_freqs[cond][jn]) for jn in joint_names]
        stds = [np.std(dom_freqs[cond][jn]) for jn in joint_names]
        offset = (i - 0.5) * width
        bars = ax.bar(
            x + offset, means, width, yerr=stds,
            color=CONDITION_COLORS[cond], alpha=0.85,
            capsize=3, label=CONDITION_LABELS[cond],
            error_kw={"linewidth": 0.8},
        )
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                    f"{m:.1f}", ha="center", va="bottom", fontsize=6)

    ax.set_xticks(x)
    ax.set_xticklabels([JOINT_DISPLAY.get(j, j) for j in joint_names])
    ax.set_ylabel("Dominant Frequency (Hz)")
    ax.set_title("Dominant Gait Frequency Comparison")
    ax.legend(frameon=False, fontsize=7)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(f"{output_path}.{ext}", dpi=300, bbox_inches="tight")
    plt.close()
    log.info(f"Dominant frequency plot saved to: {output_path}.png")


def plot_trajectory_overlay(
    mimic_angles: dict[str, np.ndarray],
    c2a_angles: dict[str, np.ndarray],
    joint_names: list[str],
    fs: float,
    clip_idx: int,
    output_path: str,
) -> None:
    """Time-domain joint angle overlay for one representative clip."""
    set_nature_style()
    n_joints = len(joint_names)
    fig, axes = plt.subplots(n_joints, 1, figsize=(6, 1.8 * n_joints), sharex=True)
    if n_joints == 1:
        axes = [axes]

    for i, jn in enumerate(joint_names):
        t_mimic = np.arange(len(mimic_angles[jn])) / fs
        t_c2a = np.arange(len(c2a_angles[jn])) / fs

        axes[i].plot(t_mimic, np.degrees(mimic_angles[jn]),
                     color=CONDITION_COLORS["mimic_mjx"], linewidth=1.2,
                     label=CONDITION_LABELS["mimic_mjx"])
        axes[i].plot(t_c2a, np.degrees(c2a_angles[jn]),
                     color=CONDITION_COLORS["code2act"], linewidth=1.2,
                     linestyle="--", label=CONDITION_LABELS["code2act"])

        axes[i].set_ylabel(f"{JOINT_DISPLAY.get(jn, jn)}\n(deg)", fontsize=7)
        if i == 0:
            axes[i].legend(frameon=False, fontsize=6)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Joint Trajectories — Walking Clip {clip_idx}",
                 fontsize=9, fontweight="bold")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(f"{output_path}.{ext}", dpi=300, bbox_inches="tight")
    plt.close()
    log.info(f"Trajectory overlay saved to: {output_path}.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(version_base=None, config_path="configs", config_name="gait_dynamics_exp")
def main(cfg: DictConfig) -> None:
    log.info("=== Gait Dynamics PSD Experiment ===")

    output_dir = Path(cfg.output.base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoints
    log.info("\n--- Loading checkpoints ---")
    ckpt_cfg, norm_state, policy_params, ppo_networks = load_moseq_checkpoint(
        cfg.checkpoint.path, step=cfg.checkpoint.get("step"),
    )
    use_rnn = bool(ckpt_cfg.network_config.get("use_rnn_decoder", False))
    num_codes = int(ckpt_cfg.network_config.num_codes)
    c2a_params = (norm_state, policy_params)

    mimic_cfg, mimic_norm, mimic_policy, mimic_ppo = load_mimic_checkpoint(
        cfg.mimic_checkpoint.path, step=cfg.mimic_checkpoint.get("step"),
    )
    mimic_params = (mimic_norm, mimic_policy)

    # Load data
    codes_data = np.load(cfg.data.codes_path)
    all_codes = codes_data["all_codes"]

    with open(cfg.data.balanced_split_path) as f:
        splits = json.load(f)
    balanced_indices = np.array(
        splits["balanced"]["train_indices"] + splits["balanced"]["test_indices"]
    )

    clip_length = int(ckpt_cfg.env_config.clip_length)
    ctrl_dt = float(ckpt_cfg.env_config.ctrl_dt)
    mocap_hz = float(ckpt_cfg.env_config.mocap_hz)
    steps_per_frame = int(round(1.0 / (mocap_hz * ctrl_dt)))
    max_control_steps = clip_length * steps_per_frame

    balanced_clips = ReferenceClips(
        data_path=cfg.data.reference_data_path,
        n_frames_per_clip=clip_length,
        keep_clips_idx=balanced_indices,
    )

    # Select walking clips
    log.info("\n--- Selecting walking clips ---")
    gd = cfg.gait_dynamics
    walk_clip_indices = select_walking_clips(
        balanced_clips, splits,
        k=int(gd.k_clips),
        min_speed=float(gd.min_xy_speed),
        ctrl_dt=ctrl_dt,
        mocap_hz=mocap_hz,
    )

    if len(walk_clip_indices) == 0:
        log.error("No walking clips found! Try lowering min_xy_speed.")
        return

    if len(walk_clip_indices) < int(gd.k_clips):
        log.warning(
            f"Only {len(walk_clip_indices)} walking clips found "
            f"(requested {gd.k_clips})"
        )

    # Create environment
    _, _, env_cfg = utils.prepare_config(ckpt_cfg)
    env_cfg.start_frame_range = [0, 0]
    env_cfg.domain_randomization.use_domain_randomization = False
    code_stack_size = int(ckpt_cfg.network_config.get("code_stack_size", 1))
    env = MoSeqImitation(
        config=env_cfg,
        clips=balanced_clips,
        kpms_codes=all_codes,
        code_stack_size=code_stack_size,
    )
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    c2a_inf_fn = make_inference_fn(ppo_networks, use_rnn=use_rnn, deterministic=True)
    mimic_inf_fn = make_mimic_inference_fn(mimic_ppo, deterministic=True)

    # Collect rollouts
    joint_names = list(gd.joints)
    fs = float(gd.psd.fs)
    nperseg = int(gd.psd.nperseg)
    fmin = float(gd.psd.fmin)
    fmax = float(gd.psd.fmax)
    seed = int(gd.seed)

    psd_data = {
        "mimic_mjx": {jn: [] for jn in joint_names},
        "code2act": {jn: [] for jn in joint_names},
    }
    dom_freqs = {
        "mimic_mjx": {jn: [] for jn in joint_names},
        "code2act": {jn: [] for jn in joint_names},
    }

    # Save first clip's angles for trajectory overlay
    first_mimic_angles = None
    first_c2a_angles = None

    for ci, clip_idx in enumerate(walk_clip_indices):
        log.info(f"\n--- Walking clip {ci+1}/{len(walk_clip_indices)} (idx={clip_idx}) ---")

        # Code2Act rollout
        codes_mocap = all_codes[clip_idx, :clip_length].astype(np.int32)
        codes_ctrl = np.repeat(codes_mocap, steps_per_frame)
        key = jax.random.PRNGKey(seed + ci)
        c2a_result = run_rollout(
            env, c2a_inf_fn, c2a_params, ppo_networks, use_rnn, key,
            max_steps=max_control_steps,
            code_override=codes_ctrl,
            reset_clip_idx=clip_idx,
            jit_reset=jit_reset, jit_step=jit_step,
            ignore_done=True,
        )

        # Mimic-MJX rollout (same clip)
        key = jax.random.PRNGKey(seed + ci + 10000)
        mimic_result = run_rollout(
            env, mimic_inf_fn, mimic_params, mimic_ppo,
            use_rnn=False, key=key,
            max_steps=max_control_steps,
            reset_clip_idx=clip_idx,
            jit_reset=jit_reset, jit_step=jit_step,
            model_type="mimic_mjx",
            ignore_done=True,
        )

        # Save raw qpos for figure scripts
        np.savez_compressed(
            output_dir / f"rollout_code2act_clip{clip_idx}.npz",
            qpos=c2a_result["qpos"],
        )
        np.savez_compressed(
            output_dir / f"rollout_mimic_mjx_clip{clip_idx}.npz",
            qpos=mimic_result["qpos"],
        )

        # Extract joint angles (control rate)
        c2a_angles = extract_joint_angles(env, c2a_result["qpos"], joint_names)
        mimic_angles = extract_joint_angles(env, mimic_result["qpos"], joint_names)

        if ci == 0:
            first_mimic_angles = mimic_angles
            first_c2a_angles = c2a_angles

        # Compute PSD per joint
        for jn in joint_names:
            if jn not in c2a_angles or jn not in mimic_angles:
                continue

            f_c2a, p_c2a = compute_psd(c2a_angles[jn], fs=fs, nperseg=nperseg)
            f_mim, p_mim = compute_psd(mimic_angles[jn], fs=fs, nperseg=nperseg)

            psd_data["code2act"][jn].append((f_c2a, p_c2a))
            psd_data["mimic_mjx"][jn].append((f_mim, p_mim))

            dom_c2a = dominant_frequency(f_c2a, p_c2a, fmin, fmax)
            dom_mim = dominant_frequency(f_mim, p_mim, fmin, fmax)
            dom_freqs["code2act"][jn].append(dom_c2a)
            dom_freqs["mimic_mjx"][jn].append(dom_mim)

            log.info(
                f"  {jn}: mimic={dom_mim:.1f} Hz, code2act={dom_c2a:.1f} Hz"
            )

    # --- Plots ---
    log.info("\n--- Generating plots ---")

    plot_psd_overlay(
        psd_data, joint_names, fmin, fmax,
        str(output_dir / "psd_overlay"),
    )

    plot_dominant_frequencies(
        dom_freqs, joint_names,
        str(output_dir / "dominant_frequencies"),
    )

    if first_mimic_angles and first_c2a_angles:
        plot_trajectory_overlay(
            first_mimic_angles, first_c2a_angles, joint_names, fs,
            clip_idx=walk_clip_indices[0],
            output_path=str(output_dir / "trajectory_overlay"),
        )

    # --- Save summary ---
    summary = {
        "walk_clip_indices": walk_clip_indices,
        "n_clips": len(walk_clip_indices),
        "dominant_frequencies": {},
    }
    for jn in joint_names:
        summary["dominant_frequencies"][jn] = {
            "mimic_mjx": {
                "mean": float(np.mean(dom_freqs["mimic_mjx"][jn])),
                "std": float(np.std(dom_freqs["mimic_mjx"][jn])),
                "per_clip": dom_freqs["mimic_mjx"][jn],
            },
            "code2act": {
                "mean": float(np.mean(dom_freqs["code2act"][jn])),
                "std": float(np.std(dom_freqs["code2act"][jn])),
                "per_clip": dom_freqs["code2act"][jn],
            },
        }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary table
    log.info(f"\n{'='*60}")
    log.info("DOMINANT FREQUENCY SUMMARY")
    log.info(f"{'='*60}")
    log.info(f"{'Joint':<20} {'Mimic-MJX':>15} {'Code2Act':>15}")
    log.info("-" * 60)
    for jn in joint_names:
        m_mim = np.mean(dom_freqs["mimic_mjx"][jn])
        s_mim = np.std(dom_freqs["mimic_mjx"][jn])
        m_c2a = np.mean(dom_freqs["code2act"][jn])
        s_c2a = np.std(dom_freqs["code2act"][jn])
        log.info(
            f"{JOINT_DISPLAY.get(jn, jn):<20} "
            f"{m_mim:>6.1f} +/- {s_mim:<5.1f} "
            f"{m_c2a:>6.1f} +/- {s_c2a:<5.1f}"
        )
    log.info("=" * 60)

    log.info("\n=== Gait Dynamics Experiment Complete ===")


if __name__ == "__main__":
    main()
