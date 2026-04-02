"""Experiment 2: Temporal order & killer demo (code-sequence experiments).

Usage:
    cd moseq_jax
    python -m experiments.run_code_sequence
"""

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import logging
import sys
from datetime import datetime
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import wandb
from omegaconf import DictConfig

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
    make_inference_fn,
    run_rollout,
)
from experiments.shared.clip_selection import load_balanced_splits, select_clips_by_behavior
from experiments.shared.code_sequences import (
    make_correct_sequences,
    make_shuffled_step_sequences,
    make_shuffled_trajectory_sequences,
)
from experiments.shared.metrics import compute_pairwise_joint_divergence
from experiments.shared.plotting import (
    set_nature_style,
    fig_to_image,
    get_trajectory_colors,
    get_code_colormap,
    CONDITION_COLORS,
    BEHAVIOR_COLORS,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

CONDITION_STYLES = {
    "correct": {"color": CONDITION_COLORS["correct"], "linestyle": "-"},
    "shuffled_step": {"color": CONDITION_COLORS["shuffled_step"], "linestyle": "-"},
    "shuffled_trajectory": {"color": CONDITION_COLORS["shuffled_trajectory"], "linestyle": "-"},
}


def plot_divergence_curves(
    curves: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    title: str = "Joint divergence over time",
) -> plt.Figure:
    """Divergence curves with std bands: conditions × modes."""
    set_nature_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    for cond, mode_curves in curves.items():
        style = CONDITION_STYLES.get(cond, {"color": "#999999", "linestyle": "-"})
        for mode, (mean_curve, std_curve) in mode_curves.items():
            ls = "-" if mode == "full" else "--"
            label = f"{cond} ({'code+z_e' if mode == 'full' else 'code only'})"
            ax.plot(mean_curve, color=style["color"], linestyle=ls, label=label, linewidth=1.2)
            ax.fill_between(
                range(len(mean_curve)),
                mean_curve - std_curve, mean_curve + std_curve,
                alpha=0.15, color=style["color"],
            )

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Mean pairwise joint L2")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=5.5, ncol=2)
    plt.tight_layout()
    return fig


def plot_root_displacement(
    panels: dict[str, list[np.ndarray]],
    title: str = "Root displacement",
) -> plt.Figure:
    """All behaviors on one plot. Colors=behavior, shapes=XY vs Z, mean+std."""
    set_nature_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    for beh, trajs in panels.items():
        color = BEHAVIOR_COLORS.get(beh, "#999999")

        # Compute cumulative XY displacement per trajectory
        xy_curves = []
        z_curves = []
        for qpos in trajs:
            min_len = len(qpos)
            xy = qpos[:, :2]
            xy_disp = np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))
            xy_disp = np.concatenate([[0], xy_disp])
            xy_curves.append(xy_disp)

            z_disp = np.cumsum(np.abs(np.diff(qpos[:, 2])))
            z_disp = np.concatenate([[0], z_disp])
            z_curves.append(z_disp)

        # Pad to same length
        max_len = max(len(c) for c in xy_curves)
        xy_padded = np.array([np.pad(c, (0, max_len - len(c)), mode="edge") for c in xy_curves])
        z_padded = np.array([np.pad(c, (0, max_len - len(c)), mode="edge") for c in z_curves])

        # XY displacement: solid line
        xy_mean, xy_std = xy_padded.mean(axis=0), xy_padded.std(axis=0)
        ax.plot(xy_mean, color=color, linestyle="-", linewidth=1.2, label=f"{beh} XY")
        ax.fill_between(range(max_len), xy_mean - xy_std, xy_mean + xy_std, alpha=0.15, color=color)

        # Z displacement: dashed line
        z_mean, z_std = z_padded.mean(axis=0), z_padded.std(axis=0)
        ax.plot(z_mean, color=color, linestyle="--", linewidth=1.2, label=f"{beh} Z")
        ax.fill_between(range(max_len), z_mean - z_std, z_mean + z_std, alpha=0.1, color=color)

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Cumulative displacement")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=5.5, ncol=2)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Temporal order experiment
# ---------------------------------------------------------------------------


def run_temporal_order(
    cfg: DictConfig,
    env,
    inf_fns: dict[str, any],
    params: tuple,
    ppo_networks,
    use_rnn: bool,
    codes: np.ndarray,
    splits: dict,
    num_codes: int,
    output_dir: Path,
    wandb_enabled: bool,
) -> None:
    """Run temporal order experiment (Claim 3)."""
    log.info("\n=== Temporal Order Experiment ===")
    K = int(cfg.temporal_order.K)
    max_steps = int(cfg.temporal_order.max_steps)
    seed = int(cfg.temporal_order.seed)

    # Select K clips (balanced)
    selected = select_clips_by_behavior(splits, "test", k_per_behavior=K // 3, seed=seed)
    clip_indices = []
    for beh in ["groom", "walk", "rear"]:
        clip_indices.extend(selected.get(beh, []))
    clip_indices = clip_indices[:K]

    # Generate code sequences
    correct_seqs = make_correct_sequences(codes, clip_indices, max_steps)
    shuffled_step_seqs = make_shuffled_step_sequences(K, max_steps, num_codes, seed=seed)
    shuffled_traj_seqs = make_shuffled_trajectory_sequences(correct_seqs, seed=seed)

    conditions = {
        "correct": correct_seqs,
        "shuffled_step": shuffled_step_seqs,
        "shuffled_trajectory": shuffled_traj_seqs,
    }

    # Run rollouts for each condition × mode
    divergence_curves: dict[str, dict[str, np.ndarray]] = {}

    for cond_name, cond_seqs in conditions.items():
        divergence_curves[cond_name] = {}

        for mode, inf_fn in inf_fns.items():
            log.info(f"  Condition: {cond_name}, Mode: {mode}")

            trajectories_qpos = []
            trajectories_codes = []

            for ki in range(K):
                key = jax.random.PRNGKey(seed + ki * 100)
                result = run_rollout(
                    env, inf_fn, params, ppo_networks, use_rnn, key,
                    max_steps=max_steps,
                    code_override=cond_seqs[ki],
                )
                trajectories_qpos.append(result["qpos"][:-1])
                trajectories_codes.append(result["code_indices"])

            # Compute divergence (mean + std)
            div_mean, div_std = compute_pairwise_joint_divergence(trajectories_qpos)
            divergence_curves[cond_name][mode] = (div_mean, div_std)

            # Ghost video (3 conditions → 3 bodies per clip not applicable here,
            # instead K bodies per condition)
            if mode == "full" and len(trajectories_qpos) >= 2:
                try:
                    from experiments.shared.ghost_rendering import (
                        build_ghost_model,
                        render_ghost_video,
                    )

                    code_colors = get_code_colormap(num_codes)
                    traj_colors = get_trajectory_colors(len(trajectories_qpos))
                    ghost_model, base_nq = build_ghost_model(
                        env,
                        num_ghosts=len(trajectories_qpos) - 1,
                        ghost_colors=traj_colors[1:],
                        camera_distance=float(cfg.rendering.camera_distance),
                        camera_elevation=float(cfg.rendering.camera_elevation),
                        camera_azimuth=float(cfg.rendering.camera_azimuth),
                        camera_fovy=float(cfg.rendering.camera_fovy),
                    )
                    ghost_path = output_dir / f"temporal_{cond_name}_{mode}.mp4"
                    render_ghost_video(
                        ghost_model, base_nq, trajectories_qpos, trajectories_codes,
                        traj_colors, ghost_path,
                        title=f"{cond_name} ({mode})",
                        fps=int(cfg.rendering.fps),
                        width=int(cfg.rendering.width),
                        height=int(cfg.rendering.height),
                        code_colors=code_colors,
                    )
                    if wandb_enabled:
                        wandb.log(
                            {f"temporal_order/{cond_name}/{mode}/ghost": wandb.Video(str(ghost_path), format="mp4")},
                            commit=False,
                        )
                except Exception as e:
                    log.warning(f"    Ghost rendering failed: {e}")

    # Plot divergence curves
    fig = plot_divergence_curves(divergence_curves)
    if wandb_enabled:
        wandb.log({"temporal_order/divergence_curves": fig_to_image(fig)}, commit=False)
    fig.savefig(output_dir / "temporal_order_divergence.png", dpi=300)
    plt.close(fig)
    log.info("  Divergence plot saved")


# ---------------------------------------------------------------------------
# Killer demo experiment
# ---------------------------------------------------------------------------


def run_killer_demo(
    cfg: DictConfig,
    env,
    inf_fn,
    params: tuple,
    ppo_networks,
    use_rnn: bool,
    codes: np.ndarray,
    clips: ReferenceClips,
    splits: dict,
    num_codes: int,
    output_dir: Path,
    wandb_enabled: bool,
) -> None:
    """Run killer demo experiment (Claim 5)."""
    log.info("\n=== Killer Demo Experiment ===")
    K = int(cfg.killer_demo.K)
    max_steps = int(cfg.killer_demo.max_steps)
    seed = int(cfg.killer_demo.seed)

    code_colors = get_code_colormap(num_codes)

    # Determine anchor heights
    # Low: median root Z across all clips frame 0
    # High: 90th percentile of max root Z
    all_z0 = clips.qpos[:, 0, 2]  # root Z at frame 0
    all_max_z = clips.qpos[:, :, 2].max(axis=1)
    low_z = float(np.median(all_z0))
    high_z = float(np.percentile(all_max_z, 90))
    log.info(f"  Anchor heights: low={low_z:.4f}, high={high_z:.4f}")

    # Collect trajectories: height → behavior → list of qpos
    height_beh_trajs: dict[str, dict[str, list[np.ndarray]]] = {}

    for beh in cfg.killer_demo.behaviors:
        log.info(f"  Behavior: {beh}")

        selected = select_clips_by_behavior(splits, "test", k_per_behavior=K, seed=seed)
        beh_indices = selected.get(beh, [])[:K]

        if len(beh_indices) < 2:
            log.warning(f"    Only {len(beh_indices)} clips for {beh}, skipping")
            continue

        code_seqs = make_correct_sequences(codes, beh_indices, max_steps)

        for height in cfg.killer_demo.start_heights:
            log.info(f"    Height: {height}")
            if height not in height_beh_trajs:
                height_beh_trajs[height] = {}

            trajectories_qpos = []
            trajectories_codes = []

            for ki in range(len(beh_indices)):
                key = jax.random.PRNGKey(seed + ki * 1000)
                result = run_rollout(
                    env, inf_fn, params, ppo_networks, use_rnn, key,
                    max_steps=max_steps,
                    code_override=code_seqs[ki],
                )
                trajectories_qpos.append(result["qpos"][:-1])
                trajectories_codes.append(result["code_indices"])

            height_beh_trajs[height][beh] = trajectories_qpos

            # Ghost video per behavior+height
            if len(trajectories_qpos) >= 2:
                try:
                    from experiments.shared.ghost_rendering import (
                        build_ghost_model,
                        render_ghost_video,
                    )

                    traj_colors = get_trajectory_colors(len(trajectories_qpos))
                    ghost_model, base_nq = build_ghost_model(
                        env,
                        num_ghosts=len(trajectories_qpos) - 1,
                        ghost_colors=traj_colors[1:],
                        camera_distance=float(cfg.rendering.camera_distance),
                        camera_elevation=float(cfg.rendering.camera_elevation),
                        camera_azimuth=float(cfg.rendering.camera_azimuth),
                        camera_fovy=float(cfg.rendering.camera_fovy),
                    )
                    ghost_path = output_dir / f"killer_{beh}_{height}.mp4"
                    render_ghost_video(
                        ghost_model, base_nq, trajectories_qpos, trajectories_codes,
                        traj_colors, ghost_path,
                        title=f"Killer: {beh} ({height} start)",
                        fps=int(cfg.rendering.fps),
                        width=int(cfg.rendering.width),
                        height=int(cfg.rendering.height),
                        code_colors=code_colors,
                    )
                    if wandb_enabled:
                        wandb.log(
                            {f"killer_demo/{beh}/{height}/ghost": wandb.Video(str(ghost_path), format="mp4")},
                            commit=False,
                        )
                except Exception as e:
                    log.warning(f"    Ghost rendering failed: {e}")

    # Combined displacement plot per height (all behaviors, mean+std)
    for height, beh_trajs in height_beh_trajs.items():
        fig = plot_root_displacement(beh_trajs, title=f"Root displacement ({height} start)")
        if wandb_enabled:
            wandb.log({f"killer_demo/{height}/root_displacement": fig_to_image(fig)}, commit=False)
        fig.savefig(output_dir / f"killer_{height}_displacement.png", dpi=300)
        plt.close(fig)
        log.info(f"  Displacement plot saved for {height}")

        # Control: original starting positions
        if cfg.killer_demo.get("include_control", True):
            log.info(f"    Control: original positions")
            trajectories_qpos = []
            trajectories_codes = []

            for ki in range(len(beh_indices)):
                key = jax.random.PRNGKey(seed + ki * 2000)
                # No code override — use env's natural codes
                result = run_rollout(
                    env, inf_fn, params, ppo_networks, use_rnn, key,
                    max_steps=max_steps,
                )
                trajectories_qpos.append(result["qpos"][:-1])
                trajectories_codes.append(result["code_indices"])

            if len(trajectories_qpos) >= 2:
                try:
                    traj_colors = get_trajectory_colors(len(trajectories_qpos))
                    ghost_model, base_nq = build_ghost_model(
                        env,
                        num_ghosts=len(trajectories_qpos) - 1,
                        ghost_colors=traj_colors[1:],
                        camera_distance=float(cfg.rendering.camera_distance),
                        camera_elevation=float(cfg.rendering.camera_elevation),
                        camera_azimuth=float(cfg.rendering.camera_azimuth),
                        camera_fovy=float(cfg.rendering.camera_fovy),
                    )
                    ghost_path = output_dir / f"killer_{beh}_control.mp4"
                    render_ghost_video(
                        ghost_model, base_nq, trajectories_qpos, trajectories_codes,
                        traj_colors, ghost_path,
                        title=f"Control: {beh} (original pos)",
                        fps=int(cfg.rendering.fps),
                        width=int(cfg.rendering.width),
                        height=int(cfg.rendering.height),
                        code_colors=code_colors,
                    )
                    if wandb_enabled:
                        wandb.log(
                            {f"killer_demo/{beh}/control/ghost": wandb.Video(str(ghost_path), format="mp4")},
                            commit=False,
                        )
                except Exception as e:
                    log.warning(f"    Control ghost rendering failed: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(version_base=None, config_path="../configs", config_name="code_sequence_exp")
def main(cfg: DictConfig) -> None:
    log.info("=== Code Sequence Experiments ===")

    output_dir = Path(cfg.output.base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wandb_enabled = cfg.wandb.get("enabled", False)
    if wandb_enabled:
        run_name = f"moseq_code_seq_{datetime.now():%y%m%d_%H%M%S}"
        wandb.init(project=cfg.wandb.project, entity=cfg.wandb.get("entity"), name=run_name, config=dict(cfg))

    # Load checkpoint
    ckpt_cfg, norm_state, policy_params, ppo_networks = load_moseq_checkpoint(cfg.checkpoint.path)
    use_rnn = bool(ckpt_cfg.network_config.get("use_rnn_decoder", False))
    num_codes = int(ckpt_cfg.network_config.num_codes)
    params = (norm_state, policy_params)

    # Load data
    codes_data = np.load(cfg.data.codes_path)
    test_codes = codes_data["test_codes"]
    splits = load_balanced_splits(cfg.data.balanced_split_path)
    test_indices = splits["balanced"]["test_indices"]

    test_clips = ReferenceClips(
        data_path=cfg.data.reference_data_path,
        n_frames_per_clip=int(ckpt_cfg.env_config.clip_length),
        keep_clips_idx=np.array(test_indices),
    )

    # Create env
    _, _, env_cfg = utils.prepare_config(ckpt_cfg)
    env_cfg.start_frame_range = [0, 0]
    env_cfg.domain_randomization.use_domain_randomization = False
    env = MoSeqImitation(config=env_cfg, clips=test_clips, kpms_codes=test_codes)

    # Build inference functions for both modes
    inf_fns = {}
    for mode in cfg.temporal_order.modes:
        z_e_scale = 1.0 if mode == "full" else 0.0
        inf_fns[mode] = make_inference_fn(
            ppo_networks, use_rnn=use_rnn, deterministic=True, z_e_scale=z_e_scale,
        )

    # --- Temporal Order ---
    run_temporal_order(
        cfg, env, inf_fns, params, ppo_networks, use_rnn,
        test_codes, splits, num_codes, output_dir, wandb_enabled,
    )

    # --- Killer Demo ---
    # Use code-only mode for clearest causal demonstration
    killer_inf_fn = inf_fns.get("code_only", inf_fns.get("full"))
    run_killer_demo(
        cfg, env, killer_inf_fn, params, ppo_networks, use_rnn,
        test_codes, test_clips, splits, num_codes, output_dir, wandb_enabled,
    )

    if wandb_enabled:
        wandb.log({}, commit=True)
        wandb.finish()

    log.info("=== Code Sequence Experiments Complete ===")


if __name__ == "__main__":
    main()
