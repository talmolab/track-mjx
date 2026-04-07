"""Experiment 2: Temporal order & killer demo (code-sequence experiments).

Compares code2act (KPMS decoder) against mimic-mjx (oracle VAE) to
demonstrate that codes are causally used: shuffling codes degrades
code2act but has no effect on mimic-mjx (which ignores codes).

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
    load_mimic_checkpoint,
    make_inference_fn,
    make_mimic_inference_fn,
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
    NATURE_COLORS,
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
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    mimic_baseline: tuple[np.ndarray, np.ndarray] | None = None,
    title: str = "Joint divergence over time",
) -> plt.Figure:
    """Divergence curves with std bands: code2act conditions + mimic-mjx baseline.

    Args:
        curves: ``{condition_name: (mean, std)}`` for code2act conditions.
        mimic_baseline: Optional ``(mean, std)`` for mimic-mjx oracle.
        title: Plot title.
    """
    set_nature_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    for cond, (mean_curve, std_curve) in curves.items():
        style = CONDITION_STYLES.get(cond, {"color": "#999999", "linestyle": "-"})
        label = f"Code2Act: {cond}"
        ax.plot(mean_curve, color=style["color"], linestyle="-", label=label, linewidth=1.2)
        ax.fill_between(
            range(len(mean_curve)),
            mean_curve - std_curve, mean_curve + std_curve,
            alpha=0.15, color=style["color"],
        )

    if mimic_baseline is not None:
        m_mean, m_std = mimic_baseline
        ax.plot(m_mean, color=NATURE_COLORS["gray"], linestyle="--",
                label="Mimic-MJX (oracle)", linewidth=1.2)
        ax.fill_between(
            range(len(m_mean)),
            m_mean - m_std, m_mean + m_std,
            alpha=0.1, color=NATURE_COLORS["gray"],
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
    inf_fn_code2act,
    inf_fn_mimic,
    params_code2act: tuple,
    params_mimic: tuple,
    ppo_networks,
    mimic_ppo,
    use_rnn: bool,
    codes: np.ndarray,
    splits: dict,
    num_codes: int,
    output_dir: Path,
    wandb_enabled: bool,
    jit_reset=None,
    jit_step=None,
) -> None:
    """Run temporal order experiment (Claim 3).

    Shuffling codes degrades code2act but NOT mimic-mjx, proving codes
    are causally used by the decoder.
    """
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

    # Run code2act rollouts for each condition
    divergence_curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for cond_name, cond_seqs in conditions.items():
        log.info(f"  Code2Act — condition: {cond_name}")

        trajectories_qpos = []
        trajectories_codes = []

        for ki in range(K):
            key = jax.random.PRNGKey(seed + ki * 100)
            result = run_rollout(
                env, inf_fn_code2act, params_code2act, ppo_networks, use_rnn, key,
                max_steps=max_steps,
                code_override=cond_seqs[ki],
                jit_reset=jit_reset, jit_step=jit_step,
                model_type="code2act",
            )
            trajectories_qpos.append(result["qpos"][:-1])
            trajectories_codes.append(result["code_indices"])

        # Compute divergence (mean + std)
        div_mean, div_std = compute_pairwise_joint_divergence(trajectories_qpos)
        divergence_curves[cond_name] = (div_mean, div_std)

        # Save per-condition trajectory data
        np.savez_compressed(
            output_dir / f"temporal_{cond_name}_code2act.npz",
            qpos=np.array(trajectories_qpos, dtype=object),
            code_indices=np.array(trajectories_codes, dtype=object),
            divergence_mean=div_mean, divergence_std=div_std,
            code_sequences=np.array(cond_seqs[:K], dtype=object),
        )

        # Ghost video for code2act (correct condition only)
        if cond_name == "correct" and len(trajectories_qpos) >= 2:
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
                ghost_path = output_dir / f"temporal_{cond_name}_code2act.mp4"
                render_ghost_video(
                    ghost_model, base_nq, trajectories_qpos, trajectories_codes,
                    traj_colors, ghost_path,
                    title=f"{cond_name} (code2act)",
                    fps=int(cfg.rendering.fps),
                    width=int(cfg.rendering.width),
                    height=int(cfg.rendering.height),
                    code_colors=code_colors,
                )
                if wandb_enabled:
                    wandb.log(
                        {f"temporal_order/{cond_name}/code2act/ghost": wandb.Video(str(ghost_path), format="mp4")},
                        commit=False,
                    )
            except Exception as e:
                log.warning(f"    Ghost rendering failed: {e}")

    # Run mimic-mjx baseline (single run, no code_override — it ignores codes)
    log.info("  Mimic-MJX oracle baseline")
    mimic_qpos = []
    for ki in range(K):
        key = jax.random.PRNGKey(seed + ki * 100)
        result = run_rollout(
            env, inf_fn_mimic, params_mimic, mimic_ppo, False, key,
            max_steps=max_steps,
            jit_reset=jit_reset, jit_step=jit_step,
            model_type="mimic_mjx",
        )
        mimic_qpos.append(result["qpos"][:-1])
    mimic_baseline = compute_pairwise_joint_divergence(mimic_qpos)

    # Save mimic baseline data
    np.savez_compressed(
        output_dir / "temporal_mimic_baseline.npz",
        qpos=np.array(mimic_qpos, dtype=object),
        divergence_mean=mimic_baseline[0], divergence_std=mimic_baseline[1],
    )

    # Save all divergence curves together for easy replotting
    div_save = {}
    for cond_name, (dm, ds) in divergence_curves.items():
        div_save[f"{cond_name}_mean"] = dm
        div_save[f"{cond_name}_std"] = ds
    div_save["mimic_mean"] = mimic_baseline[0]
    div_save["mimic_std"] = mimic_baseline[1]
    np.savez_compressed(output_dir / "temporal_divergence_all.npz", **div_save)

    # Plot divergence curves with mimic-mjx baseline
    fig = plot_divergence_curves(divergence_curves, mimic_baseline=mimic_baseline)
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
    jit_reset=None,
    jit_step=None,
) -> None:
    """Run killer demo experiment (Claim 5) — code2act only."""
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
                    jit_reset=jit_reset, jit_step=jit_step,
                )
                trajectories_qpos.append(result["qpos"][:-1])
                trajectories_codes.append(result["code_indices"])

            height_beh_trajs[height][beh] = trajectories_qpos

            # Save killer demo trajectories per behavior+height
            np.savez_compressed(
                output_dir / f"killer_{beh}_{height}.npz",
                qpos=np.array(trajectories_qpos, dtype=object),
                code_indices=np.array(trajectories_codes, dtype=object),
                code_sequences=np.array(code_seqs, dtype=object),
                behavior=beh, height=height,
            )

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
                    jit_reset=jit_reset, jit_step=jit_step,
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


@hydra.main(version_base=None, config_path="configs", config_name="code_sequence_exp")
def main(cfg: DictConfig) -> None:
    log.info("=== Code Sequence Experiments (code2act vs mimic-mjx) ===")

    output_dir = Path(cfg.output.base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wandb_enabled = cfg.wandb.get("enabled", False)
    if wandb_enabled:
        run_name = f"moseq_code_seq_{datetime.now():%y%m%d_%H%M%S}"
        wandb.init(project=cfg.wandb.project, entity=cfg.wandb.get("entity"), name=run_name, config=dict(cfg))

    # Load code2act (MoSeq decoder) checkpoint
    ckpt_cfg, norm_state, policy_params, ppo_networks = load_moseq_checkpoint(cfg.checkpoint.path)
    use_rnn = bool(ckpt_cfg.network_config.get("use_rnn_decoder", False))
    num_codes = int(ckpt_cfg.network_config.num_codes)
    code2act_params = (norm_state, policy_params)

    # Load mimic-mjx (oracle VAE) checkpoint
    mimic_cfg, mimic_norm, mimic_policy, mimic_ppo = load_mimic_checkpoint(
        cfg.mimic_checkpoint.path, step=cfg.mimic_checkpoint.get("step"),
    )
    mimic_params = (mimic_norm, mimic_policy)

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

    # Pre-compile JIT functions ONCE (critical for performance)
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    # Build inference functions
    inf_fn_code2act = make_inference_fn(
        ppo_networks, use_rnn=use_rnn, deterministic=True,
    )
    inf_fn_mimic = make_mimic_inference_fn(mimic_ppo, deterministic=True)

    # --- Temporal Order ---
    run_temporal_order(
        cfg, env, inf_fn_code2act, inf_fn_mimic,
        code2act_params, mimic_params,
        ppo_networks, mimic_ppo, use_rnn,
        test_codes, splits, num_codes, output_dir, wandb_enabled,
        jit_reset=jit_reset, jit_step=jit_step,
    )

    # --- Killer Demo (code2act only) ---
    run_killer_demo(
        cfg, env, inf_fn_code2act, code2act_params, ppo_networks, use_rnn,
        test_codes, test_clips, splits, num_codes, output_dir, wandb_enabled,
        jit_reset=jit_reset, jit_step=jit_step,
    )

    if wandb_enabled:
        wandb.log({}, commit=True)
        wandb.finish()

    log.info("=== Code Sequence Experiments Complete ===")


if __name__ == "__main__":
    main()
