"""Experiment 1: Inference rollouts comparing code2act vs mimic-mjx oracle.

Generates reward decomposition plots, K-body videos, and transition matrices.
Code2Act = KPMS decoder (codes -> RNN -> action).
Mimic-MJX = pre-trained IntentionNetwork VAE (reference -> encoder -> action).

Usage:
    cd moseq_jax
    python -m experiments.run_inference

    # Override config:
    python -m experiments.run_inference checkpoint.step=50
"""

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import json
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
from experiments.shared.metrics import (
    decompose_rewards,
)
from experiments.shared.plotting import (
    set_nature_style,
    fig_to_image,
    get_trajectory_colors,
    get_code_colormap,
    MODE_COLORS,
    MODE_LABELS,
    BEHAVIOR_COLORS,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


COMPONENT_MARKERS = {
    "total": "-",       # solid
    "coarse": "--",     # dashed
    "fine": ":",        # dotted
}

COMPONENT_LABELS = {
    "total": "Total",
    "coarse": "Coarse (root)",
    "fine": "Fine (joints+end-eff)",
}


def plot_reward_curves(
    results: dict[str, dict[str, np.ndarray]],
    max_steps: int,
) -> plt.Figure:
    """Single plot: modes as colors, components as line styles, normalized Y."""
    set_nature_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    for mode, color in MODE_COLORS.items():
        if mode not in results:
            continue
        mode_label = MODE_LABELS.get(mode, mode)
        for comp, ls in COMPONENT_MARKERS.items():
            curve = results[mode].get(comp)
            if curve is None:
                continue
            # Normalize: divide by max of total to get 0-1 range
            total_max = results[mode].get("total")
            if total_max is not None and total_max.ndim > 1:
                norm_factor = total_max.mean(axis=0).max()
            elif total_max is not None:
                norm_factor = total_max.max()
            else:
                norm_factor = 1.0
            norm_factor = max(norm_factor, 1e-8)

            mean = curve.mean(axis=0) / norm_factor if curve.ndim > 1 else curve / norm_factor
            label = f"{mode_label} — {COMPONENT_LABELS[comp]}"
            ax.plot(mean, color=color, linestyle=ls, label=label, linewidth=1.2)
            if curve.ndim > 1:
                std = curve.std(axis=0) / norm_factor
                ax.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.15, color=color)

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Normalized reward")
    ax.set_title("Reward decomposition")
    ax.legend(frameon=False, fontsize=5.5, ncol=2)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


@hydra.main(version_base=None, config_path="configs", config_name="inference")
def main(cfg: DictConfig) -> None:
    log.info("=== MoSeq Inference Experiment (code2act vs mimic-mjx) ===")

    # Output dir
    output_dir = Path(cfg.output.base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # WandB init
    wandb_enabled = cfg.wandb.get("enabled", False)
    if wandb_enabled:
        run_name = cfg.wandb.get("run_name") or f"moseq_inference_{datetime.now():%y%m%d_%H%M%S}"
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.get("entity"),
            name=run_name,
            config=dict(cfg),
        )

    # Load code2act (MoSeq decoder) checkpoint
    ckpt_path = cfg.checkpoint.path
    ckpt_step = cfg.checkpoint.get("step")
    ckpt_cfg, norm_state, policy_params, ppo_networks = load_moseq_checkpoint(
        ckpt_path, step=ckpt_step,
    )
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
    splits = load_balanced_splits(cfg.data.balanced_split_path)

    # Prepare env config (DR off, fixed start)
    _, _, env_cfg = utils.prepare_config(ckpt_cfg)
    env_cfg.start_frame_range = [0, 0]
    env_cfg.domain_randomization.use_domain_randomization = False

    # Code stack size must match checkpoint training config
    code_stack_size = int(ckpt_cfg.network_config.get("code_stack_size", 1))

    # Code colourmap for rendering
    code_colors = get_code_colormap(num_codes)

    # -------------------------------------------------------------------
    # Run inference per split
    # -------------------------------------------------------------------
    all_transition_codes: list[np.ndarray] = []  # for combined transition matrix

    for split in cfg.inference.splits:
        log.info(f"\n--- Split: {split} ---")

        split_codes = codes_data[f"{split}_codes"]
        split_indices = splits["balanced"][f"{split}_indices"]
        split_clips = ReferenceClips(
            data_path=cfg.data.reference_data_path,
            n_frames_per_clip=int(ckpt_cfg.env_config.clip_length),
            keep_clips_idx=np.array(split_indices),
        )
        env = MoSeqImitation(config=env_cfg, clips=split_clips, kpms_codes=split_codes,
                            code_stack_size=code_stack_size)

        # Pre-compile JIT functions ONCE per env (critical for performance)
        jit_reset = jax.jit(env.reset)
        jit_step = jax.jit(env.step)

        n_clips = split_codes.shape[0]
        max_clips = cfg.inference.get("max_clips")
        if max_clips is not None:
            n_clips = min(n_clips, int(max_clips))
            log.info(f"  Limiting to {n_clips} clips (max_clips={max_clips})")
        max_steps = int(cfg.inference.max_steps)
        seed = int(cfg.inference.seed)

        # Collect per-mode results
        mode_reward_curves: dict[str, dict[str, list]] = {}
        per_mode_data: dict[str, dict] = {}  # store per-mode codes/rewards

        for mode in cfg.inference.modes:
            is_mimic = mode == "mimic_mjx"
            log.info(f"  Mode: {mode}")

            if is_mimic:
                inf_fn = make_mimic_inference_fn(mimic_ppo, deterministic=True)
                mode_params = mimic_params
                mode_ppo = mimic_ppo
                mode_rnn = False
                mode_type = "mimic_mjx"
            else:
                inf_fn = make_inference_fn(
                    ppo_networks, use_rnn=use_rnn, deterministic=True,
                )
                mode_params = code2act_params
                mode_ppo = ppo_networks
                mode_rnn = use_rnn
                mode_type = "code2act"

            all_qpos: list[np.ndarray] = []
            all_rewards: list[np.ndarray] = []
            all_codes: list[np.ndarray] = []
            all_decomposed: list[dict[str, np.ndarray]] = []

            for ci in range(n_clips):
                key = jax.random.PRNGKey(seed + ci)
                result = run_rollout(
                    env, inf_fn, mode_params, mode_ppo, mode_rnn, key,
                    max_steps=max_steps,
                    jit_reset=jit_reset, jit_step=jit_step,
                    model_type=mode_type,
                )
                all_qpos.append(result["qpos"])
                all_rewards.append(result["rewards"])
                all_codes.append(result["code_indices"])
                all_decomposed.append(decompose_rewards(result["per_step_metrics"]))

                if (ci + 1) % 20 == 0 or ci == n_clips - 1:
                    log.info(f"    Clip {ci+1}/{n_clips}: reward={result['rewards'].mean():.3f}")

            # Aggregate reward curves (pad to same length)
            min_len = min(len(r) for r in all_rewards)
            reward_matrix = np.array([r[:min_len] for r in all_rewards])

            # Decomposed curves
            decomp_arrays = {}
            for comp in ["total", "coarse", "fine", "penalty"]:
                curves = [d[comp][:min_len] for d in all_decomposed]
                decomp_arrays[comp] = np.array(curves)

            mode_reward_curves[mode] = decomp_arrays

            # Save to NPZ — comprehensive data for post-hoc plotting
            save_path = output_dir / f"{split}_{mode}.npz"
            save_dict = dict(
                rewards=reward_matrix,
                qpos=np.array(all_qpos, dtype=object),
                code_indices=np.array(all_codes, dtype=object),
            )
            for comp_name, comp_arr in decomp_arrays.items():
                save_dict[f"decomp_{comp_name}"] = comp_arr
            np.savez_compressed(save_path, **save_dict)
            log.info(f"  Saved {save_path}")

            # Log per-mode scalar metrics
            mean_reward = reward_matrix.mean()
            if wandb_enabled:
                wandb.log(
                    {f"inference/{split}/{mode}/mean_reward": mean_reward},
                    commit=False,
                )

            # Store per-mode data for later analysis
            per_mode_data[mode] = {
                "all_codes": list(all_codes),
                "all_rewards": list(all_rewards),
            }

            # Collect codes for transition matrix (code2act only)
            if mode == "code2act":
                all_transition_codes.extend(all_codes)

        # ---------------------------------------------------------------
        # Reward decomposition plots (Claim 2.1)
        # ---------------------------------------------------------------
        log.info("  Plotting reward decomposition...")
        fig_rewards = plot_reward_curves(
            {m: {k: v.mean(axis=0) for k, v in curves.items()} for m, curves in mode_reward_curves.items()},
            max_steps,
        )
        if wandb_enabled:
            wandb.log({f"inference/{split}/reward_decomposition": fig_to_image(fig_rewards)}, commit=False)
        fig_rewards.savefig(output_dir / f"{split}_reward_decomposition.png", dpi=300)
        plt.close(fig_rewards)

        # ---------------------------------------------------------------
        # K-body ghost videos (Claim 2.3)
        # ---------------------------------------------------------------
        K = int(cfg.rendering.K)
        log.info(f"  Rendering K={K} body videos...")

        # Select K clips (balanced across behaviors)
        selected = select_clips_by_behavior(
            splits, split, k_per_behavior=K // 3, seed=seed,
        )
        selected_indices = []
        for beh in ["groom", "walk", "rear"]:
            selected_indices.extend(selected.get(beh, []))
        selected_indices = selected_indices[:K]

        if len(selected_indices) < K:
            log.warning(f"  Only {len(selected_indices)} clips selected (wanted {K})")

        # Run rollouts for selected clips (both modes)
        for mode in cfg.inference.modes:
            is_mimic = mode == "mimic_mjx"

            if is_mimic:
                inf_fn = make_mimic_inference_fn(mimic_ppo, deterministic=True)
                mode_params = mimic_params
                mode_ppo = mimic_ppo
                mode_rnn = False
                mode_type = "mimic_mjx"
            else:
                inf_fn = make_inference_fn(
                    ppo_networks, use_rnn=use_rnn, deterministic=True,
                )
                mode_params = code2act_params
                mode_ppo = ppo_networks
                mode_rnn = use_rnn
                mode_type = "code2act"

            trajectories_qpos = []
            trajectories_codes = []
            for ci in selected_indices:
                key = jax.random.PRNGKey(seed + ci)
                result = run_rollout(
                    env, inf_fn, mode_params, mode_ppo, mode_rnn, key,
                    max_steps=max_steps,
                    jit_reset=jit_reset, jit_step=jit_step,
                    model_type=mode_type,
                )
                trajectories_qpos.append(result["qpos"][:-1])  # remove final extra qpos
                trajectories_codes.append(result["code_indices"])

            # Ghost video
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

                    ghost_path = output_dir / f"{split}_{mode}_ghost_K{K}.mp4"
                    render_ghost_video(
                        ghost_model,
                        base_nq,
                        trajectories_qpos,
                        trajectories_codes,
                        traj_colors,
                        ghost_path,
                        title=f"{split} {mode} (K={K})",
                        fps=int(cfg.rendering.fps),
                        width=int(cfg.rendering.width),
                        height=int(cfg.rendering.height),
                        code_colors=code_colors,
                    )

                    if wandb_enabled:
                        wandb.log(
                            {f"inference/{split}/{mode}/ghost_video": wandb.Video(str(ghost_path), format="mp4")},
                            commit=False,
                        )
                    log.info(f"    Ghost video: {ghost_path}")
                except Exception as e:
                    log.warning(f"    Ghost rendering failed: {e}")

            # Solo videos
            if cfg.rendering.get("solo_videos", True):
                from experiments.shared.ghost_rendering import render_solo_video

                for vi, ci in enumerate(selected_indices):
                    if vi >= len(trajectories_qpos):
                        break
                    solo_path = output_dir / f"{split}_{mode}_solo_{vi}.mp4"
                    try:
                        render_solo_video(
                            env,
                            trajectories_qpos[vi],
                            trajectories_codes[vi],
                            solo_path,
                            fps=int(cfg.rendering.fps),
                            num_codes=num_codes,
                            title=f"Clip {ci} ({mode})",
                        )
                        if wandb_enabled:
                            wandb.log(
                                {f"inference/{split}/{mode}/solo_{vi}": wandb.Video(str(solo_path), format="mp4")},
                                commit=False,
                            )
                    except Exception as e:
                        log.warning(f"    Solo video {vi} failed: {e}")

    # Commit all WandB logs
    if wandb_enabled:
        wandb.log({}, commit=True)
        wandb.finish()
        log.info("WandB run finished")

    log.info("=== Inference experiment complete ===")


if __name__ == "__main__":
    main()
