"""Entry point for VQ-VAE training.

Usage:
    cd scratch/vqvae_jax
    python train_vqvae.py

    # Override config values:
    python train_vqvae.py network_config.num_codes=128 train_setup.train_config.num_envs=512

    # Use a different config file:
    python train_vqvae.py --config-name=vqvae_minimal
"""

import os

# Must set rendering backend before importing MuJoCo
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import sys
import functools
import logging
from pathlib import Path

# Add paths
SCRATCH_DIR = Path(__file__).parent
REPO_ROOT = SCRATCH_DIR.parent.parent
sys.path.insert(0, str(SCRATCH_DIR))
sys.path.insert(0, str(REPO_ROOT))

import hydra
import jax
import orbax.checkpoint as ocp
import wandb
from mujoco_playground import wrapper as playground_wrappers
from omegaconf import DictConfig
from vnl_playground.tasks.rodent import imitation
from vnl_playground.tasks.rodent.imitation import ReferenceClips
from vnl_playground.tasks import wrappers as rodent_wrappers

from ref_joints_imitation import RefJointsImitation

# Import from main codebase
from track_mjx.config import utils
from track_mjx.agent import checkpointing, wandb_logging
from track_mjx.agent.domain_randomization import domain_randomization_maker

# Import VQ-VAE modules from scratch
from vq_ppo_networks import (
    make_vq_intention_ppo_networks,
    make_vq_chunked_ppo_networks,
    make_vq_chunked_logging_inference_fn,
)
from vq_ppo import train as vq_train
from analysis.rendering import (
    render_rollout_to_video,
    render_per_code_videos,
    get_nature_colormap,
)


def _setup_environment() -> None:
    """Configure environment variables for JAX."""
    xla_flags = os.environ.get("XLA_FLAGS", "")
    xla_flags += " --xla_gpu_triton_gemm_any=True"
    os.environ["XLA_FLAGS"] = xla_flags
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def vq_rollout_logging_fn(
    env,
    jit_reset,
    jit_step,
    cfg,
    model_path,
    current_step,
    jit_logging_inference_fn,
    params,
    policy_params_fn_key,
    render_video=True,
    ppo_network=None,  # Added for compatibility with main PPO code
    reinit_data=None,
    jit_chunked_logging_fn=None,
    use_code_chunking=False,
):
    """Rollout logging with VQ-VAE specific metrics and code visualization.

    Wraps the standard rollout logging to add codebook usage metrics and
    renders video with code transition timeline overlay. Runs multiple rollouts
    to compute aggregated transition statistics.
    """
    import jax.numpy as jnp
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from vq_losses import compute_codebook_metrics, compute_codebook_metrics_per_depth

    # Get network config for num_codes
    num_codes = cfg.network_config.get("num_codes", 512)

    # Calculate episode length from config (same as VAE wandb_logging.py)
    physics_steps_per_ctrl = cfg.env_config.ctrl_dt / cfg.env_config.sim_dt
    steps_per_mocap_frame = (1 / cfg.env_config.mocap_hz) / (
        cfg.env_config.sim_dt * physics_steps_per_ctrl
    )
    episode_length = int(cfg.env_config.clip_length * steps_per_mocap_frame)

    # Number of rollouts to run for transition matrix aggregation
    n_rollouts = cfg.render_config.get("eval_rollouts_for_transition", 16)

    # Run multiple rollouts
    key = policy_params_fn_key
    all_rollout_indices: list[np.ndarray] = []
    all_rollout_qpos: list[np.ndarray] = []
    all_rollout_states: list[list] = []
    all_rollout_z_e: list[list] = []
    all_rollout_rewards: list[list] = []

    rvq_depth = int(cfg.network_config.get("rvq_depth", 1))
    all_rollout_indices_per_depth = [[] for _ in range(rvq_depth)]
    # Per-rollout per-depth indices (for first-rollout video rendering)
    per_rollout_depth_indices: list[list[list[int]]] = []

    for rollout_i in range(n_rollouts):
        key, subkey = jax.random.split(key)
        state = jit_reset(subkey)

        rollout_states = [state]
        rollout_indices = []
        rollout_qpos = []
        rollout_z_e = []
        rollout_rewards = []
        # Per-depth indices for RVQ logging
        rollout_indices_per_depth = [[] for _ in range(rvq_depth)]
        prev_indices = None

        # Initialize chunk state for chunked eval
        if use_code_chunking and jit_chunked_logging_fn is not None:
            held_d0_idx = jnp.zeros((), dtype=jnp.int32)
            chunk_tau = jnp.zeros((), dtype=jnp.int32)

        for _ in range(episode_length):
            key, subkey = jax.random.split(key)

            if use_code_chunking and jit_chunked_logging_fn is not None:
                action, extras, (held_d0_idx, chunk_tau) = jit_chunked_logging_fn(
                    params, state.obs, (held_d0_idx, chunk_tau), subkey
                )
            else:
                action, extras = jit_logging_inference_fn(
                    params, state.obs, subkey, prev_indices
                )

            # Collect VQ-specific data
            if "indices" in extras:
                curr_indices = extras["indices"]
                rollout_indices.append(int(curr_indices))

                # Update prev_indices: use all_indices tuple for multi-level stickiness
                if "all_indices" in extras:
                    prev_indices = extras["all_indices"]
                    # Collect per-depth indices
                    for d in range(rvq_depth):
                        if d < len(extras["all_indices"]):
                            rollout_indices_per_depth[d].append(
                                int(extras["all_indices"][d])
                            )
                else:
                    prev_indices = curr_indices

            if "z_e" in extras:
                rollout_z_e.append(extras["z_e"])

            # Collect qpos from state (handle both data and pipeline_state)
            if hasattr(state, "data"):
                rollout_qpos.append(np.array(state.data.qpos))
            elif hasattr(state, "pipeline_state"):
                rollout_qpos.append(np.array(state.pipeline_state.q))

            state = jit_step(state, action)
            rollout_states.append(state)
            rollout_rewards.append(float(state.reward))

            if state.done:
                prev_indices = None
                # Reset chunk state on done
                if use_code_chunking and jit_chunked_logging_fn is not None:
                    held_d0_idx = jnp.zeros((), dtype=jnp.int32)
                    chunk_tau = jnp.zeros((), dtype=jnp.int32)
                break

        all_rollout_indices.append(np.array(rollout_indices))
        if rollout_qpos:
            all_rollout_qpos.append(np.stack(rollout_qpos))
        all_rollout_states.append(rollout_states)
        all_rollout_z_e.append(rollout_z_e)
        all_rollout_rewards.append(rollout_rewards)
        # Save this rollout's per-depth indices and accumulate globally
        per_rollout_depth_indices.append(rollout_indices_per_depth)
        for d in range(rvq_depth):
            all_rollout_indices_per_depth[d].extend(rollout_indices_per_depth[d])

    # Store rollout data for dead code reinit (if enabled)
    if reinit_data is not None:
        # Aggregate z_e across all rollouts
        all_z_e_flat = []
        for rollout_z_e_list in all_rollout_z_e:
            all_z_e_flat.extend(rollout_z_e_list)
        if all_z_e_flat:
            reinit_data["z_e"] = np.stack([np.array(z) for z in all_z_e_flat])
        # Store accumulated per-depth indices
        if all_rollout_indices_per_depth[0]:
            reinit_data["all_indices"] = tuple(
                np.array(all_rollout_indices_per_depth[d]) for d in range(rvq_depth)
            )

    # Use first rollout for single-rollout metrics and video rendering
    indices_array = all_rollout_indices[0] if all_rollout_indices else None
    rollout_states = all_rollout_states[0] if all_rollout_states else []
    all_z_e = all_rollout_z_e[0] if all_rollout_z_e else []
    all_rewards = all_rollout_rewards[0] if all_rollout_rewards else []

    # First rollout's per-depth indices (for video + per-depth metrics)
    first_rollout_per_depth = (
        per_rollout_depth_indices[0] if per_rollout_depth_indices else None
    )

    if indices_array is not None and len(indices_array) > 0:
        # Log VQ metrics from first rollout
        indices_jnp = jnp.array(indices_array)
        perplexity, utilization, codes_used = compute_codebook_metrics(
            indices_jnp, num_codes
        )
        wandb.log(
            {
                "vq/perplexity": float(perplexity),
                "vq/codebook_utilization": float(utilization),
                "vq/codes_used": int(codes_used),
            },
            commit=False,
        )

        # Compute transition rate for first rollout
        code_transitions = np.sum(indices_array[1:] != indices_array[:-1])
        transition_rate = code_transitions / max(len(indices_array) - 1, 1)
        wandb.log(
            {
                "vq/eval_transition_rate": float(transition_rate),
                "vq/eval_transitions": int(code_transitions),
                "vq/eval_steps": len(indices_array),
            },
            commit=False,
        )

        # Log per-depth metrics for RVQ (using first rollout)
        if rvq_depth > 1 and first_rollout_per_depth:
            # Per-depth codebook metrics (perplexity, utilization, codes_used)
            valid_depths = [
                np.array(first_rollout_per_depth[d])
                for d in range(rvq_depth)
                if first_rollout_per_depth[d]
            ]
            if valid_depths:
                depth_indices = tuple(jnp.array(a) for a in valid_depths)
                depth_metrics = compute_codebook_metrics_per_depth(
                    depth_indices, num_codes
                )
                wandb.log(
                    {f"vq/{k}": float(v) for k, v in depth_metrics.items()},
                    commit=False,
                )

            # Per-depth transition rates
            for d in range(rvq_depth):
                if first_rollout_per_depth[d]:
                    d_arr = np.array(first_rollout_per_depth[d])
                    d_trans = np.sum(d_arr[1:] != d_arr[:-1])
                    d_rate = d_trans / max(len(d_arr) - 1, 1)
                    wandb.log(
                        {f"vq/eval_transition_rate_d{d}": float(d_rate)},
                        commit=False,
                    )

        # Create code sequence timeline plot
        fig, axes = plt.subplots(2, 1, figsize=(12, 4), height_ratios=[1, 2])

        # Top: code usage histogram
        code_counts = np.bincount(indices_array, minlength=num_codes)
        colors = get_nature_colormap(num_codes) / 255.0
        axes[0].bar(range(num_codes), code_counts, color=colors, edgecolor="none")
        axes[0].set_xlabel("Code Index")
        axes[0].set_ylabel("Count")
        axes[0].set_title(
            f"Code Usage (perplexity={float(perplexity):.2f}, used={int(codes_used)}/{num_codes})"
        )
        axes[0].set_xlim(-0.5, num_codes - 0.5)

        # Bottom: code sequence timeline
        timesteps = np.arange(len(indices_array))
        for i in range(len(indices_array) - 1):
            code = indices_array[i]
            axes[1].axvspan(
                timesteps[i], timesteps[i + 1], color=colors[code], alpha=0.8
            )
        # Last segment
        if len(indices_array) > 0:
            axes[1].axvspan(
                timesteps[-1],
                timesteps[-1] + 1,
                color=colors[indices_array[-1]],
                alpha=0.8,
            )

        axes[1].set_xlabel("Timestep")
        axes[1].set_ylabel("Code")
        axes[1].set_title(
            f"Code Sequence (transitions={code_transitions}, rate={transition_rate:.2%})"
        )
        axes[1].set_xlim(0, len(indices_array))
        axes[1].set_ylim(-0.5, num_codes - 0.5)

        # Add code index markers for unique codes used
        unique_codes = np.unique(indices_array)
        axes[1].set_yticks(unique_codes)

        plt.tight_layout()
        wandb.log({"vq/code_sequence": wandb.Image(fig)}, commit=False)
        plt.close(fig)

        # Log code sequence as wandb Table for detailed inspection
        table_data = [[int(t), int(c)] for t, c in enumerate(indices_array)]
        table = wandb.Table(columns=["timestep", "code_index"], data=table_data)
        wandb.log({"vq/code_sequence_table": table}, commit=False)

    if all_z_e:
        z_e = jnp.stack(all_z_e)
        # Log z_e statistics (similar to latent_mean in VAE)
        for i in range(min(5, z_e.shape[-1])):  # Log first 5 dims
            wandb.log(
                {
                    f"latents/z_e_mean{i}": float(jnp.mean(z_e[..., i])),
                    f"latents/z_e_std{i}": float(jnp.std(z_e[..., i])),
                },
                commit=False,
            )

    # Render video(s) with code overlay (runs every render_interval evals)
    if render_video:
        import mujoco

        render_fps = cfg.render_config.render_fps
        num_videos = min(
            int(cfg.render_config.get("num_eval_rollout_videos", 1)),
            n_rollouts,
        )

        for vid_i in range(num_videos):
            vid_indices = (
                all_rollout_indices[vid_i] if vid_i < len(all_rollout_indices) else None
            )
            vid_states = (
                all_rollout_states[vid_i] if vid_i < len(all_rollout_states) else []
            )
            vid_per_depth = (
                per_rollout_depth_indices[vid_i]
                if vid_i < len(per_rollout_depth_indices)
                else None
            )

            video_path = f"{model_path}/{current_step}_vid{vid_i}.mp4"

            try:
                # Build per-depth index arrays for multi-depth bar rendering
                video_indices_per_depth = None
                if rvq_depth > 1 and vid_per_depth and vid_per_depth[0]:
                    video_indices_per_depth = [
                        np.array(vid_per_depth[d]) for d in range(rvq_depth)
                    ]

                render_rollout_to_video(
                    env=env,
                    rollout_states=vid_states,
                    output_path=video_path,
                    camera=f"{cfg.render_config.render_camera_name}{env._suffix}",
                    width=640,
                    height=480,
                    fps=render_fps,
                    indices=vid_indices,
                    num_codes=num_codes,
                    code_bar_height=30,
                    indices_per_depth=video_indices_per_depth,
                )

                wandb.log(
                    {f"videos/rollout_{vid_i}": wandb.Video(video_path, format="mp4")},
                    commit=False,
                )

            except mujoco.FatalError as e:
                logging.warning(
                    f"Rendering video {vid_i} failed with MuJoCo error: {e}"
                )
            except Exception as e:
                logging.warning(f"Failed to render video {vid_i}: {e}")

        # Render per-code videos from first rollout (if enabled)
        render_per_code = cfg.render_config.get("render_per_code_videos", False)
        if render_per_code and indices_array is not None and len(indices_array) > 0:
            per_code_dir = f"{model_path}/per_code_videos/{current_step}"
            try:
                per_code_paths = render_per_code_videos(
                    env=env,
                    rollout_states=rollout_states,
                    indices=indices_array,
                    output_dir=per_code_dir,
                    num_codes=num_codes,
                    camera=f"{cfg.render_config.render_camera_name}{env._suffix}",
                    width=640,
                    height=480,
                    fps=render_fps,
                    min_frames_per_code=5,
                )

                for code_idx, vpath in per_code_paths.items():
                    wandb.log(
                        {
                            f"videos/per_code/code_{code_idx}": wandb.Video(
                                vpath, format="mp4"
                            )
                        },
                        commit=False,
                    )
                logging.info(f"Logged {len(per_code_paths)} per-code videos to wandb")
            except Exception as e:
                logging.warning(f"Failed to render per-code videos: {e}")


@hydra.main(version_base=None, config_path="configs", config_name="vqvae_minimal")
def main(cfg: DictConfig) -> None:
    """Main VQ-VAE training entry point using Hydra configs.

    Args:
        cfg: Hydra configuration containing env_config, network_config,
            train_setup, and logging_config.
    """
    _setup_environment()

    try:
        n_devices = jax.device_count(backend="gpu")
        logging.info(f"Using {n_devices} GPUs")
    except RuntimeError:
        n_devices = 1
        logging.info("Not using GPUs")

    # Determine checkpoint path
    run_id, checkpoint_path, existing_run_state = checkpointing.load_from_run_state(cfg)

    # Prepare config
    cfg, cfg_dict, env_cfg_ml = utils.prepare_config(cfg)

    # Initialize checkpoint manager
    mgr_options = ocp.CheckpointManagerOptions(
        create=True,
        step_prefix="VQPPONetwork",
    )
    ckpt_mgr = ocp.CheckpointManager(checkpoint_path, options=mgr_options)

    # Create the reference clips
    logging.info(f"Loading data: {cfg.env_config.reference_data_path}")
    balanced_split_path = cfg.env_config.get("balanced_split_path", None)
    if balanced_split_path:
        import json
        import numpy as np

        with open(balanced_split_path) as f:
            splits = json.load(f)
        train_indices = splits["balanced"]["train_indices"]
        test_indices = splits["balanced"]["test_indices"]

        train_clips = ReferenceClips(
            data_path=cfg.env_config.reference_data_path,
            n_frames_per_clip=cfg.env_config.clip_length,
            keep_clips_idx=np.array(train_indices),
        )
        test_clips = ReferenceClips(
            data_path=cfg.env_config.reference_data_path,
            n_frames_per_clip=cfg.env_config.clip_length,
            keep_clips_idx=np.array(test_indices),
        )
        logging.info(
            f"Loaded balanced splits: {len(train_indices)} train, "
            f"{len(test_indices)} test from {balanced_split_path}"
        )
    else:
        reference_clips = ReferenceClips(
            data_path=cfg.env_config.reference_data_path,
            n_frames_per_clip=cfg.env_config.clip_length,
            keep_clips_idx=cfg.env_config.keep_clips_idx,
        )

        # Create train/test split
        key_split, _ = jax.random.split(
            jax.random.PRNGKey(cfg.train_setup.train_config.seed)
        )
        train_clips, test_clips = reference_clips.split(
            train_ratio=cfg.train_setup.train_subset_ratio,
            seed=key_split,
        )

    # Create environments (dict observations, no flattening)
    use_ref_joints_encoder = bool(
        cfg.network_config.get("use_ref_joints_encoder", False)
    )
    EnvClass = RefJointsImitation if use_ref_joints_encoder else imitation.Imitation
    if use_ref_joints_encoder:
        logging.info("Using RefJointsImitation (raw ref_joints encoder input)")
    env = rodent_wrappers.LegacyObsWrapper(
        rodent_wrappers.TrackMjxObsWrapper(
            EnvClass(config=env_cfg_ml, clips=train_clips)
        )
    )
    test_env = rodent_wrappers.LegacyObsWrapper(
        rodent_wrappers.TrackMjxObsWrapper(
            EnvClass(config=env_cfg_ml, clips=test_clips)
        )
    )

    logging.info(f"Environment config: {cfg.env_config}")

    # Episode length calculation
    steps_per_frame = (1 / cfg.env_config.mocap_hz) / (cfg.env_config.ctrl_dt)
    episode_length = (
        cfg.env_config.clip_length
        - cfg.env_config.start_frame_range[-1]
        - cfg.env_config.reference_length
    ) * steps_per_frame
    logging.info(f"episode_length {episode_length}")

    logging.info("Using VQ-VAE PPO Pipeline")

    # Resolve stickiness_bias: may be float or list/ListConfig from config
    raw_bias = cfg.network_config.get("stickiness_bias", 0.0)
    try:
        # Handles list, tuple, and OmegaConf ListConfig
        stickiness_bias = tuple(float(b) for b in raw_bias)
    except TypeError:
        # Scalar float/int
        stickiness_bias = float(raw_bias)

    rvq_depth = int(cfg.network_config.get("rvq_depth", 1))
    use_rotation = bool(cfg.network_config.get("use_rotation", False))
    coupled_residual_grad = bool(cfg.network_config.get("coupled_residual_grad", False))
    codebook_entropy_weight = float(
        cfg.network_config.get("codebook_entropy_weight", 0.0)
    )
    codebook_entropy_temperature = float(
        cfg.network_config.get("codebook_entropy_temperature", 1.0)
    )
    dead_code_reinit = bool(cfg.network_config.get("dead_code_reinit", False))
    dead_code_threshold = float(cfg.network_config.get("dead_code_threshold", 0.01))
    num_codes = int(cfg.network_config.get("num_codes", 32))
    proprio_noise_scale = float(cfg.network_config.get("proprio_noise_scale", 0.0))
    use_continuous_latent = bool(cfg.network_config.get("use_continuous_latent", False))
    continuous_latent_dim = int(cfg.network_config.get("continuous_latent_dim", 4))
    kl_weight = float(cfg.network_config.get("kl_weight", 0.0))

    # Code chunking (Semi-MDP temporal commitment) config
    use_code_chunking = bool(cfg.network_config.get("use_code_chunking", False))
    code_commitment_horizon = int(cfg.network_config.get("code_commitment_horizon", 0))

    # Assertions for code chunking
    if use_code_chunking:
        assert not coupled_residual_grad, (
            "coupled_residual_grad must be False when using code chunking. "
            "Coupled gradients through held D0 codes produce zero D1 gradients "
            "at worker steps ((H-1)/H of all steps)."
        )
        assert code_commitment_horizon > 0, (
            f"code_commitment_horizon must be > 0 when use_code_chunking=True, "
            f"got {code_commitment_horizon}"
        )
        unroll_length = cfg.train_setup.train_config.unroll_length
        if unroll_length % code_commitment_horizon != 0:
            logging.warning(
                f"unroll_length ({unroll_length}) is not divisible by "
                f"code_commitment_horizon ({code_commitment_horizon}). "
                f"This causes inconsistent commitment windows at unroll boundaries."
            )

    # Shared mutable dict for dead code reinit data (populated by rollout callback)
    reinit_data = {} if dead_code_reinit else None

    # VQ-VAE network factory
    if use_code_chunking:
        network_factory = functools.partial(
            make_vq_chunked_ppo_networks,
            commitment_horizon=code_commitment_horizon,
            latent_dim=cfg.network_config.get(
                "latent_dim", cfg.network_config.intention_size
            ),
            num_codes=cfg.network_config.get("num_codes", 512),
            commitment_cost=cfg.network_config.get("commitment_cost", 0.25),
            codebook_init_scale=cfg.network_config.get("codebook_init_scale", 1.0),
            encoder_hidden_layer_sizes=tuple(cfg.network_config.encoder_layer_sizes),
            decoder_hidden_layer_sizes=tuple(cfg.network_config.decoder_layer_sizes),
            value_hidden_layer_sizes=tuple(cfg.network_config.critic_layer_sizes),
            stickiness_bias=stickiness_bias,
            rvq_depth=rvq_depth,
            use_rotation=use_rotation,
            coupled_residual_grad=coupled_residual_grad,
            proprio_noise_scale=proprio_noise_scale,
            use_continuous_latent=use_continuous_latent,
            continuous_latent_dim=continuous_latent_dim,
            use_ref_joints_encoder=use_ref_joints_encoder,
        )
    else:
        network_factory = functools.partial(
            make_vq_intention_ppo_networks,
            latent_dim=cfg.network_config.get(
                "latent_dim", cfg.network_config.intention_size
            ),
            num_codes=cfg.network_config.get("num_codes", 512),
            commitment_cost=cfg.network_config.get("commitment_cost", 0.25),
            codebook_init_scale=cfg.network_config.get("codebook_init_scale", 1.0),
            encoder_hidden_layer_sizes=tuple(cfg.network_config.encoder_layer_sizes),
            decoder_hidden_layer_sizes=tuple(cfg.network_config.decoder_layer_sizes),
            value_hidden_layer_sizes=tuple(cfg.network_config.critic_layer_sizes),
            stickiness_bias=stickiness_bias,
            rvq_depth=rvq_depth,
            use_rotation=use_rotation,
            coupled_residual_grad=coupled_residual_grad,
            proprio_noise_scale=proprio_noise_scale,
            use_continuous_latent=use_continuous_latent,
            continuous_latent_dim=continuous_latent_dim,
            use_ref_joints_encoder=use_ref_joints_encoder,
        )

    # Initialize wandb
    wandb_logging.initialize_wandb_logging(
        logging_cfg=cfg.logging_config,
        cfg=cfg,
        run_id=run_id,
        existing_run_state=existing_run_state,
    )

    # Log VQ-VAE specific config
    wandb.config.update(
        {
            "arch": "vqvae_intention",
            "num_codes": cfg.network_config.get("num_codes", 512),
            "commitment_cost": cfg.network_config.get("commitment_cost", 0.25),
            "codebook_loss_weight": cfg.network_config.get("codebook_loss_weight", 1.0),
            "stickiness_bias": stickiness_bias,
            "latent_dim": cfg.network_config.get(
                "latent_dim", cfg.network_config.intention_size
            ),
            "rvq_depth": rvq_depth,
            "use_rotation": use_rotation,
            "coupled_residual_grad": coupled_residual_grad,
            "codebook_entropy_weight": codebook_entropy_weight,
            "codebook_entropy_temperature": codebook_entropy_temperature,
            "dead_code_reinit": dead_code_reinit,
            "dead_code_threshold": dead_code_threshold,
            "proprio_noise_scale": proprio_noise_scale,
            "use_continuous_latent": use_continuous_latent,
            "kl_weight": kl_weight,
            "use_code_chunking": use_code_chunking,
            "code_commitment_horizon": code_commitment_horizon,
        }
    )

    # Save initial run state
    if existing_run_state is None:
        checkpointing.save_run_state(
            cfg=cfg,
            run_id=run_id,
            checkpoint_path=checkpoint_path,
            wandb_run_id=wandb.run.id,
        )

    # Create checkpoint callback
    checkpoint_callback = checkpointing.create_checkpoint_callback(
        cfg=cfg,
        run_id=run_id,
        checkpoint_path=checkpoint_path,
        wandb_run_id=wandb.run.id,
    )

    # Training function with VQ-VAE loss (via vq_ppo.train wrapper)
    train_fn = functools.partial(
        vq_train,
        **cfg.train_setup.train_config,
        num_evals=int(
            cfg.train_setup.train_config.num_timesteps / cfg.train_setup.eval_every
        ),
        num_resets_per_eval=cfg.train_setup.eval_every // cfg.train_setup.reset_every,
        episode_length=episode_length,
        network_factory=network_factory,
        ckpt_mgr=ckpt_mgr,
        checkpoint_to_restore=cfg.train_setup.checkpoint_to_restore,
        config_dict=cfg_dict,
        eval_env_test_set=test_env,
        freeze_decoder=cfg.train_setup.get("freeze_decoder", False),
        checkpoint_callback=checkpoint_callback,
        wrap_for_training=functools.partial(
            playground_wrappers.wrap_for_brax_training, full_reset=False
        ),
        randomization_fn=(
            domain_randomization_maker(
                floor_friction=cfg.env_config.domain_randomization.floor_friction,
                static_friction_scale=cfg.env_config.domain_randomization.static_friction_scale,
                armature_scale=cfg.env_config.domain_randomization.armature_scale,
                com_jitter=cfg.env_config.domain_randomization.com_jitter,
                link_mass_scale=cfg.env_config.domain_randomization.link_mass_scale,
                torso_mass_jitter=cfg.env_config.domain_randomization.torso_mass_jitter,
                qpos0_jitter=cfg.env_config.domain_randomization.qpos0_jitter,
            )
            if cfg.env_config.domain_randomization.use_domain_randomization
            else None
        ),
        # VQ-VAE specific parameters
        commitment_cost=cfg.network_config.get("commitment_cost", 0.25),
        codebook_loss_weight=cfg.network_config.get("codebook_loss_weight", 1.0),
        rvq_depth=rvq_depth,
        codebook_entropy_weight=codebook_entropy_weight,
        codebook_entropy_temperature=codebook_entropy_temperature,
        dead_code_reinit=dead_code_reinit,
        dead_code_threshold=dead_code_threshold,
        num_codes=num_codes,
        reinit_data=reinit_data,
        kl_weight=kl_weight,
        use_code_chunking=use_code_chunking,
        code_commitment_horizon=code_commitment_horizon,
    )

    # Set the render env start frame to always be 0
    rollout_cfg = env_cfg_ml.copy_and_resolve_references()
    rollout_cfg.start_frame_range = [0, 0]
    rollout_env = rodent_wrappers.LegacyObsWrapper(
        rodent_wrappers.TrackMjxObsWrapper(
            EnvClass(config=rollout_cfg, clips=test_clips)
        )
    )

    # Define jit reset/step functions
    jit_reset = jax.jit(rollout_env.reset)
    jit_step = jax.jit(rollout_env.step)

    # Build chunked logging inference fn lazily (needs ppo_network from training)
    _chunked_logging_fn_cache = {}

    def _get_chunked_logging_fn(ppo_network):
        """Build and cache chunked logging inference fn."""
        if "fn" not in _chunked_logging_fn_cache:
            make_fn = make_vq_chunked_logging_inference_fn(
                ppo_network, code_commitment_horizon
            )
            _chunked_logging_fn_cache["fn"] = jax.jit(make_fn(deterministic=True))
        return _chunked_logging_fn_cache["fn"]

    def _policy_params_fn_wrapper(
        current_step,
        jit_logging_inference_fn,
        params,
        policy_params_fn_key,
        render_video=True,
        ppo_network=None,
    ):
        """Wrapper that injects chunked logging fn when code chunking is enabled."""
        jit_chunked_fn = None
        if use_code_chunking and ppo_network is not None:
            jit_chunked_fn = _get_chunked_logging_fn(ppo_network)

        return vq_rollout_logging_fn(
            rollout_env,
            jit_reset,
            jit_step,
            cfg,
            checkpoint_path,
            current_step=current_step,
            jit_logging_inference_fn=jit_logging_inference_fn,
            params=params,
            policy_params_fn_key=policy_params_fn_key,
            render_video=render_video,
            ppo_network=ppo_network,
            reinit_data=reinit_data,
            jit_chunked_logging_fn=jit_chunked_fn,
            use_code_chunking=use_code_chunking,
        )

    policy_params_fn = _policy_params_fn_wrapper

    # Run training
    make_inference_fn, params, _ = train_fn(
        environment=env,
        progress_fn=wandb_logging.wandb_progress,
        policy_params_fn=policy_params_fn,
    )

    # Cleanup
    try:
        checkpointing.cleanup_run_state(cfg)
        logging.info("Training completed successfully, cleaned up run state")
    except Exception as e:
        logging.warning(f"Failed to cleanup run state: {e}")

    wandb.finish()


if __name__ == "__main__":
    main()
