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
from vnl_playground.tasks.rodent.reference_clips import ReferenceClips

# Import from main codebase
from track_mjx.config import utils
from track_mjx.agent import checkpointing, wandb_logging
from track_mjx.agent.domain_randomization import domain_randomization_maker

# Import VQ-VAE modules from scratch
from vq_ppo_networks import make_vq_intention_ppo_networks
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
):
    """Rollout logging with VQ-VAE specific metrics and code visualization.

    Wraps the standard rollout logging to add codebook usage metrics and
    renders video with code transition timeline overlay.
    """
    import jax.numpy as jnp
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from vq_losses import compute_codebook_metrics

    # Get network config for num_codes
    num_codes = cfg.network_config.get("num_codes", 512)

    # Calculate episode length from config (same as VAE wandb_logging.py)
    physics_steps_per_ctrl = cfg.env_config.ctrl_dt / cfg.env_config.sim_dt
    steps_per_mocap_frame = (1 / cfg.env_config.mocap_hz) / (
        cfg.env_config.sim_dt * physics_steps_per_ctrl
    )
    episode_length = int(cfg.env_config.clip_length * steps_per_mocap_frame)

    # Run standard rollout
    key = policy_params_fn_key
    state = jit_reset(key)

    rollout_states = [state]
    all_indices = []
    all_z_e = []
    all_rewards = []

    # Collect rollout
    for _ in range(episode_length):
        key, subkey = jax.random.split(key)
        action, extras = jit_logging_inference_fn(params, state.obs, subkey)

        # Collect VQ-specific data
        # Note: ppo.py uses ppo_networks.make_logging_inference_fn which returns
        # VAE-style keys. For VQ-VAE: z_e is stored as "latent_mean" and
        # indices are stored as "latent_logvar" due to the API mismatch.
        if "latent_logvar" in extras:
            # In VQ-VAE context, "latent_logvar" is actually the codebook indices
            all_indices.append(extras["latent_logvar"])
        if "latent_mean" in extras:
            # In VQ-VAE context, "latent_mean" is actually z_e (encoder output)
            all_z_e.append(extras["latent_mean"])

        state = jit_step(state, action)
        rollout_states.append(state)
        all_rewards.append(float(state.reward))

        if state.done:
            break

    # Convert indices to numpy array
    indices_array = None
    if all_indices:
        indices_array = np.array([int(idx) for idx in all_indices])

        # Log VQ metrics
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

        # Compute transition rate for this rollout
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

        # PCA visualization of latent space: z_e and codebook vectors
        if indices_array is not None and len(indices_array) > 0:
            from sklearn.decomposition import PCA

            # Get codebook from params (params is tuple of (normalizer_params, policy_params))
            policy_params = params[1]
            codebook = np.array(policy_params["params"]["quantizer"]["embeddings"])
            z_e_np = np.array(z_e)

            # Fit PCA on combined z_e and codebook for consistent projection
            combined = np.vstack([z_e_np, codebook])
            pca = PCA(n_components=2)
            combined_2d = pca.fit_transform(combined)

            # Split back into z_e and codebook projections
            z_e_2d = combined_2d[: len(z_e_np)]
            codebook_2d = combined_2d[len(z_e_np) :]

            # Create PCA visualization
            fig, ax = plt.subplots(figsize=(10, 8))

            # Plot z_e points colored by their assigned code
            colors = get_nature_colormap(num_codes) / 255.0
            for code_idx in range(num_codes):
                mask = indices_array == code_idx
                if np.any(mask):
                    ax.scatter(
                        z_e_2d[mask, 0],
                        z_e_2d[mask, 1],
                        c=[colors[code_idx]],
                        label=f"z_e → code {code_idx}",
                        alpha=0.6,
                        s=30,
                    )

            # Plot codebook vectors as larger stars
            for code_idx in range(num_codes):
                ax.scatter(
                    codebook_2d[code_idx, 0],
                    codebook_2d[code_idx, 1],
                    c=[colors[code_idx]],
                    marker="*",
                    s=400,
                    edgecolors="black",
                    linewidths=1.5,
                    zorder=10,
                )

            # Compute mean distance from z_e to their assigned codebook vectors
            z_e_to_codebook_dist = np.mean(
                [
                    np.linalg.norm(z_e_np[i] - codebook[indices_array[i]])
                    for i in range(len(z_e_np))
                ]
            )

            ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
            ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
            ax.set_title(
                f"Latent Space PCA: z_e (dots) and Codebook (stars)\n"
                f"Mean z_e-to-codebook distance: {z_e_to_codebook_dist:.3f}"
            )
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            wandb.log({"vq/latent_pca": wandb.Image(fig)}, commit=False)
            plt.close(fig)

            # Also log the mean distance as a metric
            wandb.log(
                {
                    "vq/eval_z_e_to_codebook_dist": float(z_e_to_codebook_dist),
                },
                commit=False,
            )

    # Render video with code overlay
    if render_video:
        import mujoco

        render_fps = cfg.render_config.render_fps
        video_path = f"{model_path}/{current_step}.mp4"

        try:
            # Use custom rendering with code transition bar
            render_rollout_to_video(
                env=env,
                rollout_states=rollout_states,
                output_path=video_path,
                camera=f"{cfg.render_config.render_camera_name}{env._suffix}",
                width=640,
                height=480,
                fps=render_fps,
                indices=indices_array,
                num_codes=num_codes,
                code_bar_height=40,
            )

            wandb.log(
                {"videos/rollout": wandb.Video(video_path, format="mp4")},
                commit=False,
            )

            # Render per-code videos showing frames grouped by code (if enabled)
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

                    # Log each per-code video to wandb
                    for code_idx, video_path in per_code_paths.items():
                        wandb.log(
                            {
                                f"videos/per_code/code_{code_idx}": wandb.Video(
                                    video_path, format="mp4"
                                )
                            },
                            commit=False,
                        )
                    logging.info(
                        f"Logged {len(per_code_paths)} per-code videos to wandb"
                    )
                except Exception as e:
                    logging.warning(f"Failed to render per-code videos: {e}")

        except mujoco.FatalError as e:
            logging.warning(f"Rendering video failed with MuJoCo error: {e}")
        except Exception as e:
            logging.warning(f"Failed to render video: {e}")


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
    (cfg, cfg_dict, env_cfg_ml) = utils.prepare_config(cfg)

    # Initialize checkpoint manager
    mgr_options = ocp.CheckpointManagerOptions(
        create=True,
        step_prefix="VQPPONetwork",
    )
    ckpt_mgr = ocp.CheckpointManager(checkpoint_path, options=mgr_options)

    # Create the reference clips
    logging.info(f"Loading data: {cfg.env_config.reference_data_path}")
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
    env = imitation.Imitation(config=env_cfg_ml, clips=train_clips)
    test_env = imitation.Imitation(config=env_cfg_ml, clips=test_clips)

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

    # VQ-VAE network factory
    # Get temporal encoder settings (with backward-compatible defaults)
    encoder_type = cfg.network_config.get("encoder_type", "mlp")
    temporal_stride = cfg.network_config.get("temporal_stride", 1)
    encoder_hidden_channels = cfg.network_config.get(
        "encoder_hidden_channels", [256, 256]
    )
    encoder_kernel_size = cfg.network_config.get("encoder_kernel_size", 3)

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
        encoder_type=encoder_type,
        temporal_stride=temporal_stride,
        encoder_hidden_channels=tuple(encoder_hidden_channels),
        encoder_kernel_size=encoder_kernel_size,
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
            "ce_stickiness_cost": cfg.network_config.get("ce_stickiness_cost", 0.0),
            "ce_stickiness_temperature": cfg.network_config.get(
                "ce_stickiness_temperature", 1.0
            ),
            "latent_dim": cfg.network_config.get(
                "latent_dim", cfg.network_config.intention_size
            ),
            # Temporal encoder settings
            "encoder_type": encoder_type,
            "temporal_stride": temporal_stride,
            "encoder_hidden_channels": list(encoder_hidden_channels),
            "encoder_kernel_size": encoder_kernel_size,
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
        ce_stickiness_cost=cfg.network_config.get("ce_stickiness_cost", 0.0),
        ce_stickiness_temperature=cfg.network_config.get(
            "ce_stickiness_temperature", 1.0
        ),
    )

    # Set the render env start frame to always be 0
    rollout_cfg = env_cfg_ml.copy_and_resolve_references()
    rollout_cfg.start_frame_range = [0, 0]
    rollout_env = imitation.Imitation(config=rollout_cfg)

    # Define jit reset/step functions
    jit_reset = jax.jit(rollout_env.reset)
    jit_step = jax.jit(rollout_env.step)

    policy_params_fn = functools.partial(
        vq_rollout_logging_fn,
        rollout_env,
        jit_reset,
        jit_step,
        cfg,
        checkpoint_path,
    )

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
