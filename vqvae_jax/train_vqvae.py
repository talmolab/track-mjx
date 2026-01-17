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
    """Rollout logging with VQ-VAE specific metrics.

    Wraps the standard rollout logging to add codebook usage metrics.
    """
    import jax.numpy as jnp
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

        if state.done:
            break

    # Log VQ metrics
    if all_indices:
        indices = jnp.stack(all_indices)
        perplexity, utilization, codes_used = compute_codebook_metrics(
            indices, num_codes
        )
        wandb.log({
            "vq/perplexity": float(perplexity),
            "vq/codebook_utilization": float(utilization),
            "vq/codes_used": int(codes_used),
        }, commit=False)

    if all_z_e:
        z_e = jnp.stack(all_z_e)
        # Log z_e statistics (similar to latent_mean in VAE)
        for i in range(min(5, z_e.shape[-1])):  # Log first 5 dims
            wandb.log({
                f"latents/z_e_mean{i}": float(jnp.mean(z_e[..., i])),
                f"latents/z_e_std{i}": float(jnp.std(z_e[..., i])),
            }, commit=False)

    # Call original logging for video rendering
    if render_video:
        wandb_logging._log_rollout_video(env, cfg, model_path, current_step, rollout_states)


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
    key_split, _ = jax.random.split(jax.random.PRNGKey(cfg.train_setup.train_config.seed))
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
    network_factory = functools.partial(
        make_vq_intention_ppo_networks,
        latent_dim=cfg.network_config.get("latent_dim", cfg.network_config.intention_size),
        num_codes=cfg.network_config.get("num_codes", 512),
        commitment_cost=cfg.network_config.get("commitment_cost", 0.25),
        codebook_init_scale=cfg.network_config.get("codebook_init_scale", 1.0),
        encoder_hidden_layer_sizes=tuple(cfg.network_config.encoder_layer_sizes),
        decoder_hidden_layer_sizes=tuple(cfg.network_config.decoder_layer_sizes),
        value_hidden_layer_sizes=tuple(cfg.network_config.critic_layer_sizes),
    )

    # Initialize wandb
    wandb_logging.initialize_wandb_logging(
        logging_cfg=cfg.logging_config,
        cfg=cfg,
        run_id=run_id,
        existing_run_state=existing_run_state,
    )

    # Log VQ-VAE specific config
    wandb.config.update({
        "arch": "vqvae_intention",
        "num_codes": cfg.network_config.get("num_codes", 512),
        "commitment_cost": cfg.network_config.get("commitment_cost", 0.25),
        "latent_dim": cfg.network_config.get("latent_dim", cfg.network_config.intention_size),
    })

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
        randomization_fn=domain_randomization_maker(
            floor_friction=cfg.env_config.domain_randomization.floor_friction,
            static_friction_scale=cfg.env_config.domain_randomization.static_friction_scale,
            armature_scale=cfg.env_config.domain_randomization.armature_scale,
            com_jitter=cfg.env_config.domain_randomization.com_jitter,
            link_mass_scale=cfg.env_config.domain_randomization.link_mass_scale,
            torso_mass_jitter=cfg.env_config.domain_randomization.torso_mass_jitter,
            qpos0_jitter=cfg.env_config.domain_randomization.qpos0_jitter,
        ) if cfg.env_config.domain_randomization.use_domain_randomization else None,
        # VQ-VAE specific parameters
        commitment_cost=cfg.network_config.get("commitment_cost", 0.25),
        codebook_loss_weight=cfg.network_config.get("codebook_loss_weight", 1.0),
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


if __name__ == "__main__":
    main()
