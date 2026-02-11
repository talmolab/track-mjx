"""Entry point for track-mjx vision PPO training.

This module provides the training entry point for vision-augmented tasks
(e.g., RodentRunGapVision) that use egocentric camera observations rendered
via mujoco_warp. Unlike the imitation-based ``train.py``, this entry point:

- Does **not** load reference motion capture clips.
- Uses ``vision_ppo.train()`` which interleaves GPU-based rendering with
  the PPO training loop.
- Passes vision-specific parameters (width, height, grayscale, camera) to
  the training function.
- Uses ``make_vision_ppo_networks`` (CNN encoder + MLP decoder, no VAE
  imitation encoder).
"""

import os

# Must set rendering backend before importing MuJoCo
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import functools
import logging

import hydra
import jax
import orbax.checkpoint as ocp
import wandb
from ml_collections import config_dict
from mujoco_playground import wrapper as playground_wrappers
from omegaconf import DictConfig, OmegaConf
from vnl_playground import registry

from track_mjx.agent import checkpointing, wandb_logging
from track_mjx.agent.ff_ppo import ppo_networks as ff_networks
from track_mjx.agent.ff_ppo import vision_ppo
from track_mjx.agent.domain_randomization import domain_randomization_maker


def _setup_environment() -> None:
    """Configure environment variables for JAX."""
    xla_flags = os.environ.get("XLA_FLAGS", "")
    xla_flags += " --xla_gpu_triton_gemm_any=True"
    os.environ["XLA_FLAGS"] = xla_flags
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


@hydra.main(
    version_base=None, config_path="config", config_name="rodent-run-gap-vision"
)
def main(cfg: DictConfig) -> None:
    """Main training entry point for vision PPO tasks.

    Initializes JAX devices, creates the vision environment (no reference
    clips), and runs vision PPO training with wandb logging.

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

    # ---- Prepare config ---------------------------------------------------
    # Vision configs do not have walker_config or reference data paths, so we
    # skip ``utils.prepare_config`` and build cfg_dict / env_cfg_ml directly.
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    env_cfg_ml = config_dict.ConfigDict(cfg_dict["env_config"])

    # Get environment name from config
    env_name = cfg.env_config.env_name

    # ---- Checkpoint discovery / restoration --------------------------------
    run_id, checkpoint_path, existing_run_state = checkpointing.load_from_run_state(cfg)

    mgr_options = ocp.CheckpointManagerOptions(
        create=True,
        step_prefix="PPONetwork",
    )
    ckpt_mgr = ocp.CheckpointManager(checkpoint_path, options=mgr_options)

    # ---- Create environment (no clips) ------------------------------------
    env = registry.load(env_name, config=env_cfg_ml, flatten_obs=False)

    logging.info(f"Environment config: {cfg.env_config}")

    # Episode length comes directly from the train_setup config
    episode_length = cfg.train_setup.episode_length
    logging.info(f"episode_length {episode_length}")

    # ---- Network factory ---------------------------------------------------
    logging.info("Using Vision PPO Pipeline")

    # Determine vision image shape from config
    vision_width = cfg.env_config.vision_width
    vision_height = cfg.env_config.vision_height
    # Grayscale defaults to True (single channel) if not specified
    grayscale = cfg.env_config.get("grayscale", True)
    vision_channels_count = 1 if grayscale else 3
    vision_shape = (vision_height, vision_width, vision_channels_count)

    # Vision network factory: CNN encoder + MLP decoder (no imitation target)
    network_factory = functools.partial(
        ff_networks.make_vision_ppo_networks,
        vision_shape=vision_shape,
        vision_latent_size=cfg.network_config.get("vision_feature_size", 128),
        decoder_hidden_layer_sizes=tuple(cfg.network_config.decoder_layer_sizes),
        value_hidden_layer_sizes=tuple(cfg.network_config.critic_layer_sizes),
        vision_channels=tuple(cfg.network_config.get("vision_channels", [32, 64, 64])),
    )

    # ---- Wandb logging -----------------------------------------------------
    wandb_logging.initialize_wandb_logging(
        logging_cfg=cfg.logging_config,
        cfg=cfg,
        run_id=run_id,
        existing_run_state=existing_run_state,
    )

    # Save initial run state after wandb initialization
    if existing_run_state is None:
        checkpointing.save_run_state(
            cfg=cfg,
            run_id=run_id,
            checkpoint_path=checkpoint_path,
            wandb_run_id=wandb.run.id,
        )

    # Create the checkpoint callback
    checkpoint_callback = checkpointing.create_checkpoint_callback(
        cfg=cfg,
        run_id=run_id,
        checkpoint_path=checkpoint_path,
        wandb_run_id=wandb.run.id,
    )

    # ---- Build training arguments ------------------------------------------
    train_kwargs = dict(
        **cfg.train_setup.train_config,
        num_evals=int(
            cfg.train_setup.train_config.num_timesteps / cfg.train_setup.eval_every
        ),
        num_resets_per_eval=cfg.train_setup.eval_every // cfg.train_setup.reset_every,
        episode_length=episode_length,
        latent_kl_weight=cfg.network_config.latent_kl_weight,
        latent_ar1_weight=cfg.network_config.latent_ar1_weight,
        network_factory=network_factory,
        ckpt_mgr=ckpt_mgr,
        checkpoint_to_restore=cfg.train_setup.checkpoint_to_restore,
        config_dict=cfg_dict,
        use_kl_schedule=cfg.network_config.kl_schedule,
        checkpoint_callback=checkpoint_callback,
        freeze_decoder=cfg.train_setup.get("freeze_decoder", False),
        get_activation=cfg.train_setup.train_config.get("get_activation", False),
        wrap_for_training=functools.partial(
            playground_wrappers.wrap_for_brax_training, full_reset=False
        ),
        # Vision-specific parameters
        vision_width=vision_width,
        vision_height=vision_height,
        grayscale=grayscale,
        camera_name=cfg.env_config.get("camera_name", "egocentric-rodent"),
        # Domain randomization
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
            if cfg.env_config.get("domain_randomization", {}).get(
                "use_domain_randomization", False
            )
            else None
        ),
    )

    train_fn = functools.partial(vision_ppo.train, **train_kwargs)

    # ---- Rollout environment for video logging -----------------------------
    rollout_env = registry.load(env_name, config=env_cfg_ml, flatten_obs=False)

    jit_reset = jax.jit(rollout_env.reset)
    jit_step = jax.jit(rollout_env.step)
    policy_params_fn = functools.partial(
        wandb_logging.rollout_logging_fn,
        rollout_env,
        jit_reset,
        jit_step,
        cfg,
        checkpoint_path,
    )

    # ---- Run training ------------------------------------------------------
    make_inference_fn, params, _ = train_fn(
        environment=env,
        progress_fn=wandb_logging.wandb_progress,
        policy_params_fn=policy_params_fn,
    )

    # Clean up run state after successful completion
    try:
        checkpointing.cleanup_run_state(cfg)
        logging.info("Training completed successfully, cleaned up run state")
    except Exception as e:
        logging.warning(f"Failed to cleanup run state: {e}")


if __name__ == "__main__":
    main()
