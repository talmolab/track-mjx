"""
Entry point for prior network training.
Load the config file, create environments, load pretrained mlp_ppo encoder/decoder,
and start prior training with KL loss.
"""

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import functools
import logging

import hydra
import jax
import orbax.checkpoint as ocp
import wandb
from mujoco_playground import wrapper as playground_wrappers
from omegaconf import DictConfig, OmegaConf
from vnl_playground.tasks.rodent import imitation
from vnl_playground.tasks.rodent.reference_clips import ReferenceClips

from track_mjx.agent import checkpointing
from track_mjx.agent import wandb_logging
from track_mjx.config import utils
from track_mjx.agent.domain_randomization import domain_randomization_maker
from track_mjx.agent.mlp_prior.rollout_logging import prior_training_rollout_logging_fn
from track_mjx.agent.mlp_prior import prior_train
from track_mjx.agent.mlp_prior import prior_networks


def _setup_environment() -> None:
    """Configure environment variables for JAX."""
    xla_flags = os.environ.get("XLA_FLAGS", "")
    xla_flags += " --xla_gpu_triton_gemm_any=True"
    os.environ["XLA_FLAGS"] = xla_flags
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def _load_from_run_state_prior(cfg: DictConfig) -> tuple:
    """Load or create run state for prior training, handling preemption and manual restoration.

    Similar to checkpointing.load_from_run_state but uses PriorNetwork step prefix.

    Args:
        cfg: Current configuration.

    Returns:
        Tuple of (run_id, checkpoint_path, existing_run_state).
    """
    from datetime import datetime
    from pathlib import Path

    # Check for existing run state
    existing_run_state = checkpointing.discover_existing_run_state(cfg)

    if existing_run_state:
        run_id = existing_run_state["run_id"]
        checkpoint_path = existing_run_state["checkpoint_path"]
        logging.info(f"Resuming from existing run: {run_id}")
        cfg.train_setup.checkpoint_to_restore = checkpoint_path
    elif cfg.train_setup.restore_from_run_state:
        import fcntl
        import json

        base_path = Path(cfg.logging_config.model_path).resolve()
        full_path = base_path / cfg.train_setup.restore_from_run_state

        with open(full_path, "r") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            existing_run_state = json.load(f)
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        run_id = existing_run_state["run_id"]
        checkpoint_path = existing_run_state["checkpoint_path"]
        logging.info(f"Restoring from specified run state: {run_id}")
        cfg.train_setup.checkpoint_to_restore = checkpoint_path
    else:
        run_id = datetime.now().strftime("%y%m%d_%H%M%S_%f")
        model_path = Path(cfg.logging_config.model_path)
        if not model_path.is_absolute():
            model_path = Path.cwd() / model_path
        checkpoint_path = str(model_path / run_id)

    # If restoring from checkpoint, load the config
    if cfg.train_setup.checkpoint_to_restore is not None:
        from omegaconf import OmegaConf

        checkpoint_to_restore = cfg.train_setup.checkpoint_to_restore
        submitted_timesteps = cfg.train_setup.train_config.num_timesteps

        # Load checkpoint config with PriorNetwork prefix
        cfg = OmegaConf.create(
            checkpointing.load_config_from_checkpoint(
                checkpoint_to_restore, step_prefix="PriorNetwork"
            )
        )
        cfg.train_setup.checkpoint_to_restore = checkpoint_to_restore

        # Allow overriding num_timesteps to extend training
        restored_timesteps = cfg.train_setup.train_config.num_timesteps
        if submitted_timesteps != restored_timesteps:
            logging.info(
                f"Updating num_timesteps: {restored_timesteps} -> {submitted_timesteps}"
            )
            cfg.train_setup.train_config.num_timesteps = submitted_timesteps

        checkpoint_path = checkpoint_to_restore
        run_id = os.path.basename(checkpoint_path)

    logging.info(f"Run ID: {run_id}")
    logging.info(f"Training checkpoint path: {checkpoint_path}")

    return (run_id, checkpoint_path, existing_run_state)


@hydra.main(version_base=None, config_path="config", config_name="rodent-prior")
def main(cfg: DictConfig):
    """Main function for prior network training using Hydra configs"""

    _setup_environment()

    try:
        n_devices = jax.device_count(backend="gpu")
        logging.info(f"Using {n_devices} GPUs")
    except:
        n_devices = 1
        logging.info("Not using GPUs")

    # Validate teacher config
    if cfg.teacher_config.checkpoint_path is None:
        raise ValueError(
            "teacher_config.checkpoint_path must be specified for prior training"
        )

    # Prepare config BEFORE load_from_run_state so the config hash is consistent
    (cfg, cfg_dict, env_cfg_ml) = utils.prepare_config(cfg)

    # Determine how to load from checkpoint
    run_id, checkpoint_path, existing_run_state = _load_from_run_state_prior(cfg)

    # Initialize checkpoint manager with PriorNetwork prefix
    mgr_options = ocp.CheckpointManagerOptions(
        create=True,
        step_prefix="PriorNetwork",
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
    # Create environments
    env = imitation.Imitation(config=env_cfg_ml, clips=train_clips)
    test_env = imitation.Imitation(config=env_cfg_ml, clips=test_clips)

    logging.info(f"Environment config: {cfg.env_config}")

    # Episode length is equal to (clip length - random init range - traj length) * steps per cur frame.
    steps_per_frame = (1 / cfg.env_config.mocap_hz) / cfg.env_config.ctrl_dt
    episode_length = (
        cfg.env_config.clip_length
        - cfg.env_config.start_frame_range[-1]
        - cfg.env_config.reference_length
    ) * steps_per_frame
    logging.info(f"episode_length {episode_length}")

    logging.info("Using Prior Training Pipeline")

    # Initialize wandb logging
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

    # Get prior config
    prior_cfg = cfg.prior_config

    # Load teacher config to get encoder/decoder network sizes
    _, _, _, teacher_cfg = prior_networks.load_frozen_encoder_decoder(
        cfg.teacher_config.checkpoint_path, step=cfg.teacher_config.checkpoint_step
    )
    # Update cfg.network_config with values from teacher (needed for rollout logging)
    teacher_net_cfg = teacher_cfg["network_config"]
    OmegaConf.set_struct(cfg.network_config, False)
    OmegaConf.update(
        cfg.network_config,
        "intention_size",
        teacher_net_cfg["intention_size"],
        merge=False,
    )
    OmegaConf.update(
        cfg.network_config,
        "encoder_layer_sizes",
        teacher_net_cfg["encoder_layer_sizes"],
        merge=False,
    )
    OmegaConf.update(
        cfg.network_config,
        "decoder_layer_sizes",
        teacher_net_cfg["decoder_layer_sizes"],
        merge=False,
    )
    OmegaConf.update(
        cfg.network_config,
        "reference_obs_size",
        teacher_net_cfg["reference_obs_size"],
        merge=False,
    )
    OmegaConf.update(
        cfg.network_config,
        "proprioceptive_obs_size",
        teacher_net_cfg["observation_size"] - teacher_net_cfg["reference_obs_size"],
        merge=False,
    )
    OmegaConf.update(
        cfg.network_config, "action_size", teacher_net_cfg["action_size"], merge=False
    )
    OmegaConf.set_struct(cfg.network_config, True)

    # Setup training function
    train_fn = functools.partial(
        prior_train.train,
        **cfg.train_setup.train_config,
        num_evals=int(
            cfg.train_setup.train_config.num_timesteps / cfg.train_setup.eval_every
        ),
        num_resets_per_eval=cfg.train_setup.eval_every // cfg.train_setup.reset_every,
        episode_length=episode_length,
        mlp_ppo_checkpoint_path=cfg.teacher_config.checkpoint_path,
        mlp_ppo_checkpoint_step=cfg.teacher_config.checkpoint_step,
        kl_weight=prior_cfg.kl_weight,
        use_kl_schedule=prior_cfg.get("use_kl_schedule", True),
        kl_schedule_params=(
            dict(prior_cfg.kl_schedule_params)
            if prior_cfg.get("use_kl_schedule", True)
            else None
        ),
        grad_clip_norm=prior_cfg.get("grad_clip_norm", 10.0),
        prior_hidden_layer_sizes=tuple(cfg.network_config.prior_layer_sizes),
        ckpt_mgr=ckpt_mgr,
        checkpoint_to_restore=cfg.train_setup.checkpoint_to_restore,
        config_dict=cfg_dict,
        eval_env_test_set=test_env,
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
        prior_rollout_config=(
            dict(cfg.prior_rollout_config)
            if hasattr(cfg, "prior_rollout_config")
            and cfg.prior_rollout_config is not None
            else None
        ),
    )

    # Set the render env start frame to always be 0
    # Use train_clips to ensure same clips as prior rollout evaluator
    rollout_cfg = env_cfg_ml.copy_and_resolve_references()
    rollout_cfg.start_frame_range = [0, 0]
    rollout_env = imitation.Imitation(config=rollout_cfg, clips=train_clips)

    # Define the jit reset/step functions for logging
    jit_reset = jax.jit(rollout_env.reset)
    jit_step = jax.jit(rollout_env.step)
    policy_params_fn = functools.partial(
        prior_training_rollout_logging_fn,
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

    # Clean up run state after successful completion
    try:
        checkpointing.cleanup_run_state(cfg)
        logging.info("Prior training completed successfully, cleaned up run state")
    except Exception as e:
        logging.warning(f"Failed to cleanup run state: {e}")


if __name__ == "__main__":
    main()
