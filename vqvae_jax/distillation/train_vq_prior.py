"""Entry point for VQ-VAE Prior Distillation training.

This script trains a Prior network to predict VQ-VAE encoder outputs
from proprioceptive observations only. The trained Prior can then be
used for "freeloop" control where no reference trajectory is needed.

Usage:
    cd vqvae_jax
    python train_vq_prior.py vqvae_config.checkpoint_path=/path/to/vqvae/checkpoint

    # Override config values:
    python train_vq_prior.py \\
        vqvae_config.checkpoint_path=/path/to/checkpoint \\
        loss_config.loss_type=l2 \\
        train_setup.train_config.learning_rate=1e-4

    # Use a different config file:
    python train_vq_prior.py --config-name=vq_prior_distill_custom
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
DISTILL_DIR = Path(__file__).parent
VQVAE_DIR = DISTILL_DIR.parent
REPO_ROOT = VQVAE_DIR.parent
sys.path.insert(0, str(DISTILL_DIR))
sys.path.insert(0, str(VQVAE_DIR))
sys.path.insert(0, str(REPO_ROOT))

import hydra
import jax
import orbax.checkpoint as ocp
import wandb
from mujoco_playground import wrapper as playground_wrappers
from omegaconf import DictConfig, OmegaConf
from vnl_playground.tasks.rodent import imitation
from vnl_playground.tasks.rodent import wrappers as vnl_wrappers
from vnl_playground.tasks.rodent.reference_clips import ReferenceClips

# Import from main codebase
from track_mjx.config import utils
from track_mjx.agent import checkpointing, wandb_logging
from track_mjx.agent.domain_randomization import domain_randomization_maker

# Import VQ-VAE Prior modules
from vq_prior_distill import train as vq_prior_train
from vq_prior_rollout import VQPriorFreelloopEvaluator


def _setup_environment() -> None:
    """Configure environment variables for JAX."""
    xla_flags = os.environ.get("XLA_FLAGS", "")
    xla_flags += " --xla_gpu_triton_gemm_any=True"
    os.environ["XLA_FLAGS"] = xla_flags
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def vq_prior_rollout_logging_fn(
    env,
    cfg,
    model_path,
    current_step,
    prior_params,
    frozen_decoder_params,
    frozen_codebook,
    policy_params_fn_key,
    render_video=True,
):
    """Rollout logging with VQ-VAE prior specific metrics.

    Logs prior output statistics and optionally renders freeloop videos.
    """
    import jax.numpy as jnp

    # Log prior params statistics
    normalizer_params, prior_weights = prior_params

    # Flatten prior weights and compute statistics
    flat_params = jax.tree_util.tree_leaves(prior_weights)
    total_params = sum(p.size for p in flat_params)
    param_norm = jnp.sqrt(sum(jnp.sum(p**2) for p in flat_params))

    wandb.log(
        {
            "prior/total_params": total_params,
            "prior/param_norm": float(param_norm),
        },
        commit=False,
    )

    # Codebook statistics
    wandb.log(
        {
            "frozen/codebook_mean": float(jnp.mean(frozen_codebook)),
            "frozen/codebook_std": float(jnp.std(frozen_codebook)),
        },
        commit=False,
    )


@hydra.main(version_base=None, config_path="configs", config_name="vq_prior_distill")
def main(cfg: DictConfig) -> None:
    """Main VQ-VAE Prior Distillation training entry point.

    Args:
        cfg: Hydra configuration containing env_config, network_config,
            vqvae_config, loss_config, train_setup, and logging_config.
    """
    _setup_environment()

    # Validate required config
    if cfg.vqvae_config.checkpoint_path is None:
        raise ValueError(
            "vqvae_config.checkpoint_path is required. "
            "Provide the path to your frozen VQ-VAE checkpoint."
        )

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
        step_prefix="VQPriorDistill",
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

    # Create environments
    env = vnl_wrappers.FlattenObsWrapper(
        imitation.Imitation(config=env_cfg_ml, clips=train_clips)
    )
    test_env = vnl_wrappers.FlattenObsWrapper(
        imitation.Imitation(config=env_cfg_ml, clips=test_clips)
    )

    logging.info(f"Environment config: {cfg.env_config}")

    # Episode length calculation
    steps_per_frame = (1 / cfg.env_config.mocap_hz) / cfg.env_config.ctrl_dt
    episode_length = (
        cfg.env_config.clip_length
        - cfg.env_config.start_frame_range[-1]
        - cfg.env_config.reference_length
    ) * steps_per_frame
    logging.info(f"episode_length {episode_length}")

    logging.info("Using VQ-VAE Prior Distillation Pipeline")
    logging.info(f"VQ-VAE checkpoint: {cfg.vqvae_config.checkpoint_path}")

    # Initialize wandb
    wandb_logging.initialize_wandb_logging(
        logging_cfg=cfg.logging_config,
        cfg=cfg,
        run_id=run_id,
        existing_run_state=existing_run_state,
    )

    # Log VQ-VAE Prior specific config
    wandb.config.update(
        {
            "arch": "vq_prior_distill",
            "vqvae_checkpoint": cfg.vqvae_config.checkpoint_path,
            "loss_type": cfg.loss_config.loss_type,
            "ar_weight": cfg.loss_config.ar_weight,
            "prior_layer_sizes": list(cfg.network_config.prior_layer_sizes),
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

    # Training function
    train_fn = functools.partial(
        vq_prior_train,
        **cfg.train_setup.train_config,
        num_evals=int(
            cfg.train_setup.train_config.num_timesteps / cfg.train_setup.eval_every
        ),
        num_resets_per_eval=cfg.train_setup.eval_every // cfg.train_setup.get("reset_every", cfg.train_setup.eval_every),
        episode_length=episode_length,
        ckpt_mgr=ckpt_mgr,
        checkpoint_to_restore=cfg.train_setup.checkpoint_to_restore,
        config_dict=cfg_dict,
        eval_env_test_set=test_env,
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
        )
        if cfg.env_config.domain_randomization.use_domain_randomization
        else None,
        # VQ-VAE checkpoint configuration
        vqvae_checkpoint_path=cfg.vqvae_config.checkpoint_path,
        vqvae_checkpoint_step=cfg.vqvae_config.checkpoint_step,
        # Prior network configuration
        prior_layer_sizes=tuple(cfg.network_config.prior_layer_sizes),
        # Loss configuration
        loss_type=cfg.loss_config.loss_type,
        ar_weight=cfg.loss_config.ar_weight,
        phi=cfg.loss_config.phi,
        use_ar_schedule=cfg.loss_config.use_ar_schedule,
        ar_schedule_params=OmegaConf.to_container(cfg.loss_config.ar_schedule_params)
        if cfg.loss_config.use_ar_schedule
        else None,
        smooth_l1_delta=cfg.loss_config.smooth_l1_delta,
        mse_weight=cfg.loss_config.mse_weight,
        cosine_weight=cfg.loss_config.cosine_weight,
        # Freeloop evaluation
        freeloop_config=OmegaConf.to_container(cfg.freeloop_config)
        if cfg.freeloop_config.enabled
        else None,
    )

    # Set the render env start frame to always be 0
    rollout_cfg = env_cfg_ml.copy_and_resolve_references()
    rollout_cfg.start_frame_range = [0, 0]
    rollout_env = vnl_wrappers.FlattenObsWrapper(
        imitation.Imitation(config=rollout_cfg)
    )

    # Define policy params logging function
    policy_params_fn = functools.partial(
        vq_prior_rollout_logging_fn,
        rollout_env,
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
