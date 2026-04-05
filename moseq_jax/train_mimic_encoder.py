"""Train a VAE intention network encoder for use as distillation target.

This script trains the standard IntentionNetwork (encoder-decoder VAE) from
the main branch PPO pipeline.  The resulting checkpoint contains encoder
params with the same structure as ``MoSeqRecurrentDecoderNetwork``'s encoder
(both use ``IntentionEncoder``), enabling direct subtree loading for
distillation training.

Usage:
    cd moseq_jax
    python train_mimic_encoder.py

    # Override config values:
    python train_mimic_encoder.py network_config.intention_size=16
"""

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import functools
import json
import logging
from pathlib import Path

MOSEQ_DIR = Path(__file__).parent
REPO_ROOT = MOSEQ_DIR.parent

import hydra
import jax
import numpy as np
import orbax.checkpoint as ocp
import wandb
from mujoco_playground import wrapper as playground_wrappers
from omegaconf import DictConfig
from vnl_playground.tasks.rodent.imitation import ReferenceClips

from track_mjx.agent.flat_imitation import FlatImitation
from track_mjx.config import utils
from track_mjx.agent import checkpointing, wandb_logging
from track_mjx.agent.ff_ppo import ppo as ff_ppo, ppo_networks as ff_networks
from track_mjx.agent.domain_randomization import domain_randomization_maker


def _setup_environment() -> None:
    """Configure environment variables for JAX."""
    xla_flags = os.environ.get("XLA_FLAGS", "")
    xla_flags += " --xla_gpu_triton_gemm_any=True"
    os.environ["XLA_FLAGS"] = xla_flags
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


@hydra.main(version_base=None, config_path="configs", config_name="mimic_encoder")
def main(cfg: DictConfig) -> None:
    """Train a VAE intention network for encoder distillation."""
    _setup_environment()

    try:
        n_devices = jax.device_count(backend="gpu")
        logging.info(f"Using {n_devices} GPUs")
    except RuntimeError:
        n_devices = 1
        logging.info("Not using GPUs")

    # Prepare config
    cfg, cfg_dict, env_cfg_ml = utils.prepare_config(cfg)

    # Checkpoint management
    run_id, checkpoint_path, existing_run_state = checkpointing.load_from_run_state(cfg)

    mgr_options = ocp.CheckpointManagerOptions(
        create=True,
        step_prefix="MimicEncoder",
    )
    ckpt_mgr = ocp.CheckpointManager(checkpoint_path, options=mgr_options)

    # Load reference clips
    logging.info(f"Loading data: {cfg.env_config.reference_data_path}")
    reference_clips = ReferenceClips(
        data_path=cfg.env_config.reference_data_path,
        n_frames_per_clip=cfg.env_config.clip_length,
        keep_clips_idx=cfg.env_config.keep_clips_idx,
    )

    # Balanced splits if available
    balanced_split_path = cfg.env_config.get("balanced_split_path", None)
    if balanced_split_path and Path(balanced_split_path).exists():
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
            f"{len(test_indices)} test"
        )
    else:
        key_split, _ = jax.random.split(
            jax.random.PRNGKey(cfg.train_setup.train_config.seed)
        )
        train_clips, test_clips = reference_clips.split(
            train_ratio=cfg.train_setup.train_subset_ratio,
            seed=key_split,
        )

    # Create environments (FlatImitation — flat dict obs, no KPMS codes)
    env = FlatImitation(config=env_cfg_ml, clips=train_clips)
    test_env = FlatImitation(config=env_cfg_ml, clips=test_clips)

    logging.info(f"Environment config: {cfg.env_config}")

    # Episode length
    steps_per_frame = (1 / cfg.env_config.mocap_hz) / cfg.env_config.ctrl_dt
    episode_length = (
        cfg.env_config.clip_length
        - cfg.env_config.start_frame_range[-1]
        - cfg.env_config.reference_length
    ) * steps_per_frame
    logging.info(f"episode_length {episode_length}")

    # Network factory — standard IntentionNetwork (feedforward encoder-decoder)
    network_factory = functools.partial(
        ff_networks.make_intention_ppo_networks,
        intention_latent_size=cfg.network_config.intention_size,
        encoder_hidden_layer_sizes=tuple(cfg.network_config.encoder_layer_sizes),
        decoder_hidden_layer_sizes=tuple(cfg.network_config.decoder_layer_sizes),
        value_hidden_layer_sizes=tuple(cfg.network_config.critic_layer_sizes),
    )

    # WandB
    wandb_logging.initialize_wandb_logging(
        logging_cfg=cfg.logging_config,
        cfg=cfg,
        run_id=run_id,
        existing_run_state=existing_run_state,
    )

    if existing_run_state is None:
        checkpointing.save_run_state(
            cfg=cfg,
            run_id=run_id,
            checkpoint_path=checkpoint_path,
            wandb_run_id=wandb.run.id,
        )

    checkpoint_callback = checkpointing.create_checkpoint_callback(
        cfg=cfg,
        run_id=run_id,
        checkpoint_path=checkpoint_path,
        wandb_run_id=wandb.run.id,
    )

    # Training
    train_fn = functools.partial(
        ff_ppo.train,
        **cfg.train_setup.train_config,
        num_evals=int(
            cfg.train_setup.train_config.num_timesteps / cfg.train_setup.eval_every
        ),
        num_resets_per_eval=cfg.train_setup.eval_every // cfg.train_setup.reset_every,
        episode_length=episode_length,
        latent_kl_weight=cfg.network_config.latent_kl_weight,
        latent_ar1_weight=cfg.network_config.get("latent_ar1_weight", 0.0),
        network_factory=network_factory,
        ckpt_mgr=ckpt_mgr,
        checkpoint_to_restore=cfg.train_setup.checkpoint_to_restore,
        config_dict=cfg_dict,
        use_kl_schedule=cfg.network_config.get("kl_schedule", True),
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
    )

    # Rollout env for logging
    rollout_cfg = env_cfg_ml.copy_and_resolve_references()
    rollout_cfg.start_frame_range = [0, 0]
    rollout_env = FlatImitation(config=rollout_cfg)

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

    make_inference_fn, params, _ = train_fn(
        environment=env,
        progress_fn=wandb_logging.wandb_progress,
        policy_params_fn=policy_params_fn,
    )

    try:
        checkpointing.cleanup_run_state(cfg)
        logging.info("Mimic encoder training completed successfully")
    except Exception as e:
        logging.warning(f"Failed to cleanup run state: {e}")

    wandb.finish()


if __name__ == "__main__":
    main()
