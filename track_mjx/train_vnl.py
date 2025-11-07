"""
Entry point for track-mjx. Load the config file, create environments, initialize network, and start training.
"""

import os
import sys

# Limit to a particular GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Either preallocate memory for JAX or disable it
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = os.environ.get(
#     "XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9"
# )
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import jax
import hydra
from omegaconf import DictConfig
import functools
import wandb
import orbax.checkpoint as ocp
from track_mjx.agent.mlp_ppo import ppo as mlp_ppo, ppo_networks as mlp_ppo_networks
import logging

from track_mjx.agent import checkpointing
from track_mjx.agent import wandb_logging
from track_mjx.analysis import render
from track_mjx import utils

from vnl_mjx.tasks.rodent import imitation
from vnl_mjx.tasks.rodent import wrappers as vnl_wrappers
from vnl_mjx.tasks.rodent import consts as rodent_consts
from vnl_mjx.tasks.rodent.reference_clips import ReferenceClips
from mujoco_playground import wrapper as playground_wrappers

@hydra.main(version_base=None, config_path="config", config_name="rodent-full-clips")
def main(cfg: DictConfig):
    """Main function using Hydra configs"""
    try:
        n_devices = jax.device_count(backend="gpu")
        logging.info(f"Using {n_devices} GPUs")
    except:
        n_devices = 1
        logging.info("Not using GPUs")

    # Determine how to load from checkpoint
    run_id, checkpoint_path, existing_run_state = checkpointing.load_from_run_state(cfg)

    # Prepare config
    (
    cfg,
    cfg_dict,
    env_cfg,
    env_cfg_ml,
    render_cfg,
    network_cfg,
    train_setup,
    train_cfg,
    logging_cfg,
    walker_cfg,
    ) = utils.prepare_config(cfg)

    # Initialize checkpoint manager
    mgr_options = ocp.CheckpointManagerOptions(
        create=True,
        step_prefix="PPONetwork",
    )
    ckpt_mgr = ocp.CheckpointManager(checkpoint_path, options=mgr_options)

    # Create the reference clips
    logging.info(f"Loading data: {cfg.data_path}")
    reference_clips = ReferenceClips(
        data_path=env_cfg.reference_data_path,
        n_frames_per_clip=env_cfg.clip_length,
        keep_clips_idx=env_cfg.keep_clips_idx,
    )
    # Create train/test split
    key_split, key = jax.random.split(jax.random.PRNGKey(train_cfg.seed))
    train_clips, test_clips = reference_clips.split(
        train_ratio=train_setup.train_subset_ratio,
        seed=key_split,
    )
    # Create environments
    env = vnl_wrappers.FlattenObsWrapper(imitation.Imitation(config=env_cfg_ml, clips=train_clips))
    test_env = vnl_wrappers.FlattenObsWrapper(imitation.Imitation(config=env_cfg_ml, clips=test_clips))

    logging.info(f"Environment config: {env_cfg}")

    # Episode length is equal to (clip length - random init range - traj length) * steps per cur frame.
    # env_args = cfg.env_config.env_args
    steps_per_frame = (1 / env_cfg.mocap_hz) / (env_cfg.ctrl_dt)
    episode_length = (
        env_cfg.clip_length
        - env_cfg.start_frame_range[-1]
        - env_cfg.reference_length
    ) * steps_per_frame
    logging.info(f"episode_length {episode_length}")

    logging.info("Using MLP Pipeline Now")
    ppo = mlp_ppo
    ppo_networks = mlp_ppo_networks
    network_factory = functools.partial(
        ppo_networks.make_intention_ppo_networks,
        intention_latent_size=network_cfg.intention_size,
        encoder_hidden_layer_sizes=tuple(network_cfg.encoder_layer_sizes),
        decoder_hidden_layer_sizes=tuple(network_cfg.decoder_layer_sizes),
        value_hidden_layer_sizes=tuple(network_cfg.critic_layer_sizes),
    )

    # Determine wandb run ID for resuming
    wandb_logging.initialize_wandb_logging(
        logging_cfg=logging_cfg,
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

    # Create the checkpoint callback with the correct wandb_run_id
    checkpoint_callback = checkpointing.create_checkpoint_callback(
        cfg=cfg,
        run_id=run_id,
        checkpoint_path=checkpoint_path,
        wandb_run_id=wandb.run.id,
    )

    train_fn = functools.partial(
        ppo.train,
        **train_cfg,
        num_evals=int(
            train_cfg.num_timesteps / train_setup.eval_every
        ),
        num_resets_per_eval=train_setup.eval_every // train_setup.reset_every,
        episode_length=episode_length,
        kl_weight=network_cfg.kl_weight,
        network_factory=network_factory,
        ckpt_mgr=ckpt_mgr,
        checkpoint_to_restore=train_setup.checkpoint_to_restore,
        config_dict=cfg_dict,
        use_kl_schedule=network_cfg.kl_schedule,
        eval_env_test_set=test_env,
        freeze_decoder=(
            False
            if "freeze_decoder" not in train_setup
            else train_setup.freeze_decoder
        ),
        checkpoint_callback=checkpoint_callback,
        wrap_for_training=functools.partial(  # Testing full reset instead of setting to initial state
            playground_wrappers.wrap_for_brax_training, full_reset=False
        ),
    )

    # Set the render env start frame to always be 0
    rollout_cfg = env_cfg_ml.copy_and_resolve_references()
    rollout_cfg.start_frame_range = [0, 0]
    rollout_env = vnl_wrappers.FlattenObsWrapper(imitation.Imitation(config=rollout_cfg))

    # define the jit reset/step functions
    jit_reset = jax.jit(rollout_env.reset)
    jit_step = jax.jit(rollout_env.step)
    renderer, mj_model, mj_data, scene_option = render.make_rollout_renderer(cfg)
    policy_params_fn = functools.partial(
        wandb_logging.rollout_logging_fn,
        rollout_env,
        jit_reset,
        jit_step,
        cfg,
        checkpoint_path,
        renderer,
        mj_model,
        mj_data,
        scene_option,
    )

    make_inference_fn, params, _ = train_fn(
        environment=env,
        progress_fn=wandb_logging.wandb_progress,
        policy_params_fn=policy_params_fn,  # fill in the rest in training
    )

    # Clean up run state after successful completion
    try:
        checkpointing.cleanup_run_state(cfg)
        logging.info("Training completed successfully, cleaned up run state")
    except Exception as e:
        logging.warning(f"Failed to cleanup run state: {e}")


if __name__ == "__main__":
    main()
