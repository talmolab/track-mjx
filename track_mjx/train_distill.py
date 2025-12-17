"""
Entry point for distillation training. 
Load the config file, create environments, load teacher model, and start distillation training.
"""

import os
import sys

# XLA and environment configuration
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import jax
import hydra
from omegaconf import DictConfig, OmegaConf
import functools
import wandb
import orbax.checkpoint as ocp
import logging

from track_mjx.agent.mlp_distill import distill, distill_networks
from track_mjx.agent import checkpointing
from track_mjx.agent import wandb_logging
from track_mjx import utils
from track_mjx.agent.domain_randomization import domain_randomization_maker

from vnl_playground.tasks.rodent import imitation
from vnl_playground.tasks.rodent import wrappers as vnl_wrappers
from vnl_playground.tasks.rodent.reference_clips import ReferenceClips
from mujoco_playground import wrapper as playground_wrappers

# Import mujoco and imageio for rendering
import mujoco
import imageio


def distill_rollout_logging_fn(
    env,
    jit_reset,
    jit_step,
    cfg: DictConfig,
    model_path: str,
    teacher_policy_fn,
    teacher_params,
    current_step: int,
    params,
    policy_params_fn_key,
    render_video: bool = True,
) -> None:
    """Rollout logging for distillation training - renders both student and teacher."""
    from jax import numpy as jp

    physics_step_per_control_step = cfg.env_config.ctrl_dt / cfg.env_config.sim_dt
    steps_per_frame = (1 / cfg.env_config.mocap_hz) / (
        cfg.env_config.sim_dt * physics_step_per_control_step
    )
    episode_length = int(cfg.env_config.clip_length * steps_per_frame)

    # Create student inference function
    normalize = lambda x, y: x
    if cfg.train_setup.train_config.normalize_observations:
        from brax.training.acme import running_statistics
        normalize = running_statistics.normalize

    student_networks = distill_networks.make_student_networks(
        observation_size=env.observation_size,
        reference_obs_size=int(env.non_proprioceptive_obs_size),
        action_size=env.action_size,
        preprocess_observations_fn=normalize,
        intention_latent_size=cfg.network_config.intention_size,
        encoder_hidden_layer_sizes=tuple(cfg.network_config.encoder_layer_sizes),
        decoder_hidden_layer_sizes=tuple(cfg.network_config.decoder_layer_sizes),
        prior_hidden_layer_sizes=tuple(cfg.network_config.get("prior_layer_sizes", cfg.network_config.encoder_layer_sizes)),
        encoder_expansion_factor=cfg.network_config.get("encoder_expansion_factor", 1),
    )

    make_student_policy = distill_networks.make_student_inference_fn(student_networks)
    student_policy = make_student_policy(params, deterministic=True)
    jit_student_policy = jax.jit(student_policy)

    # Teacher policy is already jitted and passed in
    jit_teacher_policy = teacher_policy_fn

    # Split keys for student and teacher rollouts
    _, reset_rng_student, reset_rng_teacher, act_rng_student, act_rng_teacher = jax.random.split(policy_params_fn_key, 5)

    # Run student rollout
    state = jit_reset(reset_rng_student)
    student_rollout = [state]
    encoder_latent_means = []
    encoder_latent_logvars = []
    prior_latent_means = []
    prior_latent_logvars = []
    for i in range(episode_length):
        _, act_rng_student = jax.random.split(act_rng_student)
        obs = state.obs
        ctrl, extras = jit_student_policy(obs, act_rng_student)
        ctrl = jp.squeeze(ctrl, axis=0) if ctrl.shape[0] == 1 else ctrl
        # Collect latent statistics
        encoder_latent_means.append(extras["latent_mean"])
        encoder_latent_logvars.append(extras["latent_logvar"])
        prior_latent_means.append(extras["prior_mean"])
        prior_latent_logvars.append(extras["prior_logvar"])
        state = jit_step(state, ctrl)
        student_rollout.append(state)

    # Log per-dimension latent statistics (encoder)
    encoder_latent_means = jp.stack(encoder_latent_means)
    encoder_latent_logvars = jp.stack(encoder_latent_logvars)
    encoder_means_mean = jp.mean(encoder_latent_means, axis=0)
    encoder_means_std = jp.std(encoder_latent_means, axis=0)
    encoder_logvars_mean = jp.mean(encoder_latent_logvars, axis=0)
    encoder_logvars_std = jp.std(encoder_latent_logvars, axis=0)

    # Log per-dimension latent statistics (prior)
    prior_latent_means = jp.stack(prior_latent_means)
    prior_latent_logvars = jp.stack(prior_latent_logvars)
    prior_means_mean = jp.mean(prior_latent_means, axis=0)
    prior_means_std = jp.std(prior_latent_means, axis=0)
    prior_logvars_mean = jp.mean(prior_latent_logvars, axis=0)
    prior_logvars_std = jp.std(prior_latent_logvars, axis=0)

    for i in range(encoder_means_mean.shape[0]):
        wandb.log(
            {
                # Encoder latent statistics
                f"latents/encoder_means_mean{i}": encoder_means_mean[i],
                f"latents/encoder_means_std{i}": encoder_means_std[i],
                f"latents/encoder_logvars_mean{i}": encoder_logvars_mean[i],
                f"latents/encoder_logvars_std{i}": encoder_logvars_std[i],
                # Prior latent statistics
                f"latents/prior_means_mean{i}": prior_means_mean[i],
                f"latents/prior_means_std{i}": prior_means_std[i],
                f"latents/prior_logvars_mean{i}": prior_logvars_mean[i],
                f"latents/prior_logvars_std{i}": prior_logvars_std[i],
            },
            commit=False,
        )

    # Run teacher rollout (use same initial state for fair comparison)
    state = jit_reset(reset_rng_student)  # Same reset key as student
    teacher_rollout = [state]
    for i in range(episode_length):
        _, act_rng_teacher = jax.random.split(act_rng_teacher)
        obs = state.obs
        ctrl, extras = jit_teacher_policy(teacher_params, obs, act_rng_teacher)
        ctrl = jp.squeeze(ctrl, axis=0) if ctrl.shape[0] == 1 else ctrl
        state = jit_step(state, ctrl)
        teacher_rollout.append(state)

    if render_video:
        render_fps = cfg.render_config.render_fps
        camera_name = f"{cfg.render_config.render_camera_name}-ghost"

        # Render student video
        student_video_path = f"{model_path}/{current_step}_student.mp4"
        try:
            with imageio.get_writer(student_video_path, fps=render_fps) as writer:
                video = env.render(
                    student_rollout,
                    camera=camera_name,
                    height=480,
                    width=640,
                )
                for frame in video:
                    writer.append_data(frame)

            wandb.log(
                {"videos/student_rollout": wandb.Video(student_video_path, format="mp4")},
                commit=False,
            )
        except mujoco.FatalError as e:
            logging.warning(f"Student video rendering failed due to MuJoCo error: {e}. Skipping video for this iteration.")

        # Render teacher video
        teacher_video_path = f"{model_path}/{current_step}_teacher.mp4"
        try:
            with imageio.get_writer(teacher_video_path, fps=render_fps) as writer:
                video = env.render(
                    teacher_rollout,
                    camera=camera_name,
                    height=480,
                    width=640,
                )
                for frame in video:
                    writer.append_data(frame)

            wandb.log(
                {"videos/teacher_rollout": wandb.Video(teacher_video_path, format="mp4")},
                commit=False,
            )
        except mujoco.FatalError as e:
            logging.warning(f"Teacher video rendering failed due to MuJoCo error: {e}. Skipping video for this iteration.")


@hydra.main(version_base=None, config_path="config", config_name="rodent-distill")
def main(cfg: DictConfig):
    """Main function for distillation training using Hydra configs"""
    try:
        n_devices = jax.device_count(backend="gpu")
        logging.info(f"Using {n_devices} GPUs")
    except:
        n_devices = 1
        logging.info("Not using GPUs")

    # Validate teacher config
    if cfg.teacher_config.checkpoint_path is None:
        raise ValueError("teacher_config.checkpoint_path must be specified for distillation training")

    # Determine how to load from checkpoint
    run_id, checkpoint_path, existing_run_state = checkpointing.load_from_run_state(cfg)

    # Prepare config
    (cfg, cfg_dict, env_cfg_ml) = utils.prepare_config(cfg)

    # Initialize checkpoint manager
    mgr_options = ocp.CheckpointManagerOptions(
        create=True,
        step_prefix="Network",
    )
    ckpt_mgr = ocp.CheckpointManager(checkpoint_path, options=mgr_options)

    # Create the reference clips
    logging.info(f"Loading data: {cfg.data_path}")
    reference_clips = ReferenceClips(
        data_path=cfg.env_config.reference_data_path,
        n_frames_per_clip=cfg.env_config.clip_length,
        keep_clips_idx=cfg.env_config.keep_clips_idx,
    )
    
    # Create train/test split
    key_split, key = jax.random.split(jax.random.PRNGKey(cfg.train_setup.train_config.seed))
    train_clips, test_clips = reference_clips.split(
        train_ratio=cfg.train_setup.train_subset_ratio,
        seed=key_split,
    )
    
    # Create environments
    env = vnl_wrappers.FlattenObsWrapper(imitation.Imitation(config=env_cfg_ml, clips=train_clips))
    test_env = vnl_wrappers.FlattenObsWrapper(imitation.Imitation(config=env_cfg_ml, clips=test_clips))

    logging.info(f"Environment config: {cfg.env_config}")

    # Episode length calculation
    steps_per_frame = (1 / cfg.env_config.mocap_hz) / cfg.env_config.ctrl_dt
    episode_length = (
        cfg.env_config.clip_length
        - cfg.env_config.start_frame_range[-1]
        - cfg.env_config.reference_length
    ) * steps_per_frame
    logging.info(f"episode_length {episode_length}")

    logging.info("Using Distillation Pipeline")

    # Create student network factory
    network_factory = functools.partial(
        distill_networks.make_student_networks,
        intention_latent_size=cfg.network_config.intention_size,
        encoder_hidden_layer_sizes=tuple(cfg.network_config.encoder_layer_sizes),
        decoder_hidden_layer_sizes=tuple(cfg.network_config.decoder_layer_sizes),
        prior_hidden_layer_sizes=tuple(cfg.network_config.get("prior_layer_sizes", cfg.network_config.encoder_layer_sizes)),
        encoder_expansion_factor=cfg.network_config.get("encoder_expansion_factor", 1),
    )

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

    # Get distillation config
    distill_cfg = cfg.distill_config

    # Setup training function
    train_fn = functools.partial(
        distill.train,
        **cfg.train_setup.train_config,
        num_evals=int(
            cfg.train_setup.train_config.num_timesteps / cfg.train_setup.eval_every
        ),
        num_resets_per_eval=cfg.train_setup.eval_every // cfg.train_setup.reset_every,
        episode_length=episode_length,
        teacher_checkpoint_path=cfg.teacher_config.checkpoint_path,
        teacher_checkpoint_step=cfg.teacher_config.checkpoint_step,
        action_loss_weight=distill_cfg.action_loss_weight,
        autoregressive_weight=distill_cfg.autoregressive_weight,
        kl_weight=distill_cfg.kl_weight,
        use_l2_action_loss=distill_cfg.get("use_l2_action_loss", False),
        encoder_logvar_min=distill_cfg.get("encoder_logvar_min", None),
        encoder_logvar_max=distill_cfg.get("encoder_logvar_max", None),
        prior_logvar_min=distill_cfg.get("prior_logvar_min", None),
        prior_logvar_max=distill_cfg.get("prior_logvar_max", None),
        grad_clip_norm=distill_cfg.get("grad_clip_norm", 10.0),
        network_factory=network_factory,
        ckpt_mgr=ckpt_mgr,
        checkpoint_to_restore=cfg.train_setup.checkpoint_to_restore,
        config_dict=cfg_dict,
        use_schedule=distill_cfg.use_schedule,
        schedule_params=dict(distill_cfg.schedule_params) if distill_cfg.use_schedule else None,
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
        ) if cfg.env_config.use_domain_randomization else None,
        prior_rollout_config=(
            dict(cfg.prior_rollout_config)
            if hasattr(cfg, "prior_rollout_config") and cfg.prior_rollout_config is not None
            else None
        ),
    )

    # Set the render env start frame to always be 0
    rollout_cfg = env_cfg_ml.copy_and_resolve_references()
    rollout_cfg.start_frame_range = [0, 0]
    rollout_env = vnl_wrappers.FlattenObsWrapper(imitation.Imitation(config=rollout_cfg))

    # Define the jit reset/step functions for logging
    jit_reset = jax.jit(rollout_env.reset)
    jit_step = jax.jit(rollout_env.step)
    
    policy_params_fn = functools.partial(
        distill_rollout_logging_fn,
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
        logging.info("Distillation training completed successfully, cleaned up run state")
    except Exception as e:
        logging.warning(f"Failed to cleanup run state: {e}")


if __name__ == "__main__":
    main()
