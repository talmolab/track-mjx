"""
Rollout logging for distillation training.

Observations are expected as dictionaries with keys:
- "task_obs": Reference trajectory observations (flat array)
- "proprioception": Proprioceptive state observations (flat array)
"""

import logging

import imageio
import jax
import mujoco
import wandb
from omegaconf import DictConfig

from track_mjx.agent.mlp_distill import distill_networks


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

    # Get observation sizes - support both new dict format and legacy
    obs_sizes = cfg.network_config.get("obs_sizes", None)
    if obs_sizes is None:
        # Legacy format - construct from env
        obs_sizes = {
            "task_obs": int(env.non_proprioceptive_obs_size),
            "proprioception": int(env.proprioceptive_obs_size),
        }

    distill_cfg = cfg.distill_config
    student_networks = distill_networks.make_student_networks(
        obs_sizes=obs_sizes,
        action_size=env.action_size,
        intention_latent_size=cfg.network_config.intention_size,
        encoder_hidden_layer_sizes=tuple(cfg.network_config.encoder_layer_sizes),
        decoder_hidden_layer_sizes=tuple(cfg.network_config.decoder_layer_sizes),
        prior_hidden_layer_sizes=tuple(
            cfg.network_config.get(
                "prior_layer_sizes", cfg.network_config.encoder_layer_sizes
            )
        ),
        encoder_expansion_factor=cfg.network_config.get("encoder_expansion_factor", 1),
        encoder_logvar_min=distill_cfg.get("encoder_logvar_min", None),
        encoder_logvar_max=distill_cfg.get("encoder_logvar_max", None),
        prior_logvar_min=distill_cfg.get("prior_logvar_min", None),
        prior_logvar_max=distill_cfg.get("prior_logvar_max", None),
    )

    make_student_policy = distill_networks.make_student_inference_fn(student_networks)
    student_policy = make_student_policy(params, deterministic=True)
    jit_student_policy = jax.jit(student_policy)

    # Teacher policy is already jitted and passed in
    jit_teacher_policy = teacher_policy_fn

    # Split keys for student and teacher rollouts
    _, reset_rng_student, reset_rng_teacher, act_rng_student, act_rng_teacher = (
        jax.random.split(policy_params_fn_key, 5)
    )

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
                {
                    "videos/student_rollout": wandb.Video(
                        student_video_path, format="mp4"
                    )
                },
                commit=False,
            )
        except mujoco.FatalError as e:
            logging.warning(
                f"Student video rendering failed due to MuJoCo error: {e}. Skipping video for this iteration."
            )

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
                {
                    "videos/teacher_rollout": wandb.Video(
                        teacher_video_path, format="mp4"
                    )
                },
                commit=False,
            )
        except mujoco.FatalError as e:
            logging.warning(
                f"Teacher video rendering failed due to MuJoCo error: {e}. Skipping video for this iteration."
            )
