import os
from pathlib import Path
from absl import flags
import hydra
from omegaconf import DictConfig, OmegaConf
import uuid
from typing import Callable, Any
from types import ModuleType

import functools
import jax
import wandb
import imageio
import mujoco
from brax import envs
from dm_control import mjcf as mjcf_dm
from dm_control.locomotion.walkers import rescale

from track_mjx.agent import ppo
from track_mjx.agent import ppo_networks
from track_mjx.agent import losses
from brax.io import model
import numpy as np
from jax import numpy as jp

from track_mjx.environment import wrappers
from brax.envs.base import Env

# TODO: Use MjSpec to generate the mjcf with ghost

def rollout_logging_fn(
    env,
    jit_reset,
    jit_step,
    cfg: DictConfig,
    model_path: str,
    renderer,
    mj_model,
    mj_data,
    scene_option,
    current_step: int,  # all args above this one are passed in by functools.partial
    jit_logging_inference_fn,
    params: losses.PPONetworkParams,
    policy_params_fn_key: jax.random.PRNGKey,
) -> None:
    """Logs metrics and videos for a reinforcement learning training rollout.

    Args:
        env: An instance of the base PipelineEnv envrionment.
        jit_reset: Jitted env reset function.
        jit_step: Jitted env step function.
        cfg: Configuration dictionary for the environment and agent.
        model_path: The path to save the model parameters and videos.
        renderer: A mujoco.Renderer object.
        mj_model: A mujoco.Model object for rendering.
        mj_data: A mujoco.Data object for rendering.
        scene_option: A mujoco.MjvOption object for rendering.
        current_step: The number of training steps completed.
        jit_logging_inference_fn: Jitted policy inference function.
        params: Parameters for the policy model.
        policy_params_fn_key: PRNG key.
    """
    train_config =cfg['train_setup']['train_config']
    _, reset_rng, act_rng = jax.random.split(policy_params_fn_key, 3)

    state = jit_reset(reset_rng)
    hidden_state = state.info["hidden_state"]

    rollout = [state]
    latent_means = []
    latent_logvars = []
    for i in range(int(cfg["reference_config"].clip_length * env._steps_for_cur_frame)):
        _, act_rng = jax.random.split(act_rng)
        obs = state.obs
        if train_config['use_lstm']:
            ctrl, extras, hidden_state = jit_logging_inference_fn(params, obs, act_rng, hidden_state)
        else:
            ctrl, extras,  = jit_logging_inference_fn(params, obs, act_rng, None)
        ctrl = jp.squeeze(ctrl, axis=0) if ctrl.shape[0] == 1 else ctrl
        ctrl, extras = jit_logging_inference_fn(params, obs, act_rng)
        latent_means.append(extras["latent_mean"])
        latent_logvars.append(extras["latent_logvar"])
        state = jit_step(state, ctrl)
        rollout.append(state)

    # plot the statistics of each latent dim (representing means and logvars sampled)
    latent_logvars = jp.stack(latent_logvars)
    latent_means = jp.stack(latent_means)
    latent_means_means = jp.mean(latent_means, axis=0)
    latent_logvars_means = jp.mean(latent_logvars, axis=0)
    latent_means_stds = jp.std(latent_means, axis=0)
    latent_logvars_stds = jp.std(latent_logvars, axis=0)
    for i in range(latent_means_means.shape[0]):
        wandb.log(
            {
                f"latents/latent_means_mean{i}": latent_means_means[i],
                f"latents/latent_means_std{i}": latent_means_stds[i],
                f"latents/latent_logvars_mean{i}": latent_logvars_means[i],
                f"latents/latent_logvars_std{i}": latent_logvars_stds[i],
            },
            commit=False,
        )

    for rollout_metric in cfg.logging_config.rollout_metrics:
        log_lineplot_to_wandb(
            f"eval/rollout_{rollout_metric}",
            rollout_metric,
            list(enumerate([state.metrics[rollout_metric] for state in rollout])),
            title=f"{rollout_metric} for each rollout frame",
        )

    # Render the walker with the reference expert demonstration trajectory
    qposes_rollout = np.array([state.pipeline_state.qpos for state in rollout])
    ref_traj = env._get_reference_clip(rollout[0].info)
    qposes_ref = np.repeat(
        np.hstack([ref_traj.position, ref_traj.quaternion, ref_traj.joints]),
        env._steps_for_cur_frame,
        axis=0,
    )

    # render while stepping using mujoco
    video_path = f"{model_path}/{current_step}.mp4"

    with imageio.get_writer(video_path, fps=int((1.0 / env.dt))) as video:
        for qpos1, qpos2 in zip(qposes_rollout, qposes_ref):
            mj_data.qpos = np.append(qpos1, qpos2)
            mujoco.mj_forward(mj_model, mj_data)
            renderer.update_scene(
                mj_data,
                camera=cfg["env_config"].render_camera_name,
                scene_option=scene_option,
            )
            pixels = renderer.render()
            video.append_data(pixels)

    wandb.log({"eval/rollout": wandb.Video(video_path, format="mp4")})


def log_lineplot_to_wandb(name: str, metric_name: str, data: jp.ndarray, title: str):
    """Logs a table of values and its line plot to wandb.

    Args:
        name: The name of the lineplot in wandb (i.e. eval/reward_over_rollout).
        metric_name: The key under which to log the metric.
        data: List of (x, y) tuples or two lists (frames, rewards).
        title: Title for the wandb plot.
    """
    if isinstance(data[0], tuple):
        # If data is a list of (x, y) tuples, separate it into frames and values
        frames, values = zip(*data)
    else:
        # If data is two lists, use them directly
        frames, values = data

    table = wandb.Table(
        data=[[x, y] for x, y in zip(frames, values)],
        columns=["frame", metric_name],
    )

    wandb.log(
        {
            name: wandb.plot.line(
                table,
                "frame",
                metric_name,
                title=title,
            )
        },
        commit=False,
    )
