import os
from pathlib import Path
from absl import flags
import hydra
from omegaconf import DictConfig, OmegaConf
import uuid
from typing import Callable, Any, Optional, Dict
from types import ModuleType
import logging

import functools
import jax
import wandb
import imageio
import mediapy as media
import mujoco

from track_mjx.agent.mlp_ppo import losses

from brax.io import model
from brax.envs.base import Env
import numpy as np
from jax import numpy as jp


def rollout_logging_fn(
    env,
    jit_reset,
    jit_step,
    cfg: DictConfig,
    model_path: str,
    current_step: int,  # all args above this one are passed in by functools.partial
    jit_logging_inference_fn,
    params: losses.PPONetworkParams,
    policy_params_fn_key: jax.random.PRNGKey,
    render_video: bool = True,
) -> None:
    """Logs metrics and videos for a reinforcement learning training rollout.

    Args:
        env: An instance of the base PipelineEnv envrionment. # supporting mujoco playground envs
        jit_reset: Jitted env reset function.
        jit_step: Jitted env step function.
        cfg: Configuration dictionary for the environment and agent.
        model_path: The path to save the model parameters and videos.
        current_step: The number of training steps completed.
        jit_logging_inference_fn: Jitted policy inference function.
        params: Parameters for the policy model.
        policy_params_fn_key: PRNG key.
        render_video: Whether to render the video of the rollout, defaults to True.
    """
    train_config = cfg["train_setup"]["train_config"]
    _, reset_rng, act_rng = jax.random.split(policy_params_fn_key, 3)

    state = jit_reset(reset_rng)

    rollout = [state]
    latent_means = []
    latent_logvars = []
    physics_step_per_control_step = cfg.env_config.ctrl_dt/cfg.env_config.sim_dt
    steps_per_frame = (1 / cfg.env_config.mocap_hz) / (
        cfg.env_config.sim_dt * physics_step_per_control_step
    )
    episode_length = int(cfg.env_config.clip_length * steps_per_frame)
    for i in range(episode_length):
        _, act_rng = jax.random.split(act_rng)
        obs = state.obs
        (
            ctrl,
            extras,
        ) = jit_logging_inference_fn(params, obs, act_rng)
        ctrl = jp.squeeze(ctrl, axis=0) if ctrl.shape[0] == 1 else ctrl
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
    if render_video:
        render_fps = cfg.render_config.render_fps
        video_path = f"{model_path}/{current_step}.mp4"
        # Get list of reward and termination terms from env config to log
        metric_names = env._config.reward_terms.keys()
        metric_names = ["rewards/" + metric for metric in metric_names]
        # TODO: the imitation task in vnl doesnt include termination info in metrics
        # so we're missing those here
        for rollout_metric in metric_names:
            log_lineplot_to_wandb(
                f"eval/rollout_{rollout_metric}",
                rollout_metric,
                list(enumerate([state.metrics[rollout_metric] for state in rollout])),
                title=f"{rollout_metric} for each rollout frame",
            )

        # Render the video
        try:
            with imageio.get_writer(video_path, fps=render_fps) as writer:
                video = env.render(
                    rollout,
                    camera=f"{cfg.render_config.render_camera_name}-ghost",
                    height=480,
                    width=640,
                )
                for frame in video:
                    writer.append_data(frame)

            wandb.log(
                {"videos/rollout": wandb.Video(video_path, format="mp4")},
                commit=False,
            )
        except mujoco.FatalError as e:
            logging.warning(f"Rendering video failed with mujoco error: {e}")


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

def initialize_wandb_logging(
    logging_cfg: DictConfig,
    cfg: DictConfig,
    run_id: str,
    existing_run_state: Optional[DictConfig],
) -> str:
    """
    Initialize wandb logging, handling resuming if necessary.
    
    Args:
        logging_cfg (DictConfig): Logging configuration.
        cfg (DictConfig): Full configuration.
        run_id (str): Unique identifier for the run.
        existing_run_state (Optional[DictConfig]): Existing run state if resuming.
    Returns:
        str: The wandb run ID.
    """
    # Determine wandb run ID for resuming
    wandb_run_id = f"{logging_cfg.exp_name}_{run_id}"
    if existing_run_state:
        wandb_run_id = existing_run_state["wandb_run_id"]
        wandb_resume = "must"  # Must resume the exact run
        logging.info(f"Resuming wandb run: {wandb_run_id}")
    else:
        wandb_resume = "allow"  # Allow resuming if run exists
        logging.info(f"Starting new wandb run: {wandb_run_id}")
    
    cfg_for_wandb = OmegaConf.to_container(
        cfg, resolve=True, structured_config_mode=True
    )
    
    wandb.init(
        project=logging_cfg.project_name,
        config=cfg_for_wandb,
        notes=f"",
        id=wandb_run_id,
        resume=wandb_resume,
        group=logging_cfg.group_name,
    )
    
    return wandb_run_id

def wandb_progress(
        num_steps: int,
        metrics: dict,
) -> None:
    """
    Log training progress to wandb.

    Args:
        num_steps (int): Number of training steps completed.
        metrics (dict): Dictionary of metrics to log.
    """
    metrics["num_steps_thousands"] = num_steps
    wandb.log(metrics)