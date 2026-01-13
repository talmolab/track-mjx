"""Wandb logging utilities for bowl escape task transfer training.

This module provides functions for logging training metrics, rollout videos,
and reward breakdowns to Weights & Biases.
"""

import logging
from typing import Any, Callable

import imageio
import jax
import mujoco
import wandb
from jax import numpy as jnp


def rollout_logging_fn(
    env: Any,
    jit_reset: Callable,
    jit_step: Callable,
    jit_inference_fn: Callable,
    params: Any,
    current_step: int,
    model_path: str,
    episode_length: int,
    render_camera: str = "close_profile-rodent",
    render_fps: int = 50,
) -> None:
    """Log evaluation rollout video and metrics to wandb.

    Runs a full episode using the current policy, logging per-step reward
    breakdowns and a rendered video of the rollout.

    Args:
        env: Environment instance with render() method.
        jit_reset: JIT-compiled environment reset function.
        jit_step: JIT-compiled environment step function.
        jit_inference_fn: JIT-compiled inference function with signature
            (params, obs, rng) -> (action, extras).
        params: Policy parameters for inference.
        current_step: Current training step (used for video filename).
        model_path: Directory path for saving video files.
        episode_length: Number of steps to run in the rollout.
        render_camera: Camera name for rendering.
        render_fps: Frames per second for the output video.

    Note:
        All metrics are logged with commit=False to batch with other logs.
    """
    rng = jax.random.PRNGKey(current_step)
    reset_rng, act_rng = jax.random.split(rng)

    state = jit_reset(reset_rng)
    rollout = [state]

    for _ in range(episode_length):
        _, act_rng = jax.random.split(act_rng)
        action, _ = jit_inference_fn(params, state.obs, act_rng)
        state = jit_step(state, action)
        rollout.append(state)

    # Log reward breakdown per step
    _log_reward_metrics(rollout)

    # Render and log video
    _log_rollout_video(env, model_path, current_step, rollout, render_camera, render_fps)


def _log_reward_metrics(rollout: list[Any]) -> None:
    """Log per-step reward metrics as line plots.

    Args:
        rollout: List of environment states from the episode.
    """
    # Get available reward metrics from first state
    if not rollout or not hasattr(rollout[0], "metrics"):
        return

    first_metrics = rollout[0].metrics
    reward_keys = [k for k in first_metrics.keys() if k.startswith("rewards/")]

    for metric_name in reward_keys:
        try:
            data = [(i, float(s.metrics.get(metric_name, 0.0))) for i, s in enumerate(rollout)]
            table = wandb.Table(data=data, columns=["frame", metric_name])
            wandb.log(
                {f"eval/{metric_name}": wandb.plot.line(table, "frame", metric_name)},
                commit=False,
            )
        except Exception as e:
            logging.warning(f"Failed to log reward metric {metric_name}: {e}")


def _log_rollout_video(
    env: Any,
    model_path: str,
    current_step: int,
    rollout: list[Any],
    render_camera: str,
    render_fps: int,
) -> None:
    """Render rollout video and log to wandb.

    Args:
        env: Environment with render() method.
        model_path: Directory to save video file.
        current_step: Training step for filename.
        rollout: List of environment states from the episode.
        render_camera: Camera name for rendering.
        render_fps: Frames per second for output video.
    """
    video_path = f"{model_path}/{current_step}.mp4"

    try:
        # Use render_camera directly - it should already include any suffix
        # (e.g., "close_profile-rodent" from config)
        with imageio.get_writer(video_path, fps=render_fps) as writer:
            frames = env.render(
                rollout,
                camera=render_camera,
                height=480,
                width=640,
            )
            for frame in frames:
                writer.append_data(frame)

        wandb.log(
            {"videos/rollout": wandb.Video(video_path, format="mp4")},
            commit=False,
        )
    except mujoco.FatalError as e:
        logging.warning(f"Rendering video failed with MuJoCo error: {e}")
    except Exception as e:
        logging.warning(f"Failed to log rollout video: {e}")


def wandb_progress(num_steps: int, metrics: dict[str, Any]) -> None:
    """Log training progress metrics to Weights & Biases.

    Args:
        num_steps: Current training step count.
        metrics: Dictionary of metric names to values to log.
    """
    metrics["num_steps"] = num_steps
    wandb.log(metrics)
