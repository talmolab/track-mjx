"""Wandb logging utilities for maintain_velocity task transfer training.

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
    render_height: int = 480,
    render_width: int = 640,
    render_every: int = 1,
    eval_count: int = 0,
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
        render_height: Height in pixels for the output video.
        render_width: Width in pixels for the output video.
        render_every: Render video every N evaluation steps.
        eval_count: Current evaluation count (0-indexed).

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

    _log_reward_metrics(rollout, current_step)

    # Render video on first eval and every render_every evals thereafter
    should_render = eval_count == 0 or eval_count % render_every == 0
    if should_render:
        _log_rollout_video(
            env,
            model_path,
            current_step,
            rollout,
            render_camera,
            render_fps,
            render_height,
            render_width,
        )


def _log_reward_metrics(rollout: list[Any], current_step: int) -> None:
    """Log per-step reward metrics as line plots with step-based versioning.

    Creates a combined table with all reward metrics that can be viewed
    with a slider in WandB to compare different evaluation steps.

    Args:
        rollout: List of environment states from the episode.
        current_step: Current training step for versioning.
    """
    if not rollout or not hasattr(rollout[0], "metrics"):
        return

    first_metrics = rollout[0].metrics
    reward_keys = sorted([k for k in first_metrics.keys() if k.startswith("rewards/")])

    if not reward_keys:
        return

    columns = ["frame"] + reward_keys
    data = []
    for i, state in enumerate(rollout):
        row = [i] + [float(state.metrics.get(k, 0.0)) for k in reward_keys]
        data.append(row)

    try:
        table = wandb.Table(data=data, columns=columns)
        wandb.log(
            {"eval/reward_curves": table},
            commit=False,
        )
        for metric_name in reward_keys:
            wandb.log(
                {
                    f"eval/{metric_name}": wandb.plot.line(
                        table,
                        "frame",
                        metric_name,
                        title=f"{metric_name} (step {current_step})",
                    )
                },
                commit=False,
            )
    except Exception as e:
        logging.warning(f"Failed to log reward metrics: {e}")


def _log_rollout_video(
    env: Any,
    model_path: str,
    current_step: int,
    rollout: list[Any],
    render_camera: str,
    render_fps: int,
    render_height: int = 480,
    render_width: int = 640,
) -> None:
    """Render rollout video and log to wandb.

    Args:
        env: Environment with render() method.
        model_path: Directory to save video file.
        current_step: Training step for filename.
        rollout: List of environment states from the episode.
        render_camera: Camera name for rendering.
        render_fps: Frames per second for output video.
        render_height: Height in pixels for output video.
        render_width: Width in pixels for output video.
    """
    video_path = f"{model_path}/{current_step}.mp4"

    try:
        with imageio.get_writer(video_path, fps=render_fps) as writer:
            frames = env.render(
                rollout,
                camera=render_camera,
                height=render_height,
                width=render_width,
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
