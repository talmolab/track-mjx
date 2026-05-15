"""Weights & Biases logging utilities for PPO training.

This module provides functions for logging training metrics, rollout videos,
and latent space statistics to Weights & Biases during reinforcement learning
training runs.
"""

import logging
from typing import Any, Callable

import imageio
import jax
import mujoco
import wandb
from jax import numpy as jnp
from omegaconf import DictConfig, OmegaConf

from track_mjx.agent.ff_ppo import losses


def rollout_logging_fn(
    env: Any,
    jit_reset: Callable,
    jit_step: Callable,
    cfg: DictConfig,
    model_path: str,
    current_step: int,
    jit_logging_inference_fn: Callable,
    params: Any,  # PPONetworkParams or RecurrentPPONetworkParams
    policy_params_fn_key: jax.Array,
    render_video: bool = True,
    ppo_network: Any = None,
) -> None:
    """Log evaluation rollout metrics and video to Weights & Biases.

    Runs a full episode using the current policy, logging latent space statistics
    (mean/std of VAE latents), per-step reward breakdowns, and optionally a
    rendered video of the rollout.

    Args:
        env: VNL environment instance with render() and _config attributes.
        jit_reset: JIT-compiled environment reset function.
        jit_step: JIT-compiled environment step function.
        cfg: Full configuration containing env_config and render_config.
        model_path: Directory path for saving video files.
        current_step: Current training step (used for video filename).
        jit_logging_inference_fn: JIT-compiled inference function. For feedforward:
            (params, obs, rng) -> (action, extras). For recurrent:
            (params, obs, hidden, rng) -> (action, extras, new_hidden).
            Extras must contain "latent_mean" and "latent_logvar" keys.
        params: PPO network parameters for the policy.
        policy_params_fn_key: JAX PRNG key for stochastic operations.
        render_video: If True, render and log a video of the rollout.
        ppo_network: PPO network container. Required for recurrent policies
            to initialize hidden state.

    Note:
        All metrics are logged with commit=False to batch with other logs.
        The caller should call wandb.log() with commit=True afterward.
    """
    _, reset_rng, act_rng = jax.random.split(policy_params_fn_key, 3)
    state = jit_reset(reset_rng)

    rollout = [state]
    latent_means = []
    latent_logvars = []

    # Calculate episode length from config
    physics_steps_per_ctrl = cfg.env_config.ctrl_dt / cfg.env_config.sim_dt
    steps_per_mocap_frame = (1 / cfg.env_config.mocap_hz) / (
        cfg.env_config.sim_dt * physics_steps_per_ctrl
    )

    episode_length = int(cfg.env_config.clip_length * steps_per_mocap_frame)

    # Check if using recurrent policy
    arch_name = cfg.network_config.get("arch_name", "intention")
    is_recurrent = arch_name == "recurrent_intention"

    # Initialize hidden state for recurrent policies
    if is_recurrent:
        if ppo_network is None:
            raise ValueError(
                "ppo_network must be provided for recurrent policy rollouts"
            )
        hidden = ppo_network.policy_network.init_hidden(1)

    for _ in range(episode_length):
        _, act_rng = jax.random.split(act_rng)

        if is_recurrent:
            ctrl, extras, hidden = jit_logging_inference_fn(
                params, state.obs, hidden, act_rng
            )
        else:
            ctrl, extras = jit_logging_inference_fn(params, state.obs, act_rng)

        ctrl = jnp.squeeze(ctrl, axis=0) if ctrl.shape[0] == 1 else ctrl

        latent_means.append(extras["latent_mean"])
        latent_logvars.append(extras["latent_logvar"])

        state = jit_step(state, ctrl)
        rollout.append(state)

    # Log latent space statistics per dimension
    latent_means = jnp.stack(latent_means)
    latent_logvars = jnp.stack(latent_logvars)

    means_mean = jnp.mean(latent_means, axis=0)
    means_std = jnp.std(latent_means, axis=0)
    logvars_mean = jnp.mean(latent_logvars, axis=0)
    logvars_std = jnp.std(latent_logvars, axis=0)

    for i in range(means_mean.shape[0]):
        wandb.log(
            {
                f"latents/latent_means_mean{i}": means_mean[i],
                f"latents/latent_means_std{i}": means_std[i],
                f"latents/latent_logvars_mean{i}": logvars_mean[i],
                f"latents/latent_logvars_std{i}": logvars_std[i],
            },
            commit=False,
        )

    # Cumulative episode-reward scalars. Brax-PPO already emits these via its
    # internal Evaluator + progress_fn, so for PPO this is redundant-but-harmless
    # (a separate scalar key under eval/ rather than overwriting the Brax one).
    # For DMPO it's the only source, since DMPO has no Brax Evaluator.
    # Skip rollout[0] (reset state, reward=0).
    _log_cumulative_reward_scalars(env, current_step, rollout)

    if render_video:
        _log_rollout_video(env, cfg, model_path, current_step, rollout)


def _log_cumulative_reward_scalars(
    env: Any,
    current_step: int,
    rollout: list[Any],
) -> None:
    """Log eval/rollout_episode_reward and per-term cumulatives as wandb scalars.

    Sums state.reward and state.metrics["rewards/<term>"] across the rollout
    (skipping the reset state at index 0). Logged with commit=False so they
    batch with the next train-metrics commit.
    """
    if len(rollout) <= 1:
        return
    try:
        ep_reward = float(sum(jnp.asarray(s.reward) for s in rollout[1:]))
        wandb.log(
            {"eval/rollout_episode_reward": ep_reward},
            commit=False,
        )
    except Exception as e:
        logging.warning("Failed to log episode_reward scalar: %s", e)
    cfg_terms = getattr(getattr(env, "_config", None), "reward_terms", None) or {}
    for term in cfg_terms.keys():
        metric_key = f"rewards/{term}"
        try:
            term_total = float(
                sum(jnp.asarray(s.metrics[metric_key]) for s in rollout[1:])
            )
        except (KeyError, AttributeError, TypeError):
            continue
        wandb.log(
            {f"eval/rollout_episode_reward_{term}": term_total},
            commit=False,
        )


def _log_rollout_video(
    env: Any,
    cfg: DictConfig,
    model_path: str,
    current_step: int,
    rollout: list[Any],
) -> None:
    """Render rollout video and log reward metrics to wandb.

    Args:
        env: Environment with render() method and _config attribute.
        cfg: Configuration with render_config section.
        model_path: Directory to save video file.
        current_step: Training step for filename.
        rollout: List of environment states from the episode.
    """
    render_fps = cfg.render_config.render_fps
    video_path = f"{model_path}/{current_step}.mp4"

    # Log per-step reward breakdowns
    metric_names = [f"rewards/{k}" for k in env._config.reward_terms.keys()]
    for metric in metric_names:
        log_lineplot_to_wandb(
            name=f"eval/rollout_{metric}",
            metric_name=metric,
            data=list(enumerate([s.metrics[metric] for s in rollout])),
            title=f"{metric} per rollout frame",
        )

    # Resolution is overridable in render_config; default to 720p so videos
    # are readable for small walkers (stick bug is ~5 cm, was illegible at 480p).
    height = int(cfg.render_config.get("height", 720))
    width = int(cfg.render_config.get("width", 1280))
    try:
        with imageio.get_writer(video_path, fps=render_fps, quality=8) as writer:
            video = env.render(
                rollout,
                camera=f"{cfg.render_config.render_camera_name}{env._suffix}",
                height=height,
                width=width,
            )
            for frame in video:
                writer.append_data(frame)

        wandb.log(
            {"videos/rollout": wandb.Video(video_path, format="mp4")},
            commit=False,
        )
    except mujoco.FatalError as e:
        logging.warning(f"Rendering video failed with MuJoCo error: {e}")


def log_lineplot_to_wandb(
    name: str,
    metric_name: str,
    data: list[tuple[int, float]] | tuple[list[int], list[float]],
    title: str,
) -> None:
    """Log a line plot to Weights & Biases.

    Creates a wandb Table and plots it as a line chart. Useful for visualizing
    metrics over time (e.g., reward per timestep during a rollout).

    Args:
        name: Wandb log key (e.g., "eval/reward_over_rollout").
        metric_name: Column name for the y-axis values.
        data: Either a list of (x, y) tuples, or a tuple of (x_list, y_list).
        title: Title displayed on the wandb plot.

    Note:
        Logged with commit=False; caller should batch with other logs.
    """
    if isinstance(data[0], tuple):
        frames, values = zip(*data)
    else:
        frames, values = data

    table = wandb.Table(
        data=[[x, y] for x, y in zip(frames, values)],
        columns=["frame", metric_name],
    )

    wandb.log(
        {name: wandb.plot.line(table, "frame", metric_name, title=title)},
        commit=False,
    )


def initialize_wandb_logging(
    logging_cfg: DictConfig,
    cfg: DictConfig,
    run_id: str,
    existing_run_state: DictConfig | None,
) -> str:
    """Initialize a Weights & Biases run, with optional resume support.

    Creates a new wandb run or resumes an existing one if restoring from a
    checkpoint. The full config is logged to wandb for experiment tracking.

    Args:
        logging_cfg: Logging config with project_name, exp_name, group_name.
        cfg: Full experiment configuration to log to wandb.
        run_id: Unique identifier for this run (used in wandb run ID).
        existing_run_state: If resuming, dict with "wandb_run_id" key.
            If None, starts a new run.

    Returns:
        The wandb run ID (for saving in checkpoints to enable future resume).
    """
    wandb_run_id = f"{logging_cfg.exp_name}_{run_id}"

    if existing_run_state:
        wandb_run_id = existing_run_state["wandb_run_id"]
        wandb_resume = "must"
        logging.info(f"Resuming wandb run: {wandb_run_id}")
    else:
        wandb_resume = "allow"
        logging.info(f"Starting new wandb run: {wandb_run_id}")

    cfg_dict = OmegaConf.to_container(cfg, resolve=True, structured_config_mode=True)

    wandb.init(
        project=logging_cfg.project_name,
        config=cfg_dict,
        notes="",
        id=wandb_run_id,
        resume=wandb_resume,
        group=logging_cfg.group_name,
    )

    return wandb_run_id


def wandb_progress(num_steps: int, metrics: dict[str, Any]) -> None:
    """Log training progress metrics to Weights & Biases.

    Adds the step count to metrics and commits the log entry.

    Args:
        num_steps: Current training step count (logged as num_steps_thousands).
        metrics: Dictionary of metric names to values to log.

    Note:
        This function commits the log (unlike other logging functions in this
        module), so it should be called last in a logging batch.
    """
    metrics["num_steps_thousands"] = num_steps
    wandb.log(metrics)
