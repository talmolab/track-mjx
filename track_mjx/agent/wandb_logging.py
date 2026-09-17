"""Weights & Biases logging utilities for PPO training.

This module provides functions for logging training metrics, rollout videos,
and latent space statistics to Weights & Biases during reinforcement learning
training runs.
"""

import logging
from collections.abc import Callable
from typing import Any

import imageio
import jax
import mujoco
import numpy as np
import wandb
from jax import numpy as jnp
from omegaconf import DictConfig, OmegaConf

# Per-term episode-reward failures already warned about for the current wandb
# run, as (term, exception type) pairs, so that a repeat of the same failure is
# quiet but a new kind of failure is not. Cleared when the run id changes, so a
# second run in the same process (a multirun job, a re-run in a notebook) warns
# again.
_warned_reward_terms: set[tuple[str, str]] = set()
_warned_run_id: str | None = None


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
    physics_steps_per_ctrl = env._config.ctrl_dt / env._config.sim_dt
    steps_per_mocap_frame = (1 / env._config.mocap_hz) / (
        env._config.sim_dt * physics_steps_per_ctrl
    )
    episode_length = int(env._clip_length() * steps_per_mocap_frame)

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

    # Cumulative episode-reward scalars, summed up to the first done step.
    # These are the return of this one rollout episode, under their own keys.
    # They are not the same quantity as Brax's eval/episode_reward, which PPO
    # logs via progress_fn as a mean over its evaluator's episodes. Trainers
    # that don't run a Brax evaluator have no other episode-reward scalar.
    _log_cumulative_reward_scalars(env, current_step, rollout)

    if cfg.logging_config.get("log_histograms", False):
        log_histogram_to_wandb("eval/histograms/latent_means", means_mean.tolist())
        log_histogram_to_wandb("eval/histograms/latent_stds", means_std.tolist())

    if render_video:
        _log_rollout_video(env, cfg, model_path, current_step, rollout)


def _log_cumulative_reward_scalars(
    env: Any,
    current_step: int,
    rollout: list[Any],
) -> None:
    """Log eval/rollout_episode_reward and per-term cumulatives as wandb scalars.

    Sums ``state.reward`` and ``state.metrics["rewards/<term>"]`` over the first
    episode of the rollout, skipping the reset state at index 0. Like Brax's
    evaluator, the step on which ``done`` first becomes true is counted and
    every later step is not; if ``done`` never fires, the whole rollout is
    summed. Logged with commit=False so they batch with the next train-metrics
    commit.

    This runs inside the training loop, so it never raises. A scalar that
    cannot be computed or logged is skipped with a warning naming it and the
    exception. Per-term warnings are emitted once per term, exception type and
    wandb run, so a term that fails the same way on every eval does not flood
    the log, while a new kind of failure is still reported.

    Args:
        env: Environment whose ``_config.reward_terms`` names the per-term
            metrics.
        current_step: Current training step (unused; kept for call-site
            symmetry with the other rollout loggers).
        rollout: Unbatched environment states, starting with the reset state.
    """
    try:
        episode = _first_episode(rollout)
    except Exception as e:
        logging.warning(
            "Failed to log episode_reward scalars: cannot read done flags: %s: %s",
            type(e).__name__,
            e,
        )
        return
    if not episode:
        return

    try:
        ep_reward = float(sum(jnp.asarray(s.reward) for s in episode))
        wandb.log({"eval/rollout_episode_reward": ep_reward}, commit=False)
    except Exception as e:
        logging.warning(
            "Failed to log episode_reward scalar: %s: %s", type(e).__name__, e
        )

    try:
        cfg_terms = getattr(getattr(env, "_config", None), "reward_terms", None)
        if isinstance(cfg_terms, (str, bytes)):
            raise TypeError(
                f"expected a collection of term names, got {type(cfg_terms).__name__}"
            )
        terms = list(cfg_terms or ())
    except Exception as e:
        if _should_warn(("<reward_terms>", type(e).__name__)):
            logging.warning(
                "Failed to log per-term episode_reward scalars: cannot read "
                "reward_terms: %s: %s (not logged again for this run)",
                type(e).__name__,
                e,
            )
        return

    if not terms and _should_warn(("<reward_terms>", "empty")):
        logging.warning(
            "Not logging per-term episode_reward scalars: the environment "
            "config lists no reward terms (not logged again for this run)"
        )
    for term in terms:
        try:
            metric_key = f"rewards/{term}"
            term_total = float(sum(jnp.asarray(s.metrics[metric_key]) for s in episode))
            wandb.log(
                {f"eval/rollout_episode_reward_{term}": term_total},
                commit=False,
            )
        except Exception as e:
            label = _term_label(term)
            if _should_warn((label, type(e).__name__)):
                logging.warning(
                    "Failed to log episode_reward_%s scalar: %s: %s "
                    "(the same failure is not logged again for this run)",
                    label,
                    type(e).__name__,
                    e,
                )


def _should_warn(key: tuple[str, str]) -> bool:
    """Whether to warn about this failure, recording it so repeats stay quiet.

    Keys are (term, exception type) pairs. The record is dropped whenever the
    wandb run changes, so each run reports its failures once.

    Args:
        key: Identifies the failure being warned about.

    Returns:
        True if the caller should emit the warning.
    """
    global _warned_run_id

    run_id = _current_run_id()
    if run_id != _warned_run_id:
        _warned_run_id = run_id
        _warned_reward_terms.clear()
    if key in _warned_reward_terms:
        return False
    _warned_reward_terms.add(key)
    return True


def _current_run_id() -> str:
    """Return a label for the active wandb run, without ever raising.

    wandb may be uninitialised, in which case ``wandb.run`` is None.
    """
    try:
        return _term_label(getattr(getattr(wandb, "run", None), "id", None))
    except Exception:
        return "<unreadable wandb run>"


def _term_label(term: Any) -> str:
    """Return a printable name for a reward term, without ever raising."""
    for to_text in (str, repr):
        try:
            return to_text(term)
        except Exception:
            pass
    return "<unprintable reward term>"


def _first_episode(rollout: list[Any]) -> list[Any]:
    """Return the post-reset states that belong to the rollout's first episode.

    The rollout environment is not auto-reset, so it keeps stepping after
    ``done``. The first episode runs from index 1 up to and including the first
    state whose ``done`` flag is set. This matches Brax's evaluator, which
    starts an ``active`` mask at 1 on reset, weights each step's reward by the
    mask before that step, and then updates it cumulatively with
    ``active *= 1 - done``: once ``done`` fires the mask stays 0, even if a
    later ``done`` flag clears. The reset state's own ``done`` flag is ignored,
    as in Brax.

    The done flags are stacked and copied to the host in a single transfer.

    Args:
        rollout: Unbatched environment states, starting with the reset state.

    Returns:
        The states whose rewards count towards the episode return.

    Raises:
        ValueError: If the done flags are not scalars, or are not finite. A
            non-finite flag would make Brax's ``active`` mask NaN from that step
            on, so there is no episode boundary to honor.
    """
    steps = rollout[1:]
    if not steps:
        return []
    done = np.asarray(jnp.stack([jnp.asarray(s.done) for s in steps]))
    if done.shape != (len(steps),):
        raise ValueError(f"expected scalar done flags, got shape {done.shape[1:]}")
    if not np.isfinite(done).all():
        raise ValueError("expected finite done flags, got a non-finite value")
    ended = np.flatnonzero(done)
    return steps[: ended[0] + 1] if ended.size else steps


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
    reward_names = [f"rewards/{k}" for k in env._config.reward_terms]
    metric_names = cfg.logging_config.get("rollout_metrics", reward_names)
    for metric in metric_names:
        if metric in rollout[0].metrics:
            values = [s.metrics[metric] for s in rollout]
            log_lineplot_to_wandb(
                name=f"eval/rollout_{metric}",
                metric_name=metric,
                data=list(enumerate(values)),
                title=f"{metric} per rollout frame",
            )
        else:
            logging.warning(
                f"Rollout metric '{metric}' not found in environment metrics."
            )

    # Log histograms of per-step reward distributions
    if cfg.logging_config.get("log_histograms", False):
        episode_rewards = [float(s.reward) for s in rollout]
        log_histogram_to_wandb("eval/histograms/episode_reward", episode_rewards)
        for metric in metric_names:
            if metric in rollout[0].metrics:
                values = [float(s.metrics[metric]) for s in rollout]
                log_histogram_to_wandb(
                    f"eval/histograms/{metric}",
                    values,
                )

    try:
        with imageio.get_writer(video_path, fps=render_fps) as writer:
            video = env.render(
                trajectory=rollout,
                camera=f"{cfg.render_config.render_camera_name}{env._suffix}",
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


def log_histogram_to_wandb(
    name: str,
    data: list[float],
) -> None:
    """Log a histogram to Weights & Biases.

    Uses ``wandb.Histogram`` for lightweight native histogram rendering
    without creating intermediate tables.

    Args:
        name: Wandb log key (e.g., "eval/episode_reward_hist").
        data: Values to include in the histogram.

    Note:
        Logged with commit=False; caller should batch with other logs.
    """
    wandb.log({name: wandb.Histogram(data)}, commit=False)


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
    summary_keys = set(wandb.run.summary.keys())
    for metric in metrics:
        if metric not in summary_keys:
            mode = "max" if "reward" in metric or "episode_length" in metric else "mean"
            wandb.run.define_metric(metric, summary=mode)
            summary_keys.add(metric)
    wandb.log(metrics)
