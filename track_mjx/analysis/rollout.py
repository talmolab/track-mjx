"""Rollout generation utilities for VNL imitation learning environments.

This module provides functions to create VNL environments and generate policy
rollouts with optional logging of activations, metrics, and sensor data.
Rollouts can be used for evaluation, visualization, or data collection.

Supports both feedforward and recurrent policies.
"""

from typing import Any, Callable

import jax
from mujoco_playground._src import mjx_env
from jax import numpy as jnp
from ml_collections import config_dict
from omegaconf import DictConfig, OmegaConf
from vnl_playground import registry
from vnl_playground.tasks import wrappers as vnl_wrappers


def create_environment(env_config: dict[str, Any] | DictConfig) -> mjx_env.MjxEnv:
    """Create a VNL imitation learning environment from a configuration.

    Uses the vnl-playground registry to create the environment with reference
    clips, returning nested dictionary observations:
        {'state': {'task_obs': ..., 'proprioception': ...},
         'privileged_state': {'task_obs': ..., 'proprioception': ...}}

    Args:
        env_config: Environment configuration only (not the full training
            config). Must contain env_name, reference_data_path, clip_length,
            and optionally keep_clips_idx.

    Returns:
        A Brax-compatible environment with nested dictionary observations.

    Example:
        >>> cfg, cfg_dict, env_cfg_ml = utils.prepare_config(cfg)
        >>> env = create_environment(cfg.env_config)
        >>> state = env.reset(jax.random.PRNGKey(0))
        >>> print(state.obs.keys())  # dict_keys(['state', 'privileged_state'])
    """
    if isinstance(env_config, DictConfig):
        env_cfg_dict = OmegaConf.to_container(env_config, resolve=True)
    else:
        env_cfg_dict = dict(env_config)

    env_cfg_ml = config_dict.ConfigDict(env_cfg_dict)

    env_name = env_cfg_ml.env_name
    reference_clips = registry.load_reference_clips(
        env_name,
        data_path=env_cfg_ml.reference_data_path,
        n_frames_per_clip=env_cfg_ml.clip_length,
        keep_clips_idx=env_cfg_ml.get("keep_clips_idx", None),
    )
    return vnl_wrappers.TrackMjxObsWrapper(
        registry.load(
            env_name, config=env_cfg_ml, clips=reference_clips, flatten_obs=False
        )
    )


def create_rollout_generator(
    cfg: dict[str, Any] | DictConfig,
    env: mjx_env.MjxEnv,
    inference_fn: Callable[[jnp.ndarray, jax.Array], tuple[jnp.ndarray, dict]],
    log_full_states: bool = False,
    log_activations: bool = False,
    log_metrics: bool = False,
    log_sensor_data: bool = False,
    init_hidden_fn: Callable[[int], Any] | None = None,
    cell_type: str | None = None,
) -> Callable[[int | None, int], dict[str, Any]]:
    """Create a JIT-compatible rollout generator using jax.lax.scan.

    Uses jax.lax.scan for efficient JIT compilation and batching with jax.vmap.
    Supports both feedforward and recurrent policies.

    Args:
        cfg: Full configuration dict containing env_config with timing parameters
            (mocap_hz, ctrl_dt, clip_length) and network_config with arch_name,
            obs_sizes, and architecture hyperparameters.
        env: The VNL environment to run rollouts in.
        inference_fn: Policy function. For feedforward: (obs, rng) -> (action, extras).
            For recurrent: (obs, hidden, rng) -> (action, extras, new_hidden).
            The extras dict should contain "activations" if log_activations=True.
        log_full_states: If True, include full State objects in the rollout output.
            WARNING: This uses significant memory when batched.
        log_activations: If True, collect neural network activations at each step.
        log_metrics: If True, collect all metrics from state.metrics at each step.
        log_sensor_data: If True, collect contact forces and sensor readings.
        init_hidden_fn: Hidden state initializer for recurrent policies. Has
            signature (batch_size: int) -> list[HiddenState]. If None, assumes
            feedforward policy. Use checkpointing.load_recurrent_extras to obtain.
        cell_type: RNN cell type string ('gru', 'lstm', 'simple') for recurrent
            policies. Required when init_hidden_fn is provided.

    Returns:
        A generate_rollout function with signature:
            generate_rollout(clip_idx=None, seed=42) -> dict

        The returned dict contains JAX arrays (not Python lists):
            - qposes_ref: Reference poses, shape [num_steps, qpos_dim]
            - qposes_rollout: Simulated poses, shape [num_steps, qpos_dim]
            - ctrl: Control actions, shape [num_steps-1, ctrl_dim]
            - state_rewards: Rewards, shape [num_steps]
            - rollout_states: (if log_full_states) Stacked State objects
            - activations: (if log_activations) Network activations per step
            - rollout_metrics: (if log_metrics) Dict of metric arrays
            - joint_forces: (if log_sensor_data) Contact forces per step
            - sensor_readings: (if log_sensor_data) Sensor data per step

    Example:
        >>> generate_rollout = create_rollout_generator(cfg, env, policy_fn)
        >>> # For single rollout:
        >>> rollout_data = generate_rollout(clip_idx=5, seed=123)
        >>> # For batched rollouts:
        >>> jit_vmap_rollout = jax.jit(jax.vmap(generate_rollout))
        >>> batch_data = jit_vmap_rollout(jnp.arange(100))
    """
    is_recurrent = init_hidden_fn is not None

    # Calculate total steps
    mocap_dt = 1.0 / cfg.env_config.mocap_hz
    steps_per_frame = int(mocap_dt / cfg.env_config.ctrl_dt)
    num_steps = cfg.env_config.clip_length * steps_per_frame - 1
    mocap_hz = cfg.env_config.mocap_hz

    # Get reference clips from the unwrapped environment
    unwrapped_env = env
    while hasattr(unwrapped_env, "_env"):
        unwrapped_env = unwrapped_env._env
    reference_clips = unwrapped_env.reference_clips
    reference_qpos = reference_clips.qpos

    def _build_step_output(state, ctrl, extras):
        """Build the per-step output dict."""
        time_in_frames = state.data.time * mocap_hz
        frame = jnp.floor(time_in_frames + state.info["start_frame"]).astype(int)
        clip = jnp.asarray(state.info["reference_clip"]).astype(int)
        step_output = {
            "qpos": state.data.qpos,
            "ctrl": ctrl,
            "reward": state.reward,
            "frame": frame,
            "clip": clip,
        }

        if log_activations:
            step_output["activations"] = extras["activations"]

        if log_metrics:
            step_output["metrics"] = state.metrics

        if log_sensor_data:
            step_output["joint_forces"] = state.data.cfrc_ext
            step_output["sensor_readings"] = state.data.sensordata

        return step_output

    if is_recurrent:

        def step_fn(carry, _):
            """Single step function for recurrent policies."""
            state, act_rng, hidden = carry
            act_rng, next_rng = jax.random.split(act_rng)

            ctrl, extras, new_hidden = inference_fn(state.obs, hidden, act_rng)
            step_output = _build_step_output(state, ctrl, extras)
            next_state = env.step(state, ctrl)

            if log_full_states:
                return (next_state, next_rng, new_hidden), (step_output, state)
            else:
                return (next_state, next_rng, new_hidden), step_output

    else:

        def step_fn(carry, _):
            """Single step function for feedforward policies."""
            state, act_rng = carry
            act_rng, next_rng = jax.random.split(act_rng)

            ctrl, extras = inference_fn(state.obs, act_rng)
            step_output = _build_step_output(state, ctrl, extras)
            next_state = env.step(state, ctrl)

            if log_full_states:
                return (next_state, next_rng), (step_output, state)
            else:
                return (next_state, next_rng), step_output

    def generate_rollout(clip_idx: int | None = None, seed: int = 42) -> dict[str, Any]:
        """Generate a single episode rollout.

        Args:
            clip_idx: Reference clip index to track. If None, samples randomly.
            seed: Random seed for JAX PRNG initialization.

        Returns:
            Dictionary containing JAX arrays (stacked over timesteps).
        """
        rollout_key = jax.random.PRNGKey(seed)
        rollout_key, reset_rng, act_rng = jax.random.split(rollout_key, 3)

        # Reset environment (clip_idx=None lets the environment sample randomly)
        initial_state = env.reset(reset_rng, clip_idx=clip_idx, start_frame=0)

        # Build initial carry
        if is_recurrent:
            # Initialize hidden state for a single rollout (unbatched)
            hidden = jax.tree.map(lambda x: x[0], init_hidden_fn(1))
            initial_carry = (initial_state, act_rng, hidden)
        else:
            initial_carry = (initial_state, act_rng)

        # Run rollout using scan
        if log_full_states:
            carry_out, (outputs, states) = jax.lax.scan(
                step_fn, initial_carry, None, length=num_steps
            )
        else:
            carry_out, outputs = jax.lax.scan(
                step_fn, initial_carry, None, length=num_steps
            )

        final_state = carry_out[0]

        # Get final step outputs
        final_time_in_frames = final_state.data.time * mocap_hz
        final_frame = jnp.floor(
            final_time_in_frames + final_state.info["start_frame"]
        ).astype(int)
        final_clip = jnp.asarray(final_state.info["reference_clip"]).astype(int)
        rollout_frames = jnp.concatenate([outputs["frame"], final_frame[None]], axis=0)
        rollout_clips = jnp.concatenate([outputs["clip"], final_clip[None]], axis=0)
        qposes_ref = reference_qpos[rollout_clips, rollout_frames]

        # Build result dict with core fields
        result = {
            "qposes_ref": qposes_ref,
            "qposes_rollout": jnp.concatenate(
                [outputs["qpos"], final_state.data.qpos[None]], axis=0
            ),
            "ctrl": outputs["ctrl"],
            "state_rewards": jnp.concatenate(
                [outputs["reward"], final_state.reward[None]], axis=0
            ),
        }

        # Add optional fields
        if log_full_states:
            result["rollout_states"] = states

        if log_activations:
            result["activations"] = outputs["activations"]

        if log_metrics:
            result["rollout_metrics"] = outputs["metrics"]

        if log_sensor_data:
            result["joint_forces"] = outputs["joint_forces"]
            result["sensor_readings"] = outputs["sensor_readings"]

        return result

    return generate_rollout
