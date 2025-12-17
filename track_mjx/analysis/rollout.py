"""Rollout generation utilities for VNL imitation learning environments.

This module provides functions to create VNL environments and generate policy
rollouts with optional logging of activations, metrics, and sensor data.
Rollouts can be used for evaluation, visualization, or data collection.
"""

from typing import Any, Callable

import jax
from brax.envs.base import Env
from jax import numpy as jnp
from ml_collections import config_dict
from omegaconf import DictConfig, OmegaConf
from vnl_playground.tasks.rodent import imitation
from vnl_playground.tasks.rodent import wrappers as vnl_wrappers


def create_environment(cfg_dict: dict[str, Any] | DictConfig) -> Env:
    """Create a VNL imitation learning environment from a configuration.

    Wraps the VNL Imitation environment with a FlattenObsWrapper for
    compatibility with standard RL algorithms expecting flat observations.

    Args:
        cfg_dict: Configuration dictionary. Can be either:
            - Full config with "data_path" key: extracts "env_config" section
            - Direct env_config dict: used as-is

    Returns:
        A Brax-compatible environment with flattened observations.

    Example:
        >>> env = create_environment(cfg)
        >>> state = env.reset(jax.random.PRNGKey(0))
    """
    if "data_path" in cfg_dict:
        env_cfg = cfg_dict["env_config"]
        env_cfg_ml = config_dict.ConfigDict(
            OmegaConf.to_container(env_cfg, resolve=True)
        )
    else:
        env_cfg_ml = config_dict.ConfigDict(cfg_dict)

    return vnl_wrappers.FlattenObsWrapper(imitation.Imitation(config=env_cfg_ml))


def create_rollout_generator(
    cfg: dict[str, Any] | DictConfig,
    env: Env,
    inference_fn: Callable[[jnp.ndarray, jax.Array], tuple[jnp.ndarray, dict]],
    log_activations: bool = False,
    log_metrics: bool = False,
    log_sensor_data: bool = False,
) -> Callable[[int | None, int], dict[str, Any]]:
    """Create a JIT-compiled rollout generator for a given environment and policy.

    Returns a function that generates full episode rollouts using pre-compiled
    JAX functions for efficient repeated evaluation.

    Args:
        cfg: Full configuration dict containing env_config with timing parameters
            (mocap_hz, ctrl_dt, clip_length).
        env: The VNL environment to run rollouts in.
        inference_fn: Policy function with signature (obs, rng) -> (action, extras).
            The extras dict should contain "activations" if log_activations=True.
        log_activations: If True, collect neural network activations at each step.
        log_metrics: If True, collect all metrics from state.metrics at each step.
        log_sensor_data: If True, collect contact forces and sensor readings.

    Returns:
        A generate_rollout function with signature:
            generate_rollout(clip_idx=None, seed=42) -> dict

    Example:
        >>> generate_rollout = create_rollout_generator(cfg, env, policy_fn)
        >>> rollout_data = generate_rollout(clip_idx=5, seed=123)
        >>> print(len(rollout_data["qposes_rollout"]))
    """
    jit_inference_fn = jax.jit(inference_fn)
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    def generate_rollout(
        clip_idx: int | None = None, seed: int = 42
    ) -> dict[str, Any]:
        """Generate a single episode rollout.

        Runs the policy in the environment for the full clip length, collecting
        states, actions, rewards, and optional diagnostic data.

        Args:
            clip_idx: Reference clip index to track. If None, samples randomly.
            seed: Random seed for JAX PRNG initialization.

        Returns:
            Dictionary containing:
                - rollout_states: List of Brax State objects (for rendering)
                - qposes_ref: Reference motion capture poses per timestep
                - qposes_rollout: Actual simulated poses per timestep
                - ctrl: Control actions applied at each step
                - state_rewards: Reward at each timestep
                - rollout_metrics: (if log_metrics) Dict of metric lists
                - activations: (if log_activations) Network activations per step
                - joint_forces: (if log_sensor_data) Contact forces per step
                - sensor_readings: (if log_sensor_data) Sensor data per step
        """
        rollout_key = jax.random.PRNGKey(seed)
        rollout_key, reset_rng, act_rng = jax.random.split(rollout_key, 3)

        state = jit_reset(reset_rng, clip_idx=clip_idx, start_frame=0)

        # Calculate total steps: (clip_length * mocap_frames_per_ctrl_step) - 1
        mocap_dt = 1.0 / cfg.env_config.mocap_hz
        steps_per_frame = int(mocap_dt / cfg.env_config.ctrl_dt)
        num_steps = cfg.env_config.clip_length * steps_per_frame - 1

        rollout_states = [state]
        ctrls = []
        activations = []
        joint_forces = []
        sensor_readings = []

        for _ in range(num_steps):
            _, act_rng = jax.random.split(act_rng)
            ctrl, extras = jit_inference_fn(state.obs, act_rng)
            ctrls.append(ctrl)

            if log_activations:
                activations.append(extras["activations"])

            state = jit_step(state, ctrl)
            rollout_states.append(state)

            if log_sensor_data:
                joint_forces.append(state.data.cfrc_ext)
                sensor_readings.append(state.data.sensordata)

        # Extract reference poses for comparison
        qposes_ref = []
        for s in rollout_states:
            time_in_frames = s.data.time * env._config.mocap_hz
            frame = jnp.floor(time_in_frames + s.info["start_frame"]).astype(int)
            clip = s.info["reference_clip"]
            ref = env.reference_clips.at(clip=clip, frame=frame)
            qposes_ref.append(ref.qpos)

        result = {
            "rollout_states": rollout_states,
            "qposes_ref": qposes_ref,
            "qposes_rollout": [s.data.qpos for s in rollout_states],
            "ctrl": ctrls,
            "state_rewards": [s.reward for s in rollout_states],
        }

        if log_metrics:
            metric_keys = rollout_states[0].metrics.keys()
            result["rollout_metrics"] = {
                f"{k}s": [s.metrics[k] for s in rollout_states] for k in metric_keys
            }

        if log_activations and activations:
            result["activations"] = activations

        if log_sensor_data:
            if joint_forces:
                result["joint_forces"] = joint_forces
            if sensor_readings:
                result["sensor_readings"] = sensor_readings

        return result

    return generate_rollout
