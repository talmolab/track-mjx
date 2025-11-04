"""
Functions to load environment and run a rollout with a given policy.
"""

import jax
from brax.envs.base import Env
from track_mjx.environment.walker.fly import Fly
from track_mjx.environment.walker.stick import Stick
from track_mjx.environment.walker.rodent import Rodent
from brax import envs
from typing import Dict, Callable
import hydra
import logging
from track_mjx.environment.task.reward import RewardConfig
from jax import numpy as jnp

from track_mjx.environment.task.multi_clip_tracking import MultiClipTracking
from track_mjx.environment.task.single_clip_tracking import SingleClipTracking
from track_mjx.environment import wrappers
from track_mjx.io import load

from vnl_mjx.tasks.rodent import wrappers as vnl_wrappers
from vnl_mjx.tasks.rodent import imitation
from ml_collections import config_dict

from omegaconf import DictConfig, OmegaConf


def create_environment(cfg_dict: Dict | DictConfig) -> Env:
    """
    Creates the environment based on the provided configuration dictionary.

    Args:
        cfg_dict (Dict | DictConfig): Configuration dictionary for the environment.
    Returns:
        Env: The created environment instance.
    """
    # If the config is the full config, extract the env_config
    if "data_path" in cfg_dict:
        env_cfg = cfg_dict["env_config"]
        env_cfg_ml = config_dict.ConfigDict(
            OmegaConf.to_container(env_cfg, resolve=True)
        )
    else:
        env_cfg_ml = config_dict.ConfigDict(cfg_dict)
    
    env = vnl_wrappers.FlattenObsWrapper(imitation.Imitation(config=env_cfg_ml))

    return env


def create_rollout_generator(
    cfg: Dict | DictConfig,
    env: Env,
    inference_fn: Callable,
    log_activations: bool = False,
    log_metrics: bool = False,
    log_sensor_data: bool = False,
) -> Callable[[int | None], Dict]:
    """
    Creates a rollout generator with JIT-compiled functions.

    Args:
        cfg (Dict | DictConfig): Configuration dictionary for the rollout.
        env (Env): The environment to generate rollouts for.
        inference_fn (Callable): The inference function to compute controls.
        log_activations (bool): Whether to log neural network activations.
        log_metrics (bool): Whether to log rollout metrics.
        log_sensor_data (bool): Whether to log sensor readings.

    Returns:
        Callable: A generate_rollout function that can be called with configuration.
    """

    # JIT-compile the necessary functions
    jit_inference_fn = jax.jit(inference_fn)
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    def generate_rollout(clip_idx: int | None = None, seed: int = 42) -> Dict:
        """
        Generates a rollout using pre-compiled JIT functions.

        Args:
            clip_idx (Optional[int]): Specific clip ID to generate the rollout for.
            seed (int): Random seed for jax PRNGKey.
            log_activations (bool): Whether to log neural network activations.
            log_metrics (bool): Whether to log rollout metrics.
            log_sensor_data (bool): Whether to log sensor readings.

        Returns:
            Dict: A dictionary containing rollout data.
        """

        # Initialize PRNG keys
        rollout_key = jax.random.PRNGKey(seed)
        rollout_key, reset_rng, act_rng = jax.random.split(rollout_key, 3)

        # Reset the environment
        init_state = jit_reset(reset_rng, clip_idx=clip_idx, start_frame=0)

        # Calculate number of steps
        mocap_dt = float(1 / int(cfg.env_config.mocap_hz))
        steps_for_cur_frame = int(float(mocap_dt) / float(cfg.env_config.ctrl_dt))
        num_steps = int(
            int(cfg.env_config.clip_length) * steps_for_cur_frame - 1
        )

        def _step_fn_mlp(carry, _):
            state, act_rng = carry
            act_rng, new_rng = jax.random.split(act_rng)
            ctrl, extras = jit_inference_fn(state.obs, act_rng)
            next_state = jit_step(state, ctrl)

            # Collect optional data based on logging flags
            joint_force = (
                next_state.data.cfrc_ext if log_sensor_data else None
            )
            sensor_reading = (
                next_state.data.sensordata if log_sensor_data else None
            )
            activations = extras["activations"] if log_activations else None

            return (next_state, new_rng), (
                next_state,
                ctrl,
                activations,
                joint_force,
                sensor_reading,
            )

        # Initialize variables
        states = None
        ctrls = None
        activations = None
        joint_forces = None
        sensor_readings = None
        stacked_hidden = None

        # Run rollout for mlp
        init_carry = (init_state, jax.random.PRNGKey(0))
        (final_state, _), (
            states,
            ctrls,
            activations,
            joint_forces,
            sensor_readings,
        ) = jax.lax.scan(_step_fn_mlp, init_carry, None, length=num_steps)

        def prepend(element, arr):
            # Scalar elements shouldn't be modified
            if arr.ndim == 0:
                return arr
            return jnp.concatenate([element[None], arr])

        rollout_states = jax.tree.map(prepend, init_state, states)

        def _get_ref_qpos(state):
            time_in_frames = state.data.time * env._config.mocap_hz
            frame = jnp.floor(time_in_frames + state.info["start_frame"]).astype(int)
            clip = state.info["reference_clip"]
            ref = env.reference_clips.at(clip=clip, frame=frame)
            return ref.qpos

        # Reference and rollout qposes (always logged)
        qposes_ref = jax.vmap(_get_ref_qpos)(rollout_states)

        # Collect qposes from states (always logged)
        qposes_rollout = jax.vmap(lambda s: s.data.qpos)(rollout_states)

        # Extract state rewards (always logged)
        state_rewards = jax.vmap(lambda s: s.reward)(rollout_states)

        # Build return dictionary with required data
        result = {
            "qposes_ref": qposes_ref,
            "qposes_rollout": qposes_rollout,
            "ctrl": ctrls,
            "state_rewards": state_rewards,
        }

        # Add optional data if requested
        if log_metrics:
            rollout_metrics = {}
            for rollout_metric in cfg.logging_config.rollout_metrics:
                rollout_metrics[f"{rollout_metric}s"] = jax.vmap(
                    lambda s: s.metrics[rollout_metric]
                )(rollout_states)
            result["rollout_metrics"] = rollout_metrics

        if log_activations and activations is not None:
            result["activations"] = activations

        if log_sensor_data:
            if joint_forces is not None:
                result["joint_forces"] = joint_forces
            if sensor_readings is not None:
                result["sensor_readings"] = sensor_readings

        return result

    return jax.jit(generate_rollout)
