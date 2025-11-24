"""
Functions to load environment and run a rollout with a given policy.
"""

import jax
from brax.envs.base import Env
from typing import Dict, Callable
from jax import numpy as jnp

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

        Returns:
            Dict: A dictionary containing rollout data.
        """

        # Initialize PRNG keys
        rollout_key = jax.random.PRNGKey(seed)
        rollout_key, reset_rng, act_rng = jax.random.split(rollout_key, 3)

        # Reset the environment
        state = jit_reset(reset_rng, clip_idx=clip_idx, start_frame=0)

        # Calculate number of steps
        mocap_dt = float(1 / int(cfg.env_config.mocap_hz))
        steps_for_cur_frame = int(float(mocap_dt) / float(cfg.env_config.ctrl_dt))
        num_steps = int(
            int(cfg.env_config.clip_length) * steps_for_cur_frame - 1
        )

        # Initialize lists to collect data
        rollout_states = [state]
        ctrls = []
        activations = []
        joint_forces = []
        sensor_readings = []

        # Run rollout using a for loop
        for i in range(num_steps):
            _, act_rng = jax.random.split(act_rng)
            ctrl, extras = jit_inference_fn(state.obs, act_rng)
            
            # Store control
            ctrls.append(ctrl)
            
            # Collect activations if requested
            if log_activations:
                activations.append(extras["activations"])
            
            # Step the environment
            state = jit_step(state, ctrl)
            rollout_states.append(state)
            
            # Collect sensor data if requested
            if log_sensor_data:
                joint_forces.append(state.data.cfrc_ext)
                sensor_readings.append(state.data.sensordata)

        # Extract reference qposes from states
        qposes_ref = []
        for state in rollout_states:
            time_in_frames = state.data.time * env._config.mocap_hz
            frame = jnp.floor(time_in_frames + state.info["start_frame"]).astype(int)
            clip = state.info["reference_clip"]
            ref = env.reference_clips.at(clip=clip, frame=frame)
            qposes_ref.append(ref.qpos)

        # Collect qposes from states
        qposes_rollout = [state.data.qpos for state in rollout_states]

        # Extract state rewards
        state_rewards = [state.reward for state in rollout_states]

        # Build return dictionary with required data
        result = {
            "rollout_states": rollout_states,  # List of states for env.render
            "qposes_ref": qposes_ref,
            "qposes_rollout": qposes_rollout,
            "ctrl": ctrls,
            "state_rewards": state_rewards,
        }

        # Add optional data if requested
        if log_metrics:
            rollout_metrics = {}
            metric_keys = rollout_states[0].metrics.keys()
            for rollout_metric in metric_keys:
                rollout_metrics[f"{rollout_metric}s"] = [
                    state.metrics[rollout_metric] for state in rollout_states
                ]
            result["rollout_metrics"] = rollout_metrics

        if log_activations and activations:
            result["activations"] = activations

        if log_sensor_data:
            if joint_forces:
                result["joint_forces"] = joint_forces
            if sensor_readings:
                result["sensor_readings"] = sensor_readings

        return result

    return generate_rollout
