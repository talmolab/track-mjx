"""
Functions to load environment and run a rollout with a given policy.
"""

import numpy as np
import jax
from brax.envs.base import Env
from track_mjx.environment.walker.rodent import Rodent
from track_mjx.environment.walker.fly import Fly
from track_mjx.environment.reacher.mouse_arm import MouseArm
from brax import envs
from typing import Dict, Callable
import hydra
import logging
from track_mjx.environment.task.reward import RewardConfig, RewardConfigReach
from jax import numpy as jnp

from track_mjx.environment.task.multi_clip_tracking import MultiClipTracking
from track_mjx.environment.task.single_clip_tracking import SingleClipTracking
from track_mjx.environment.task.multi_clip_reaching import MultiClipReaching
from track_mjx.environment.task.single_clip_reaching import SingleClipReaching
from track_mjx.environment import wrappers
from track_mjx.io import load
from track_mjx.io.load import load_reaching_data, make_reaching_data, make_multiclip_reaching_data

from omegaconf import DictConfig

envs.register_environment("mouse_arm_multi_clip", MultiClipReaching)
envs.register_environment("mouse_arm_single_clip", SingleClipReaching)
envs.register_environment("rodent_single_clip", SingleClipTracking)
envs.register_environment("rodent_multi_clip", MultiClipTracking)
envs.register_environment("fly_multi_clip", MultiClipTracking)


def create_environment(cfg_dict: Dict | DictConfig) -> Env:
    env_args = cfg_dict["env_config"]["env_args"]
    env_rewards = cfg_dict["env_config"]["reward_weights"]
    traj_config = cfg_dict["reference_config"]

    reference_data_path = hydra.utils.to_absolute_path(cfg_dict["data_path"])
    logging.info(f"Loading data: {reference_data_path}")
    
    # SIMPLE: Just use the right loader for the right task
    if "reacher_config" in cfg_dict:
        # Reaching task - load_reaching_data handles everything
        reference_clip = load_reaching_data(reference_data_path)
        logging.info(f"Loaded reaching data with joints shape: {reference_clip.joints.shape}")
    else:
        # Tracking task
        try:
            reference_clip = load.make_multiclip_data(
                reference_data_path, n_frames_per_clip=traj_config.clip_length
            )
        except KeyError:
            logging.info("Loading from stac-mjx format failed. Loading from ReferenceClip format.")
            reference_clip = load.load_reference_clip_data(reference_data_path)

    # Body setup
    walker_map = {"rodent": Rodent, "fly": Fly}
    reacher_map = {"mouse_arm": MouseArm}

    if "walker_config" in cfg_dict:
        body_config = cfg_dict["walker_config"]
        body_class = walker_map[cfg_dict["env_config"]["walker_name"]]
    elif "reacher_config" in cfg_dict:
        body_config = cfg_dict["reacher_config"]
        body_class = reacher_map[cfg_dict["env_config"]["reacher_name"]]
    else:
        raise KeyError("Expected 'walker_config' or 'reacher_config' in configuration.")

    walker = body_class(**body_config)

    # Reward config
    if "energy_cost_weight" not in env_rewards:
        env_rewards["energy_cost_weight"] = 0.0

    if "walker_config" in cfg_dict:
        reward_config = RewardConfig(**env_rewards)
    elif "reacher_config" in cfg_dict:
        reward_config = RewardConfigReach(**env_rewards)
    else:
        raise KeyError("Expected 'walker_config' or 'reacher_config' in configuration.")

    # Create environment
    if "walker_config" in cfg_dict:
        env = envs.get_environment(
            env_name=cfg_dict["env_config"]["env_name"],
            reference_clip=reference_clip,
            walker=walker,
            reward_config=reward_config,
            **env_args,
            **traj_config,
        )
    elif "reacher_config" in cfg_dict:
        env = envs.get_environment(
            env_name=cfg_dict["env_config"]["env_name"],
            reference_clip=reference_clip,
            reacher=walker,  # pass as reacher
            reward_config=reward_config,
            **env_args,
            **traj_config,
        )
    else:
        raise KeyError("Expected 'walker_config' or 'reacher_config' in configuration.")
    
    # Debug logging
    logging.info(f"Created environment: {type(env)}")
    if hasattr(env, '_n_clips'):
        logging.info(f"Environment _n_clips: {env._n_clips}")
    if hasattr(env, 'sys'):
        logging.info(f"Environment sys.nq: {env.sys.nq}, sys.nu: {env.sys.nu}, sys.nv: {env.sys.nv}")
    
    return env


def create_rollout_generator(
    cfg: Dict | DictConfig,
    environment: Env,
    inference_fn: Callable,
    model: str = "mlp",
    log_activations: bool = False,
    log_metrics: bool = False,
    log_sensor_data: bool = False,
) -> Callable[[int | None], Dict]:
    """
    Creates a rollout generator with JIT-compiled functions.

    Args:
        environment (Env): The environment to generate rollouts for.
        inference_fn (Callable): The inference function to compute controls.

    Returns:
        Callable: A generate_rollout function that can be called with configuration.
    """
    ref_traj_config = cfg["reference_config"]
    # Wrap the environment
    # TODO this logic is used in a few different places, make it a function?
    rollout_env = environment  # Initialize with base environment

    if type(environment) == MultiClipTracking or type(environment) == MultiClipReaching:
        rollout_env = wrappers.RenderRolloutWrapperMulticlipTracking(environment)
    elif type(environment) == SingleClipReaching or type(environment) == SingleClipTracking:
        rollout_env = wrappers.RenderRolloutWrapperSingleclipTracking(environment)

    if cfg["train_setup"]["train_config"]["use_lstm"]:
        rollout_env = wrappers.RenderRolloutWrapperTrackingLSTM(environment)

    # JIT-compile the necessary functions
    jit_inference_fn = jax.jit(inference_fn)
    
    # Debug version: add logging and error handling for reset
    def debug_reset(rng, clip_idx=None):
        logging.info(f"Debug reset called with clip_idx={clip_idx}")
        logging.info(f"Environment type: {type(rollout_env)}")
        if hasattr(rollout_env, '_n_clips'):
            logging.info(f"Environment _n_clips: {rollout_env._n_clips}")
        try:
            result = rollout_env.reset(rng, clip_idx=clip_idx)
            logging.info(f"Reset successful, state.obs.shape: {result.obs.shape}")
            return result
        except Exception as e:
            logging.error(f"Reset failed with error: {e}")
            raise
    
    jit_reset = jax.jit(debug_reset)
    jit_step = jax.jit(rollout_env.step)

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
        init_state = jit_reset(reset_rng, clip_idx=clip_idx)

        num_steps = (
            int(ref_traj_config.clip_length * environment._steps_for_cur_frame) - 1
        )

        def _step_fn_mlp(carry, _):
            state, act_rng = carry
            act_rng, new_rng = jax.random.split(act_rng)
            ctrl, extras = jit_inference_fn(state.obs, act_rng)
            next_state = jit_step(state, ctrl)

            # Collect optional data based on logging flags
            joint_force = (
                next_state.pipeline_state.cfrc_ext if log_sensor_data else None
            )
            sensor_reading = (
                next_state.pipeline_state.sensordata if log_sensor_data else None
            )
            activations = extras["activations"] if log_activations else None

            return (next_state, new_rng), (
                next_state,
                ctrl,
                activations,
                joint_force,
                sensor_reading,
            )

        def _step_fn_lstm(carry, _):
            state, act_rng, hidden = carry
            act_rng, new_rng = jax.random.split(act_rng)
            ctrl, extras, new_hidden = jit_inference_fn(state.obs, act_rng, hidden)
            ctrl = jnp.squeeze(ctrl, axis=0)
            next_state = jit_step(state, ctrl)

            # Collect optional data based on logging flags
            joint_force = (
                next_state.pipeline_state.cfrc_ext if log_sensor_data else None
            )
            sensor_reading = (
                next_state.pipeline_state.sensordata if log_sensor_data else None
            )
            activations = extras["activations"] if log_activations else None

            return (next_state, new_rng, new_hidden), (
                next_state,
                ctrl,
                hidden,
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

        if model == "mlp":
            # Run rollout for mlp
            init_carry = (init_state, jax.random.PRNGKey(0))
            (final_state, _), (
                states,
                ctrls,
                activations,
                joint_forces,
                sensor_readings,
            ) = jax.lax.scan(_step_fn_mlp, init_carry, None, length=num_steps)

        elif model == "lstm":
            # Run rollout for lstm
            init_carry = (
                init_state,
                jax.random.PRNGKey(0),
                init_state.info["hidden_state"],
            )
            (final_state, _, final_hidden_state), (
                states,
                ctrls,
                stacked_hidden,
                activations,
                joint_forces,
                sensor_readings,
            ) = jax.lax.scan(_step_fn_lstm, init_carry, None, length=num_steps)

        def prepend(element, arr):
            # Scalar elements shouldn't be modified
            if arr.ndim == 0:
                return arr
            return jnp.concatenate([element[None], arr])

        rollout_states = jax.tree.map(prepend, init_state, states)

        # Reference and rollout qposes (always logged)
        ref_traj = rollout_env._get_reference_clip(init_state.info)

        # Check if we're dealing with ReferenceClipReach (reaching task)
        if hasattr(ref_traj, 'position') and hasattr(ref_traj, 'quaternion'):
            # Tracking task
            qposes_ref = jnp.repeat(
                jnp.hstack([ref_traj.position, ref_traj.quaternion, ref_traj.joints]),
                int(environment._steps_for_cur_frame),
                axis=0,
            )
        else:
            # Reaching task - only has joints
            qposes_ref = jnp.repeat(
                ref_traj.joints,
                int(environment._steps_for_cur_frame),
                axis=0,
            )

        # Collect qposes from states (always logged)
        qposes_rollout = jax.vmap(lambda s: s.pipeline_state.qpos)(rollout_states)

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
