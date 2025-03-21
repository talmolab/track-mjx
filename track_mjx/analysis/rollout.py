"""
rollout file that contains the function to run the trained model on the environment.
"""

import numpy as np
import jax
from brax.envs.base import Env
from track_mjx.environment.walker.rodent import Rodent
from track_mjx.environment.walker.fly import Fly
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


from omegaconf import DictConfig


def create_environment(cfg_dict: Dict | DictConfig) -> Env:
    envs.register_environment("rodent_single_clip", SingleClipTracking)
    envs.register_environment("rodent_multi_clip", MultiClipTracking)
    envs.register_environment("fly_multi_clip", MultiClipTracking)

    env_args = cfg_dict["env_config"]["env_args"]
    env_rewards = cfg_dict["env_config"]["reward_weights"]
    walker_config = cfg_dict["walker_config"]
    traj_config = cfg_dict["reference_config"]

    reference_data_path = hydra.utils.to_absolute_path(cfg_dict["data_path"])
    logging.info(f"Loading data: {reference_data_path}")
    try:
        reference_clip = load.make_multiclip_data(reference_data_path)
    except KeyError:
        logging.info(
            f"Loading from stac-mjx format failed. Loading from ReferenceClip format."
        )
        reference_clip = load.load_reference_clip_data(reference_data_path)

    walker_map = {
        "rodent": Rodent,
        "fly": Fly,
    }
    walker_class = walker_map[cfg_dict["walker_type"]]
    walker = walker_class(**walker_config)

    reward_config = RewardConfig(**env_rewards)
    # Automatically match dict keys and func needs
    env = envs.get_environment(
        env_name=cfg_dict["env_config"]["env_name"],
        reference_clip=reference_clip,
        walker=walker,
        reward_config=reward_config,
        **env_args,
        **traj_config,
    )
    return env


def create_rollout_generator(
    cfg: Dict | DictConfig, environment: Env, inference_fn: Callable
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
    if type(environment) == MultiClipTracking:
        rollout_env = wrappers.RenderRolloutWrapperMulticlipTracking(environment)
    elif type(environment) == SingleClipTracking:
        rollout_env = wrappers.RenderRolloutWrapperSingleclipTracking(environment)

    # JIT-compile the necessary functions
    jit_inference_fn = jax.jit(inference_fn)
    jit_reset = jax.jit(rollout_env.reset)
    jit_step = jax.jit(rollout_env.step)

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
        init_state = jit_reset(reset_rng, clip_idx=clip_idx)

        num_steps = (
            int(ref_traj_config.clip_length * environment._steps_for_cur_frame) - 1
        )

        def _step_fn(carry, _):
            state, act_rng = carry
            act_rng, new_rng = jax.random.split(act_rng)
            ctrl, extras = jit_inference_fn(state.obs, act_rng)
            next_state = jit_step(state, ctrl)
            return (next_state, new_rng), (next_state, ctrl, extras["activations"])

        # Run rollout
        init_carry = (init_state, jax.random.PRNGKey(0))
        (final_state, _), (states, ctrls, activations) = jax.lax.scan(
            _step_fn, init_carry, None, length=num_steps
        )

        def prepend(element, arr):
            # Scalar elements shouldn't be modified
            if arr.ndim == 0:
                return arr

            return jnp.concatenate([element[None], arr])

        rollout_states = jax.tree.map(prepend, init_state, states)

        # Get metrics
        rollout_metrics = {}
        for rollout_metric in cfg.logging_config.rollout_metrics:
            rollout_metrics[f"{rollout_metric}s"] = jax.vmap(
                lambda s: s.metrics[rollout_metric]
            )(rollout_states)

        # Reference and rollout qposes
        ref_traj = rollout_env._get_reference_clip(init_state.info)
        qposes_ref = jnp.repeat(
            jnp.hstack([ref_traj.position, ref_traj.quaternion, ref_traj.joints]),
            int(environment._steps_for_cur_frame),
            axis=0,
        )

        # Collect qposes from states
        qposes_rollout = jax.vmap(lambda s: s.pipeline_state.qpos)(rollout_states)

        return {
            "rollout_metrics": rollout_metrics,
            "observations": jax.vmap(lambda s: s.obs)(rollout_states),
            "ctrl": ctrls,
            "activations": activations,
            "qposes_ref": qposes_ref,
            "qposes_rollout": qposes_rollout,
            "info": jax.vmap(lambda s: s.info)(rollout_states),
        }

    return jax.jit(generate_rollout)
