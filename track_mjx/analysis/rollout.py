"""
rollout file that contains the function to run the trained model on the environment.
"""

import numpy as np
import jax
from brax.envs.base import Env
from track_mjx.environment.walker.rodent import Rodent
from track_mjx.environment.walker.fly import Fly
from track_mjx.environment.walker.mouse_arm import MouseArm
import pickle
from brax import envs
from brax.training.acme import running_statistics, specs
from typing import Dict, List, Callable, Tuple
import orbax.checkpoint as ocp
import functools
from track_mjx.agent import custom_ppo_networks
import hydra
import logging
from track_mjx.io import preprocess as preprocessing
from track_mjx.environment.task.reward import RewardConfig
from jax import numpy as jnp

from track_mjx.environment.task.multi_clip_tracking import MultiClipTracking
from track_mjx.environment.task.single_clip_tracking import SingleClipTracking
from track_mjx.environment import custom_wrappers
from track_mjx.agent import custom_losses as ppo_losses
from track_mjx.io import load
import h5py

from omegaconf import DictConfig, OmegaConf


def restore_config(checkpoint_path: str) -> DictConfig:
    """
    Restore the config from the checkpoint

    Args:
        checkpoint_path (str): The path to the checkpoint

    Returns:
        DictConfig: The restored config
    """
    options = ocp.CheckpointManagerOptions(step_prefix="PPONetwork")
    with ocp.CheckpointManager(
        checkpoint_path,
        options=options,
    ) as mngr:
        print(f"latest checkpoint step: {mngr.latest_step()}")
        config = mngr.restore(
            mngr.latest_step(),
            args=ocp.args.Composite(config=ocp.args.JsonRestore(None)),
        )["config"]
    return config


def create_environment(cfg_dict: Dict | DictConfig) -> Env:
    envs.register_environment("mouse_arm_multi_clip", MultiClipTracking)
    envs.register_environment("rodent_single_clip", SingleClipTracking)
    envs.register_environment("rodent_multi_clip", MultiClipTracking)
    envs.register_environment("fly_multi_clip", MultiClipTracking)

    env_args = cfg_dict["env_config"]["env_args"]
    env_rewards = cfg_dict["env_config"]["reward_weights"]
    train_config = cfg_dict["train_setup"]["train_config"]
    walker_config = cfg_dict["walker_config"]
    traj_config = cfg_dict["reference_config"]

    if isinstance(env_args["reset_noise_scale"], str):
        env_args["reset_noise_scale"] = float(env_args["reset_noise_scale"])

    # TODO(Scott): move this to track_mjx.io module
    input_data_path = hydra.utils.to_absolute_path(cfg_dict["data_path"])
    logging.info(f"Loading data: {input_data_path}")

    # with open(input_data_path, "rb") as file:
    #     reference_clip = pickle.load(file)

    # Split the data path to handle multiple files
    file_paths = input_data_path.split(",")
    print(f"Using first file path: {file_paths[0]}")

    # Just pass the path strings directly - don't open the files yourself
    reference_clip, clip_lengths = load.make_multiclip_data([file_paths[0]])

    # Ensure clip_lengths are integers not strings
    clip_lengths = jnp.array(clip_lengths, dtype=jnp.int32)
    print(f"Loaded reference clip with {clip_lengths[0]} frames")

    # Set up walker
    walker_config = cfg_dict["walker_config"]
    print(f"Joint names: {walker_config['joint_names']}")
    walker = MouseArm(**walker_config)

    # Show important values for debugging
    env_args = cfg_dict["env_config"]["env_args"]
    print(
        f"env._steps_for_cur_frame: {env_args.get('physics_steps_per_control_step', '?')}"
    )

    walker_map = {
        "mouse-arm": MouseArm,
        "rodent": Rodent,
        "fly": Fly,
    }
    walker_class = walker_map[cfg_dict["walker_type"]]
    walker = walker_class(**walker_config)

    reward_config = RewardConfig(
        # too_far_dist=env_rewards.too_far_dist,
        # bad_pose_dist=env_rewards.bad_pose_dist,
        # bad_quat_dist=env_rewards.bad_quat_dist,
        ctrl_cost_weight=env_rewards.ctrl_cost_weight,
        ctrl_diff_cost_weight=env_rewards.ctrl_diff_cost_weight,
        # pos_reward_weight=env_rewards.pos_reward_weight,
        # quat_reward_weight=env_rewards.quat_reward_weight,
        joint_reward_weight=env_rewards.joint_reward_weight,
        # angvel_reward_weight=env_rewards.angvel_reward_weight,
        bodypos_reward_weight=env_rewards.bodypos_reward_weight,
        endeff_reward_weight=env_rewards.endeff_reward_weight,
        # healthy_z_range=env_rewards.healthy_z_range,
        # pos_reward_exp_scale=env_rewards.pos_reward_exp_scale,
        # quat_reward_exp_scale=env_rewards.quat_reward_exp_scale,
        joint_reward_exp_scale=env_rewards.joint_reward_exp_scale,
        # angvel_reward_exp_scale=env_rewards.angvel_reward_exp_scale,
        bodypos_reward_exp_scale=env_rewards.bodypos_reward_exp_scale,
        endeff_reward_exp_scale=env_rewards.endeff_reward_exp_scale,
        # penalty_pos_distance_scale=jnp.array(env_rewards.penalty_pos_distance_scale),
        jerk_cost_weight=env_rewards.jerk_cost_weight,
        energy_cost=env_rewards.energy_cost,
    )
    # Automatically match dict keys and func needs
    env = envs.get_environment(
        env_name=cfg_dict["env_config"]["env_name"],
        reference_clip=reference_clip,
        walker=walker,
        reward_config=reward_config,
        clip_lengths=clip_lengths,
        **env_args,
        **traj_config,
    )
    return env


def create_abstract_policy(
    environment: Env, cfg_dict: Dict | DictConfig
) -> Tuple[Tuple, Callable]:
    """
    Create the policy function for the environment

    Args:
        environment (Env): The environment to create the policy for
        cfg_dict (Dict): The config dictionary, direct translation from the
            yaml file

    Returns:
        Tuple[Tuple, Callable]: The abstract policy and the policy function
    """
    network_factory = functools.partial(
        custom_ppo_networks.make_intention_ppo_networks,
        encoder_hidden_layer_sizes=tuple(
            cfg_dict["network_config"]["encoder_layer_sizes"]
        ),
        decoder_hidden_layer_sizes=tuple(
            cfg_dict["network_config"]["decoder_layer_sizes"]
        ),
        value_hidden_layer_sizes=tuple(
            cfg_dict["network_config"]["critic_layer_sizes"]
        ),
    )

    reset_fn = environment.reset
    key_envs = jax.random.key(1)
    env_state = reset_fn(key_envs)

    ppo_network = network_factory(
        env_state.obs.shape[-1],
        int(env_state.info["reference_obs_size"]),
        environment.action_size,
    )

    make_policy = custom_ppo_networks.make_inference_fn(ppo_network)
    key_policy, key_value = jax.random.split(jax.random.key(1))

    init_params = ppo_losses.PPONetworkParams(
        policy=ppo_network.policy_network.init(key_policy),
        value=ppo_network.value_network.init(key_value),
    )

    abstract_policy = (
        running_statistics.init_state(
            specs.Array(env_state.obs.shape[-1:], jnp.dtype("float32"))
        ),
        init_params.policy,
    )

    return abstract_policy, make_policy


def restore_policy(checkpoint_path: str, abstract_policy: Tuple) -> Tuple:
    options = ocp.CheckpointManagerOptions(step_prefix="PPONetwork")
    with ocp.CheckpointManager(checkpoint_path, options=options) as mngr:
        latest_step = mngr.latest_step()
        print(f"[restore_policy] Latest checkpoint step: {latest_step}")
        policy = mngr.restore(
            latest_step,
            args=ocp.args.Composite(policy=ocp.args.StandardRestore(abstract_policy)),
        )["policy"]

    # Debug: Print a summary of the restored policy parameters
    def print_tree(tree, prefix=""):
        flat, _ = jax.tree_util.tree_flatten(tree)
        for i, x in enumerate(flat):
            if hasattr(x, "shape"):
                print(
                    f"{prefix}Param {i}: shape={x.shape}, dtype={x.dtype}, mean={jnp.nanmean(x):.6f}"
                )
            else:
                print(f"{prefix}Param {i}: {x}")

    print("[restore_policy] Restored policy parameters:")
    print_tree(policy, prefix="Policy: ")

    return policy


def create_inference_fn(environment: Env, cfg_dict: Dict | DictConfig) -> Callable:
    # Wrap the environment to get a rollout version
    rollout_env = custom_wrappers.RenderRolloutWrapperTracking(environment)

    # Create the abstract policy (defines the network architecture and initial parameters)
    abstract_policy, make_policy = create_abstract_policy(rollout_env, cfg_dict)

    # Debug: Print a summary of the abstract policy parameters.
    print("Abstract policy summary:")

    def print_stats(x):
        if hasattr(x, "shape"):
            print(
                f"Shape: {x.shape}, dtype: {x.dtype}, mean: {jnp.nanmean(x):.6f}, min: {jnp.nanmin(x):.6f}, max: {jnp.nanmax(x):.6f}"
            )

    jax.tree_util.tree_map(print_stats, abstract_policy)

    # Load the checkpoint
    checkpoint_path = cfg_dict["train_setup"]["checkpoint_to_restore"]
    policy = restore_policy(checkpoint_path, abstract_policy)

    # Check if restored policy contains any NaNs
    def contains_nans(tree):
        flat, _ = jax.tree_util.tree_flatten(tree)
        return any(jnp.any(jnp.isnan(x)) for x in flat if hasattr(x, "dtype"))

    if contains_nans(policy):
        raise ValueError("Restored policy parameters contain NaNs!")
    else:
        print("Restored policy parameters look good!")

        # Print a detailed summary of the restored parameters
        def print_param_summary(x):
            if hasattr(x, "shape"):
                print(
                    f"Param: shape={x.shape}, dtype={x.dtype}, mean={jnp.nanmean(x):.6f}, min={jnp.nanmin(x):.6f}, max={jnp.nanmax(x):.6f}"
                )

        jax.tree_util.tree_map(print_param_summary, policy)

    # Build and jit the inference function - Use DETERMINISTIC mode to reduce shakiness
    jit_inference_fn = jax.jit(
        make_policy(policy, deterministic=True, get_activation=True)
    )

    # Simple validation that inference function works
    test_state = rollout_env.reset(jax.random.PRNGKey(0))
    test_obs = test_state.obs
    ctrl, _ = jit_inference_fn(test_obs, jax.random.PRNGKey(42))

    if jnp.any(jnp.isnan(ctrl)):
        raise ValueError("Inference function returned NaN control values!")

    return jit_inference_fn


def create_rollout_generator(
    ref_traj_config: Dict | DictConfig, environment: Env, inference_fn: Callable
) -> Callable[[int | None], Dict]:
    """
    Creates a rollout generator with JIT-compiled functions.

    Args:
        environment (Env): The environment to generate rollouts for.
        inference_fn (Callable): The inference function to compute controls.

    Returns:
        Callable: A generate_rollout function that can be called with configuration.
    """
    # Wrap the environment
    rollout_env = custom_wrappers.RenderRolloutWrapperTracking(environment)

    # JIT-compile the necessary functions
    jit_inference_fn = jax.jit(inference_fn)
    jit_reset = jax.jit(rollout_env.reset)
    jit_step = jax.jit(rollout_env.step)

    def generate_rollout(clip_idx: int | None = None, seed: int = 42) -> Dict:
        """
        Generates a rollout using pre-compiled JIT functions.

        Args:
            cfg_dict (Dict): Configuration dictionary.
            clip_idx (Optional[int]): Specific clip ID to generate the rollout for.

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
            if jnp.any(jnp.isnan(next_state.pipeline_state.qpos)):
                raise ValueError("qpos became NaN after stepping the environment!")
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

        # Compute rewards and metrics
        rewards = {
            # "pos_rewards": jax.vmap(lambda s: s.metrics["pos_reward"])(rollout_states),
            "endeff_rewards": jax.vmap(lambda s: s.metrics["endeff_reward"])(
                rollout_states
            ),
            # "quat_rewards": jax.vmap(lambda s: s.metrics["quat_reward"])(
            #     rollout_states
            # ),
            # "angvel_rewards": jax.vmap(lambda s: s.metrics["angvel_reward"])(
            #     rollout_states
            # ),
            "bodypos_rewards": jax.vmap(lambda s: s.metrics["bodypos_reward"])(
                rollout_states
            ),
            "joint_rewards": jax.vmap(lambda s: s.metrics["joint_reward"])(
                rollout_states
            ),
            # "summed_pos_distances": jax.vmap(lambda s: s.info["summed_pos_distance"])(
            #     rollout_states
            # ),
            "joint_distances": jax.vmap(lambda s: s.info["joint_distance"])(
                rollout_states
            ),
            "energy_costs": jax.vmap(lambda s: s.metrics["energy_cost"])(
                rollout_states
            ),
            # "torso_heights": jax.vmap(
            #     lambda s: s.pipeline_state.xpos[environment.walker._torso_idx][2]
            # )(rollout_states),
        }

        # Reference and rollout qposes
        ref_traj = rollout_env._get_reference_clip(init_state.info)
        qposes_ref = jnp.tile(
            jnp.concatenate(
                [ref_traj.position, ref_traj.quaternion, ref_traj.joints], axis=-1
            ),
            (int(environment._steps_for_cur_frame), 1),
        )

        # Collect qposes from states
        qposes_rollout = jax.vmap(lambda s: s.pipeline_state.qpos)(rollout_states)

        return {
            "rewards": rewards,
            "observations": jax.vmap(lambda s: s.obs)(rollout_states),
            "ctrl": ctrls,
            "activations": activations,
            "qposes_ref": qposes_ref,
            "qposes_rollout": qposes_rollout,
            "info": jax.vmap(lambda s: s.info)(rollout_states),
        }

    return jax.jit(generate_rollout)
