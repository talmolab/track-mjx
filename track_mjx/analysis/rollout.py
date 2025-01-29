"""
rollout file that contains the function to run the trained model on the environment.
"""

import numpy as np
import jax
from brax.envs.base import Env
from track_mjx.environment.walker.rodent import Rodent
from track_mjx.environment.walker.fly import Fly
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
            mngr.latest_step(), args=ocp.args.Composite(config=ocp.args.JsonRestore(None))
        )["config"]
    return config

def create_environment(cfg_dict: Dict | DictConfig) -> Env:
    envs.register_environment("rodent_single_clip", SingleClipTracking)
    envs.register_environment("rodent_multi_clip", MultiClipTracking)
    envs.register_environment("fly_multi_clip", MultiClipTracking)

    env_args = cfg_dict["env_config"]["env_args"]
    env_rewards = cfg_dict["env_config"]["reward_weights"]
    train_config = cfg_dict["train_setup"]["train_config"]
    walker_config = cfg_dict["walker_config"]
    traj_config = cfg_dict["reference_config"]

    # TODO: Fix this dependency issue
    import sys

    sys.modules["preprocessing"] = preprocessing
    # TODO(Scott): move this to track_mjx.io module
    input_data_path = hydra.utils.to_absolute_path(cfg_dict["data_path"])
    logging.info(f"Loading data: {input_data_path}")
    with open(input_data_path, "rb") as file:
        reference_clip = pickle.load(file)
    walker_map = {
        "rodent": Rodent,
        "fly": Fly,
    }
    walker_class = walker_map[cfg_dict["walker_type"]]
    walker = walker_class(**walker_config)

    reward_config = RewardConfig(
        too_far_dist=env_rewards.too_far_dist,
        bad_pose_dist=env_rewards.bad_pose_dist,
        bad_quat_dist=env_rewards.bad_quat_dist,
        ctrl_cost_weight=env_rewards.ctrl_cost_weight,
        ctrl_diff_cost_weight=env_rewards.ctrl_diff_cost_weight,
        pos_reward_weight=env_rewards.pos_reward_weight,
        quat_reward_weight=env_rewards.quat_reward_weight,
        joint_reward_weight=env_rewards.joint_reward_weight,
        angvel_reward_weight=env_rewards.angvel_reward_weight,
        bodypos_reward_weight=env_rewards.bodypos_reward_weight,
        endeff_reward_weight=env_rewards.endeff_reward_weight,
        healthy_z_range=env_rewards.healthy_z_range,
        pos_reward_exp_scale=env_rewards.pos_reward_exp_scale,
        quat_reward_exp_scale=env_rewards.quat_reward_exp_scale,
        joint_reward_exp_scale=env_rewards.joint_reward_exp_scale,
        angvel_reward_exp_scale=env_rewards.angvel_reward_exp_scale,
        bodypos_reward_exp_scale=env_rewards.bodypos_reward_exp_scale,
        endeff_reward_exp_scale=env_rewards.endeff_reward_exp_scale,
        penalty_pos_distance_scale=jnp.array(env_rewards.penalty_pos_distance_scale),
    )
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


def create_abstract_policy(environment: Env, cfg_dict: Dict | DictConfig) -> Tuple[Tuple, Callable]:
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
        encoder_hidden_layer_sizes=tuple(cfg_dict["network_config"]["encoder_layer_sizes"]),
        decoder_hidden_layer_sizes=tuple(cfg_dict["network_config"]["decoder_layer_sizes"]),
        value_hidden_layer_sizes=tuple(cfg_dict["network_config"]["critic_layer_sizes"]),
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
        running_statistics.init_state(specs.Array(env_state.obs.shape[-1:], jnp.dtype("float32"))),
        init_params.policy,
    )

    return abstract_policy, make_policy


def restore_policy(checkpoint_path: str, abstract_policy: Tuple) -> Tuple:
    """
    Restore the policy from the checkpoint

    Args:
        checkpoint_path (str): The path to the checkpoint
        abstract_policy (Callable): The abstract policy for loading

    Returns:
        Tuple: The restored policy
    """
    options = ocp.CheckpointManagerOptions(step_prefix="PPONetwork")
    with ocp.CheckpointManager(
        checkpoint_path,
        options=options,
    ) as mngr:
        print(f"latest checkpoint step: {mngr.latest_step()}")
        policy = mngr.restore(
            mngr.latest_step(), args=ocp.args.Composite(policy=ocp.args.StandardRestore(abstract_policy))
        )["policy"]
    return policy

def create_inference_fn(environment: Env, cfg_dict: Dict | DictConfig) -> Callable:
    rollout_env = custom_wrappers.RenderRolloutWrapperTracking(environment)
    abstract_policy, make_policy = create_abstract_policy(rollout_env, cfg_dict)
    # load the checkpoint
    policy = restore_policy(cfg_dict["train_setup"]["checkpoint_to_restore"], abstract_policy)
    jit_inference_fn = jax.jit(make_policy(policy, deterministic=True, get_activation=True))
    return jit_inference_fn


def generate_rollout(environment: Env, cfg_dict: Dict | DictConfig, inference_fn: Callable, clip_idx: int | None = None) -> Dict:
    """
    Generate a rollout for a given clip id, with the loaded checkpoint

    Args:
        environment (Env): The environment to generate the rollout for
        clip_id (int): The clip id to generate the rollout for, if None, will generate a random clip

    returns:
        ctrls (List): The controls for the rollout
        extrases (List): The extra outputs from the policy for the rollout
        rewards (Dict): The rewards for the rollout
    """
    ref_trak_config = cfg_dict["reference_config"]
    # Wrap the env in the brax autoreset and episode wrappers
    rollout_env = custom_wrappers.RenderRolloutWrapperTracking(environment)
    
    jit_inference_fn = jax.jit(inference_fn)

    # define the jit reset/step functions
    jit_reset = jax.jit(rollout_env.reset)
    jit_step = jax.jit(rollout_env.step)

    rollout_key = jax.random.PRNGKey(42)
    rollout_key, reset_rng, act_rng = jax.random.split(rollout_key, 3)
    # do a rollout on the saved model
    state = jit_reset(reset_rng, clip_idx=clip_idx)

    rollout_states = [state]
    ctrls, activations, rewards = [], [], {}
    for i in range(
        int(ref_trak_config.clip_length * environment._steps_for_cur_frame) - 1
    ):  # why is this? what's the observation for the last few step?
        _, act_rng = jax.random.split(act_rng)
        obs = state.obs
        ctrl, extras = jit_inference_fn(obs, act_rng)
        state = jit_step(state, ctrl)
        rollout_states.append(state)
        ctrls.append(ctrl)
        activations.append(extras["activations"])

    # might include those reward term in the visual rendering
    pos_rewards = [state.metrics["pos_reward"] for state in rollout_states]
    endeff_rewards = [state.metrics["endeff_reward"] for state in rollout_states]
    quat_rewards = [state.metrics["quat_reward"] for state in rollout_states]
    angvel_rewards = [state.metrics["angvel_reward"] for state in rollout_states]
    bodypos_rewards = [state.metrics["bodypos_reward"] for state in rollout_states]
    joint_rewards = [state.metrics["joint_reward"] for state in rollout_states]
    summed_pos_distances = [state.info["summed_pos_distance"] for state in rollout_states]
    joint_distances = [state.info["joint_distance"] for state in rollout_states]
    torso_heights = [state.pipeline_state.xpos[environment.walker._torso_idx][2] for state in rollout_states]
    rewards = {
        "pos_rewards": pos_rewards,
        "endeff_rewards": endeff_rewards,
        "quat_rewards": quat_rewards,
        "angvel_rewards": angvel_rewards,
        "bodypos_rewards": bodypos_rewards,
        "joint_rewards": joint_rewards,
        "summed_pos_distances": summed_pos_distances,
        "joint_distances": joint_distances,
        "torso_heights": torso_heights,
    }
    # get qposes for both rollout and reference
    ref_traj = rollout_env._get_reference_clip(rollout_states[0].info)
    qposes_ref = np.repeat(
        np.hstack([ref_traj.position, ref_traj.quaternion, ref_traj.joints]),
        environment._steps_for_cur_frame,
        axis=0,
    )
    qposes_rollout = np.array([state.pipeline_state.qpos for state in rollout_states])
    # processed_states = [process_state_to_save(state) for state in rollout_states]
    observations = [state.obs for state in rollout_states]
    output = {
        "rewards": rewards,
        "observations": observations,
        # "states": processed_states,
        "ctrl": ctrls,
        "activations": activations,
        "qposes_ref": qposes_ref,
        "qposes_rollout": qposes_rollout,
        "info": [state.info for state in rollout_states],
    }

    output = jax.tree.map(lambda x: np.array(x), output)
    return output
