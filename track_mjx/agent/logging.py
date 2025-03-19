import os
from pathlib import Path
from absl import flags
import hydra
from omegaconf import DictConfig, OmegaConf
import uuid
from typing import Callable, Any, Optional
from types import ModuleType

import functools
import jax
import wandb
import imageio
import mujoco
from brax import envs
from dm_control import mjcf as mjcf_dm
from dm_control.locomotion.walkers import rescale

from track_mjx.agent import custom_ppo_networks
from track_mjx.agent import custom_losses
from brax.io import model
import numpy as np
from jax import numpy as jp

from track_mjx.environment import custom_wrappers
from brax.envs.base import Env

from flax import linen as nn


def log_metric_to_wandb(metric_name: str, data: jp.ndarray, title: str = ""):
    """Logs a list of metrics to wandb with a specified title.

    Args:
        metric_name: The key under which to log the metric.
        data: List of (x, y) tuples or two lists (frames, rewards).
        title: Title for the wandb plot.
    """
    if isinstance(data[0], tuple):
        # If data is a list of (x, y) tuples, separate it into frames and values
        frames, values = zip(*data)
    else:
        # If data is two lists, use them directly
        frames, values = data

    table = wandb.Table(
        data=[[x, y] for x, y in zip(frames, values)],
        columns=["frame", metric_name],
    )

    wandb.log(
        {
            f"eval/rollout_{metric_name}": wandb.plot.line(
                table,
                "frame",
                metric_name,
                title=title or f"{metric_name} for each rollout frame",
            )
        },
        commit=False,
    )


def make_rollout_renderer(env, cfg):
    walker_config = cfg["walker_config"]
    pair_render_xml_path = env.walker._pair_rendering_xml_path
    _XML_PATH = (
        Path(__file__).resolve().parent.parent
        / "environment"
        / "walker"
        / pair_render_xml_path
    )

    if cfg.env_config.walker_name == "rodent":
        # TODO: Make this ghost rendering walker agonist
        root = mjcf_dm.from_path(_XML_PATH)
        rescale.rescale_subtree(
            root,
            walker_config.rescale_factor / 0.8,
            walker_config.rescale_factor / 0.8,
        )

        mj_model = mjcf_dm.Physics.from_mjcf_model(root).model.ptr
    elif cfg.env_config.walker_name == "fly":
        spec = mujoco.MjSpec()
        spec = spec.from_file(str(_XML_PATH))

        # in training scaled by this amount as well
        for geom in spec.geoms:
            if geom.size is not None:
                geom.size *= walker_config.rescale_factor
            if geom.pos is not None:
                geom.pos *= walker_config.rescale_factor

        mj_model = spec.compile()
    else:
        raise ValueError(f"Unknown walker_name: {cfg.env_config.walker_name}")

    mj_model.opt.solver = {
        "cg": mujoco.mjtSolver.mjSOL_CG,
        "newton": mujoco.mjtSolver.mjSOL_NEWTON,
    }["cg"]

    mj_model.opt.iterations = 6
    mj_model.opt.ls_iterations = 6
    mj_data = mujoco.MjData(mj_model)

    site_id = [
        mj_model.site(i).id
        for i in range(mj_model.nsite)
        if "-0" in mj_model.site(i).name
    ]
    for id in site_id:
        mj_model.site(id).rgba = [1, 0, 0, 1]

    # visual mujoco rendering
    scene_option = mujoco.MjvOption()
    scene_option.sitegroup[:] = [1, 1, 1, 1, 1, 0]
    # scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    # scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

    # save rendering and log to wandb
    mujoco.mj_kinematics(mj_model, mj_data)
    renderer = mujoco.Renderer(mj_model, height=512, width=512)

    return renderer, mj_model, mj_data, scene_option


def rollout_logging_fn(
    env,
    jit_reset,
    jit_step,
    cfg: DictConfig,
    model_path: str,
    renderer,
    mj_model,
    mj_data,
    scene_option,
    current_step: int,  # all args above this one are passed in by functools.partial
    jit_logging_inference_fn,
    params: custom_losses.PPONetworkParams,
    policy_params_fn_key: jax.random.PRNGKey,
) -> None:
    """Logs metrics and videos for a reinforcement learning training rollout.

    Args:
        env: An instance of the base PipelineEnv envrionment.
        jit_reset: Jitted env reset function.
        jit_step: Jitted env step function.
        cfg: Configuration dictionary for the environment and agent.
        model_path: The path to save the model parameters and videos.
        renderer: A mujoco.Renderer object.
        mj_model: A mujoco.Model object for rendering.
        mj_data: A mujoco.Data object for rendering.
        scene_option: A mujoco.MjvOption object for rendering.
        current_step: The number of training steps completed.
        jit_logging_inference_fn: Jitted policy inference function.
        params: Parameters for the policy model.
        policy_params_fn_key: PRNG key.
    """

    ref_trak_config = cfg["reference_config"]
    env_config = cfg["env_config"]
    train_config =cfg['train_setup']['train_config']
    network_config = cfg["network_config"]

    _, reset_rng, act_rng = jax.random.split(policy_params_fn_key, 3)

    state = jit_reset(reset_rng)
    
    #TODO: make this a scan actor_step function
    
    hidden_state =state.info["hidden_state"]
    
    print(f'In rendering, hidden shape is {hidden_state[0].shape}')
    
    rollout = [state]
    for i in range(int(ref_trak_config.clip_length * env._steps_for_cur_frame)):
        _, act_rng = jax.random.split(act_rng)
        obs = state.obs
        
        if train_config['use_lstm']:
            ctrl, extras, hidden_state = jit_logging_inference_fn(params, obs, act_rng, hidden_state)
        else:
            ctrl, extras,  = jit_logging_inference_fn(params, obs, act_rng, None)
            
        state = jit_step(state, ctrl)
            
        rollout.append(state)

    pos_rewards = [state.metrics["pos_reward"] for state in rollout]
    endeff_rewards = [state.metrics["endeff_reward"] for state in rollout]
    quat_rewards = [state.metrics["quat_reward"] for state in rollout]
    angvel_rewards = [state.metrics["angvel_reward"] for state in rollout]
    bodypos_rewards = [state.metrics["bodypos_reward"] for state in rollout]
    joint_rewards = [state.metrics["joint_reward"] for state in rollout]
    summed_pos_distances = [state.info["summed_pos_distance"] for state in rollout]
    joint_distances = [state.info["joint_distance"] for state in rollout]
    torso_heights = [
        state.pipeline_state.xpos[env.walker._torso_idx][2] for state in rollout
    ]
    rewards = [state.reward for state in rollout]
    
    log_metric_to_wandb(
        "rendering_rewards",
        list(enumerate(rewards)),
        title="rendering_rewards for each rollout frame",
    )
    log_metric_to_wandb(
        "pos_rewards",
        list(enumerate(pos_rewards)),
        title="pos_rewards for each rollout frame",
    )
    log_metric_to_wandb(
        "endeff_rewards",
        list(enumerate(endeff_rewards)),
        title="endeff_rewards for each rollout frame",
    )
    log_metric_to_wandb(
        "quat_rewards",
        list(enumerate(quat_rewards)),
        title="quat_rewards for each rollout frame",
    )
    log_metric_to_wandb(
        "angvel_rewards",
        list(enumerate(angvel_rewards)),
        title="angvel_rewards for each rollout frame",
    )
    log_metric_to_wandb(
        "bodypos_rewards",
        list(enumerate(bodypos_rewards)),
        title="bodypos_rewards for each rollout frame",
    )
    log_metric_to_wandb(
        "joint_rewards",
        list(enumerate(joint_rewards)),
        title="joint_rewards for each rollout frame",
    )
    log_metric_to_wandb(
        "summed_pos_distances",
        list(enumerate(summed_pos_distances)),
        title="summed_pos_distances for each rollout frame",
    )
    log_metric_to_wandb(
        "joint_distances",
        list(enumerate(joint_distances)),
        title="joint_distances for each rollout frame",
    )
    log_metric_to_wandb(
        "torso_heights",
        list(enumerate(torso_heights)),
        title="torso_heights for each rollout frame",
    )

    # Render the walker with the reference expert demonstration trajectory
    qposes_rollout = np.array([state.pipeline_state.qpos for state in rollout])
    ref_traj = env._get_reference_clip(rollout[0].info)
    print(f"clip_id:{rollout[0].info}")
    qposes_ref = np.repeat(
        np.hstack([ref_traj.position, ref_traj.quaternion, ref_traj.joints]),
        env._steps_for_cur_frame,
        axis=0,
    )

    # render while stepping using mujoco
    video_path = f"{model_path}/{current_step}.mp4"

    with imageio.get_writer(video_path, fps=int((1.0 / env.dt))) as video:
        for qpos1, qpos2 in zip(qposes_rollout, qposes_ref):
            mj_data.qpos = np.append(qpos1, qpos2)
            mujoco.mj_forward(mj_model, mj_data)
            renderer.update_scene(
                mj_data, camera=env_config.render_camera_name, scene_option=scene_option
            )
            pixels = renderer.render()
            video.append_data(pixels)

    wandb.log({"eval/rollout": wandb.Video(video_path, format="mp4")})
    
    # eval_metrics = rollout[-1].info['eval_metrics']
    # eval_metrics.active_episodes.block_until_ready()
    # metrics = {}
    # for fn in [np.mean, np.std]:
    #     suffix = '_std' if fn == np.std else ''
    #     metrics.update(
    #         {
    #             f'eval/episode_{name}{suffix}': (
    #                 fn(value)
    #             )
    #             for name, value in eval_metrics.episode_metrics.items()
    #         }
    #     )
    # metrics['eval/avg_episode_length'] = np.mean(eval_metrics.episode_steps)
    # wandb.log(metrics)


def render_rollout(
    num_steps: int,
    make_policy: Callable[
        [custom_losses.PPONetworkParams, bool],
        Callable[
            [jax.numpy.ndarray, jax.random.PRNGKey],
            tuple[jax.numpy.ndarray, dict[str, Any]],
        ],
    ],
    params: custom_losses.PPONetworkParams,
    rollout_key: jax.random.PRNGKey,
    cfg: DictConfig,
    env: Env,
    model_path: str,
):
    """
    Rendering rollout to visualize the walker with the reference expert demonstration trajectory, without wandb logging.
    This will save the rendered video to the <model_path>_<num_steps>.mp4, it will also return all the frames for notebook inline rendering.
    """
    ref_trak_config = cfg["reference_config"]
    env_config = cfg["env_config"]
    walker_config = cfg["walker_config"]

    # Wrap the env in the brax autoreset and episode wrappers
    rollout_env = custom_wrappers.RenderRolloutWrapperTracking(env)

    # define the jit reset/step functions
    jit_reset = jax.jit(rollout_env.reset)
    jit_step = jax.jit(rollout_env.step)

    jit_inference_fn = jax.jit(make_policy(params, deterministic=True))
    rollout_key, reset_rng, act_rng = jax.random.split(rollout_key, 3)

    # do a rollout on the saved model
    state = jit_reset(reset_rng)

    rollout = [state]
    for i in range(int(ref_trak_config.clip_length * env._steps_for_cur_frame)):
        _, act_rng = jax.random.split(act_rng)
        obs = state.obs
        ctrl, extras = jit_inference_fn(obs, act_rng)
        state = jit_step(state, ctrl)
        rollout.append(state)

    # might include those reward term in the visual rendering
    # pos_rewards = [state.metrics["pos_reward"] for state in rollout]
    # endeff_rewards = [state.metrics["endeff_reward"] for state in rollout]
    # quat_rewards = [state.metrics["quat_reward"] for state in rollout]
    # angvel_rewards = [state.metrics["angvel_reward"] for state in rollout]
    # bodypos_rewards = [state.metrics["bodypos_reward"] for state in rollout]
    # joint_rewards = [state.metrics["joint_reward"] for state in rollout]
    # summed_pos_distances = [state.info["summed_pos_distance"] for state in rollout]
    # joint_distances = [state.info["joint_distance"] for state in rollout]
    # torso_heights = [state.pipeline_state.xpos[env.walker._torso_idx][2] for state in rollout]

    # Render the walker with the reference expert demonstration trajectory
    qposes_rollout = np.array([state.pipeline_state.qpos for state in rollout])
    ref_traj = rollout_env._get_reference_clip(rollout[0].info)
    print(f"clip_id:{rollout[0].info}")
    qposes_ref = np.repeat(
        np.hstack([ref_traj.position, ref_traj.quaternion, ref_traj.joints]),
        env._steps_for_cur_frame,
        axis=0,
    )

    pair_render_xml_path = env.walker._pair_rendering_xml_path
    _XML_PATH = (
        Path(__file__).resolve().parent.parent
        / "environment"
        / "walker"
        / pair_render_xml_path
    )

    if cfg.env_config.walker_name == "rodent":
        # TODO: Make this ghost rendering walker agonist
        root = mjcf_dm.from_path(_XML_PATH)
        rescale.rescale_subtree(
            root,
            0.9 / 0.8,
            0.9 / 0.8,
        )

        mj_model = mjcf_dm.Physics.from_mjcf_model(root).model.ptr
        mj_model.opt.solver = {
            "cg": mujoco.mjtSolver.mjSOL_CG,
            "newton": mujoco.mjtSolver.mjSOL_NEWTON,
        }["cg"]
        mj_model.opt.iterations = 6
        mj_model.opt.ls_iterations = 6
        mj_data = mujoco.MjData(mj_model)
    elif cfg.env_config.walker_name == "flybody":
        spec = mujoco.MjSpec()
        spec = spec.from_file(str(_XML_PATH))

        # in training scaled by this amount as well
        for geom in spec.geoms:
            if geom.size is not None:
                geom.size *= walker_config.rescale_factor
            if geom.pos is not None:
                geom.pos *= walker_config.rescale_factor

        mj_model = spec.compile()
    else:
        raise ValueError(f"Unknown walker_name: {cfg.env_config.walker_name}")

    mj_model.opt.solver = {
        "cg": mujoco.mjtSolver.mjSOL_CG,
        "newton": mujoco.mjtSolver.mjSOL_NEWTON,
    }["cg"]

    mj_model.opt.iterations = 6
    mj_model.opt.ls_iterations = 6
    mj_data = mujoco.MjData(mj_model)

    site_id = [
        mj_model.site(i).id
        for i in range(mj_model.nsite)
        if "-0" in mj_model.site(i).name
    ]
    for id in site_id:
        mj_model.site(id).rgba = [1, 0, 0, 1]

    # visual mujoco rendering
    scene_option = mujoco.MjvOption()
    scene_option.sitegroup[:] = [1, 1, 1, 1, 1, 0]
    # scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    # scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

    # save rendering and log to wandb
    mujoco.mj_kinematics(mj_model, mj_data)
    renderer = mujoco.Renderer(mj_model, height=512, width=512)

    # render while stepping using mujoco
    video_path = f"{model_path}/{num_steps}.mp4"
    frames = []
    with imageio.get_writer(video_path, fps=int((1.0 / env.dt))) as video:
        for qpos1, qpos2 in zip(qposes_rollout, qposes_ref):

            # TODO: ValueError: could not broadcast input array from shape (148,) into shape (74,)
            mj_data.qpos = np.append(qpos1, qpos2)
            mujoco.mj_forward(mj_model, mj_data)
            renderer.update_scene(
                mj_data, camera=env_config.render_camera_name, scene_option=scene_option
            )
            pixels = renderer.render()
            frames.append(pixels)
            video.append_data(pixels)
    return frames
