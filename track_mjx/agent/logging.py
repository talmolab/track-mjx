import os
from pathlib import Path
from absl import flags
import hydra
from omegaconf import DictConfig, OmegaConf
import uuid
from typing import Callable, Any

import functools
import jax
import wandb
import imageio
import mujoco
from brax import envs
from dm_control import mjcf as mjcf_dm
from dm_control.locomotion.walkers import rescale

from track_mjx.agent import custom_ppo
from track_mjx.agent import custom_ppo_networks
from track_mjx.agent import custom_losses
from brax.io import model
import numpy as np
import pickle
import warnings
from jax import numpy as jp

from track_mjx.environment import custom_wrappers
from brax.envs.base import PipelineEnv


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


def setup_training_logging(
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
    env: PipelineEnv,
    wandb: Callable[[int, dict[str, jp.ndarray]], None],
    model_path: str,
) -> None:
    """Logs metrics and videos for a reinforcement learning training rollout.

    Args:
        num_steps: The number of training steps completed.
        make_policy: A function to create the policy with parameters and deterministic mode (reference to custom_ppo_networks).
        params: Parameters for the policy model (reference to custom_losses).
        rollout_key: A PRNG key for generating rollout randomness.
        cfg: Configuration dictionary for the environment and agent.
        env: An instance of the base PipelineEnv envrionment.
        wandb: The Weights and Biases logging function (reference to train).
        model_path: The path to save the model parameters and videos.
    """

    ref_trak_config = cfg["reference_config"]
    env_config = cfg["env_config"]

    # Wrap the env in the brax autoreset and episode wrappers
    rollout_env = custom_wrappers.RenderRolloutWrapperTracking(env)

    # define the jit reset/step functions
    jit_reset = jax.jit(rollout_env.reset)
    jit_step = jax.jit(rollout_env.step)

    os.makedirs(model_path, exist_ok=True)
    model.save_params(f"{model_path}/{num_steps}", params)
    jit_inference_fn = jax.jit(make_policy(params, deterministic=True))
    rollout_key, reset_rng, act_rng = jax.random.split(rollout_key, 3)

    state = jit_reset(reset_rng)

    rollout = [state]
    for i in range(int(ref_trak_config.clip_length * env._steps_for_cur_frame)):
        _, act_rng = jax.random.split(act_rng)
        obs = state.obs
        ctrl, extras = jit_inference_fn(obs, act_rng)
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
    ref_traj = rollout_env._get_reference_clip(rollout[0].info)
    print(f"clip_id:{rollout[0].info}")
    qposes_ref = np.repeat(
        np.hstack([ref_traj.position, ref_traj.quaternion, ref_traj.joints]),
        env._steps_for_cur_frame,
        axis=0,
    )
    
    pair_render_xml_path = env.walker._pair_rendering_xml_path
    _XML_PATH = Path(__file__).resolve().parent.parent / "environment" / "walker" / pair_render_xml_path

    # root = mjcf_dm.from_path(_XML_PATH)
    # rescale.rescale_subtree(
    #     root,
    #     0.9 / 0.8,
    #     0.9 / 0.8,
    # )
    # mj_model = mjcf_dm.Physics.from_mjcf_model(root).model.ptr
    
    spec = mujoco.MjSpec()
    spec = spec.from_file(str(_XML_PATH))
    
    # in training scaled by this amount as well
    scaling_factor = 0.9
    for geom in spec.geoms:
        if geom.size is not None:
            geom.size *= scaling_factor
        if geom.pos is not None:
            geom.pos *= scaling_factor

    mj_model = spec.compile()

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

    scene_option = mujoco.MjvOption()
    scene_option.sitegroup[:] = [1, 1, 1, 1, 1, 0]
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

    # save rendering and log to wandb
    mujoco.mj_kinematics(mj_model, mj_data)
    renderer = mujoco.Renderer(mj_model, height=512, width=512)

    # render while stepping using mujoco
    video_path = f"{model_path}/{num_steps}.mp4"

    with imageio.get_writer(video_path, fps=env_config.render_fps) as video:
        for qpos1, qpos2 in zip(qposes_rollout, qposes_ref):

            # TODO: ValueError: could not broadcast input array from shape (148,) into shape (74,)
            mj_data.qpos = np.append(qpos1, qpos2)
            mujoco.mj_forward(mj_model, mj_data)
            renderer.update_scene(mj_data, camera=env_config.render_camera_name)
            pixels = renderer.render()
            video.append_data(pixels)

    wandb.log({"eval/rollout": wandb.Video(video_path, format="mp4")})
