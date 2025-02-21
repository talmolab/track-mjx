import os
from pathlib import Path
from absl import flags
import hydra
from omegaconf import DictConfig, OmegaConf
import uuid
from typing import Callable, Any
from types import ModuleType

import functools
import jax
import wandb
import imageio
import mujoco
from brax import envs
from dm_control import mjcf as mjcf_dm
from dm_control.locomotion.walkers import rescale

from track_mjx.agent import ppo
from track_mjx.agent import ppo_networks
from track_mjx.agent import losses
from brax.io import model
import numpy as np
from jax import numpy as jp

from track_mjx.environment import wrappers
from brax.envs.base import Env


def log_lineplot_to_wandb(name: str, metric_name: str, data: jp.ndarray, title: str):
    """Logs a table of values and its line plot to wandb.

    Args:
        name: The name of the lineplot in wandb (i.e. eval/reward_over_rollout).
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
            name: wandb.plot.line(
                table,
                "frame",
                metric_name,
                title=title,
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
    params: losses.PPONetworkParams,
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

    _, reset_rng, act_rng = jax.random.split(policy_params_fn_key, 3)

    state = jit_reset(reset_rng)

    rollout = [state]
    latent_means = []
    latent_logvars = []
    for i in range(int(ref_trak_config.clip_length * env._steps_for_cur_frame)):
        _, act_rng = jax.random.split(act_rng)
        obs = state.obs
        ctrl, extras = jit_logging_inference_fn(params, obs, act_rng)
        latent_means.append(extras["latent_mean"])
        latent_logvars.append(extras["latent_logvar"])
        state = jit_step(state, ctrl)
        rollout.append(state)

    # plot the statistics of each latent dim (representing means and logvars sampled)
    latent_logvars = jp.stack(latent_logvars)
    latent_means = jp.stack(latent_means)
    latent_means_means = jp.mean(latent_means, axis=0)
    latent_logvars_means = jp.mean(latent_logvars, axis=0)
    latent_means_stds = jp.std(latent_means, axis=0)
    latent_logvars_stds = jp.std(latent_logvars, axis=0)
    for i in range(latent_means_means.shape[0]):
        wandb.log(
            {
                f"latents/latent_means_mean{i}": latent_means_means[i],
                f"latents/latent_means_std{i}": latent_means_stds[i],
                f"latents/latent_logvars_mean{i}": latent_logvars_means[i],
                f"latents/latent_logvars_std{i}": latent_logvars_stds[i],
            },
            commit=False,
        )

    for rollout_metric in cfg.logging_config.rollout_metrics:
        log_lineplot_to_wandb(
            f"eval/rollout_{rollout_metric}",
            rollout_metric,
            list(enumerate([state.metrics[rollout_metric] for state in rollout])),
            title=f"{rollout_metric} for each rollout frame",
        )
    # torso_heights = [
    #     state.pipeline_state.xpos[env.walker._torso_idx][2] for state in rollout
    # ]

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


def render_rollout(
    num_steps: int,
    make_policy: Callable[
        [losses.PPONetworkParams, bool],
        Callable[
            [jax.numpy.ndarray, jax.random.PRNGKey],
            tuple[jax.numpy.ndarray, dict[str, Any]],
        ],
    ],
    params: losses.PPONetworkParams,
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
    rollout_env = wrappers.RenderRolloutWrapperTracking(env)

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
