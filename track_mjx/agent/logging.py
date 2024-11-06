import os
from absl import flags
import hydra
from omegaconf import DictConfig
import uuid

import functools
import jax
from typing import Dict
import wandb
import imageio
import mujoco
from brax import envs
from dm_control import mjcf as mjcf_dm
from dm_control.locomotion.walkers import rescale

import track_mjx.agent.custom_ppo as ppo
from track_mjx.agent import custom_ppo
from brax.io import model
import numpy as np
import pickle
import warnings
from jax import numpy as jp

from track_mjx.environment import RodentMultiClipTracking, RodentTracking
from track_mjx.io.preprocess.mjx_preprocess import process_clip_to_train
from track_mjx.io import preprocess as preprocessing  # the pickle file needs it
from track_mjx.environment import custom_wrappers
from track_mjx.agent import custom_ppo_networks


def policy_params_fn(
        cfg, env, wandb, num_steps, make_policy, params, rollout_key, model_path
        ):
    """Main logging functions for policy params,
    cfg, wandb, model_path, and env currently func.partial from train.py"""

    # Wrap the env in the brax autoreset and episode wrappers
    # rollout_env = custom_wrappers.AutoResetWrapperTracking(env)
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
    for i in range(int(250 * rollout_env._steps_for_cur_frame)):
        _, act_rng = jax.random.split(act_rng)
        obs = state.obs
        ctrl, extras = jit_inference_fn(obs, act_rng)
        state = jit_step(state, ctrl)
        rollout.append(state)

    pos_rewards = [state.metrics["pos_reward"] for state in rollout]
    table = wandb.Table(
        data=[[x, y] for (x, y) in zip(range(len(pos_rewards)), pos_rewards)],
        columns=["frame", "pos_rewards"],
    )
    wandb.log(
        {
            "eval/rollout_pos_rewards": wandb.plot.line(
                table,
                "frame",
                "pos_rewards",
                title="pos_rewards for each rollout frame",
            )
        },
        commit=False,
    )

    endeff_rewards = [state.metrics["endeff_reward"] for state in rollout]
    table = wandb.Table(
        data=[[x, y] for (x, y) in zip(range(len(endeff_rewards)), endeff_rewards)],
        columns=["frame", "endeff_rewards"],
    )
    wandb.log(
        {
            "eval/rollout_endeff_rewards": wandb.plot.line(
                table,
                "frame",
                "endeff_rewards",
                title="endeff_rewards for each rollout frame",
            )
        },
        commit=False,
    )

    quat_rewards = [state.metrics["quat_reward"] for state in rollout]
    table = wandb.Table(
        data=[[x, y] for (x, y) in zip(range(len(quat_rewards)), quat_rewards)],
        columns=["frame", "quat_rewards"],
    )
    wandb.log(
        {
            "eval/rollout_quat_rewards": wandb.plot.line(
                table,
                "frame",
                "quat_rewards",
                title="quat_rewards for each rollout frame",
            )
        },
        commit=False,
    )

    angvel_rewards = [state.metrics["angvel_reward"] for state in rollout]
    table = wandb.Table(
        data=[[x, y] for (x, y) in zip(range(len(angvel_rewards)), angvel_rewards)],
        columns=["frame", "angvel_rewards"],
    )
    wandb.log(
        {
            "eval/rollout_angvel_rewards": wandb.plot.line(
                table,
                "frame",
                "angvel_rewards",
                title="angvel_rewards for each rollout frame",
            )
        },
        commit=False,
    )

    bodypos_rewards = [state.metrics["bodypos_reward"] for state in rollout]
    table = wandb.Table(
        data=[[x, y] for (x, y) in zip(range(len(bodypos_rewards)), bodypos_rewards)],
        columns=["frame", "bodypos_rewards"],
    )
    wandb.log(
        {
            "eval/rollout_bodypos_rewards": wandb.plot.line(
                table,
                "frame",
                "bodypos_rewards",
                title="bodypos_rewards for each rollout frame",
            )
        },
        commit=False,
    )

    joint_rewards = [state.metrics["joint_reward"] for state in rollout]
    table = wandb.Table(
        data=[[x, y] for (x, y) in zip(range(len(joint_rewards)), joint_rewards)],
        columns=["frame", "joint_rewards"],
    )
    wandb.log(
        {
            "eval/rollout_joint_rewards": wandb.plot.line(
                table,
                "frame",
                "joint_rewards",
                title="joint_rewards for each rollout frame",
            )
        },
        commit=False,
    )

    summed_pos_distances = [state.info["summed_pos_distance"] for state in rollout]
    table = wandb.Table(
        data=[
            [x, y]
            for (x, y) in zip(range(len(summed_pos_distances)), summed_pos_distances)
        ],
        columns=["frame", "summed_pos_distances"],
    )
    wandb.log(
        {
            "eval/rollout_summed_pos_distances": wandb.plot.line(
                table,
                "frame",
                "summed_pos_distances",
                title="summed_pos_distances for each rollout frame",
            )
        },
        commit=False,
    )

    joint_distances = [state.info["joint_distance"] for state in rollout]
    table = wandb.Table(
        data=[[x, y] for (x, y) in zip(range(len(joint_distances)), joint_distances)],
        columns=["frame", "joint_distances"],
    )
    wandb.log(
        {
            "eval/rollout_joint_distances": wandb.plot.line(
                table,
                "frame",
                "joint_distances",
                title="joint_distances for each rollout frame",
            )
        },
        commit=False,
    )

    torso_heights = [state.pipeline_state.xpos[env._torso_idx][2] for state in rollout]
    table = wandb.Table(
        data=[[x, y] for (x, y) in zip(range(len(torso_heights)), torso_heights)],
        columns=["frame", "torso_heights"],
    )
    wandb.log(
        {
            "eval/rollout_torso_heights": wandb.plot.line(
                table,
                "frame",
                "torso_heights",
                title="torso_heights for each rollout frame",
            )
        },
        commit=False,
    )

    # Render the walker with the reference expert demonstration trajectory
    os.environ["MUJOCO_GL"] = "osmesa"
    qposes_rollout = np.array([state.pipeline_state.qpos for state in rollout])

    # def f(x):
    #     if len(x.shape) != 1:
    #         return jax.lax.dynamic_slice_in_dim(
    #             x,
    #             0,
    #             250,
    #         )
    #     return jp.array([])

    # ref_traj = jax.tree_util.tree_map(f, reference_clip)
    ref_traj = rollout_env._get_reference_clip(rollout[0].info)
    print(f"clip_id:{rollout[0].info}")
    qposes_ref = np.repeat(
        np.hstack([ref_traj.position, ref_traj.quaternion, ref_traj.joints]),
        env._steps_for_cur_frame,
        axis=0,
    )

    # Trying to align them when using the reset wrapper...
    # Doesn't work bc reset wrapper handles the done under the hood so it's always 0 :(
    # done_array = np.array([state.done for state in rollout])
    # reset_indices = np.where(done_array == 1.0)[0]
    # if reset_indices.shape[0] == 0:
    #     aligned_traj = qposes_ref
    # else:
    #     aligned_traj = np.zeros_like(qposes_rollout)
    #     # Set the first segment
    #     aligned_traj[: reset_indices[0] + 1] = qposes_ref[: reset_indices[0] + 1]

    #     # Iterate through reset points
    #     for i in range(len(reset_indices) - 1):
    #         start = reset_indices[i] + 1
    #         end = reset_indices[i + 1] + 1
    #         length = end - start
    #         aligned_traj[start:end] = qposes_ref[:length]

    #     # Set the last segment
    #     if reset_indices[-1] < len(done_array) - 1:
    #         start = reset_indices[-1] + 1
    #         length = len(done_array) - start
    #         aligned_traj[start:] = qposes_ref[:length]

    _XML_PATH = os.path.join(
        os.path.dirname(__file__),
        cfg.env_config.ghost_xml_path,
    )  # TODO Better relative path scripts
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

    # save rendering and log to wandb
    mujoco.mj_kinematics(mj_model, mj_data)
    renderer = mujoco.Renderer(mj_model, height=512, width=512)

    # render while stepping using mujoco
    video_path = f"{model_path}/{num_steps}.mp4"

    with imageio.get_writer(video_path, fps=int((1.0 / env.dt))) as video:
        for qpos1, qpos2 in zip(qposes_rollout, qposes_ref):
            mj_data.qpos = np.append(qpos1, qpos2)
            mujoco.mj_forward(mj_model, mj_data)
            renderer.update_scene(mj_data, camera=f"close_profile")
            pixels = renderer.render()
            video.append_data(pixels)

    wandb.log({"eval/rollout": wandb.Video(video_path, format="mp4")})
