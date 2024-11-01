"""
Entries point for track-mjx. Load the config file, create environments, initialize network, and start training.
"""

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
from track_mjx.io import preprocess as preprocessing # the pickle file needs it
from track_mjx.environment import custom_wrappers

# import custom_ppo_networks
from track_mjx.agent import custom_ppo_networks

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os

from absl import flags

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
os.environ["MUJOCO_GL"] = "egl"

FLAGS = flags.FLAGS

try:
    n_devices = jax.device_count(backend="gpu")
    os.environ["XLA_FLAGS"] = (
        "--xla_gpu_enable_triton_softmax_fusion=true " "--xla_gpu_triton_gemm_any=True "
    )
    print(f"Using {n_devices} GPUs")
except:
    n_devices = 1
    print("Not using GPUs")

flags.DEFINE_enum("solver", "cg", ["cg", "newton"], "constraint solver")
flags.DEFINE_integer("iterations", 4, "number of solver iterations")
flags.DEFINE_integer("ls_iterations", 4, "number of linesearch iterations")


# TODO(Scott): Move this to hydra config file structure
config = {
    "env_name": "multi clip",
    "algo_name": "ppo",
    "task_name": "run",
    "num_envs": 4096 * n_devices,
    "num_timesteps": 100_000, #20_000_000_000,
    "eval_every": 10_000, #200_000_000,
    "episode_length": 200,
    "batch_size": 2048 * n_devices,
    "num_minibatches": 4 * n_devices,
    "num_updates_per_batch": 4,
    "learning_rate": 1e-4,
    "kl_weight": 5e-3,
    "clipping_epsilon": 0.2,
    "torque_actuators": False,
    "physics_steps_per_control_step": 5,
    "too_far_dist": 0.01,
    "bad_pose_dist": 20,
    "bad_quat_dist": 1,
    "ctrl_cost_weight": 0.02,
    "ctrl_diff_cost_weight": 0.02,
    "pos_reward_weight": 1.0,
    "quat_reward_weight": 1.0,
    "joint_reward_weight": 1.0,
    "angvel_reward_weight": 0.0,
    "bodypos_reward_weight": 0.0,
    "endeff_reward_weight": 1.0,
    "healthy_z_range": (0.0325, 0.5),
    "run_platform": "Harvard",
    "solver": "cg",
    "iterations": 4,
    "ls_iterations": 4,
}

envs.register_environment("single clip", RodentTracking)
envs.register_environment("multi clip", RodentMultiClipTracking)


# clip_id = 84  # 84 is the walking in half circle one
# reference_path = f"clips/{clip_id}.p"

# if not os.path.exists(reference_path):
#     os.makedirs(os.path.dirname(reference_path), exist_ok=True)

#     # Process rodent clip and save as pickle
#     reference_clip = process_clip_to_train(
#         stac_path="./transform_snips_new.p",
#         start_step=clip_id * 250,
#         clip_length=250,
#         mjcf_path="./models/rodent_new.xml",
#     )
#     with open(reference_path, "wb") as file:
#         # Use pickle.dump() to save the data to the file
#         pickle.dump(reference_clip, file)
# else:
#     with open(reference_path, "rb") as file:
#         # Use pickle.load() to load the data from the file
#         reference_clip = pickle.load(file)

clip_id = -1

# TODO(Scott): move this to track_mjx.io module
with open("/root/vast/scott-yang/track-mjx/data/twoClips.p", "rb") as file:
    # Use pickle.load() to load the data from the file
    reference_clip = pickle.load(file)


# TODO(Scott): move this to track_mjx.env module
# instantiate the environment
env = envs.get_environment(
    config["env_name"],
    reference_clip=reference_clip,
    torque_actuators=config["torque_actuators"],
    solver=config["solver"],
    iterations=config["iterations"],
    ls_iterations=config["ls_iterations"],
    too_far_dist=config["too_far_dist"],
    bad_pose_dist=config["bad_pose_dist"],
    bad_quat_dist=config["bad_quat_dist"],
    ctrl_cost_weight=config["ctrl_cost_weight"],
    ctrl_diff_cost_weight=config["ctrl_diff_cost_weight"],
    pos_reward_weight=config["pos_reward_weight"],
    quat_reward_weight=config["quat_reward_weight"],
    joint_reward_weight=config["joint_reward_weight"],
    angvel_reward_weight=config["angvel_reward_weight"],
    bodypos_reward_weight=config["bodypos_reward_weight"],
    endeff_reward_weight=config["endeff_reward_weight"],
    healthy_z_range=config["healthy_z_range"],
    physics_steps_per_control_step=config["physics_steps_per_control_step"],
)

# Episode length is equal to (clip length - random init range - traj length) * steps per cur frame
# Will work on not hardcoding these values later
episode_length = (250 - 50 - 5) * env._steps_for_cur_frame
print(f"episode_length {episode_length}")

train_fn = functools.partial(
    ppo.train,
    num_timesteps=config["num_timesteps"],
    num_evals=int(config["num_timesteps"] / config["eval_every"]),
    num_resets_per_eval=1,
    reward_scaling=1,
    episode_length=episode_length,
    normalize_observations=True,
    action_repeat=1,
    clipping_epsilon=config["clipping_epsilon"],
    unroll_length=20,
    num_minibatches=config["num_minibatches"],
    num_updates_per_batch=config["num_updates_per_batch"],
    discounting=0.95,
    learning_rate=config["learning_rate"],
    kl_weight=config["kl_weight"],
    entropy_cost=1e-2,
    num_envs=config["num_envs"],
    batch_size=config["batch_size"],
    seed=0,
    network_factory=functools.partial(
        custom_ppo_networks.make_intention_ppo_networks,
        encoder_hidden_layer_sizes=(512, 512),
        decoder_hidden_layer_sizes=(512, 512),
        value_hidden_layer_sizes=(512, 512),
    ),
)

import uuid

# Generates a completely random UUID (version 4)
run_id = uuid.uuid4()
model_path = f"./model_checkpoints/{run_id}"



# TODO(Scott): Move this to vnl_track.agent.logging module
run = wandb.init(project="vnl_debug", config=config, notes=f"clip_id: {clip_id}")


wandb.run.name = (
    f"{config['env_name']}_{config['task_name']}_{config['algo_name']}_{run_id}"
)


def wandb_progress(num_steps, metrics):
    metrics["num_steps"] = num_steps
    wandb.log(metrics, commit=False)


# Wrap the env in the brax autoreset and episode wrappers
# rollout_env = custom_wrappers.AutoResetWrapperTracking(env)
rollout_env = custom_wrappers.RenderRolloutWrapperTracking(env)
# define the jit reset/step functions
jit_reset = jax.jit(rollout_env.reset)
jit_step = jax.jit(rollout_env.step)

# TODO(Scott): Move this to vnl_track.agent.logging module
def policy_params_fn(
    num_steps, make_policy, params, rollout_key, model_path=model_path
):
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

    _XML_PATH = os.path.join(os.path.dirname(__file__),
                         'environment/walker/assets/rodent_ghostpair_scale080.xml') # TODO Better relative path scripts
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


make_inference_fn, params, _ = train_fn(
    environment=env, progress_fn=wandb_progress, policy_params_fn=policy_params_fn
)

final_save_path = f"{model_path}/brax_ppo_rodent_run_finished"
model.save_params(final_save_path, params)
print(f"Run finished. Model saved to {final_save_path}")
