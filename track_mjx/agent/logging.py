import os
from pathlib import Path
import jax
import wandb
import imageio
import mujoco
import numpy as np
from jax import numpy as jnp
from dm_control import mjcf as mjcf_dm
from dm_control.locomotion.walkers import rescale


# Restored logging file with rollout and renderer functions.
def log_metric_to_wandb(metric_name: str, data, title: str = ""):
    # ...existing code...
    if isinstance(data[0], tuple):
        frames, values = zip(*data)
    else:
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
    # ...existing code...
    walker_config = cfg["walker_config"]
    pair_render_xml_path = env.walker._pair_rendering_xml_path
    _XML_PATH = (
        Path(__file__).resolve().parent.parent
        / "environment"
        / "walker"
        / pair_render_xml_path
    )

    if cfg.env_config.walker_name == "rodent":
        root = mjcf_dm.from_path(_XML_PATH)
        rescale.rescale_subtree(
            root,
            walker_config.rescale_factor / 0.8,
            walker_config.rescale_factor / 0.8,
        )
        mj_model = mjcf_dm.Physics.from_mjcf_model(root).model.ptr
    elif cfg.env_config.walker_name == "mouse_arm":
        root = mjcf_dm.from_path(_XML_PATH)
        mj_model = mjcf_dm.Physics.from_mjcf_model(root).model.ptr
    elif cfg.env_config.walker_name == "fly":
        spec = mujoco.MjSpec().from_file(str(_XML_PATH))
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
    mj_model.opt.iterations = 20
    mj_model.opt.ls_iterations = 20
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
    mujoco.mj_kinematics(mj_model, mj_data)
    renderer = mujoco.Renderer(mj_model, height=480, width=640)
    return renderer, mj_model, mj_data, scene_option


def rollout_logging_fn(
    env,
    jit_reset,
    jit_step,
    cfg,
    model_path,
    renderer,
    mj_model,
    mj_data,
    scene_option,
    current_step: int,
    jit_logging_inference_fn,
    params,
    policy_params_fn_key,
    max_frames: int = None,
):
    # ...existing code...
    ref_trak_config = cfg["reference_config"]
    env_config = cfg["env_config"]
    _, reset_rng, act_rng = jax.random.split(policy_params_fn_key, 3)
    state = jit_reset(reset_rng)
    rollout = [state]
    for i in range(int(ref_trak_config.clip_length * env._steps_for_cur_frame)):
        _, act_rng = jax.random.split(act_rng)
        obs = state.obs
        ctrl, extras = jit_logging_inference_fn(params, obs, act_rng)
        state = jit_step(state, ctrl)
        if jnp.any(jnp.isnan(state.pipeline_state.qpos)):
            raise ValueError("qpos is NaN after stepping the environment!")
        rollout.append(state)
    endeff_rewards = [state.metrics["endeff_reward"] for state in rollout]
    bodypos_rewards = [state.metrics["bodypos_reward"] for state in rollout]
    joint_rewards = [state.metrics["joint_reward"] for state in rollout]
    joint_distances = [state.info["joint_distance"] for state in rollout]
    jerk_costs = [state.info.get("jerk_cost", 0) for state in rollout]
    log_metric_to_wandb(
        "endeff_rewards",
        list(enumerate(endeff_rewards)),
        title="endeff_rewards for each rollout frame",
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
        "joint_distances",
        list(enumerate(joint_distances)),
        title="joint_distances for each rollout frame",
    )
    log_metric_to_wandb(
        "jerk_costs",
        list(enumerate(jerk_costs)),
        title="jerk_cost for each rollout frame",
    )
    qposes_rollout = np.array([state.pipeline_state.qpos for state in rollout])
    ref_traj = env._get_reference_clip(rollout[0].info)
    qposes_ref = np.repeat(
        np.hstack([ref_traj.joints]), env._steps_for_cur_frame, axis=0
    )
    video_path = f"{model_path}/{current_step}.mp4"
    desired_fps = cfg.env_config.render_fps
    sim_fps = 1.0 / env.dt
    downsample_rate = int(sim_fps / desired_fps)
    # Replace the video creation loop with this simplified version:
    with imageio.get_writer(video_path, fps=desired_fps) as video:
        for idx, (qpos1, qpos2) in enumerate(zip(qposes_rollout, qposes_ref)):

            if idx % downsample_rate != 0:
                continue

            if max_frames is not None and idx >= max_frames:
                break

            mj_data.qpos = np.append(qpos1, qpos2)
            mujoco.mj_forward(mj_model, mj_data)
            renderer.update_scene(
                mj_data, camera=env_config.render_camera_name, scene_option=scene_option
            )
            pixels = renderer.render()
            video.append_data(pixels)
    wandb.log({"eval/rollout": wandb.Video(video_path, format="mp4")})
