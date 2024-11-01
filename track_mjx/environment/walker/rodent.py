import jax
from jax import numpy as jp

from brax.envs.base import PipelineEnv, State
from brax.io import mjcf as mjcf_brax
from brax import math as brax_math
from dm_control.locomotion.walkers import rescale
from dm_control import mjcf as mjcf_dm

from jax.numpy import inf, ndarray
import mujoco
from mujoco import mjx

import numpy as np

import os

from track_mjx.io.preprocess.mjx_preprocess import ReferenceClip

_XML_PATH = "/root/vast/scott-yang/Brax-Rodent-Run/models/rodent.xml"
_MOCAP_HZ = 50
_JOINT_NAMES = [
    "vertebra_1_extend",
    "hip_L_supinate",
    "hip_L_abduct",
    "hip_L_extend",
    "knee_L",
    "ankle_L",
    "toe_L",
    "hip_R_supinate",
    "hip_R_abduct",
    "hip_R_extend",
    "knee_R",
    "ankle_R",
    "toe_R",
    "vertebra_C11_extend",
    "vertebra_cervical_1_bend",
    "vertebra_axis_twist",
    "atlas",
    "mandible",
    "scapula_L_supinate",
    "scapula_L_abduct",
    "scapula_L_extend",
    "shoulder_L",
    "shoulder_sup_L",
    "elbow_L",
    "wrist_L",
    "scapula_R_supinate",
    "scapula_R_abduct",
    "scapula_R_extend",
    "shoulder_R",
    "shoulder_sup_R",
    "elbow_R",
    "wrist_R",
    "finger_R",
]
_BODY_NAMES = [
    "torso",
    "pelvis",
    "upper_leg_L",
    "lower_leg_L",
    "foot_L",
    "upper_leg_R",
    "lower_leg_R",
    "foot_R",
    "skull",
    "jaw",
    "scapula_L",
    "upper_arm_L",
    "lower_arm_L",
    "finger_L",
    "scapula_R",
    "upper_arm_R",
    "lower_arm_R",
    "finger_R",
]

_END_EFF_NAMES = [
    "foot_L",
    "foot_R",
    "hand_L",
    "hand_R",
    "skull",
]


def _bounded_quat_dist(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Computes a quaternion distance limiting the difference to a max of pi/2.

    This function supports an arbitrary number of batch dimensions, B.

    Args:
        source: a quaternion, shape (B, 4).
        target: another quaternion, shape (B, 4).

    Returns:
        Quaternion distance, shape (B, 1).
    """
    source /= jp.linalg.norm(source, axis=-1, keepdims=True)
    target /= jp.linalg.norm(target, axis=-1, keepdims=True)
    # "Distance" in interval [-1, 1].
    dist = 2 * jp.einsum("...i,...i", source, target) ** 2 - 1
    # Clip at 1 to avoid occasional machine epsilon leak beyond 1.
    dist = jp.minimum(1.0, dist)
    # Divide by 2 and add an axis to ensure consistency with expected return
    # shape and magnitude.
    return 0.5 * jp.arccos(dist)[..., np.newaxis]


class RodentTracking(PipelineEnv):
    """Single clip rodent tracking"""

    def __init__(
        self,
        reference_clip,
        torque_actuators: bool = False,
        ref_len: int = 5,
        too_far_dist=0.1,
        bad_pose_dist=jp.inf,
        bad_quat_dist=jp.inf,
        ctrl_cost_weight=0.01,
        ctrl_diff_cost_weight=0.01,
        pos_reward_weight=1.0,
        quat_reward_weight=1.0,
        joint_reward_weight=1.0,
        angvel_reward_weight=1.0,
        bodypos_reward_weight=1.0,
        endeff_reward_weight=1.0,
        healthy_z_range=(0.03, 0.5),
        physics_steps_per_control_step=10,
        reset_noise_scale=1e-3,
        solver="cg",
        iterations: int = 6,
        ls_iterations: int = 6,
        **kwargs,
    ):
        root = mjcf_dm.from_path(_XML_PATH)

        # Convert to torque actuators
        if torque_actuators:
            for actuator in root.find_all("actuator"):
                actuator.gainprm = [actuator.forcerange[1]]
                del actuator.biastype
                del actuator.biasprm

        rescale.rescale_subtree(
            root,
            0.9,
            0.9,
        )
        mj_model = mjcf_dm.Physics.from_mjcf_model(root).model.ptr
        mj_model.opt.solver = {
            "cg": mujoco.mjtSolver.mjSOL_CG,
            "newton": mujoco.mjtSolver.mjSOL_NEWTON,
        }[solver.lower()]
        mj_model.opt.iterations = iterations
        mj_model.opt.ls_iterations = ls_iterations

        mj_model.opt.jacobian = 0

        sys = mjcf_brax.load_model(mj_model)

        kwargs["n_frames"] = kwargs.get("n_frames", physics_steps_per_control_step)
        kwargs["backend"] = "mjx"

        max_physics_steps_per_control_step = int(
            (1.0 / (_MOCAP_HZ * mj_model.opt.timestep))
        )

        super().__init__(sys, **kwargs)
        if max_physics_steps_per_control_step % physics_steps_per_control_step != 0:
            raise ValueError(
                f"physics_steps_per_control_step ({physics_steps_per_control_step}) must be a factor of ({max_physics_steps_per_control_step})"
            )

        self._steps_for_cur_frame = (
            max_physics_steps_per_control_step / physics_steps_per_control_step
        )
        # print(f"self._steps_for_cur_frame: {self._steps_for_cur_frame}")

        self._torso_idx = mujoco.mj_name2id(
            mj_model, mujoco.mju_str2Type("body"), "torso"
        )

        self._joint_idxs = jp.array(
            [
                mujoco.mj_name2id(mj_model, mujoco.mju_str2Type("joint"), joint)
                for joint in _JOINT_NAMES
            ]
        )

        self._body_idxs = jp.array(
            [
                mujoco.mj_name2id(mj_model, mujoco.mju_str2Type("body"), body)
                for body in _BODY_NAMES
            ]
        )

        self._endeff_idxs = jp.array(
            [
                mujoco.mj_name2id(mj_model, mujoco.mju_str2Type("body"), body)
                for body in _END_EFF_NAMES
            ]
        )

        self._reference_clip = reference_clip
        self._bad_pose_dist = bad_pose_dist
        self._too_far_dist = too_far_dist
        self._bad_quat_dist = bad_quat_dist
        self._ref_len = ref_len
        self._pos_reward_weight = pos_reward_weight
        self._quat_reward_weight = quat_reward_weight
        self._joint_reward_weight = joint_reward_weight
        self._angvel_reward_weight = angvel_reward_weight
        self._bodypos_reward_weight = bodypos_reward_weight
        self._endeff_reward_weight = endeff_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._ctrl_diff_cost_weight = ctrl_diff_cost_weight
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale

    def reset(self, rng) -> State:
        """Resets the environment to an initial state."""
        _, start_rng, rng = jax.random.split(rng, 3)

        start_frame = jax.random.randint(start_rng, (), 0, 44)

        info = {
            "cur_frame": start_frame,
            "steps_taken_cur_frame": 0,
            "summed_pos_distance": 0.0,
            "quat_distance": 0.0,
            "joint_distance": 0.0,
            "prev_ctrl": jp.zeros((self.sys.nv,)),
        }

        return self.reset_from_clip(rng, info, noise=True)

    def reset_from_clip(self, rng, info, noise=True) -> State:
        """Reset based on a reference clip."""
        _, rng1, rng2 = jax.random.split(rng, 3)

        # Get reference clip and select the start frame
        reference_frame = jax.tree_map(
            lambda x: x[info["cur_frame"]], self._get_reference_clip(info)
        )

        low, hi = -self._reset_noise_scale, self._reset_noise_scale

        # Add pos
        qpos_with_pos = jp.array(self.sys.qpos0).at[:3].set(reference_frame.position)

        # Add quat
        new_qpos = qpos_with_pos.at[3:7].set(reference_frame.quaternion)

        # Add noise
        qpos = new_qpos + jp.where(
            noise,
            jax.random.uniform(rng1, (self.sys.nq,), minval=low, maxval=hi),
            jp.zeros((self.sys.nq,)),
        )

        qvel = jp.where(
            noise,
            jax.random.uniform(rng1, (self.sys.nv,), minval=low, maxval=hi),
            jp.zeros((self.sys.nv,)),
        )

        data = self.pipeline_init(qpos, qvel)

        reference_obs, proprioceptive_obs = self._get_obs(data, info)

        # Used to intialize our intention network
        info["reference_obs_size"] = reference_obs.shape[-1]

        obs = jp.concatenate([reference_obs, proprioceptive_obs])

        reward, done, zero = jp.zeros(3)
        metrics = {
            "pos_reward": zero,
            "quat_reward": zero,
            "joint_reward": zero,
            "angvel_reward": zero,
            "bodypos_reward": zero,
            "endeff_reward": zero,
            "reward_ctrlcost": zero,
            "ctrl_diff_cost": zero,
            "too_far": zero,
            "bad_pose": zero,
            "bad_quat": zero,
            "fall": zero,
        }

        return State(data, obs, reward, done, metrics, info)

    def step(self, state: State, action: jp.ndarray) -> State:
        """Runs one timestep of the environment's dynamics."""
        data0 = state.pipeline_state
        data = self.pipeline_step(data0, action)

        # Logic for moving to next frame to track to maintain timesteps alignment
        # TODO: Update this to just refer to model.timestep
        info = state.info.copy()
        info["steps_taken_cur_frame"] += 1
        info["cur_frame"] += jp.where(
            info["steps_taken_cur_frame"] == self._steps_for_cur_frame, 1, 0
        )
        info["steps_taken_cur_frame"] *= jp.where(
            info["steps_taken_cur_frame"] == self._steps_for_cur_frame, 0, 1
        )

        # Gets reference clip and indexes to current frame
        reference_clip = jax.tree_map(
            lambda x: x[info["cur_frame"]], self._get_reference_clip(info)
        )

        pos_distance = data.qpos[:3] - reference_clip.position
        pos_reward = self._pos_reward_weight * jp.exp(-400 * jp.sum(pos_distance**2))

        quat_distance = jp.sum(
            _bounded_quat_dist(data.qpos[3:7], reference_clip.quaternion) ** 2
        )
        quat_reward = self._quat_reward_weight * jp.exp(-4.0 * quat_distance)

        joint_distance = jp.sum((data.qpos[7:] - reference_clip.joints) ** 2)
        joint_reward = self._joint_reward_weight * jp.exp(-0.25 * joint_distance)
        info["joint_distance"] = joint_distance

        angvel_reward = self._angvel_reward_weight * jp.exp(
            -0.5 * jp.sum((data.qvel[3:6] - reference_clip.angular_velocity) ** 2)
        )

        bodypos_reward = self._bodypos_reward_weight * jp.exp(
            -8.0
            * jp.sum(
                (
                    data.xpos[self._body_idxs]
                    - reference_clip.body_positions[self._body_idxs]
                ).flatten()
                ** 2
            )
        )

        endeff_reward = self._endeff_reward_weight * jp.exp(
            -500
            * jp.sum(
                (
                    data.xpos[self._endeff_idxs]
                    - reference_clip.body_positions[self._endeff_idxs]
                ).flatten()
                ** 2
            )
        )

        min_z, max_z = self._healthy_z_range
        is_healthy = jp.where(data.xpos[self._torso_idx][2] < min_z, 0.0, 1.0)
        is_healthy = jp.where(data.xpos[self._torso_idx][2] > max_z, 0.0, is_healthy)
        fall = 1.0 - is_healthy

        summed_pos_distance = jp.sum((pos_distance * jp.array([1.0, 1.0, 0.2])) ** 2)
        too_far = jp.where(summed_pos_distance > self._too_far_dist, 1.0, 0.0)
        info["summed_pos_distance"] = summed_pos_distance
        info["quat_distance"] = quat_distance
        bad_pose = jp.where(joint_distance > self._bad_pose_dist, 1.0, 0.0)
        bad_quat = jp.where(quat_distance > self._bad_quat_dist, 1.0, 0.0)
        ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))
        ctrl_diff_cost = self._ctrl_diff_cost_weight * jp.sum(
            jp.square(info["prev_ctrl"] - action)
        )
        info["prev_ctrl"] = action
        reference_obs, proprioceptive_obs = self._get_obs(data, info)
        obs = jp.concatenate([reference_obs, proprioceptive_obs])
        reward = (
            joint_reward
            + pos_reward
            + quat_reward
            + angvel_reward
            + bodypos_reward
            + endeff_reward
            - ctrl_cost
            - ctrl_diff_cost
        )

        # Raise done flag if terminating
        done = jp.max(jp.array([fall, too_far, bad_pose, bad_quat]))

        # Handle nans during sim by resetting env
        reward = jp.nan_to_num(reward)
        obs = jp.nan_to_num(obs)

        from jax.flatten_util import ravel_pytree

        flattened_vals, _ = ravel_pytree(data)
        num_nans = jp.sum(jp.isnan(flattened_vals))
        nan = jp.where(num_nans > 0, 1.0, 0.0)
        done = jp.max(jp.array([nan, done]))

        state.metrics.update(
            pos_reward=pos_reward,
            quat_reward=quat_reward,
            joint_reward=joint_reward,
            angvel_reward=angvel_reward,
            bodypos_reward=bodypos_reward,
            endeff_reward=endeff_reward,
            reward_ctrlcost=-ctrl_cost,
            ctrl_diff_cost=ctrl_diff_cost,
            too_far=too_far,
            bad_pose=bad_pose,
            bad_quat=bad_quat,
            fall=fall,
        )

        return state.replace(
            pipeline_state=data, obs=obs, reward=reward, done=done, info=info
        )

    def _get_reference_clip(self, info) -> ReferenceClip:
        """Returns reference clip; to be overridden in child classes"""
        return self._reference_clip

    def _get_reference_trajectory(self, info) -> ReferenceClip:
        """Slices ReferenceClip into the observation trajectory"""

        # Get the relevant slice of the reference clip
        def f(x):
            if len(x.shape) != 1:
                return jax.lax.dynamic_slice_in_dim(
                    x,
                    info["cur_frame"] + 1,
                    self._ref_len,
                )
            return jp.array([])

        return jax.tree_util.tree_map(f, self._get_reference_clip(info))

    def _get_obs(self, data: mjx.Data, info) -> jp.ndarray:
        """Observes rodent body position, velocities, and angles."""

        ref_traj = self._get_reference_trajectory(info)

        track_pos_local = jax.vmap(
            lambda a, b: brax_math.rotate(a, b), in_axes=(0, None)
        )(
            ref_traj.position - data.qpos[:3],
            data.qpos[3:7],
        ).flatten()

        quat_dist = jax.vmap(
            lambda a, b: brax_math.relative_quat(a, b), in_axes=(0, None)
        )(
            ref_traj.quaternion,
            data.qpos[3:7],
        ).flatten()

        joint_dist = (ref_traj.joints - data.qpos[7:])[:, self._joint_idxs].flatten()

        body_pos_dist_local = jax.vmap(
            lambda a, b: jax.vmap(brax_math.rotate, in_axes=(0, None))(a, b),
            in_axes=(0, None),
        )(
            (ref_traj.body_positions - data.xpos)[:, self._body_idxs],
            data.qpos[3:7],
        ).flatten()

        # print(track_pos_local.shape)
        # print(quat_dist.shape)
        # print(joint_dist.shape)
        # print(body_pos_dist_local.shape)

        reference_obs = jp.concatenate(
            [
                track_pos_local,
                quat_dist,
                joint_dist,
                body_pos_dist_local,
            ]
        )

        print(reference_obs.shape)

        # jax.debug.print("track_pos_local: {}", track_pos_local)
        # jax.debug.print("quat_dist: {}", quat_dist)
        # jax.debug.print("joint_dist: {}", joint_dist)
        # jax.debug.print("body_pos_dist_local: {}", body_pos_dist_local)

        prorioceptive_obs = jp.concatenate(
            [
                data.qpos,
                data.qvel,
            ]
        )
        return reference_obs, prorioceptive_obs


class RodentMultiClipTracking(RodentTracking):
    def __init__(
        self,
        reference_clip,
        torque_actuators: bool = False,
        ref_len: int = 5,
        too_far_dist=0.1,
        bad_pose_dist=jp.inf,
        bad_quat_dist=jp.inf,
        ctrl_cost_weight=0.01,
        ctrl_diff_cost_weight=0.01,
        pos_reward_weight=1,
        quat_reward_weight=1,
        joint_reward_weight=1,
        angvel_reward_weight=1,
        bodypos_reward_weight=1,
        endeff_reward_weight=1,
        healthy_z_range=(0.03, 0.5),
        physics_steps_per_control_step=10,
        reset_noise_scale=0.001,
        solver="cg",
        iterations: int = 6,
        ls_iterations: int = 6,
        **kwargs,
    ):
        super().__init__(
            None,
            torque_actuators,
            ref_len,
            too_far_dist,
            bad_pose_dist,
            bad_quat_dist,
            ctrl_cost_weight,
            ctrl_diff_cost_weight,
            pos_reward_weight,
            quat_reward_weight,
            joint_reward_weight,
            angvel_reward_weight,
            bodypos_reward_weight,
            endeff_reward_weight,
            healthy_z_range,
            physics_steps_per_control_step,
            reset_noise_scale,
            solver,
            iterations,
            ls_iterations,
            **kwargs,
        )

        self._reference_clips = reference_clip
        self._n_clips = reference_clip.position.shape[0]

    def reset(self, rng) -> State:
        """Resets the environment to an initial state."""
        _, start_rng, clip_rng, rng = jax.random.split(rng, 4)

        start_frame = jax.random.randint(start_rng, (), 0, 44)
        clip_idx = jax.random.randint(clip_rng, (), 0, self._n_clips)
        # clip_idx = 492
        info = {
            "clip_idx": clip_idx,
            "cur_frame": start_frame,
            "steps_taken_cur_frame": 0,
            "summed_pos_distance": 0.0,
            "quat_distance": 0.0,
            "joint_distance": 0.0,
            "prev_ctrl": jp.zeros((self.sys.nu,)),
        }

        return self.reset_from_clip(rng, info, noise=True)

    def _get_reference_clip(self, info) -> ReferenceClip:
        """Gets clip based on info["clip_idx"]"""

        return jax.tree_map(lambda x: x[info["clip_idx"]], self._reference_clips)
