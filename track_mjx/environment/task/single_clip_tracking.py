"""Base tracking task heer, base env is brax pipeline env"""

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

import collections

import typing
from typing import Any, Callable, Mapping, Optional, Sequence, Set, Text, Union

from track_mjx.io.preprocess.mjx_preprocess import ReferenceClip
from track_mjx.environment.task.reward import compute_tracking_rewards

_MOCAP_HZ = 50


class RodentTracking(PipelineEnv):
    """Single clip walker tracking, agonist of the walker"""

    def __init__(
        self,
        reference_clip,
        walker,
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
        self.walker = walker(torque_actuators)
        self.walker._initialize_indices()

        mj_model = self.walker._mjcf_model.model.ptr
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
        print(f"self._steps_for_cur_frame: {self._steps_for_cur_frame}")

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

        # reward calculation
        (
            pos_reward,
            quat_reward,
            joint_reward,
            angvel_reward,
            bodypos_reward,
            endeff_reward,
            ctrl_cost,
            ctrl_diff_cost,
            too_far,
            bad_pose,
            bad_quat,
            fall,
            info,
        ) = compute_tracking_rewards(
            data=data,
            reference_clip=reference_clip,
            walker=self.walker,
            action=action,
            info=info,
            healthy_z_range=self._healthy_z_range,
            too_far_dist=self._too_far_dist,
            bad_pose_dist=self._bad_pose_dist,
            bad_quat_dist=self._bad_quat_dist,
            pos_reward_weight=self._pos_reward_weight,
            quat_reward_weight=self._quat_reward_weight,
            joint_reward_weight=self._joint_reward_weight,
            angvel_reward_weight=self._angvel_reward_weight,
            bodypos_reward_weight=self._bodypos_reward_weight,
            endeff_reward_weight=self._endeff_reward_weight,
            ctrl_cost_weight=self._ctrl_cost_weight,
            ctrl_diff_cost_weight=self._ctrl_diff_cost_weight,
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

        # TODO: consider not get index, but full get data from walker class?

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

        joint_dist = (ref_traj.joints - data.qpos[7:])[
            :, self.walker._joint_idxs
        ].flatten()

        body_pos_dist_local = jax.vmap(
            lambda a, b: jax.vmap(brax_math.rotate, in_axes=(0, None))(a, b),
            in_axes=(0, None),
        )(
            (ref_traj.body_positions - data.xpos)[:, self.walker._body_idxs],
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
