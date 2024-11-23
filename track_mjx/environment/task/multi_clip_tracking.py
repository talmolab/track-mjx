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
from typing import Any

from track_mjx.io.preprocess.mjx_preprocess import ReferenceClip
from track_mjx.environment.walker.base import BaseWalker
from track_mjx.environment.task.single_clip_tracking import SingleClipTracking


class MultiClipTracking(SingleClipTracking):
    """Multi clip walker tracking using SingleTracking env, agonist of the walker"""

    def __init__(
        self,
        reference_clip: ReferenceClip,
        walker: BaseWalker,
        ref_len: int = 5,
        too_far_dist: float = 0.1,
        bad_pose_dist: float = jp.inf,
        bad_quat_dist: float = jp.inf,
        ctrl_cost_weight: float = 0.01,
        ctrl_diff_cost_weight: float = 0.01,
        pos_reward_weight: float = 1.0,
        quat_reward_weight: float = 1.0,
        joint_reward_weight: float = 1.0,
        angvel_reward_weight: float = 1.0,
        bodypos_reward_weight: float = 1.0,
        endeff_reward_weight: float = 1.0,
        healthy_z_range: tuple[float, float] = (0.03, 0.5),
        pos_reward_exp_scale: float = 400.0,
        quat_reward_exp_scale: float = 4.0,
        joint_reward_exp_scale: float = 0.25,
        angvel_reward_exp_scale: float = 0.5,
        bodypos_reward_exp_scale: float = 8.0,
        endeff_reward_exp_scale: float = 500.0,
        penalty_pos_distance_scale: jp.ndarray = jp.array([1.0, 1.0, 0.2]),
        physics_steps_per_control_step: int = 10,
        reset_noise_scale: float = 1e-3,
        solver: str = "cg",
        iterations: int = 6,
        ls_iterations: int = 6,
        **kwargs: Any,
    ):
        """Initializes the MultiTracking environment.

        Args:
            reference_clip: The reference trajectory data.
            walker: The base walker model.
            torque_actuators: Whether to use torque actuators.
            ref_len: Length of the reference trajectory.
            too_far_dist: Threshold for "too far" penalty.
            bad_pose_dist: Threshold for "bad pose" penalty.
            bad_quat_dist: Threshold for "bad quaternion" penalty.
            ctrl_cost_weight: Weight for control cost.
            ctrl_diff_cost_weight: Weight for control difference cost.
            pos_reward_weight: Weight for position reward.
            quat_reward_weight: Weight for quaternion reward.
            joint_reward_weight: Weight for joint reward.
            angvel_reward_weight: Weight for angular velocity reward.
            bodypos_reward_weight: Weight for body position reward.
            endeff_reward_weight: Weight for end-effector reward.
            healthy_z_range: Range for a healthy z-position.
            pos_reward_exp_scale: Scaling factor for position rewards.
            quat_reward_exp_scale: Scaling factor for quaternion rewards.
            joint_reward_exp_scale: Scaling factor for joint rewards.
            angvel_reward_exp_scale: Scaling factor for angular velocity rewards.
            bodypos_reward_exp_scale: Scaling factor for body position rewards.
            endeff_reward_exp_scale: Scaling factor for end-effector rewards.
            penalty_pos_distance_scale: Scaling factor for positional penalties as an array.
            physics_steps_per_control_step: Number of physics steps per control step.
            reset_noise_scale: Scale of noise for reset.
            solver: Solver type for Mujoco.
            iterations: Maximum number of solver iterations.
            ls_iterations: Maximum number of line search iterations.
            **kwargs: Additional arguments for the PipelineEnv initialization.
        """
        super().__init__(
            None,
            walker,
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
            pos_reward_exp_scale,
            quat_reward_exp_scale,
            joint_reward_exp_scale,
            angvel_reward_exp_scale,
            bodypos_reward_exp_scale,
            endeff_reward_exp_scale,
            penalty_pos_distance_scale,
            physics_steps_per_control_step,
            reset_noise_scale,
            solver,
            iterations,
            ls_iterations,
            **kwargs,
        )

        self._reference_clips = reference_clip
        self._n_clips = reference_clip.position.shape[0]

    def reset(self, rng: jp.ndarray) -> State:
        """
        Resets the environment to an initial state.

        Args:
            rng (jp.ndarray): Random key for reproducibility.

        Returns:
            State: The initial state of the environment.
        """
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

    def _get_reference_clip(self, info: dict[str, jp.ndarray]) -> ReferenceClip:
        """
        Retrieves the reference clip corresponding to the current clip index.

        Args:
            info: Dictionary containing clip information.

        Returns:
            ReferenceClip: The reference clip for the given index.
        """

        return jax.tree_map(lambda x: x[info["clip_idx"]], self._reference_clips)
