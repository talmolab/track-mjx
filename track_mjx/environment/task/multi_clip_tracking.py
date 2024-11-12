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
from track_mjx.environment.task.single_clip_tracking import RodentTracking


class RodentMultiClipTracking(RodentTracking):
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
            walker,
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
