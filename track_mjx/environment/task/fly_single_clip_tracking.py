"""Base fly tracking task here, base env is brax pipeline env"""

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
from track_mjx.environment.walker.base import BaseWalker
from track_mjx.environment.task.single_clip_tracking import SingleClipTracking


class FlyTracking(SingleClipTracking):
    """Single clip fly tracking"""

    def __init__(
        self,
        reference_clip: ReferenceClip,
        walker: BaseWalker,
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
        healthy_z_range=(-0.03, 0.1),
        physics_steps_per_control_step=10,
        reset_noise_scale=1e-3,
        solver="cg",
        iterations: int = 6,
        ls_iterations: int = 6,
        **kwargs: Any,
    ):
        """Initializes the FlyTracking environment."""
        super().__init__(
            reference_clip=reference_clip,
            walker=walker,
            ref_len=ref_len,
            too_far_dist=too_far_dist,
            bad_pose_dist=bad_pose_dist,
            bad_quat_dist=bad_quat_dist,
            ctrl_cost_weight=ctrl_cost_weight,
            ctrl_diff_cost_weight=ctrl_diff_cost_weight,
            pos_reward_weight=pos_reward_weight,
            quat_reward_weight=quat_reward_weight,
            joint_reward_weight=joint_reward_weight,
            angvel_reward_weight=angvel_reward_weight,
            bodypos_reward_weight=bodypos_reward_weight,
            endeff_reward_weight=endeff_reward_weight,
            healthy_z_range=healthy_z_range,
            physics_steps_per_control_step=physics_steps_per_control_step,
            reset_noise_scale=reset_noise_scale,
            solver=solver,
            iterations=iterations,
            ls_iterations=ls_iterations,
            **kwargs,
        )
