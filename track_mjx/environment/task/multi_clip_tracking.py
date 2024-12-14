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

from track_mjx.environment.task.reward import RewardConfig


class MultiClipTracking(SingleClipTracking):
    """Multi clip walker tracking using SingleTracking env, agonist of the walker"""

    def __init__(
        self,
        reference_clip: ReferenceClip,
        walker: BaseWalker,
        reward_config: RewardConfig,
        ref_len: int = 5,
        physics_steps_per_control_step: int = 10,
        reset_noise_scale: float = 1e-3,
        solver: str = "cg",
        iterations: int = 6,
        ls_iterations: int = 6,
        mj_model_timestep: float = 2e-4,
        mocap_hz: int = 50,
        **kwargs: Any,
    ):
        """Initializes the MultiTracking environment.

        Args:
            reference_clip: The reference trajectory data.
            walker: The base walker model.
            torque_actuators: Whether to use torque actuators.
            reward_config: Reward configuration.
            ref_len: Length of the reference trajectory.
            physics_steps_per_control_step: Number of physics steps per control step.
            reset_noise_scale: Scale of noise for reset.
            solver: Solver type for Mujoco.
            iterations: Maximum number of solver iterations.
            ls_iterations: Maximum number of line search iterations.
            mj_model_timestep: fundamental time increment of the MuJoCo physics simulation
            mocap_hz: cycles per second for the reference data
            **kwargs: Additional arguments for the PipelineEnv initialization.
        """
        super().__init__(
            None,
            walker,
            reward_config,
            ref_len,
            physics_steps_per_control_step,
            reset_noise_scale,
            solver,
            iterations,
            ls_iterations,
            mj_model_timestep,
            mocap_hz,
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
