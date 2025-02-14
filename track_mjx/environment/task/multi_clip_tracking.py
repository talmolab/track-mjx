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
from track_mjx.environment.walker.mouse_arm import MouseArm
from track_mjx.environment.task.single_clip_tracking import SingleClipTracking

from track_mjx.environment.task.reward import RewardConfig


class MultiClipTracking(SingleClipTracking):
    """Multi clip walker tracking using SingleTracking env, agonist of the walker"""

    def __init__(
        self,
        reference_clip: ReferenceClip | None,
        walker: MouseArm,
        reward_config: RewardConfig | None,
        physics_steps_per_control_step: int,
        reset_noise_scale: float,
        solver: str = "cg",
        iterations: int = 4,
        ls_iterations: int = 4,
        mj_model_timestep: float = 0.002,
        mocap_hz: int = 50,
        clip_length: int = 250,
        random_init_range: int = 50,
        traj_length: int = 5,
        **kwargs: Any,
    ):
        """Initializes the MultiTracking environment.

        Args:
            reference_clip (ReferenceClip, Optional): The reference trajectory data. None is used when in pure rendering mode.
            walker: The base walker model.
            torque_actuators: Whether to use torque actuators.
            reward_config: Reward configuration.
            physics_steps_per_control_step: Number of physics steps per control step.
            reset_noise_scale: Scale of noise for reset.
            solver: Solver type for Mujoco.
            iterations: Maximum number of solver iterations.
            ls_iterations: Maximum number of line search iterations.
            mj_model_timestep: fundamental time increment of the MuJoCo physics simulation
            mocap_hz: cycles per second for the reference data
            clip_length: clip length of the tracking clips
            random_init_range: the initiated range
            traj_length: one trajectory length
            **kwargs: Additional arguments for the PipelineEnv initialization.
        """
        super().__init__(
            None,
            walker,
            reward_config,
            physics_steps_per_control_step,
            reset_noise_scale,
            solver,
            iterations,
            ls_iterations,
            mj_model_timestep,
            mocap_hz,
            clip_length,
            random_init_range,
            traj_length,
            **kwargs,
        )
        if reference_clip is not None:
            self._reference_clips = reference_clip
            self._n_clips = reference_clip.body_positions.shape[0]
        else:
            print("No reference clip provided, in pure rendering mode.")

    def reset(self, rng: jp.ndarray, clip_idx: int | None = None) -> State:
        """
        Resets the environment to an initial state.

        Args:
            rng (jp.ndarray): Random key for reproducibility.
            clip_idx (int, optional): Index of the clip to reset to. Defaults to None.

        Returns:
            State: The initial state of the environment.
        """
        _, start_rng, clip_rng, rng = jax.random.split(rng, 4)

        start_frame = jax.random.randint(start_rng, (), 0, 44)
        if clip_idx is None:
            clip_idx = jax.random.randint(clip_rng, (), 0, self._n_clips)  # type: ignore
        info = {
            "clip_idx": clip_idx,
            "start_frame": start_frame,
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

        return jax.tree.map(lambda x: x[info["clip_idx"]], self._reference_clips)

class MouseArmMultiClipTracking(SingleClipTracking):
    """Multi clip walker tracking task for mouse arm using SingleTracking env, agonist of the walker"""

    def __init__(
        self,
        reference_clip: ReferenceClip | None,
        walker: MouseArm,
        reward_config: RewardConfig | None,
        physics_steps_per_control_step: int,
        reset_noise_scale: float,
        solver: str = "cg",
        iterations: int = 4,
        ls_iterations: int = 4,
        mj_model_timestep: float = 0.002,
        mocap_hz: int = 50,
        clip_length: int = 250,
        random_init_range: int = 50,
        traj_length: int = 5,
        **kwargs: Any,
    ):
        """Initializes the MultiTracking environment.

        Args:
            reference_clip (ReferenceClip, Optional): The reference trajectory data. None is used when in pure rendering mode.
            walker: The base walker model.
            torque_actuators: Whether to use torque actuators.
            reward_config: Reward configuration.
            physics_steps_per_control_step: Number of physics steps per control step.
            reset_noise_scale: Scale of noise for reset.
            solver: Solver type for Mujoco.
            iterations: Maximum number of solver iterations.
            ls_iterations: Maximum number of line search iterations.
            mj_model_timestep: fundamental time increment of the MuJoCo physics simulation
            mocap_hz: cycles per second for the reference data
            clip_length: clip length of the tracking clips
            random_init_range: the initiated range
            traj_length: one trajectory length
            **kwargs: Additional arguments for the PipelineEnv initialization.
        """
        super().__init__(
            None,
            walker,
            reward_config,
            physics_steps_per_control_step,
            reset_noise_scale,
            solver,
            iterations,
            ls_iterations,
            mj_model_timestep,
            mocap_hz,
            clip_length,
            random_init_range,
            traj_length,
            **kwargs,
        )
        if reference_clip is not None:
            self._reference_clips = reference_clip
            self._n_clips = reference_clip.body_positions.shape[0]
        else:
            print("No reference clip provided, in pure rendering mode.")

    def reset(self, rng: jp.ndarray, clip_idx: int | None = None) -> State:
        """
        Resets the environment to an initial state.

        Args:
            rng (jp.ndarray): Random key for reproducibility.
            clip_idx (int, optional): Index of the clip to reset to. Defaults to None.

        Returns:
            State: The initial state of the environment.
        """
        _, start_rng, clip_rng, rng = jax.random.split(rng, 4)

        start_frame = jax.random.randint(start_rng, (), 0, 44)
        if clip_idx is None:
            clip_idx = jax.random.randint(clip_rng, (), 0, self._n_clips)  # type: ignore
        info = {
            "clip_idx": clip_idx,
            "start_frame": start_frame,
            "summed_pos_distance": 0.0,
            "quat_distance": 0.0,
            "joint_distance": 0.0,
            "prev_ctrl": jp.zeros((self.sys.nu,)),
        }

        return self.reset_from_clip(rng, info, noise=True)
    
    def reset_from_clip(
        self, rng: jp.ndarray, info: dict[str, Any], noise: bool = True
    ) -> State:
        """Resets the environment using a reference clip.

        Args:
            rng: Random number generator state.
            info: Information dictionary.
            noise: Whether to add noise during reset. Defaults to True.

        Returns:
            State: The reset environment state.
        """
        _, rng1, rng2 = jax.random.split(rng, 3)

        # Get reference clip and select the start frame
        reference_frame = jax.tree.map(
            lambda x: x[info["start_frame"]], self._get_reference_clip(info)
        )

        low, hi = -self._reset_noise_scale, self._reset_noise_scale

        # # Add pos
        # qpos_with_pos = jp.array(self.sys.qpos0).at[:3].set(reference_frame.position)

        # # Add quat
        # new_qpos = qpos_with_pos.at[3:7].set(reference_frame.quaternion)

        # # Add noise
        # qpos = new_qpos + jp.where(
        #     noise,
        #     jax.random.uniform(rng1, (self.sys.nq,), minval=low, maxval=hi),
        #     jp.zeros((self.sys.nq,)),
        # )

        new_qpos = jp.concatenate(
            (
                #reference_frame.position,
                #reference_frame.quaternion,
                reference_frame.joints,
            ),
            axis=0,
        )
        qpos = new_qpos + jax.random.uniform(
            rng1, (self.sys.nq,), minval=low, maxval=hi
        )

        qvel = jp.where(
            noise,
            jax.random.uniform(rng1, (self.sys.nv,), minval=low, maxval=hi),
            jp.zeros((self.sys.nv,)),
        )

        data = self.pipeline_init(qpos, qvel)

        reference_obs, proprioceptive_obs = self._get_obs(data, info)

        # Used to initialize our intention network
        info["reference_obs_size"] = reference_obs.shape[-1]

        obs = jp.concatenate([reference_obs, proprioceptive_obs])

        reward, done, zero = jp.zeros(3)
        metrics = {
            #"pos_reward": zero,
            #"quat_reward": zero,
            "joint_reward": zero,
            #"angvel_reward": zero,
            "bodypos_reward": zero,
            "endeff_reward": zero,
            "reward_ctrlcost": zero,
            "ctrl_diff_cost": zero,
            #"too_far": zero,
            #"bad_pose": zero,
            #"bad_quat": zero,
            #"fall": zero,
            #"nan": zero,
        }

        return State(data, obs, reward, done, metrics, info)

    def _get_reference_clip(self, info: dict[str, jp.ndarray]) -> ReferenceClip:
        """
        Retrieves the reference clip corresponding to the current clip index.

        Args:
            info: Dictionary containing clip information.

        Returns:
            ReferenceClip: The reference clip for the given index.
        """

        return jax.tree.map(lambda x: x[info["clip_idx"]], self._reference_clips)
