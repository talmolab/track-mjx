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
        reference_clip: ReferenceClip | None,
        walker: BaseWalker,
        reward_config: RewardConfig | None,
        physics_steps_per_control_step: int,
        reset_noise_scale: float,
        solver: str = "cg",
        iterations: int = 20,
        ls_iterations: int = 20,
        mj_model_timestep: float = 0.002,
        mocap_hz: int = 50,
        clip_length: int = 250,
        random_init_range: int = 50,
        traj_length: int = 5,
        clip_lengths: list[int] = None,
        termination_joint_distance_threshold: float = 0.4,
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
            clip_lengths: list of lengths for each clip
            termination_joint_distance_threshold: Threshold for early termination based on joint distance.
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
            # Convert clip_lengths to JAX array to allow proper indexing with traced integers
            self._clip_lengths = (
                jax.numpy.array(clip_lengths) if clip_lengths is not None else None
            )
        else:
            print("No reference clip provided, in pure rendering mode.")

        self._termination_threshold = (
            termination_joint_distance_threshold  # STORE THE THRESHOLD
        )

    def _normalize_state_info(self, state: State) -> State:
        """Ensures that state.info has a consistent structure with all required keys using JAX-compatible operations."""
        info = state.info

        # Create a normalized info dictionary with all required keys
        # We use a JAX-compatible approach without conditionals or loops
        normalized_info = {}

        # Core required fields - always include these with appropriate defaults
        normalized_info["clip_idx"] = jp.array(0)
        normalized_info["clip_length"] = jp.array(150)
        normalized_info["start_frame"] = jp.array(0)
        normalized_info["joint_distance"] = jp.array(0.0)
        normalized_info["quat_distance"] = jp.array(0.0)
        normalized_info["energy_cost"] = jp.array(0.0)
        normalized_info["prev_ctrl"] = jp.zeros((self.sys.nu,))
        normalized_info["prev_prev_ctrl"] = jp.zeros((self.sys.nu,))
        normalized_info["prev_qacc"] = jp.zeros((self.sys.nv,))
        normalized_info["step"] = jp.array(0)
        normalized_info["reference_obs_size"] = jp.array(0)

        # Reward fields that might be present
        normalized_info["bodypos_reward"] = jp.array(0.0)
        normalized_info["endeff_reward"] = jp.array(0.0)
        normalized_info["reward_ctrlcost"] = jp.array(0.0)
        normalized_info["ctrl_diff_cost"] = jp.array(0.0)
        normalized_info["jerk_cost"] = jp.array(0.0)
        normalized_info["joint_reward"] = jp.array(0.0)

        # Override defaults with actual values where they exist
        # This uses JAX's functional updating through dictionary merging
        # First create the info dict with all keys needed
        # Then update with the original info to preserve any existing values
        return state.replace(info={**normalized_info, **info})

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
            "clip_length": self._clip_lengths[clip_idx],
            "quat_distance": 0.0,
            "joint_distance": 0.0,
            "prev_ctrl": jp.zeros((self.sys.nu,)),
            "prev_prev_ctrl": jp.zeros((self.sys.nu,)),
            "prev_qacc": jp.zeros((self.sys.nv,)),  # Added to preserve carry structure
            "step": jp.array(0),  # Step counter initialization
            "energy_cost": jp.array(0.0),  # Initialize energy_cost field
            "step": jp.array(0),
        }

        state = self.reset_from_clip(rng, info, noise=True)
        # Make sure both clip_length AND step are preserved in state.info
        state = state.replace(
            info={
                **state.info,
                "clip_length": info["clip_length"],
                "step": info["step"],
            }
        )
        # Normalize state structure
        state = self._normalize_state_info(state)
        return state

    def step(self, state: State, action: jp.ndarray) -> State:
        """
        Steps the environment forward by one timestep.

        Args:
            state (State): The current state of the environment.
            action (jp.ndarray): The action to take.

        Returns:
            State: The new state of the environment.
        """
        # Normalize state.info to ensure consistent structure
        state = self._normalize_state_info(state)

        # Increment the step counter
        state = state.replace(info={**state.info, "step": state.info["step"] + 1})

        # IMPORTANT: Clip actions to allowed range before applying
        clipped_action = jp.clip(action, -0.1, 0.1)

        # Call parent step method with normalized state
        state = super().step(state, clipped_action)

        # Renormalize after parent step to ensure structure consistency
        state = self._normalize_state_info(state)

        # Handle NaNs or all-zero observations
        condition = jp.logical_or(jp.any(jp.isnan(state.obs)), jp.all(state.obs == 0))
        state = jax.lax.cond(
            condition,
            lambda st: st.replace(done=jp.array(True, dtype=st.done.dtype)),
            lambda st: st,
            state,
        )

        # Early termination based on joint distance threshold
        state = jax.lax.cond(
            state.info["joint_distance"] > self._termination_threshold,
            lambda st: st.replace(done=jp.array(True, dtype=st.done.dtype)),
            lambda st: st,
            state,
        )

        # Early termination when reaching end of reference clip
        state = jax.lax.cond(
            state.info["step"] >= state.info["clip_length"],
            lambda st: st.replace(done=jp.array(True, dtype=st.done.dtype)),
            lambda st: st,
            state,
        )

        # Update info with previous controls
        new_info = {
            **state.info,
            "prev_prev_ctrl": state.info["prev_ctrl"],
            "prev_ctrl": clipped_action,  # Store clipped action
        }
        state = state.replace(info=new_info)

        # Final normalization to guarantee consistent structure
        state = self._normalize_state_info(state)

        return state

    def _get_reference_clip(self, info: dict[str, jp.ndarray]) -> ReferenceClip:
        """
        Retrieves the reference clip corresponding to the current clip index.

        Args:
            info: Dictionary containing clip information.

        Returns:
            ReferenceClip: The reference clip for the given index.
        """

        return jax.tree.map(lambda x: x[info["clip_idx"]], self._reference_clips)
