import jax
from jax import numpy as jp

from brax.envs.base import State
from typing import Any

from track_mjx.io.load import ReferenceClip
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
            self._n_clips = reference_clip.position.shape[0]
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
