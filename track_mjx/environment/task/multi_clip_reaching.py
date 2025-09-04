import jax
from jax import numpy as jp

from brax.envs.base import State
from typing import Any

from track_mjx.io.load import ReferenceClipReach
from track_mjx.environment.reacher.base import BaseReacher
from track_mjx.environment.task.single_clip_reaching import SingleClipReaching
from track_mjx.environment.task.reward import RewardConfigReach


class MultiClipReaching(SingleClipReaching):
    """Overrides reset function and reference clip retrieval to handle multiple clips."""

    def __init__(
        self,
        reference_clip: ReferenceClipReach | None,
        reacher: BaseReacher,
        reward_config: RewardConfigReach | None,
        physics_steps_per_control_step: int,
        reset_noise_scale: float,
        solver: str = "cg",
        iterations: int = 30,
        ls_iterations: int = 30,
        mj_model_timestep: float = 0.001,
        mocap_hz: int = 200,
        clip_length: int = 100,
        random_init_range: int = 5,
        traj_length: int = 5,
        use_emg: bool = False,  # Add this parameter
        **kwargs: Any,
    ):
        # Pass use_emg to the parent constructor
        super().__init__(
            None,
            reacher,
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
            use_emg=use_emg,  # Pass this parameter
            **kwargs,
        )
        if reference_clip is not None:
            self._reference_clips = reference_clip
            self._n_clips = reference_clip.joints.shape[0]  # Changed from position to joints
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
        _, start_rng, clip_rng = jax.random.split(rng, 3)

        start_frame = jax.random.randint(start_rng, (), 0, 44)
        if clip_idx is None:
            clip_idx = jax.random.randint(clip_rng, (), 0, self._n_clips)  # type: ignore
        info = {
            "clip_idx": clip_idx,
            "start_frame": start_frame,
            "prev_ctrl": jp.zeros((self.sys.nu,)),
        }

        return self.reset_from_clip(rng, info, noise=True)

    def _get_reference_clip(self, info: dict[str, jp.ndarray]) -> ReferenceClipReach:
        """
        Retrieves the reference clip corresponding to the current clip index.

        Args:
            info: Dictionary containing clip information.

        Returns:
            ReferenceClip: The reference clip for the given index.
        """

        return jax.tree.map(lambda x: x[info["clip_idx"]], self._reference_clips)
