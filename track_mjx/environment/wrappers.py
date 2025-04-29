from typing import Callable, Dict, Optional, Tuple

from brax.base import System
from brax.envs.base import Env, State, Wrapper
from brax.envs.wrappers.training import (
    EpisodeWrapper,
    VmapWrapper,
    DomainRandomizationVmapWrapper,
)
import jax
from jax import numpy as jp
from mujoco import mjx
from brax.v1.envs import env as brax_env
from mujoco.mjx._src import smooth


def wrap(
    env: Env,
    episode_length: int = 1000,
    action_repeat: int = 1,
    randomization_fn: Optional[Callable[[System], Tuple[System, System]]] = None,
) -> Wrapper:
    """Common wrapper pattern for all training agents.

    Args:
      env: environment to be wrapped
      episode_length: length of episode
      action_repeat: how many repeated actions to take per step
      randomization_fn: randomization function that produces a vectorized system
        and in_axes to vmap over

    Returns:
      An environment that is wrapped with Episode and AutoReset wrappers.  If the
      environment did not already have batch dimensions, it is additional Vmap
      wrapped.
    """
    env = EpisodeWrapper(env, episode_length, action_repeat)
    if randomization_fn is None:
        env = VmapWrapper(env)
    else:
        env = DomainRandomizationVmapWrapper(env, randomization_fn)
    env = AutoResetWrapperTracking(env)
    return env


# TODO: Rename these wrappers to be more concise/descriptive
class RenderRolloutVmapWrapper(brax_env.Wrapper):
    """Vectorizes Brax env given a clip index, used with RenderRolloutWrapperMulticlipTracking."""

    def __init__(self, env: Env, batch_size: Optional[int] = None):
        super().__init__(env)
        self.batch_size = batch_size

    def reset(self, rng: jax.Array, clip_idx: jax.Array = None) -> State:
        if self.batch_size is not None:
            rng = jax.random.split(rng, self.batch_size)
        if clip_idx is None:
            clip_idx = jnp.zeros((rng.shape[0],), dtype=jnp.int32)

        return jax.vmap(self.env.reset)(rng, clip_idx)

    def step(self, state: State, action: jax.Array) -> State:
        return jax.vmap(self.env.step)(state, action)


class RenderRolloutWrapperSingleclipTracking(Wrapper):
    """Always resets to the first frame of the clips for complete rollouts."""

    def reset(self, rng: jax.Array, start_frame: int = 0) -> State:
        """
        Resets the environment to an initial state.
        Args:
            rng (jax.Array): Random key for reproducibility.
            clip_idx (int | None, optional): clip index to reset to. if None, randomly choose the clip . Defaults to None.
        Returns:
            State: The initial state of the environment.
        """

        info = {
            "start_frame": start_frame,
            "prev_ctrl": jp.zeros((self.sys.nu,)),
        }

        return self.reset_from_clip(rng, info)


class RenderRolloutWrapperMulticlipTracking(Wrapper):
    """Always resets to the first frame of the clips for complete rollouts."""

    def reset(self, rng: jax.Array, clip_idx: int | None = None) -> State:
        """
        Resets the environment to an initial state.

        Args:
            rng (jax.Array): Random key for reproducibility.
            clip_idx (int | None, optional): clip index to reset to. if None, randomly choose the clip . Defaults to None.

        Returns:
            State: The initial state of the environment.
        """
        _, clip_rng, rng = jax.random.split(rng, 3)
        if clip_idx is None:
            clip_idx = jax.random.randint(clip_rng, (), 0, self._n_clips)  # type: ignore
        info = {
            "clip_idx": clip_idx,
            "start_frame": 0,
            "prev_ctrl": jp.zeros((self.sys.nu,)),
        }

        return self.reset_from_clip(rng, info)


# Single clip
class AutoResetWrapperTracking(Wrapper):
    """Automatically resets Brax envs that are done."""

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        state.info["first_pipeline_state"] = state.pipeline_state
        state.info["first_obs"] = state.obs
        state.info["first_prev_ctrl"] = state.info["prev_ctrl"]
        return state

    def step(self, state: State, action: jax.Array) -> State:
        if "steps" in state.info:
            steps = state.info["steps"]
            steps = jp.where(state.done, jp.zeros_like(steps), steps)
            state.info.update(steps=steps)
        state = state.replace(done=jp.zeros_like(state.done))
        state = self.env.step(state, action)

        def where_done(x, y):
            done = state.done
            if done.shape:
                done = jp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))  # type: ignore
            return jp.where(done, x, y)

        pipeline_state = jax.tree.map(
            where_done, state.info["first_pipeline_state"], state.pipeline_state
        )
        obs = where_done(state.info["first_obs"], state.obs)
        state.info["prev_ctrl"] = where_done(
            state.info["first_prev_ctrl"],
            state.info["prev_ctrl"],
        )
        return state.replace(pipeline_state=pipeline_state, obs=obs)


class EvalClipWrapperTracking(Wrapper):
    """Always resets to 0, at a specific clip"""

    def reset(self, rng: jax.Array, clip_idx=0) -> State:
        _, rng = jax.random.split(rng)

        info = {
            "clip_idx": clip_idx,
            "start_frame": 0,
            "prev_ctrl": jp.zeros((self.sys.nu,)),
        }

        return self.reset_from_clip(rng, info, noise=False)


class AutoAlignWrapperTracking(Wrapper):
    """When done (reset), align pose with reference trajectory.
    Only works with the multiclip tracking env after multiclipvmapwrapper.
    """

    def reset(self, rng: jax.Array, clip_idx: jax.Array = None) -> State:
        state = self.env.reset(rng, clip_idx)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        if "steps" in state.info:
            steps = state.info["steps"]
            steps = jp.where(state.done, jp.zeros_like(steps), steps)
            state.info.update(steps=steps)
        state = state.replace(done=jp.zeros_like(state.done))
        state = self.env.step(state, action)

        def where_done(x, y):
            done = state.done
            if done.shape:
                done = jp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))  # type: ignore
            return jp.where(done, x, y)

        new_qpos = jp.concatenate(
            (
                state.info["reference_frame"].position,
                state.info["reference_frame"].quaternion,
                state.info["reference_frame"].joints,
            ),
            axis=-1,
        )
        new_qvel = jp.concatenate(
            (
                state.info["reference_frame"].velocity,
                state.info["reference_frame"].angular_velocity,
                state.info["reference_frame"].joints_velocity,
            ),
            axis=-1,
        )
        aligned_pipeline_state = state.pipeline_state.replace(
            qpos=new_qpos, qvel=new_qvel
        )
        aligned_pipeline_state = jax.vmap(smooth.kinematics, in_axes=(None, 0))(
            self._mjx_model, aligned_pipeline_state
        )
        pipeline_state = jax.tree.map(
            where_done, aligned_pipeline_state, state.pipeline_state
        )

        reference_obs, proprioceptive_obs = jax.vmap(self._get_obs)(
            pipeline_state, state.info
        )
        obs = jp.concatenate([reference_obs, proprioceptive_obs], axis=-1)
        return state.replace(pipeline_state=pipeline_state, obs=obs)


class HighLevelWrapper(Wrapper):
    """Takes a decoder inference function and uses it to get the ctrl used in the sim step"""

    def __init__(self, env, decoder_inference_fn, reference_obs_size):
        self._decoder_inference_fn = decoder_inference_fn
        self._reference_obs_size = reference_obs_size
        super().__init__(env)

    def step(self, state: State, latents: jax.Array) -> State:
        obs = state.obs

        # TODO replace reference obs size
        action, _ = self._decoder_inference_fn(
            jp.concatenate(
                [latents, obs[..., self._reference_obs_size :]],
                axis=-1,
            ),
        )
        return self.env.step(state, action)
