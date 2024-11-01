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


# class AutoResetWrapperTracking(Wrapper):
#     """Automatically resets RodentMultiClipTracking envs that are done.

#     Each reset selects a new random clip_idx to ensure varied initial conditions.
#     """

#     def reset(self, rng: jax.Array) -> State:
#         """Resets the environment and initializes the 'first_' info."""
#         state = self.env.reset(rng)
#         # Save rng
#         state.info["reset_rng"] = rng
#         return state

#     def step(self, state: State, action: jax.Array) -> State:
#         if "steps" in state.info:
#             steps = state.info["steps"]
#             steps = jp.where(state.done, jp.zeros_like(steps), steps)
#             state.info.update(steps=steps)
#         state = state.replace(done=jp.zeros_like(state.done))
#         state = self.env.step(state, action)


#         def where_done(x, y):
#             done = state.done
#             if done.shape:
#                 done = jp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))  # type: ignore
#             return jp.where(done, x, y)

#         info = state.info
#         def f():
#             new_state = self.env.reset(state.info["reset_rng"])

#         return where_done(, state)


class RenderRolloutWrapperTracking(Wrapper):
    """Always resets to 0"""

    def reset(self, rng: jax.Array) -> State:
        _, clip_rng, rng = jax.random.split(rng, 3)

        clip_idx = jax.random.randint(clip_rng, (), 0, self._n_clips)
        info = {
            "clip_idx": clip_idx,
            "cur_frame": 0,
            "steps_taken_cur_frame": 0,
            "summed_pos_distance": 0.0,
            "quat_distance": 0.0,
            "joint_distance": 0.0,
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
        state.info["first_cur_frame"] = state.info["cur_frame"]
        state.info["first_steps_taken_cur_frame"] = state.info["steps_taken_cur_frame"]
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
        state.info["cur_frame"] = where_done(
            state.info["first_cur_frame"],
            state.info["cur_frame"],
        )
        state.info["steps_taken_cur_frame"] = where_done(
            state.info["first_steps_taken_cur_frame"],
            state.info["steps_taken_cur_frame"],
        )
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
            "cur_frame": 0,
            "steps_taken_cur_frame": 0,
            "summed_pos_distance": 0.0,
            "quat_distance": 0.0,
            "joint_distance": 0.0,
            "prev_ctrl": jp.zeros((self.sys.nu,)),
        }

        return self.reset_from_clip(rng, info, noise=False)


# Single clip
# class RenderRolloutWrapperTracking(Wrapper):
#     """Always resets to 0"""

#     def reset(self, rng: jax.Array) -> State:
#         info = {
#             "cur_frame": 0,
#             "steps_taken_cur_frame": 0,
#             "summed_pos_distance": 0.0,
#             "quat_distance": 0.0,
#             "joint_distance": 0.0,
#         }

#         return self.reset_from_clip(rng, info)
