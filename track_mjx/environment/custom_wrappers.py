from typing import Callable, Dict, Optional

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
from flax import linen as nn


def wrap(
    env: Env,
    episode_length: int = 1000,
    action_repeat: int = 1,
    randomization_fn: Optional[Callable[[System], tuple[System, System]]] = None,
    use_lstm: bool = True,
    hidden_state_dim: int = 128,
) -> Wrapper:
    """Common wrapper pattern for all training agents.

    Args:
      env: environment to be wrapped
      episode_length: length of episode
      action_repeat: how many repeated actions to take per step
      randomization_fn: randomization function that produces a vectorized system
        and in_axes to vmap over
      use_lstm: boolean argument for using lstm
      hidden_state_dim: number of hidden states in lstm setting

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
    
    if use_lstm:
        env = LSTMAutoResetWrapperTracking(env, lstm_features=hidden_state_dim)
    else:
        env = AutoResetWrapperTracking(env)
        
    return env


class LSTMAutoResetWrapperTracking(Wrapper):
    """Automatically resets Brax envs that are done and tracks LSTM hidden states."""

    def __init__(self, env: Env, lstm_features: int = 128):
        """Initializes the wrapper with LSTM tracking."""
        super().__init__(env)
        self.lstm_features = lstm_features

    def initialize_hidden_state(self, rng: jax.Array, num_envs: int) -> tuple[jp.ndarray, jp.ndarray]:
        """Initializes LSTM hidden state (h_t, c_t) for each environment."""
        lstm_cell = nn.LSTMCell(features=self.lstm_features)
        return lstm_cell.initialize_carry(rng, (num_envs, self.lstm_features))  # shape: (num_envs, hidden_dim)

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment and reinitializes LSTM hidden states."""
        state = self.env.reset(rng)
        state.info["first_pipeline_state"] = state.pipeline_state
        state.info["first_obs"] = state.obs
        state.info["first_prev_ctrl"] = state.info["prev_ctrl"]

        # initialize hidden state per environment
        num_envs = state.obs.shape[0]
        hidden_state = self.initialize_hidden_state(jax.random.PRNGKey(0), num_envs)
        state.info["hidden_state"] = hidden_state
        
        print(f'DEBUG IN ENV: OBS SHAPE {state.obs.shape[0]}, HIDDEN SHAPE {hidden_state[1].shape}')
        
        return state

    def step(self, state: State, action: jax.Array) -> State:
        """Steps through the environment and resets hidden states for done envs."""
        
        if "steps" in state.info:
            steps = state.info["steps"]
            steps = jp.where(state.done, jp.zeros_like(steps), steps)
            state.info.update(steps=steps)

        state = state.replace(done=jp.zeros_like(state.done))
        state = self.env.step(state, action)

        def where_done(x, y):
            """Reinitialize hidden state where done == True."""
            done = state.done
            if done.shape:
                done = jp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))  # broadcasts over batch dim
            return jp.where(done, x, y)  # reset only where `done == True`

        pipeline_state = jax.tree.map(
            where_done, state.info["first_pipeline_state"], state.pipeline_state
        )
        obs = where_done(state.info["first_obs"], state.obs)
        state.info["prev_ctrl"] = where_done(
            state.info["first_prev_ctrl"],
            state.info["prev_ctrl"],
        )

        # reset LSTM hidden states for completed environments
        num_envs = state.obs.shape[0]
        new_hidden_state = self.initialize_hidden_state(jax.random.PRNGKey(0), num_envs)
        hidden_state = jax.tree_map(
            lambda x, y: where_done(x, y),
            new_hidden_state,
            state.info.get("hidden_state", new_hidden_state),  # default to new hidden if missing
        )

        state.info["hidden_state"] = hidden_state
        return state.replace(pipeline_state=pipeline_state, obs=obs)
    

# class RenderRolloutWrapperTracking(Wrapper):
#     """Always resets to the first frame of the clips for complete rollouts."""

#     def reset(self, rng: jax.Array, clip_idx: int | None = None) -> State:
#         """
#         Resets the environment to an initial state.

#         Args:
#             rng (jax.Array): Random key for reproducibility.
#             clip_idx (int | None, optional): clip index to reset to. if None, randomly choose the clip . Defaults to None.

#         Returns:
#             State: The initial state of the environment.
#         """
#         _, clip_rng, rng = jax.random.split(rng, 3)
#         if clip_idx is None:
#             clip_idx = jax.random.randint(clip_rng, (), 0, self._n_clips)  # type: ignore
#         info = {
#             "clip_idx": clip_idx,
#             "start_frame": 0,
#             "summed_pos_distance": 0.0,
#             "quat_distance": 0.0,
#             "joint_distance": 0.0,
#             "prev_ctrl": jp.zeros((self.sys.nu,)),
#         }

#         return self.reset_from_clip(rng, info)

class RenderRolloutWrapperTracking(Wrapper):
    """Always resets to the first frame of the clips for complete rollouts, with LSTM hidden states."""

    def reset(self, rng: jax.Array, clip_idx: int | None = None) -> State:
        """
        Resets the environment to an initial state.

        Args:
            rng (jax.Array): Random key for reproducibility.
            clip_idx (int | None, optional): clip index to reset to. If None, randomly choose a clip.

        Returns:
            State: The initial state of the environment.
        """
        _, clip_rng, rng = jax.random.split(rng, 3)

        if clip_idx is None:
            clip_idx = jax.random.randint(clip_rng, (), 0, self._n_clips)  # type: ignore

        info = {
            "clip_idx": clip_idx,
            "start_frame": 0,
            "summed_pos_distance": 0.0,
            "quat_distance": 0.0,
            "joint_distance": 0.0,
            "prev_ctrl": jp.zeros((self.sys.nu,)),
        }

        hidden_state = nn.LSTMCell(features=128).initialize_carry(jax.random.PRNGKey(0), ())
        info["hidden_state"] = hidden_state

        state = self.reset_from_clip(rng, info)
        return state


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
            "summed_pos_distance": 0.0,
            "quat_distance": 0.0,
            "joint_distance": 0.0,
            "prev_ctrl": jp.zeros((self.sys.nu,)),
        }

        return self.reset_from_clip(rng, info, noise=False)
