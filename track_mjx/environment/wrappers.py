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
from brax.v1.envs import env as brax_env
from mujoco.mjx._src import smooth
from flax import linen as nn


def wrap(
    env: Env,
    episode_length: int = 1000,
    action_repeat: int = 1,
    randomization_fn: Optional[Callable[[System], tuple[System, System]]] = None,
    use_lstm: bool = True,
    hidden_state_dim: int = 128,
    hidden_layer_num: int = 2,
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
        env = LSTMAutoResetWrapperTracking(
            env, lstm_features=hidden_state_dim, hidden_layer_num=hidden_layer_num
        )
    else:
        env = AutoResetWrapperTracking(env)

    return env


class LSTMAutoResetWrapperTracking(Wrapper):
    """Automatically resets Brax envs that are done and tracks LSTM hidden states. Goes after vmap"""

    def __init__(self, env: Env, lstm_features: int = 128, hidden_layer_num: int = 2):
        """Initializes the wrapper with LSTM tracking."""
        super().__init__(env)
        self.lstm_features = lstm_features
        self.hidden_layer_num = hidden_layer_num

    def initialize_hidden_state(
        self, rng: jax.Array, num_envs: int
    ) -> tuple[jp.ndarray, jp.ndarray]:
        """Initializes LSTM hidden state (h_t, c_t) for each environment."""

        lstm_cell = nn.LSTMCell(features=self.lstm_features)

        def init_single_env(_rng):
            def init_single_layer(layer_rng):
                return lstm_cell.initialize_carry(layer_rng, ())

            layer_rngs = jax.random.split(_rng, self.hidden_layer_num)
            c_list, h_list = zip(*[init_single_layer(r) for r in layer_rngs])
            return jp.stack(h_list), jp.stack(c_list)  # [num_layers, hidden_dim]

        env_rngs = jax.random.split(rng, num_envs)
        h_stack, c_stack = jax.vmap(init_single_env)(
            env_rngs
        )  # [num_envs, num_layers, hidden_dim]

        return h_stack, c_stack

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
                done = jp.reshape(
                    done, [x.shape[0]] + [1] * (len(x.shape) - 1)
                )  # broadcasts over batch dim
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
        # num_envs = state.obs.shape[0]
        # new_hidden_state = self.initialize_hidden_state(jax.random.PRNGKey(0), num_envs)
        # hidden_state = jax.tree_map(
        #     lambda x, y: where_done(x, y),
        #     new_hidden_state,
        #     state.info.get("hidden_state", new_hidden_state),  # default to new hidden if missing
        # )

        # state.info["hidden_state"] = hidden_state

        return state.replace(pipeline_state=pipeline_state, obs=obs)


class RenderRolloutWrapperTrackingLSTM(Wrapper):
    """Always resets to the first frame of the clips for complete rollouts."""
    
    def __init__(self, env, lstm_features: int = 128, hidden_layer_num: int =2):
        """Initialize the wrapper.
        
        Args:
            env: The environment to wrap.
            lstm_features (int): The size of the LSTM hidden state.
        """
        super().__init__(env)
        self.lstm_features = lstm_features
        self.hidden_layer_num = hidden_layer_num
    
    def initialize_hidden_state(self, rng: jax.Array) -> tuple[jp.ndarray, jp.ndarray]:
        """Initializes LSTM hidden states for the environment."""
        lstm_cell = nn.LSTMCell(features=self.lstm_features)
        # return lstm_cell.initialize_carry(rng, ())
    
        def init_single_env(_rng):
            def init_single_layer(layer_rng):
                return lstm_cell.initialize_carry(layer_rng, ())
            layer_rngs = jax.random.split(_rng, self.hidden_layer_num)
            c_list, h_list = zip(*[init_single_layer(r) for r in layer_rngs])
            return jp.stack(h_list), jp.stack(c_list)  # [num_layers, hidden_dim]
        
        num_envs = 1
        env_rngs = jax.random.split(rng, num_envs)
        h_stack, c_stack = jax.vmap(init_single_env)(env_rngs)

        print(f'[DEBUG] In env_wrapper, the rendering initialized hidden shape: {h_stack.shape}')
        return h_stack, c_stack
        

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
            "summed_pos_distance": 0.0,
            "quat_distance": 0.0,
            "joint_distance": 0.0,
            "prev_ctrl": jp.zeros((self.sys.nu,)),
        }
        
        hidden_state = self.initialize_hidden_state(jax.random.PRNGKey(0))
        info["hidden_state"] = hidden_state

        return self.reset_from_clip(rng, info)
    

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
