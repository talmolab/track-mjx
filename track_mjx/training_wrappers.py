"""Training wrappers whose bookkeeping is independent of PRNG representation."""

from typing import Any

import jax
import jax.numpy as jnp
from brax.envs.wrappers import training as brax_training
from mujoco_playground import wrapper as playground_wrapper

from track_mjx.imitation import is_sparse_imitation

_PRESERVE_INFO_KEY = "AutoResetWrapper_preserve_info"


class EpisodeWrapper(brax_training.EpisodeWrapper):
    """Tracks episode metrics using the environment state's batch shape."""

    def reset(self, rng: jax.Array):
        state = self.env.reset(rng)
        state.info["steps"] = jnp.zeros_like(state.done)
        state.info["truncation"] = jnp.zeros_like(state.done)
        state.info["episode_done"] = jnp.zeros_like(state.done)
        episode_metrics = {
            "sum_reward": jnp.zeros_like(state.reward),
            "length": jnp.zeros_like(state.done),
        }
        episode_metrics.update(
            {name: jnp.zeros_like(value) for name, value in state.metrics.items()}
        )
        state.info["episode_metrics"] = episode_metrics
        return state

    def step(self, state, action: jax.Array):
        state = super().step(state, action)
        # VNL's reference boundary is a time limit for PPO, just like the wrapper's.
        state.info["truncation"] = jnp.maximum(
            state.info["truncation"], state.info.get("truncated", 0)
        )
        preserved = state.info.get(_PRESERVE_INFO_KEY, {})
        if "truncation" in preserved:
            state.info[_PRESERVE_INFO_KEY] = {
                **preserved,
                "truncation": state.info["truncation"],
            }
        return state


class AutoResetWrapper(playground_wrapper.BraxAutoResetWrapper):
    """Auto-reset wrapper with per-environment typed-key bookkeeping."""

    def reset(self, rng: jax.Array):
        rng_key = jax.vmap(jax.random.split)(rng)
        next_rng, reset_key = rng_key[..., 0], rng_key[..., 1]
        state = self.env.reset(reset_key)
        state.info[f"{self._info_key}_first_data"] = state.data
        state.info[f"{self._info_key}_first_obs"] = state.obs
        state.info[f"{self._info_key}_rng"] = next_rng
        state.info[f"{self._info_key}_done_count"] = jnp.zeros_like(
            state.done, dtype=int
        )
        if self._full_reset:
            # Upstream's preservation hook keeps terminal flags across info resets.
            state.info[_PRESERVE_INFO_KEY] = {
                **state.info.get(_PRESERVE_INFO_KEY, {}),
                "truncation": state.info["truncation"],
            }
        return state

    def step(self, state, action: jax.Array):
        state = super().step(state, action)
        if self._full_reset:
            state.info["truncation"] = state.info[_PRESERVE_INFO_KEY]["truncation"]
        elif "reference_step" in state.info:
            # Cached auto-reset restores physics and observations but preserves info.
            info = state.info.copy()
            info["reference_step"] = jnp.where(
                state.done,
                jnp.zeros_like(info["reference_step"]),
                info["reference_step"],
            )
            state = state.replace(info=info)
        return state


def wrap_for_brax_training(
    env: Any,
    vision: bool = False,
    num_vision_envs: int = 1,
    episode_length: int = 1000,
    action_repeat: int = 1,
    randomization_fn=None,
    full_reset: bool = False,
):
    """Wrap an environment for Brax training using typed JAX keys."""
    # Sparse DP history belongs to one episode and must reset with its physics.
    full_reset = full_reset or is_sparse_imitation(env)
    if vision:
        env = playground_wrapper.MadronaWrapper(env, num_vision_envs, randomization_fn)
    elif randomization_fn is None:
        env = brax_training.VmapWrapper(env)
    else:
        env = playground_wrapper.BraxDomainRandomizationVmapWrapper(
            env, randomization_fn
        )
    env = EpisodeWrapper(env, episode_length, action_repeat)
    return AutoResetWrapper(env, full_reset=full_reset)
