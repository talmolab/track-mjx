"""Environment wrapper for MoSeq high-level decoder transfer training.

Wraps a MoSeqImitation environment to enable hierarchical RL where:
- The high-level policy outputs latent intentions (replacing the action space)
- A frozen pretrained decoder converts intentions + proprioception into low-level actions
- The policy observes only kpms_code + proprioception (no reference trajectory)
"""

from typing import Any

import jax
import jax.numpy as jnp
from brax import envs


class MoSeqHighLevelWrapper(envs.Wrapper):
    """Wrapper that routes latent intentions through a frozen decoder.

    The high-level policy sees only kpms_code and proprioception.
    Its "action" is a latent intention vector. The wrapper concatenates
    the intention with proprioception and passes it to the frozen decoder,
    which produces the actual motor commands.

    Attributes:
        _decoder_inference_fn: Frozen decoder callable.
        _intention_size: Dimension of latent intention space.
    """

    def __init__(
        self,
        env: envs.Env,
        decoder_inference_fn,
        intention_size: int,
    ):
        super().__init__(env)
        self._decoder_inference_fn = decoder_inference_fn
        self._intention_size = intention_size

        # Create dummy decoder extras for initial reset
        # (decoder hasn't run yet, but info dict needs consistent pytree)
        dummy_obs = jnp.zeros(intention_size + self._get_proprio_size(env))
        _, self._dummy_decoder_extras = decoder_inference_fn(dummy_obs)

    def _get_proprio_size(self, env: envs.Env) -> int:
        """Get proprioception size from a sample reset."""
        sample_state = env.reset(jax.random.PRNGKey(0))
        proprio = sample_state.obs["state"]["proprioception"]
        return int(jax.flatten_util.ravel_pytree(proprio)[0].shape[0])

    @property
    def action_size(self) -> int:
        return self._intention_size

    @property
    def observation_size(self) -> dict[str, dict[str, int]]:
        """Return obs sizes for just kpms_code and proprioception."""
        full_obs_size = self.env.observation_size
        state_sizes = full_obs_size["state"]
        return {
            "state": {
                "kpms_code": state_sizes["kpms_code"],
                "proprioception": state_sizes["proprioception"],
            }
        }

    @property
    def unwrapped(self):
        return self

    def _filter_obs(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Keep only kpms_code and proprioception for the policy."""
        state_obs = obs["state"]
        return {
            "state": {
                "kpms_code": state_obs["kpms_code"],
                "proprioception": state_obs["proprioception"],
            }
        }

    def reset(self, rng: jax.Array) -> envs.State:
        state = self.env.reset(rng)
        state.info["_full_obs"] = state.obs
        state.info["decoder_extras"] = self._dummy_decoder_extras
        return state.replace(obs=self._filter_obs(state.obs))

    def step(self, state: envs.State, action: jax.Array) -> envs.State:
        # action = latent intention from high-level policy
        # Get proprioception from stored full obs
        proprio = jnp.nan_to_num(
            jax.flatten_util.ravel_pytree(
                state.info["_full_obs"]["state"]["proprioception"]
            )[0]
        )
        decoder_input = jnp.concatenate([action, proprio], axis=-1)
        ctrl, extras = self._decoder_inference_fn(decoder_input)

        next_state = self.env.step(state, ctrl)
        next_state.info["decoder_extras"] = extras
        next_state.info["_full_obs"] = next_state.obs
        return next_state.replace(obs=self._filter_obs(next_state.obs))
