"""Environment wrappers for bowl escape task transfer training.

Two wrapper classes for different transfer learning modes:
- DecoderHighLevelWrapper: Policy outputs latent → frozen decoder → ctrl
- PriorDecoderHighLevelWrapper: Policy outputs residual → add to prior mean → decoder → ctrl

Supports both legacy flat observations and new dict observations format.
"""

from typing import Any, Callable, Mapping

import jax
import jax.numpy as jp
from mujoco_playground import wrapper
from mujoco_playground._src import mjx_env

from track_mjx.agent.task_transfer.bowl_escape.observation_utils import (
    concat_flat_dict_obs,
    flatten_obs_dict,
)


def _get_proprio(obs: jax.Array | Mapping[str, Any], proprio_size: int) -> jax.Array:
    """Extract proprioception from observation, handling both dict and flat formats.

    Args:
        obs: Either a flat array (legacy) or dict with 'proprioception' key (new).
        proprio_size: Size of proprioceptive observations (used for flat format).

    Returns:
        Flattened proprioceptive observation array.
    """
    if isinstance(obs, Mapping):
        # New dict format - use flatten_obs_dict to handle nested structures
        flat_obs = flatten_obs_dict(obs)
        return flat_obs["proprioception"]
    else:
        # Legacy flat format - proprio is last proprio_size elements
        return obs[..., -proprio_size:]


def _flatten_obs_for_policy(obs: jax.Array | Mapping[str, Any]) -> jax.Array:
    """Flatten observations for the high-level policy network.

    Args:
        obs: Either a flat array (legacy) or dict with observation keys (new).

    Returns:
        Flat observation array suitable for the PPO policy network.
    """
    if isinstance(obs, Mapping):
        # New dict format - flatten and concatenate
        flat_obs = flatten_obs_dict(obs)
        return concat_flat_dict_obs(flat_obs)
    else:
        # Legacy flat format - already flat
        return obs


class DecoderHighLevelWrapper(wrapper.Wrapper):
    """Mode 1: Policy outputs latent intentions → frozen decoder → ctrl.

    This wrapper translates high-level latent intentions into low-level
    control signals using a frozen decoder from a pretrained checkpoint.

    The policy learns to output latent vectors that the decoder converts
    to motor commands.

    Supports both legacy flat observations and new dict observations format.
    """

    def __init__(
        self,
        env: mjx_env.MjxEnv,
        decoder_inference_fn: Callable,
        latent_size: int,
        proprio_size: int,
    ):
        """Initialize the decoder wrapper.

        Args:
            env: Base environment (with dict or flat observations).
            decoder_inference_fn: Function (latent_proprio) -> (action, extras)
                where latent_proprio is [latent, proprio] concatenated.
            latent_size: Dimension of the latent intention space.
            proprio_size: Size of proprioceptive observations.
        """
        self._decoder_fn = decoder_inference_fn
        self._latent_size = latent_size
        self._proprio_size = proprio_size
        super().__init__(env)

    def reset(
        self,
        rng: jax.Array,
        **kwargs: Any,
    ) -> mjx_env.State:
        """Reset the environment.

        Args:
            rng: Random key for environment reset.
            **kwargs: Additional arguments passed to base env reset.

        Returns:
            Initial environment state with decoder extras initialized.
        """
        state = self.env.reset(rng, **kwargs)

        # Initialize decoder_extras with correct pytree structure by running
        # decoder once with dummy latent. This ensures lax.scan carry structure
        # matches between reset and step.
        proprio = _get_proprio(state.obs, self._proprio_size)
        dummy_latent = jp.zeros(proprio.shape[:-1] + (self._latent_size,))
        latent_proprio = jp.concatenate([dummy_latent, proprio], axis=-1)
        _, extras = self._decoder_fn(latent_proprio)
        state.info["decoder_extras"] = extras

        # Flatten observations for the high-level policy network
        state = state.replace(obs=_flatten_obs_for_policy(state.obs))

        return state

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        """Take a step using latent intention as action.

        Args:
            state: Current environment state.
            action: Latent intention vector from the policy.

        Returns:
            Next environment state after applying decoded control.
        """
        # Extract proprioception from observation (handles both dict and flat)
        proprio = _get_proprio(state.obs, self._proprio_size)

        # Concatenate latent and proprio for decoder
        latent_proprio = jp.concatenate([action, proprio], axis=-1)

        # Apply frozen decoder to get control signal
        ctrl, extras = self._decoder_fn(latent_proprio)

        # Store decoder extras in state info
        state.info["decoder_extras"] = extras

        # Step the base environment with decoded control
        next_state = super().step(state, ctrl)

        # Flatten observations for the high-level policy network
        next_state = next_state.replace(obs=_flatten_obs_for_policy(next_state.obs))

        return next_state

    @property
    def action_size(self) -> int:
        """Return the latent size as the action size for the policy."""
        return self._latent_size


class PriorDecoderHighLevelWrapper(wrapper.Wrapper):
    """Mode 2: Policy outputs residual → add to prior mean → decoder → ctrl.

    This wrapper combines a learned residual policy with a frozen prior network.
    The final latent is: residual + prior_mean + optional_noise

    This allows the policy to learn task-specific corrections while leveraging
    the pretrained prior's knowledge of natural movements.

    Supports both legacy flat observations and new dict observations format.
    """

    def __init__(
        self,
        env: mjx_env.MjxEnv,
        prior_inference_fn: Callable,
        decoder_inference_fn: Callable,
        latent_size: int,
        proprio_size: int,
        deterministic_prior: bool = True,
        noise_logvar: float = -2.0,
    ):
        """Initialize the prior+decoder wrapper.

        Args:
            env: Base environment (with dict or flat observations).
            prior_inference_fn: Function (proprio) -> (mean, logvar)
            decoder_inference_fn: Function (latent_proprio) -> (action, extras)
            latent_size: Dimension of the latent intention space.
            proprio_size: Size of proprioceptive observations.
            deterministic_prior: If True, use prior mean only (no noise).
            noise_logvar: Fixed log-variance for noise sampling (used when
                deterministic_prior=False).
        """
        self._prior_fn = prior_inference_fn
        self._decoder_fn = decoder_inference_fn
        self._latent_size = latent_size
        self._proprio_size = proprio_size
        self._deterministic = deterministic_prior
        self._noise_logvar = noise_logvar
        super().__init__(env)

    def reset(
        self,
        rng: jax.Array,
        **kwargs: Any,
    ) -> mjx_env.State:
        """Reset the environment.

        Args:
            rng: Random key for environment reset.
            **kwargs: Additional arguments passed to base env reset.

        Returns:
            Initial environment state with prior/decoder extras initialized.
        """
        state = self.env.reset(rng, **kwargs)

        # Extract proprioception (handles both dict and flat formats)
        proprio = _get_proprio(state.obs, self._proprio_size)

        # Initialize prior_mean/prior_logvar with correct shape by running prior
        prior_mean, prior_logvar = self._prior_fn(proprio)
        state.info["prior_mean"] = prior_mean
        state.info["prior_logvar"] = prior_logvar

        # Initialize decoder_extras with correct pytree structure
        dummy_latent = jp.zeros_like(prior_mean)
        latent_proprio = jp.concatenate([dummy_latent, proprio], axis=-1)
        _, decoder_extras = self._decoder_fn(latent_proprio)
        state.info["decoder_extras"] = decoder_extras

        # Initialize final_latent with correct shape
        state.info["final_latent"] = jp.zeros_like(prior_mean)
        state.info["rng"] = rng

        # Flatten observations for the high-level policy network
        state = state.replace(obs=_flatten_obs_for_policy(state.obs))

        return state

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        """Take a step using residual + prior mean as latent.

        Args:
            state: Current environment state.
            action: Residual vector from the policy.

        Returns:
            Next environment state after applying decoded control.
        """
        # Extract proprioception from observation (handles both dict and flat)
        proprio = _get_proprio(state.obs, self._proprio_size)

        # Get prior distribution from proprioception
        prior_mean, prior_logvar = self._prior_fn(proprio)

        # Compute final latent
        if self._deterministic:
            latent = action + prior_mean
        else:
            # Sample noise with fixed logvar
            rng = state.info.get("rng", jax.random.PRNGKey(0))
            rng, noise_rng = jax.random.split(rng)
            std = jp.exp(0.5 * self._noise_logvar)
            noise = jax.random.normal(noise_rng, shape=prior_mean.shape) * std
            latent = action + prior_mean + noise
            state.info["rng"] = rng

        # Concatenate latent and proprio for decoder
        latent_proprio = jp.concatenate([latent, proprio], axis=-1)

        # Apply frozen decoder
        ctrl, decoder_extras = self._decoder_fn(latent_proprio)

        # Store extras in state info
        state.info["decoder_extras"] = decoder_extras
        state.info["prior_mean"] = prior_mean
        state.info["prior_logvar"] = prior_logvar
        state.info["final_latent"] = latent

        # Step the base environment
        next_state = super().step(state, ctrl)

        # Flatten observations for the high-level policy network
        next_state = next_state.replace(obs=_flatten_obs_for_policy(next_state.obs))

        return next_state

    @property
    def action_size(self) -> int:
        """Return the latent size as the action size for the policy."""
        return self._latent_size
