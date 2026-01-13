"""Environment wrappers for bowl escape task transfer training.

Two wrapper classes for different transfer learning modes:
- DecoderHighLevelWrapper: Policy outputs latent → frozen decoder → ctrl
- PriorDecoderHighLevelWrapper: Policy outputs residual → add to prior mean → decoder → ctrl
"""

from typing import Any, Callable, Mapping

import jax
import jax.numpy as jp
from mujoco_playground import wrapper
from mujoco_playground._src import mjx_env


class DecoderHighLevelWrapper(wrapper.Wrapper):
    """Mode 1: Policy outputs latent intentions → frozen decoder → ctrl.

    This wrapper translates high-level latent intentions into low-level
    control signals using a frozen decoder from a pretrained checkpoint.

    The policy learns to output latent vectors that the decoder converts
    to motor commands.
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
            env: Base environment (should be FlattenObsWrapper wrapped).
            decoder_inference_fn: Function (latent_proprio) -> (action, extras)
                where latent_proprio is [latent, proprio] concatenated.
            latent_size: Dimension of the latent intention space.
            proprio_size: Size of proprioceptive observations (last N elements of obs).
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
        state.info["decoder_extras"] = {}
        return state

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        """Take a step using latent intention as action.

        Args:
            state: Current environment state.
            action: Latent intention vector from the policy.

        Returns:
            Next environment state after applying decoded control.
        """
        # Extract proprioception from observation (last proprio_size elements)
        proprio = state.obs[..., -self._proprio_size :]

        # Concatenate latent and proprio for decoder
        latent_proprio = jp.concatenate([action, proprio], axis=-1)

        # Apply frozen decoder to get control signal
        ctrl, extras = self._decoder_fn(latent_proprio)

        # Store decoder extras in state info
        state.info["decoder_extras"] = extras

        # Step the base environment with decoded control
        return super().step(state, ctrl)

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
            env: Base environment (should be FlattenObsWrapper wrapped).
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
        state.info["decoder_extras"] = {}
        state.info["prior_mean"] = jp.zeros(self._latent_size)
        state.info["prior_logvar"] = jp.zeros(self._latent_size)
        state.info["rng"] = rng
        return state

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        """Take a step using residual + prior mean as latent.

        Args:
            state: Current environment state.
            action: Residual vector from the policy.

        Returns:
            Next environment state after applying decoded control.
        """
        # Extract proprioception from observation
        proprio = state.obs[..., -self._proprio_size :]

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
        return super().step(state, ctrl)

    @property
    def action_size(self) -> int:
        """Return the latent size as the action size for the policy."""
        return self._latent_size
