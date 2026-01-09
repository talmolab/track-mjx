"""
Student network module for distillation training.

Observations are expected as dictionaries with keys:
- "imitation_target": Reference trajectory observations (flat array)
- "proprioception": Proprioceptive state observations (flat array)
"""

from collections.abc import Mapping
from typing import Sequence, Tuple, Union

from brax.training import networks
from brax.training import types
from brax.training.types import PRNGKey

import jax
import jax.numpy as jnp
from jax import random

from flax import linen as nn

from track_mjx.agent.observation_utils import (
    normalize_dict_obs,
    flatten_obs_dict,
    concat_flat_dict_obs,
)


class Encoder(nn.Module):
    """outputs in the form of distributions in latent space"""

    layer_sizes: Sequence[int]
    latents: int  # intention size
    activation: networks.ActivationFn = nn.silu
    kernel_init: networks.Initializer = jax.nn.initializers.lecun_uniform()
    bias: bool = True
    expansion_factor: int = 1  # PULSE uses 5x expansion before mean/logvar heads

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, get_activation: bool = False
    ) -> Union[
        Tuple[jnp.ndarray, jnp.ndarray], Tuple[Tuple[jnp.ndarray, jnp.ndarray], dict]
    ]:
        activations = {}
        # For each layer in the sequence
        for i, hidden_size in enumerate(self.layer_sizes):
            x = nn.Dense(
                hidden_size,
                name=f"hidden_{i}",
                kernel_init=self.kernel_init,
                use_bias=self.bias,
            )(x)
            x = self.activation(x)
            x = nn.LayerNorm()(x)
            if get_activation:
                activations[f"layer_{i}"] = x

        # Expansion layer before mean/logvar heads (PULSE-style)
        if self.expansion_factor > 1:
            expansion_dim = self.latents * self.expansion_factor
            x = nn.Dense(
                expansion_dim,
                name="expansion",
                kernel_init=self.kernel_init,
                use_bias=self.bias,
            )(x)
            x = self.activation(x)
            x = nn.LayerNorm()(x)
            if get_activation:
                activations["expansion"] = x

        mean_x = nn.Dense(self.latents, name="fc2_mean")(x)
        logvar_x = nn.Dense(self.latents, name="fc2_logvar")(x)

        if get_activation:
            activations["mean"] = mean_x
            activations["logvar"] = logvar_x
            return (mean_x, logvar_x), activations
        return mean_x, logvar_x


class Decoder(nn.Module):
    """decode with action output"""

    layer_sizes: Sequence[int]
    activation: networks.ActivationFn = nn.silu
    kernel_init: networks.Initializer = jax.nn.initializers.lecun_uniform()
    activate_final: bool = False
    bias: bool = True

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, get_activation: bool = False
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, dict]]:
        activations = {}
        for i, hidden_size in enumerate(self.layer_sizes):
            x = nn.Dense(
                hidden_size,
                name=f"hidden_{i}",
                kernel_init=self.kernel_init,
                use_bias=self.bias,
            )(x)
            if i != len(self.layer_sizes) - 1 or self.activate_final:
                x = self.activation(x)
                x = nn.LayerNorm()(x)
                if get_activation:
                    activations[f"layer_{i}"] = x
        if get_activation:
            return x, activations
        return x, {}


def reparameterize(rng, mean, logvar):
    std = jnp.exp(0.5 * logvar)
    eps = random.normal(rng, logvar.shape)
    return mean + eps * std


class Prior(nn.Module):
    """Prior network that outputs distributions in latent space from proprioceptive observations"""

    layer_sizes: Sequence[int]
    latents: int  # intention size
    activation: networks.ActivationFn = nn.silu
    kernel_init: networks.Initializer = jax.nn.initializers.lecun_uniform()
    bias: bool = True

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, get_activation: bool = False
    ) -> Union[
        Tuple[jnp.ndarray, jnp.ndarray], Tuple[Tuple[jnp.ndarray, jnp.ndarray], dict]
    ]:
        activations = {}
        # For each layer in the sequence
        for i, hidden_size in enumerate(self.layer_sizes):
            x = nn.Dense(
                hidden_size,
                name=f"hidden_{i}",
                kernel_init=self.kernel_init,
                use_bias=self.bias,
            )(x)
            x = self.activation(x)
            x = nn.LayerNorm()(x)
            if get_activation:
                activations[f"layer_{i}"] = x

        mean_x = nn.Dense(self.latents, name="fc2_mean")(x)
        logvar_x = nn.Dense(self.latents, name="fc2_logvar")(x)

        if get_activation:
            activations["mean"] = mean_x
            activations["logvar"] = logvar_x
            return (mean_x, logvar_x), activations
        return mean_x, logvar_x


class StudentNetwork(nn.Module):
    """Full VAE model with prior, encoder, and decoder.

    Now accepts dict observations with "imitation_target" and "proprioception" keys.
    """

    encoder_layers: Sequence[int]
    decoder_layers: Sequence[int]
    prior_layers: Sequence[int]
    latents: int = 60
    # Log-variance clamping (PULSE uses min=-5, max=2)
    encoder_logvar_min: float | None = None
    encoder_logvar_max: float | None = None
    prior_logvar_min: float | None = None
    prior_logvar_max: float | None = None
    # Encoder expansion factor (PULSE uses 5x expansion before mean/logvar heads)
    encoder_expansion_factor: int = 1

    def setup(self):
        self.encoder = Encoder(
            layer_sizes=self.encoder_layers,
            latents=self.latents,
            expansion_factor=self.encoder_expansion_factor,
        )
        self.decoder = Decoder(layer_sizes=self.decoder_layers)
        self.prior = Prior(layer_sizes=self.prior_layers, latents=self.latents)

    def _clamp_encoder_logvar(self, logvar: jnp.ndarray) -> jnp.ndarray:
        """Apply clamping to encoder log-variance if bounds are set."""
        if self.encoder_logvar_min is not None or self.encoder_logvar_max is not None:
            return jnp.clip(
                logvar, a_min=self.encoder_logvar_min, a_max=self.encoder_logvar_max
            )
        return logvar

    def _clamp_prior_logvar(self, logvar: jnp.ndarray) -> jnp.ndarray:
        """Apply clamping to prior log-variance if bounds are set."""
        if self.prior_logvar_min is not None or self.prior_logvar_max is not None:
            return jnp.clip(
                logvar, a_min=self.prior_logvar_min, a_max=self.prior_logvar_max
            )
        return logvar

    def __call__(
        self,
        obs: Mapping[str, jnp.ndarray],
        key,
        deterministic: bool = False,
        get_activation: bool = False,
    ):
        """Apply student network.

        Args:
            obs: Dict with "imitation_target" and "proprioception" keys.
            key: Random key for sampling.
            deterministic: If True, use mean of latent distribution.
            get_activation: If True, return activations.

        Returns:
            Tuple of (action, latent_mean, latent_logvar, prior_mean, prior_logvar)
            or with activations dict if get_activation=True.
        """
        _, encoder_rng = jax.random.split(key)
        # Access observations by key
        traj = obs["imitation_target"]
        egocentric_obs = obs["proprioception"]

        if get_activation:
            # Concatenate proprioceptive observations with trajectory for encoder
            encoder_input = jnp.concatenate([traj, egocentric_obs], axis=-1)
            (latent_mean, latent_logvar), encoder_activations = self.encoder(
                encoder_input, get_activation=True
            )
            # Apply encoder logvar clamping (PULSE-style)
            latent_logvar = self._clamp_encoder_logvar(latent_logvar)

            # Prior takes only proprioceptive observations
            (prior_mean, prior_logvar), prior_activations = self.prior(
                egocentric_obs, get_activation=True
            )
            # Apply prior logvar clamping (PULSE-style)
            prior_logvar = self._clamp_prior_logvar(prior_logvar)

            # Uses mean in the case of deterministic evaluation
            if deterministic:
                z = latent_mean
            else:
                z = reparameterize(encoder_rng, latent_mean, latent_logvar)

            concatenated = jnp.concatenate([z, egocentric_obs], axis=-1)
            action, decoder_activations = self.decoder(
                concatenated, get_activation=True
            )
            return (
                action,
                latent_mean,
                latent_logvar,
                prior_mean,
                prior_logvar,
                {
                    "encoder": encoder_activations,
                    "decoder": decoder_activations,
                    "prior": prior_activations,
                    "egocentric_obs": egocentric_obs,
                    "traj_obs": traj,
                    "intention": z,
                    "prior_mean": prior_mean,
                    "prior_logvar": prior_logvar,
                },
            )
        else:
            # Concatenate proprioceptive observations with trajectory for encoder
            encoder_input = jnp.concatenate([traj, egocentric_obs], axis=-1)
            latent_mean, latent_logvar = self.encoder(
                encoder_input, get_activation=False
            )
            # Apply encoder logvar clamping (PULSE-style)
            latent_logvar = self._clamp_encoder_logvar(latent_logvar)

            # Prior takes only proprioceptive observations
            prior_mean, prior_logvar = self.prior(egocentric_obs, get_activation=False)
            # Apply prior logvar clamping (PULSE-style)
            prior_logvar = self._clamp_prior_logvar(prior_logvar)

            # Uses mean in the case of deterministic evaluation
            if deterministic:
                z = latent_mean
            else:
                z = reparameterize(encoder_rng, latent_mean, latent_logvar)

            action, _ = self.decoder(jnp.concatenate([z, egocentric_obs], axis=-1))
            return action, latent_mean, latent_logvar, prior_mean, prior_logvar


def make_student_policy(
    action_param_size: int,
    latent_size: int,
    obs_sizes: Mapping[str, int],
    preprocess_observations_fn=None,
    encoder_hidden_layer_sizes: Sequence[int] = (1024, 1024),
    decoder_hidden_layer_sizes: Sequence[int] = (1024, 1024),
    prior_hidden_layer_sizes: Sequence[int] = (1024, 1024),
    encoder_logvar_min: float | None = None,
    encoder_logvar_max: float | None = None,
    prior_logvar_min: float | None = None,
    prior_logvar_max: float | None = None,
    encoder_expansion_factor: int = 1,
) -> networks.FeedForwardNetwork:
    """
    Create a policy network with student module including prior, encoder, and decoder.

    Args:
        action_param_size (int): the parameter size of the action space,
            usually double of the action size to model both the mean and
            variance of the action distribution
        latent_size (int): the size of the latent space
        obs_sizes (Mapping[str, int]): dict with "imitation_target" and
            "proprioception" sizes
        preprocess_observations_fn: function to preprocess dict observations.
            Should accept (obs_dict, normalizer_params) and return normalized obs_dict.
        encoder_hidden_layer_sizes (Sequence[int], optional): sizes of encoder
            hidden layers. Defaults to (1024, 1024).
        decoder_hidden_layer_sizes (Sequence[int], optional): sizes of decoder
            hidden layers. Defaults to (1024, 1024).
        prior_hidden_layer_sizes (Sequence[int], optional): sizes of prior
            hidden layers. Defaults to (1024, 1024).
        encoder_logvar_min (float | None, optional): min clamp for encoder
            log-variance. Defaults to None (no clamping).
        encoder_logvar_max (float | None, optional): max clamp for encoder
            log-variance. Defaults to None (no clamping).
        prior_logvar_min (float | None, optional): min clamp for prior
            log-variance. Defaults to None (no clamping).
        prior_logvar_max (float | None, optional): max clamp for prior
            log-variance. Defaults to None (no clamping).
        encoder_expansion_factor (int, optional): expansion factor for encoder
            before mean/logvar heads (PULSE uses 5). Defaults to 1 (no expansion).

    Returns:
        networks.FeedForwardNetwork: the created policy network
    """
    # Default preprocessor just flattens and returns as-is
    if preprocess_observations_fn is None:

        def preprocess_observations_fn(obs, processor_params):
            flat_obs = flatten_obs_dict(obs)
            return normalize_dict_obs(flat_obs, processor_params)

    policy_module = StudentNetwork(
        encoder_layers=list(encoder_hidden_layer_sizes),
        decoder_layers=list(decoder_hidden_layer_sizes)
        + [action_param_size],  # add action size to the last layer
        prior_layers=list(prior_hidden_layer_sizes),
        latents=latent_size,
        encoder_logvar_min=encoder_logvar_min,
        encoder_logvar_max=encoder_logvar_max,
        prior_logvar_min=prior_logvar_min,
        prior_logvar_max=prior_logvar_max,
        encoder_expansion_factor=encoder_expansion_factor,
    )

    def apply(
        processor_params,
        policy_params,
        obs,
        key,
        deterministic: bool = False,
        get_activation: bool = False,
    ):
        """Applies the policy network with observation normalizer.

        The output is the action distribution parameters.
        """
        obs = preprocess_observations_fn(obs, processor_params)
        return policy_module.apply(
            policy_params,
            obs=obs,
            key=key,
            deterministic=deterministic,
            get_activation=get_activation,
        )

    # Create dummy dict observations for initialization
    dummy_obs = {
        "imitation_target": jnp.zeros((1, obs_sizes["imitation_target"])),
        "proprioception": jnp.zeros((1, obs_sizes["proprioception"])),
    }
    dummy_key = jax.random.PRNGKey(0)

    return networks.FeedForwardNetwork(
        init=lambda key: policy_module.init(key, dummy_obs, dummy_key),
        apply=apply,
    )
