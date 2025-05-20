from typing import Sequence, Tuple, Union

from brax.training import networks
from brax.training import types
from brax.training.types import PRNGKey

import jax
import jax.numpy as jnp
from jax import random

from flax import linen as nn


class Encoder(nn.Module):
    """outputs in the form of distributions in latent space"""

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


class IntentionNetwork(nn.Module):
    """Full VAE model, encode -> decode with sampled actions"""

    encoder_layers: Sequence[int]
    decoder_layers: Sequence[int]
    reference_obs_size: int
    latents: int = 60

    def setup(self):
        self.encoder = Encoder(layer_sizes=self.encoder_layers, latents=self.latents)
        self.decoder = Decoder(layer_sizes=self.decoder_layers)

    def __call__(self, obs, key, get_activation: bool = False):
        _, encoder_rng = jax.random.split(key)
        traj = obs[..., : self.reference_obs_size]

        if get_activation:
            (latent_mean, latent_logvar), encoder_activations = self.encoder(
                traj, get_activation=True
            )
            z = reparameterize(encoder_rng, latent_mean, latent_logvar)
            egocentric_obs = obs[..., self.reference_obs_size :]
            concatenated = jnp.concatenate([z, egocentric_obs], axis=-1)
            action, decoder_activations = self.decoder(
                concatenated, get_activation=True
            )
            return (
                action,
                latent_mean,
                latent_logvar,
                {
                    "encoder": encoder_activations,
                    "decoder": decoder_activations,
                    "egocentric_obs": egocentric_obs,
                    "traj_obs": traj,
                    "intention": z,
                },
            )
        else:
            latent_mean, latent_logvar = self.encoder(traj, get_activation=False)
            z = reparameterize(encoder_rng, latent_mean, latent_logvar)
            action, _ = self.decoder(
                jnp.concatenate([z, obs[..., self.reference_obs_size :]], axis=-1)
            )
            return action, latent_mean, latent_logvar


def make_intention_policy(
    action_param_size: int,
    latent_size: int,
    total_obs_size: int,
    reference_obs_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    encoder_hidden_layer_sizes: Sequence[int] = (1024, 1024),
    decoder_hidden_layer_sizes: Sequence[int] = (1024, 1024),
) -> networks.FeedForwardNetwork:
    """
    Create a policy network with intention module.

    Args:
        action_param_size (int): the parameter size of the action space, usually double of the action size to model both the mean and variance of the action distribution
        latent_size (int): the size of the latent space
        total_obs_size (int): the total size of observations
        reference_obs_size (int): the size of reference observations
        preprocess_observations_fn (types.PreprocessObservationFn, optional): function to preprocess observations. Defaults to types.identity_observation_preprocessor.
        encoder_hidden_layer_sizes (Sequence[int], optional): sizes of encoder hidden layers. Defaults to (1024, 1024).
        decoder_hidden_layer_sizes (Sequence[int], optional): sizes of decoder hidden layers. Defaults to (1024, 1024).

    Returns:
        networks.FeedForwardNetwork: the created policy network
    """

    policy_module = IntentionNetwork(
        encoder_layers=list(encoder_hidden_layer_sizes),
        decoder_layers=list(decoder_hidden_layer_sizes)
        + [action_param_size],  # add action size to the last layer
        reference_obs_size=reference_obs_size,
        latents=latent_size,
    )

    def apply(processor_params, policy_params, obs, key, get_activation: bool = False):
        """Applies the policy network with observation normalizer, the output is the action distribution parameters."""
        obs = preprocess_observations_fn(obs, processor_params)
        return policy_module.apply(
            policy_params, obs=obs, key=key, get_activation=get_activation
        )

    dummy_total_obs = jnp.zeros((1, total_obs_size))
    dummy_key = jax.random.PRNGKey(0)

    return networks.FeedForwardNetwork(
        init=lambda key: policy_module.init(key, dummy_total_obs, dummy_key),
        apply=apply,
    )


def make_decoder_policy(
    param_size: int,
    decoder_obs_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    decoder_hidden_layer_sizes: Sequence[int] = (1024, 1024),
) -> Decoder:
    """Creates an encoder policy network."""

    policy_module = Decoder(
        layer_sizes=list(decoder_hidden_layer_sizes) + [param_size],
    )

    def apply(processor_params, policy_params, obs):
        temp_obs = obs
        obs = preprocess_observations_fn(
            obs[..., -processor_params.mean.shape[-1] :], processor_params
        )
        obs = jnp.concatenate(
            [temp_obs[..., : -processor_params.mean.shape[-1]], obs], axis=-1
        )
        return policy_module.apply(policy_params, x=obs)

    dummy_total_obs = jnp.zeros((1, decoder_obs_size))
    dummy_key = jax.random.PRNGKey(0)

    return networks.FeedForwardNetwork(
        init=lambda key: policy_module.init(key, dummy_total_obs, dummy_key),
        apply=apply,
    )