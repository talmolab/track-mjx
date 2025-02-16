import dataclasses
from typing import Any, Callable, Sequence, Tuple, Union
import warnings

from brax.training import networks
from brax.training import types
from brax.training import distribution
from brax.training.networks import MLP
import brax.training.agents.ppo.networks as ppo_networks
from brax.training.types import PRNGKey

import jax
import jax.numpy as jnp
from jax import random

import flax
from flax import linen as nn


class Encoder(nn.Module):
    """outputs in the form of distributions in latent space"""

    layer_sizes: Sequence[int]
    latents: int  # intention size
    activation: networks.ActivationFn = nn.relu
    kernel_init: networks.Initializer = jax.nn.initializers.lecun_uniform()
    bias: bool = True

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, get_activation: bool = False
    ) -> Union[Tuple[jnp.ndarray, jnp.ndarray], Tuple[Tuple[jnp.ndarray, jnp.ndarray], dict]]:
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
    activation: networks.ActivationFn = nn.relu
    kernel_init: networks.Initializer = jax.nn.initializers.lecun_uniform()
    activate_final: bool = False
    bias: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, get_activation: bool = False) -> Union[jnp.ndarray, Tuple[jnp.ndarray, dict]]:
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
        return x

class LSTMDecoder(nn.Module):
    """LSTM-based decoder for sequential action generation."""
    layer_sizes: Sequence[int]
    hidden_dim: int = 128
    activation: networks.ActivationFn = nn.relu
    kernel_init: networks.Initializer = jax.nn.initializers.lecun_uniform()
    bias: bool = True

    @nn.compact
    def __call__(self, x, hidden_state, get_activation: bool = False):
        activations = {}

        # LSTM layer
        lstm = nn.LSTMCell(features=self.hidden_dim, name="lstm")
        new_hidden_state, x = lstm(hidden_state, x)

        for i, hidden_size in enumerate(self.layer_sizes):
            x = nn.Dense(
                hidden_size, name=f"hidden_{i}", kernel_init=self.kernel_init, use_bias=self.bias
            )(x)
            x = self.activation(x)
            x = nn.LayerNorm()(x)
            if get_activation:
                activations[f"layer_{i}"] = x
        
        if get_activation:
            return x, new_hidden_state, activations
        return x, new_hidden_state


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
        self.lstm_decoder = LSTMDecoder(layer_sizes=self.decoder_layers, hidden_dim=128) #TODO: hard codeed now, change it later

    def __call__(self, obs, key, hidden_state, get_activation, use_lstm):
        _, encoder_rng = jax.random.split(key)
        traj = obs[..., : self.reference_obs_size]
        
        if get_activation:
            (latent_mean, latent_logvar), encoder_activations = self.encoder(traj, get_activation=get_activation)
            z = reparameterize(encoder_rng, latent_mean, latent_logvar)
            concatenated = jnp.concatenate([z, obs[..., self.reference_obs_size :]], axis=-1)
            
            if use_lstm:
                action, new_hidden_state, decoder_activations = self.lstm_decoder(concatenated, hidden_state, get_activation=get_activation)
                return action, latent_mean, latent_logvar, new_hidden_state, {"encoder": encoder_activations, "decoder": decoder_activations, "intention": z}
            else:
                action, decoder_activations = self.decoder(concatenated, get_activation=get_activation)
                return action, latent_mean, latent_logvar, {"encoder": encoder_activations, "decoder": decoder_activations, "intention": z}
        else:
            latent_mean, latent_logvar = self.encoder(traj, get_activation=get_activation)
            z = reparameterize(encoder_rng, latent_mean, latent_logvar)
            
            if use_lstm:
                action, new_hidden_state = self.lstm_decoder(jnp.concatenate([z, obs[..., self.reference_obs_size:]], axis=-1), hidden_state)
                return action, latent_mean, latent_logvar, new_hidden_state
            
            else:
                action = self.decoder(
                    jnp.concatenate([z, obs[..., self.reference_obs_size :]], axis=-1)
                )
                return action, latent_mean, latent_logvar


def make_intention_policy(
    param_size: int,
    latent_size: int,
    total_obs_size: int,
    reference_obs_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    encoder_hidden_layer_sizes: Sequence[int] = (1024, 1024),
    decoder_hidden_layer_sizes: Sequence[int] = (1024, 1024),
    get_activation: bool = True,
    use_lstm: bool = True,
) -> networks.FeedForwardNetwork:
    """Creates an intention policy network."""

    policy_module = IntentionNetwork(
        encoder_layers=list(encoder_hidden_layer_sizes),
        decoder_layers=list(decoder_hidden_layer_sizes) + [param_size],
        reference_obs_size=reference_obs_size,
        latents=latent_size,
    )

    # def apply(processor_params, policy_params, obs, key, get_activation: bool = False):
    #     """Applies the policy network with observation normalizer."""
    #     obs = preprocess_observations_fn(obs, processor_params)
    #     return policy_module.apply(policy_params, obs=obs, key=key, get_activation=get_activation)
    
    def apply(processor_params, policy_params, obs, key, hidden_state, get_activation, use_lstm):
        """Applies the policy network with observation normalizer."""
        obs = preprocess_observations_fn(obs, processor_params)
        return policy_module.apply(policy_params, obs=obs, key=key, hidden_state=hidden_state, get_activation=get_activation, use_lstm=use_lstm)

    dummy_total_obs = jnp.zeros((1, total_obs_size))
    dummy_key = jax.random.PRNGKey(0)
    dummy_hidden_state = nn.LSTMCell(features=128).initialize_carry(
        jax.random.PRNGKey(0), (1,)
    )

    return networks.FeedForwardNetwork(
        init=lambda key: policy_module.init(key, dummy_total_obs, dummy_key, dummy_hidden_state, get_activation, use_lstm),
        apply=apply,
    )
