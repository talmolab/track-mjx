from typing import Sequence

from brax.training import networks
from brax.training import types
from brax.training.types import PRNGKey

import jax
import jax.numpy as jnp

from flax import nnx


class Encoder(nnx.Module):
    """Encoder Module for intention network, with NNX"""

    def __init__(
        self,
        input_size: int,
        layer_sizes: Sequence[int],
        latents: int,
        activation=nnx.relu,
        kernel_init: networks.Initializer = jax.nn.initializers.lecun_uniform(),
        use_bias: bool = True,
        *,
        rngs: nnx.Rngs,
    ):
        """Module Initializer to create an Encoder module for intention network

        Args:
            input_size (int): input shape for the encoder
            layer_sizes (Sequence[int]): layer size of the Encoder object, with fully connected weights
            latents (int): output shape for the encoder, latent space size
            activation (networks.ActivationFn, optional): activation function added between each linear layer, Defaults to nnx.relu.
            kernel_init (networks.Initializer, optional): kernel initializer for the network Defaults to jax.nn.initializers.lecun_uniform().
            use_bias (bool, optional): whether to use bias in the linear layer. Defaults to True.
        """
        self.layers = []

        for i, hidden_size in enumerate(layer_sizes[:-1]):
            if i != 0:
                input_size = hidden_size
            self.layers.append(
                nnx.Linear(input_size, hidden_size, kernel_init=kernel_init, use_bias=use_bias, rngs=rngs)
            )
            self.layers.append(activation)
            self.layers.append(nnx.LayerNorm(hidden_size, rngs=rngs))
        self.out_mean = nnx.Linear(layer_sizes[-1], latents, kernel_init=kernel_init, use_bias=use_bias, rngs=rngs)
        self.out_logvar = nnx.Linear(layer_sizes[-1], latents, kernel_init=kernel_init, use_bias=use_bias, rngs=rngs)

    def __call__(self, x: jnp.ndarray):
        """Call function for the Encoder module

        Args:
            x (jnp.ndarray): tasks specific input to the encoder, e.g. trajectory of the imitation clips

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: mean and logvar of the latent space
        """
        for layer in self.layers:
            x = layer(x)
        mean_x = self.out_mean(x)
        logvar_x = self.out_logvar(x)
        return mean_x, logvar_x


class Decoder(nnx.Module):
    """Decoder Module for intention network, with NNX"""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        layer_sizes: Sequence[int],
        activation: networks.ActivationFn = nnx.relu,
        kernel_init: networks.Initializer = jax.nn.initializers.lecun_uniform(),
        activate_final: bool = False,
        use_bias: bool = True,
        *,
        rngs: nnx.Rngs,
    ):
        """Module Initializer to create a Decoder module for intention network

        Args:
            input_size (int): input shape for the decoder
            output_size (int): output shape for the decoder, should be aligned with the action size
            layer_sizes (Sequence[int]): layer size of the Decoder object, with fully connected weights
            rngs (nnx.Rngs): random number generator for the network
            activation (networks.ActivationFn, optional): activation function between the layers . Defaults to nnx.relu.
            kernel_init (networks.Initializer, optional): kernel initializer function. Defaults to jax.nn.initializers.lecun_uniform().
            activate_final (bool, optional): whether to activate the final layer. Defaults to False.
            use_bias (bool, optional): whether to use bias to each layer. Defaults to True.
        """
        self.layers = []
        for i, hidden_size in enumerate(layer_sizes):
            if i != 0:
                input_size = hidden_size
            self.layers.append(
                nnx.Linear(input_size, hidden_size, kernel_init=kernel_init, use_bias=use_bias, rngs=rngs)
            )
            if i != len(layer_sizes) - 1 or activate_final:
                self.layers.append(activation)
                self.layers.append(nnx.LayerNorm(hidden_size, rngs=rngs))

    def __call__(self, x: jnp.ndarray):
        """_summary_

        Args:
            x (jnp.ndarray): input array concatenated from the intention and egocentrics observations

        Returns:
            jnp.ndarray : action output from the decoder
        """
        for layer in self.layers:
            x = layer(x)
        return x


def reparameterize(rng, mean, logvar):
    std = jnp.exp(0.5 * logvar)
    eps = jax.random.normal(rng, logvar.shape)
    return mean + eps * std


class IntentionNetwork(nnx.Module):
    """Intention Network with Encoder and Decoder, along with reparameterization trick"""

    def __init__(
        self,
        reference_obs_size: int,
        egocentric_obs_size: int,
        action_size: int,
        encoder_layers: Sequence[int],
        decoder_layers: Sequence[int],
        latents: int = 60,
        *,
        rngs: nnx.Rngs,
    ):
        """Module Initializer to create an Intention Network with Encoder and Decoder

        Args:
            reference_obs_size (int): reference observation size, e.g. the size of trajectory of the imitation clips
            egocentric_obs_size (int): egocentric observation size, e.g. the size of the current observation from the walker model
            action_size (int): action size, e.g. the size of the action space
            encoder_layers (Sequence[int]): encoder layer sizes for the Encoder object
            decoder_layers (Sequence[int]): decoder layer sizes for the Decoder object
            rngs (nnx.Rngs): random number generator for the network
            latents (int, optional): intention dimensions . Defaults to 60.
        """
        self.reference_obs_size = reference_obs_size

        self.encoder = Encoder(input_size=reference_obs_size, layer_sizes=encoder_layers, latents=latents, rngs=rngs)
        self.decoder = Decoder(
            input_size=latents + egocentric_obs_size, output_size=action_size, layer_sizes=decoder_layers, rngs=rngs
        )

    def __call__(self, observations: jnp.ndarray, rng: nnx.Rngs):
        """Call function for the Intention Network

        Args:
            observations (jnp.ndarray): concatenated observations from the reference and egocentric observations

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]: action, mean and logvar of the latent space
        """
        reference_obs = observations[..., : self.reference_obs_size]
        latent_mean, latent_logvar = self.encoder(reference_obs)
        z = reparameterize(rng, latent_mean, latent_logvar)
        action = self.decoder(jnp.concatenate([z, observations[..., self.reference_obs_size :]], axis=-1))
        return action, latent_mean, latent_logvar


def make_intention_policy(
    action_size: int,
    latent_size: int,
    total_obs_size: int,
    reference_obs_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    encoder_hidden_layer_sizes: Sequence[int] = (1024, 1024),
    decoder_hidden_layer_sizes: Sequence[int] = (1024, 1024),
) -> networks.FeedForwardNetwork:
    """Create an Intention Policy with Encoder and Decoder

    Args:
        action_size (int): the size of the output
        latent_size (int): _description_
        total_obs_size (int): _description_
        reference_obs_size (int): _description_
        preprocess_observations_fn (types.PreprocessObservationFn, optional): _description_. Defaults to types.identity_observation_preprocessor.
        encoder_hidden_layer_sizes (Sequence[int], optional): _description_. Defaults to (1024, 1024).
        decoder_hidden_layer_sizes (Sequence[int], optional): _description_. Defaults to (1024, 1024).

    Returns:
        IntentionNetwork: _description_
    """

    policy_module = IntentionNetwork(
        encoder_layers=list(encoder_hidden_layer_sizes),
        decoder_layers=list(decoder_hidden_layer_sizes),
        reference_obs_size=reference_obs_size,
        egocentric_obs_size=total_obs_size - reference_obs_size,
        latents=latent_size,
        action_size=action_size,
        rngs=nnx.Rngs(),
    )

    def apply(processor_params, policy_params, obs, key):
        obs = preprocess_observations_fn(obs, processor_params)
        return policy_module(observations=obs, rng=key)

    return networks.FeedForwardNetwork(
        init=lambda x: x,  # this is a dummy init function, as we are using NNX, where the init is done in the module itself
        apply=apply,
    )
