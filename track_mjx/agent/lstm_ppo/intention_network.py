from typing import Sequence, Tuple, Callable, Union, Any
import dataclasses

from brax.training import networks
from brax.training import types
from brax.training.types import PRNGKey

import jax
import jax.numpy as jnp
from jax import random

from flax import linen as nn


@dataclasses.dataclass
class LSTMNetwork:
    init: Callable[..., Any]
    apply: Callable[..., Any]


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
    ) -> Union[
        Tuple[jnp.ndarray, jnp.ndarray], Tuple[Tuple[jnp.ndarray, jnp.ndarray], dict]
    ]:
        activations = {}
        # For each layer in the sequence
        for i, hidden_size in enumerate(self.layer_sizes):
            # jax.debug.print("[DEBUG inside Encoder] layer {}, x mean: {}", i, x.mean())
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


class LSTMDecoder(nn.Module):
    """LSTM-based decoder for sequential action generation."""

    layer_sizes: Sequence[int]
    hidden_dim: int = 128
    hidden_layer_num: int = 2
    activation: networks.ActivationFn = nn.relu
    kernel_init: networks.Initializer = jax.nn.initializers.lecun_uniform()
    bias: bool = True

    @nn.compact
    def __call__(self, x, hidden_state, get_activation: bool = False):
        activations = {}
        h, c = hidden_state

        h_new, c_new = [], []
        for layer_idx in range(self.hidden_layer_num):

            # LSTM layer, returned (new_c, new_h), new_h if call cirectly, no need to init + carry in nn.compact
            # c_t is the memory h_t is the hidden layers, same in NN or in LSTM, so need to connect another fully connected out
            lstm = nn.LSTMCell(
                features=self.hidden_dim,
                name=f"lstm_{layer_idx}",
                kernel_init=self.kernel_init,
            )

            h_i = h[:, layer_idx, :]
            c_i = c[:, layer_idx, :]

            (new_c_i, new_h_i), x = lstm((c_i, h_i), x)

            h_new.append(new_h_i)
            c_new.append(new_c_i)

        # flax does not allow control the output size independently of the hidden state size.
        x = nn.Dense(
            self.layer_sizes[-1],
            name="lstm_projection",
            kernel_init=self.kernel_init,
            use_bias=self.bias,
        )(x)
        activations["lstm_projection"] = x

        stacke_h_new = jnp.stack(h_new, axis=1)
        stacke_c_new = jnp.stack(c_new, axis=1)

        if get_activation:
            # hidden is stored as (num_hidden_layers, 128)
            return x, (stacke_h_new, stacke_c_new), activations
        return x, (stacke_h_new, stacke_c_new), {} # hidden_states still tuple


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
    hidden_states: int = 128
    hidden_layer_num: int = 2

    def setup(self):
        self.encoder = Encoder(layer_sizes=self.encoder_layers, latents=self.latents)
        self.lstm_decoder = LSTMDecoder(
            layer_sizes=self.decoder_layers,
            hidden_dim=self.hidden_states,
            hidden_layer_num=self.hidden_layer_num,
        )

    def __call__(self, obs, key, hidden_state, get_activation):
        _, encoder_rng = jax.random.split(key)
        traj = obs[..., : self.reference_obs_size]

        if get_activation:
            (latent_mean, latent_logvar), encoder_activations = self.encoder(
                traj, get_activation=get_activation
            )
            z = latent_mean
            egocentric_obs = obs[..., self.reference_obs_size :]
            concatenated = jnp.concatenate(
                [z, egocentric_obs], axis=-1
            )
            print("In intention_network using LSTM + Activation")
            action, new_hidden_state, decoder_activations = self.lstm_decoder(
                concatenated, hidden_state, get_activation=get_activation
            )
            return (
                action,
                latent_mean,
                latent_logvar,
                new_hidden_state,
                {
                    "encoder": encoder_activations,
                    "decoder": decoder_activations,
                    "intention": z,
                    "hidden_state": new_hidden_state,
                },
            )
                
        else:
            latent_mean, latent_logvar = self.encoder(
                traj, get_activation=get_activation
            )
            z = latent_mean
            egocentric_obs = obs[..., self.reference_obs_size :]
            concatenated = jnp.concatenate(
                [z, egocentric_obs], axis=-1
            )
            print("In intention_network using just LSTM, no Activation")
            action, new_hidden_state, decoder_activations = self.lstm_decoder(concatenated, hidden_state)
            return action, latent_mean, latent_logvar, new_hidden_state


def make_intention_policy(
    action_param_size: int,
    latent_size: int,
    hidden_state_size: int,
    hidden_layer_num: int,
    total_obs_size: int,
    reference_obs_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    encoder_hidden_layer_sizes: Sequence[int] = (1024, 1024),
    decoder_hidden_layer_sizes: Sequence[int] = (1024, 1024),
    get_activation: bool = True,
) -> LSTMNetwork:
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
        LSTMNetwork: the created policy network
    """

    policy_module = IntentionNetwork(
        encoder_layers=list(encoder_hidden_layer_sizes),
        decoder_layers=list(decoder_hidden_layer_sizes)
        + [action_param_size],  # add action size to the last layer
        reference_obs_size=reference_obs_size,
        latents=latent_size,
        hidden_states=hidden_state_size,
        hidden_layer_num=hidden_layer_num,
    )

    def apply(
        processor_params,
        policy_params,
        obs,
        key,
        hidden_state,
        get_activation,
    ):
        """Applies the policy network with observation normalizer, the output is the action distribution parameters."""
        obs = preprocess_observations_fn(obs, processor_params)
        return policy_module.apply(
            policy_params,
            obs=obs,
            key=key,
            hidden_state=hidden_state,
            get_activation=get_activation,
        )

    # dummy variables here, actual pass in in training loops
    dummy_total_obs = jnp.zeros((1, total_obs_size))
    dummy_key = jax.random.PRNGKey(0)

    # lambda function here to pass in hidden from training loop
    return LSTMNetwork(
        init=lambda key, hidden_state: policy_module.init(
            key, dummy_total_obs, dummy_key, hidden_state, get_activation
        ),
        apply=apply,
    )