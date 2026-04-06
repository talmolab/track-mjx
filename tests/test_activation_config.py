"""Tests for configurable activation functions."""
import pytest
import flax.linen as nn
from track_mjx.agent.ff_ppo.intention_network import get_activation_fn


class TestGetActivationFn:
    """Resolve activation name strings to callables."""

    def test_silu(self):
        assert get_activation_fn("silu") is nn.silu

    def test_relu(self):
        assert get_activation_fn("relu") is nn.relu

    def test_tanh(self):
        assert get_activation_fn("tanh") is nn.tanh

    def test_gelu(self):
        assert get_activation_fn("gelu") is nn.gelu

    def test_elu(self):
        assert get_activation_fn("elu") is nn.elu

    def test_case_insensitive(self):
        assert get_activation_fn("SiLU") is nn.silu
        assert get_activation_fn("ReLU") is nn.relu

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown activation"):
            get_activation_fn("leaky_relu_xyz")


import jax
import jax.numpy as jnp

from track_mjx.agent.ff_ppo.intention_network import (
    Encoder,
    Decoder,
    IntentionNetwork,
    get_activation_fn,
)


class TestEncoderActivation:
    """Encoder respects the activation parameter."""

    def test_encoder_default_is_silu(self):
        enc = Encoder(layer_sizes=[64, 32], latents=8)
        assert enc.activation is nn.silu

    def test_encoder_with_relu(self):
        enc = Encoder(layer_sizes=[64, 32], latents=8, activation=nn.relu)
        assert enc.activation is nn.relu

    def test_encoder_forward_with_relu(self):
        enc = Encoder(layer_sizes=[64, 32], latents=8, activation=nn.relu)
        key = jax.random.PRNGKey(0)
        x = jnp.ones((2, 16))
        params = enc.init(key, x)
        mean, logvar = enc.apply(params, x)
        assert mean.shape == (2, 8)
        assert logvar.shape == (2, 8)


class TestDecoderActivation:
    """Decoder respects the activation parameter."""

    def test_decoder_default_is_silu(self):
        dec = Decoder(layer_sizes=[64, 32])
        assert dec.activation is nn.silu

    def test_decoder_with_relu(self):
        dec = Decoder(layer_sizes=[64, 32], activation=nn.relu)
        assert dec.activation is nn.relu

    def test_decoder_forward_with_relu(self):
        dec = Decoder(layer_sizes=[64, 32], activation=nn.relu)
        key = jax.random.PRNGKey(0)
        x = jnp.ones((2, 16))
        params = dec.init(key, x)
        out, _ = dec.apply(params, x)
        assert out.shape == (2, 32)


class TestIntentionNetworkActivation:
    """IntentionNetwork passes activation to submodules."""

    def test_intention_network_threads_activation(self):
        net = IntentionNetwork(
            encoder_layers=[64],
            decoder_layers=[64, 32],
            latents=8,
            activation=nn.relu,
        )
        assert net.activation is nn.relu
