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
