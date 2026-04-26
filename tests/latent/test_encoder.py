"""Tests for MotionEncoder."""
import jax
import jax.numpy as jnp

from track_mjx.agent.latent_ppo.networks.encoder import MotionEncoder


def test_encoder_outputs_shapes():
    enc = MotionEncoder(layer_sizes=(64, 32), latent_dim=16)
    rng = jax.random.PRNGKey(0)
    x = jnp.ones((4, 10, 77))  # batch=4, w=10, feat=77
    params = enc.init(rng, x)
    mean, logvar = enc.apply(params, x)
    assert mean.shape == (4, 16)
    assert logvar.shape == (4, 16)


def test_encoder_flattens_window():
    """Encoder must accept (B, w, feat) and produce (B, latent)."""
    enc = MotionEncoder(layer_sizes=(8,), latent_dim=4)
    rng = jax.random.PRNGKey(0)
    x = jnp.ones((2, 3, 5))
    params = enc.init(rng, x)
    mean, _ = enc.apply(params, x)
    assert mean.shape == (2, 4)
