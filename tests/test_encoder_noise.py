"""Tests for encoder input noise in IntentionNetwork."""
import jax
import jax.numpy as jnp
from track_mjx.agent.ff_ppo.intention_network import IntentionNetwork


def test_intention_network_has_encoder_noise_attribute():
    net = IntentionNetwork(
        encoder_layers=[64],
        decoder_layers=[64, 22],
        latents=8,
        encoder_noise_std=0.1,
    )
    assert net.encoder_noise_std == 0.1


def test_intention_network_encoder_noise_default_zero():
    net = IntentionNetwork(
        encoder_layers=[64],
        decoder_layers=[64, 22],
        latents=8,
    )
    assert net.encoder_noise_std == 0.0


def test_encoder_noise_deterministic_no_effect():
    """Deterministic mode should produce identical outputs regardless of key."""
    net = IntentionNetwork(
        encoder_layers=[64],
        decoder_layers=[64, 22],
        latents=8,
        encoder_noise_std=0.5,
    )
    obs = {
        "imitation_target": jnp.ones((1, 20)),
        "proprioception": jnp.ones((1, 10)),
    }
    key = jax.random.PRNGKey(0)
    params = net.init(key, obs, key)
    out1, _, _ = net.apply(params, obs=obs, key=jax.random.PRNGKey(1), deterministic=True)
    out2, _, _ = net.apply(params, obs=obs, key=jax.random.PRNGKey(2), deterministic=True)
    assert jnp.allclose(out1, out2, atol=1e-6)


def test_encoder_noise_stochastic_changes_output():
    """Stochastic mode with encoder noise should produce different outputs for different keys."""
    net = IntentionNetwork(
        encoder_layers=[64],
        decoder_layers=[64, 22],
        latents=8,
        encoder_noise_std=0.5,
        proprioception_noise_std=0.0,
    )
    obs = {
        "imitation_target": jnp.ones((1, 20)),
        "proprioception": jnp.ones((1, 10)),
    }
    key = jax.random.PRNGKey(0)
    params = net.init(key, obs, key)
    out1, _, _ = net.apply(params, obs=obs, key=jax.random.PRNGKey(1), deterministic=False)
    out2, _, _ = net.apply(params, obs=obs, key=jax.random.PRNGKey(2), deterministic=False)
    assert not jnp.allclose(out1, out2, atol=1e-4)


def test_encoder_noise_zero_std_no_effect():
    """With encoder_noise_std=0.0, encoder means should be identical across keys."""
    net = IntentionNetwork(
        encoder_layers=[64],
        decoder_layers=[64, 22],
        latents=8,
        encoder_noise_std=0.0,
        proprioception_noise_std=0.0,
    )
    obs = {
        "imitation_target": jnp.ones((1, 20)),
        "proprioception": jnp.ones((1, 10)),
    }
    key = jax.random.PRNGKey(0)
    params = net.init(key, obs, key)
    out1, mean1, _ = net.apply(params, obs=obs, key=jax.random.PRNGKey(1), deterministic=False)
    out2, mean2, _ = net.apply(params, obs=obs, key=jax.random.PRNGKey(2), deterministic=False)
    assert jnp.allclose(mean1, mean2, atol=1e-6)
