"""Tests for proprioception noise in VisionIntentionNetwork."""
import jax
import jax.numpy as jnp
from track_mjx.agent.ff_ppo.intention_network import VisionIntentionNetwork


def test_vision_intention_network_has_noise_attribute():
    net = VisionIntentionNetwork(
        encoder_layers=[64], decoder_layers=[64, 22], latents=8,
        proprioception_noise_std=0.1,
    )
    assert net.proprioception_noise_std == 0.1


def test_vision_intention_network_noise_default_zero():
    net = VisionIntentionNetwork(
        encoder_layers=[64], decoder_layers=[64, 22], latents=8,
    )
    assert net.proprioception_noise_std == 0.0


def test_vision_intention_network_deterministic_no_noise():
    net = VisionIntentionNetwork(
        encoder_layers=[64], decoder_layers=[64, 22], latents=8,
        proprioception_noise_std=0.5,
    )
    obs = {
        "imitation_target": jnp.ones((1, 20)),
        "proprioception": jnp.ones((1, 10)),
        "vision": jnp.ones((1, 32, 32, 3)),  # 32x32 required for 4-layer CNN
    }
    key = jax.random.PRNGKey(0)
    params = net.init(key, obs, key)
    out1, _, _ = net.apply(params, obs=obs, key=jax.random.PRNGKey(1), deterministic=True)
    out2, _, _ = net.apply(params, obs=obs, key=jax.random.PRNGKey(2), deterministic=True)
    assert jnp.allclose(out1, out2, atol=1e-6)
