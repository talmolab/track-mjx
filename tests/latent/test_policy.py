import jax, jax.numpy as jnp

from track_mjx.agent.latent_ppo.networks.policy import (
    LatentMimicPolicy, LatentMimicValue,
)


def test_policy_shapes():
    pi = LatentMimicPolicy(layer_sizes=(64, 32), action_dim=12)
    rng = jax.random.PRNGKey(0)
    obs = {
        "proprioception": jnp.ones((4, 30)),
        "o_history": jnp.ones((4, 60)),
        "z_target": jnp.ones((4, 16)),
    }
    params = pi.init(rng, obs)
    out = pi.apply(params, obs)
    assert out.shape == (4, 24)


def test_value_shape():
    v = LatentMimicValue(layer_sizes=(64, 32))
    rng = jax.random.PRNGKey(0)
    obs = {
        "proprioception": jnp.ones((4, 30)),
        "o_history": jnp.ones((4, 60)),
        "z_target": jnp.ones((4, 16)),
    }
    params = v.init(rng, obs)
    out = v.apply(params, obs)
    assert out.shape == (4,)
