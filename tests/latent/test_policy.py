import jax, jax.numpy as jnp

from track_mjx.agent.latent_ppo.networks.policy import (
    LatentMimicPolicy, LatentMimicValue,
)


def test_policy_shapes():
    pi = LatentMimicPolicy(layer_sizes=(64, 32), action_dim=12)
    rng = jax.random.PRNGKey(0)
    # ff_ppo schema after flatten_obs_dict: imitation_target = z_target,
    # proprioception = base_proprio + o_history concatenated.
    obs = {
        "proprioception": jnp.ones((4, 90)),       # base 30 + o_history 60
        "imitation_target": jnp.ones((4, 16)),     # z_target
    }
    params = pi.init(rng, obs)
    out = pi.apply(params, obs)
    assert out.shape == (4, 24)


def test_value_shape():
    v = LatentMimicValue(layer_sizes=(64, 32))
    rng = jax.random.PRNGKey(0)
    obs = {
        "proprioception": jnp.ones((4, 90)),
        "imitation_target": jnp.ones((4, 16)),
    }
    params = v.init(rng, obs)
    out = v.apply(params, obs)
    assert out.shape == (4,)
