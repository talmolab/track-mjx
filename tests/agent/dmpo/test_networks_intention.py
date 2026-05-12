"""IntentionDMPOPolicy + FlatDMPOCritic shape and finite-output tests."""
import jax
import jax.numpy as jnp
import pytest

from track_mjx.agent.dmpo.config import DMPOConfig
from track_mjx.agent.dmpo.networks_intention import (
    IntentionDMPOPolicy,
    FlatDMPOCritic,
    make_dmpo_intention_networks,
)


@pytest.fixture
def cfg():
    return DMPOConfig(num_envs=4, batch_size=4, sequence_length=2,
                      min_replay_size=4, max_replay_size=64,
                      num_atoms=11, vmin=-10.0, vmax=10.0,
                      critic_layer_sizes=(16,))


def _toy_obs(batch_dims=(), imit=8, proprio=4):
    return {
        "imitation_target": jnp.ones(batch_dims + (imit,), jnp.float32) * 0.5,
        "proprioception": jnp.ones(batch_dims + (proprio,), jnp.float32) * -0.3,
    }


def test_intention_policy_shapes(cfg):
    """Forward pass produces a Gaussian over action_size."""
    policy = IntentionDMPOPolicy(
        encoder_layer_sizes=(16, 16),
        decoder_layer_sizes=(16,),
        intention_size=4,
        action_size=3,
    )
    obs = _toy_obs()
    params = policy.init(jax.random.PRNGKey(0), obs)
    dist = policy.apply(params, obs)
    assert dist.mean().shape == (3,)
    assert dist.stddev().shape == (3,)
    assert jnp.all(jnp.isfinite(dist.mean()))
    assert jnp.all(jnp.isfinite(dist.stddev()))


def test_intention_policy_encode(cfg):
    """`.encode` returns (mean, logvar) without going through the decoder."""
    policy = IntentionDMPOPolicy(
        encoder_layer_sizes=(16, 16),
        decoder_layer_sizes=(16,),
        intention_size=4,
        action_size=3,
    )
    obs = _toy_obs()
    params = policy.init(jax.random.PRNGKey(0), obs)
    mean, logvar = policy.apply(params, obs, method=IntentionDMPOPolicy.encode)
    assert mean.shape == (4,)
    assert logvar.shape == (4,)
    assert jnp.all(jnp.isfinite(mean))


def test_flat_critic_shapes(cfg):
    """Critic returns a C51 distribution with `num_atoms` support points."""
    critic = FlatDMPOCritic(
        layer_sizes=(16,),
        num_atoms=cfg.num_atoms,
        vmin=cfg.vmin,
        vmax=cfg.vmax,
    )
    obs = _toy_obs()
    action = jnp.zeros((3,))
    params = critic.init(jax.random.PRNGKey(0), obs, action)
    dist = critic.apply(params, obs, action)
    # CategoricalCriticHead returns a DiscreteValuedTfpDistribution with
    # `values` of length num_atoms.
    assert dist.values.shape == (cfg.num_atoms,)
    assert jnp.all(jnp.isfinite(dist.mean()))


def test_factory_returns_dmpo_networks(cfg):
    """make_dmpo_intention_networks returns a DMPONetworks(policy, critic)."""
    from track_mjx.agent.dmpo.networks import DMPONetworks
    nets = make_dmpo_intention_networks(
        obs_sizes={"imitation_target": 8, "proprioception": 4},
        action_size=3,
        cfg=cfg,
        network_cfg={
            "encoder_layer_sizes": [16, 16],
            "decoder_layer_sizes": [16],
            "intention_size": 4,
            "activation": "silu",
        },
    )
    assert isinstance(nets, DMPONetworks)
    obs = _toy_obs()
    params = nets.policy.init(jax.random.PRNGKey(0), obs)
    dist = nets.policy.apply(params, obs)
    assert jnp.all(jnp.isfinite(dist.mean()))


def test_batched_forward(cfg):
    """vmap'd forward pass over a batch produces the right shape."""
    policy = IntentionDMPOPolicy(
        encoder_layer_sizes=(16,),
        decoder_layer_sizes=(16,),
        intention_size=4,
        action_size=3,
    )
    obs_unbatched = _toy_obs()
    params = policy.init(jax.random.PRNGKey(0), obs_unbatched)
    obs_batch = _toy_obs(batch_dims=(7,))
    dist = jax.vmap(policy.apply, in_axes=(None, 0))(params, obs_batch)
    assert dist.mean().shape == (7, 3)
