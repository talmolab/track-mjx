"""Shape/init parity tests for _VisionScratchPolicyNet and _VisionScratchCriticNet.

These networks consume the dict observation emitted by
``vnl_playground.tasks.wrappers.EndToEndWrapper``:

    {"vision":          float32[H, W, 2C],
     "imitation_target": float32[task_obs_dim],
     "proprioception":   float32[proprio_dim]}

The policy returns ``tfd.MultivariateNormalDiag`` over ``action_size`` (38 for
the rat). The critic returns the categorical critic head distribution
(``DiscreteValuedTfpDistribution`` over ``num_atoms`` atoms).
"""

import jax
import jax.numpy as jnp
import pytest
from tensorflow_probability.substrates import jax as tfp

from track_mjx.agent.dmpo.config import DMPOConfig
from track_mjx.agent.dmpo.networks_vision_scratch import (
    _VisionScratchPolicyNet,
    _VisionScratchCriticNet,
    make_dmpo_vision_scratch_networks,
)

tfd = tfp.distributions

_VISION_SHAPE = (32, 32, 2)
_TASK_OBS_SIZE = 7
_PROPRIO_SIZE = 261
_ACTION_SIZE = 38

_OBS = {
    "vision": jnp.zeros(_VISION_SHAPE, dtype=jnp.float32),
    "imitation_target": jnp.zeros((_TASK_OBS_SIZE,), dtype=jnp.float32),
    "proprioception": jnp.zeros((_PROPRIO_SIZE,), dtype=jnp.float32),
}


def _cfg():
    return DMPOConfig()


def test_policy_returns_diag_normal_with_correct_action_shape():
    rng = jax.random.PRNGKey(0)
    net = _VisionScratchPolicyNet(
        layer_sizes=(256, 256, 256),
        action_size=_ACTION_SIZE,
        vision_shape=_VISION_SHAPE,
        cnn_feature_size=32,
        cnn_channels=(4, 8, 16, 32),
        mono_channels=1,
        shared_weights=True,
    )
    params = net.init(rng, _OBS)
    dist = net.apply(params, _OBS)
    assert isinstance(dist, tfd.Distribution)
    sample = dist.sample(seed=jax.random.PRNGKey(1))
    assert sample.shape == (_ACTION_SIZE,)
    assert dist.distribution.mean().shape == (_ACTION_SIZE,)
    assert jnp.all(dist.distribution.stddev() > 0.0)
    assert bool(jnp.all(jnp.isfinite(dist.distribution.mean())))


def test_critic_returns_categorical_distribution_with_correct_atom_count():
    rng = jax.random.PRNGKey(0)
    cfg = _cfg()
    net = _VisionScratchCriticNet(
        layer_sizes=(512, 512, 256),
        num_atoms=cfg.num_atoms,
        vmin=cfg.vmin,
        vmax=cfg.vmax,
        vision_shape=_VISION_SHAPE,
        cnn_feature_size=32,
        cnn_channels=(4, 8, 16, 32),
        mono_channels=1,
        shared_weights=True,
    )
    action = jnp.zeros((_ACTION_SIZE,), dtype=jnp.float32)
    params = net.init(rng, _OBS, action)
    dist = net.apply(params, _OBS, action)
    logits = dist.logits_parameter()
    assert logits.shape[-1] == cfg.num_atoms
    value = dist.mean()
    assert value.shape == ()
    assert bool(jnp.isfinite(value))


def test_factory_returns_dmpo_networks_with_correct_init_shapes():
    nets = make_dmpo_vision_scratch_networks(
        task_obs_size=_TASK_OBS_SIZE,
        proprio_size=_PROPRIO_SIZE,
        action_size=_ACTION_SIZE,
        vision_shape=_VISION_SHAPE,
        cfg=_cfg(),
    )
    rng = jax.random.PRNGKey(0)
    rng_p, rng_c = jax.random.split(rng)
    pol_params = nets.policy.init(rng_p, _OBS)
    crit_params = nets.critic.init(rng_c, _OBS, jnp.zeros((_ACTION_SIZE,)))
    dist_p = nets.policy.apply(pol_params, _OBS)
    dist_c = nets.critic.apply(crit_params, _OBS, jnp.zeros((_ACTION_SIZE,)))
    assert dist_p.distribution.mean().shape == (_ACTION_SIZE,)
    assert dist_c.logits_parameter().shape[-1] == _cfg().num_atoms
