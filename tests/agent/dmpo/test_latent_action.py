"""Latent-action DMPO transfer: the policy net and the latent -> motor map."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from track_mjx.agent.dmpo.latent_action import LatentActionPolicyNet, latent_to_ctrl


def _obs(key, vision_shape=(32, 32, 2), task=12, proprio=20):
    k1, k2, k3 = jax.random.split(key, 3)
    return {
        "vision": jax.random.uniform(k1, vision_shape),
        "imitation_target": jax.random.normal(k2, (task,)),
        "proprioception": jax.random.normal(k3, (proprio,)),
    }


def test_policy_starts_at_zero_mean_with_the_configured_spread():
    net = LatentActionPolicyNet(
        latent_size=16,
        vision_shape=(32, 32, 2),
        policy_head_layer_sizes=(32, 32),
        cnn_feature_size=8,
        cnn_channels=(4, 8),
        mono_channels=1,
        shared_weights=True,
        init_std=0.7,
    )
    obs = _obs(jax.random.key(0))
    params = net.init(jax.random.key(1), obs)
    dist = net.apply(params, obs)
    np.testing.assert_allclose(
        np.asarray(dist.mean()), 0.0, atol=1e-7
    )  # zero-init last layer
    np.testing.assert_allclose(np.asarray(dist.stddev()), 0.7, rtol=1e-5)
    assert dist.event_shape == (16,)
    # the spread is a trainable parameter
    assert "raw_std" in params["params"]


def test_latent_to_ctrl_is_prior_mean_plus_scaled_delta_through_the_decoder():
    L, P, A = 4, 6, 3
    W = jax.random.normal(jax.random.key(2), (L + P, 2 * A))

    def prior_fn(p):
        return jnp.tanh(p[:L]), jnp.zeros(L)

    def decoder_logits_fn(zp):
        out = zp @ W
        return out, {}

    proprio = jax.random.normal(jax.random.key(3), (P,))
    delta = jnp.array([0.5, -1.0, 0.0, 0.25])
    got = latent_to_ctrl(prior_fn, decoder_logits_fn, proprio, delta, 2.0, A)
    z = jnp.tanh(proprio[:L]) + 2.0 * delta
    want = jnp.tanh((jnp.concatenate([z, proprio]) @ W)[:A])
    np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-6)
    # delta = 0 reproduces the frozen pipeline's mode action at the prior mean
    got0 = latent_to_ctrl(prior_fn, decoder_logits_fn, proprio, jnp.zeros(L), 2.0, A)
    want0 = jnp.tanh((jnp.concatenate([jnp.tanh(proprio[:L]), proprio]) @ W)[:A])
    np.testing.assert_allclose(np.asarray(got0), np.asarray(want0), rtol=1e-6)


@pytest.mark.parametrize("bad", ["policy_head_slow_core", "dual_prior"])
def test_entry_rejects_unsupported_combinations_names(bad):
    # The entry's guard lists these by name; kept as a cheap tripwire that the strings stay in sync.
    import pathlib

    src = (
        pathlib.Path(__file__).resolve().parents[3]
        / "scripts"
        / "train_dmpo_highlvl_kl_anchor.py"
    ).read_text()
    assert "transfer.mode=latent_action does not support" in src
    assert bad in src
