"""End-to-end integration test for the kl-anchor warm-start invariant.

Builds the full computational chain (imit normalizer → frozen prior+decoder →
a_imit; trainable kl-anchor policy with warm-started weights → bind(μ_θ) →
a_taken) and asserts that a_taken ≈ a_imit at step 0. Equivalently asserts
that the wrapper's anchor reward saturates near 1.0 at step 0.

This is the test that would have caught the obs-normalization mismatch and
the σ-parameterization mismatch on day one. We keep it separate from the
unit test in test_networks_kl_anchor.py because it touches more pieces
(prior_utils inference fns, anchor reward formula).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from brax.training import distribution
from brax.training.acme import running_statistics

from scamper.agent.imitation.intention_network import Decoder as ScamperDecoder
from scamper.agent.mlp_prior.prior_networks import Prior as ScamperPrior

from track_mjx.agent.dmpo.action_utils import bind
from track_mjx.agent.dmpo.config import DMPOConfig
from track_mjx.agent.dmpo.networks_kl_anchor import make_dmpo_kl_anchor_networks
from track_mjx.agent.observation_utils import (
    DictRunningStatisticsState,
    init_dict_normalizer,
)
from track_mjx.agent.dmpo.normalizer_seeding import seed_proprio_from_imit


@pytest.fixture
def cfg():
    return DMPOConfig(num_envs=4, unroll_length=4, batch_size=4, sequence_length=4)


def test_anchor_reward_near_one_at_step_zero(cfg):
    """Simulate one wrapper step at t=0 and assert r_anchor > 0.99.

    The anchor reward is exp(-w * ||a_taken - a_imit||² / action_size *
    action_size) = exp(-w * Σ_i (a_taken_i - a_imit_i)²). With a perfectly
    aligned warm-start, the squared-error sum is ≈ 0 and r_anchor ≈ 1.0.
    """
    rng = jax.random.PRNGKey(13)
    proprio_size = 32
    task_obs_size = 12
    action_size = 18
    vision_shape = (16, 16, 2)
    latent_size = 8
    prior_layers = (32, 32)
    decoder_layers = (32, 32)
    w_anchor = 0.5  # matches velocity_only_kl_anchor.yaml default

    # 1. Build random imit prior + decoder + non-trivial proprio normalizer.
    frozen_prior = ScamperPrior(layer_sizes=list(prior_layers), latents=latent_size)
    frozen_decoder = ScamperDecoder(
        layer_sizes=list(decoder_layers) + [2 * action_size],
    )
    rng, k_p, k_d, k_n, k_o = jax.random.split(rng, 5)
    prior_params = frozen_prior.init(k_p, jnp.zeros((1, proprio_size)))["params"]
    decoder_params = frozen_decoder.init(
        k_d, jnp.zeros((1, latent_size + proprio_size))
    )["params"]
    proprio_mean = jax.random.normal(k_n, (proprio_size,)) * 0.3
    proprio_std = 0.4 + jnp.abs(jax.random.normal(k_n, (proprio_size,))) * 0.3
    imit_norm = DictRunningStatisticsState(
        imitation_target=running_statistics.RunningStatisticsState(
            mean=jnp.zeros((512,), dtype=jnp.float32),
            std=jnp.ones((512,), dtype=jnp.float32),
            count=jnp.array(1_000_000.0, dtype=jnp.float32),
            summed_variance=jnp.ones((512,), dtype=jnp.float32),
            std_eps=1e-6,
            mode=running_statistics.NormalizationMode.WELFORD,
        ),
        proprioception=running_statistics.RunningStatisticsState(
            mean=proprio_mean,
            std=proprio_std,
            count=jnp.array(1_000_000.0, dtype=jnp.float32),
            summed_variance=proprio_std * proprio_std * 1_000_000.0,
            std_eps=1e-6,
            mode=running_statistics.NormalizationMode.WELFORD,
        ),
    )

    # 2. Build the DMPO normalizer (fresh) and seed proprio.
    obs_template = {
        "vision": jnp.zeros(vision_shape, dtype=jnp.float32),
        "imitation_target": jnp.zeros((task_obs_size,), dtype=jnp.float32),
        "proprioception": jnp.zeros((proprio_size,), dtype=jnp.float32),
    }
    dmpo_norm_fresh = init_dict_normalizer(obs_template)
    dmpo_norm = seed_proprio_from_imit(dmpo_norm_fresh, imit_norm)

    # 3. Sample raw obs (env units).
    raw_proprio = jax.random.normal(k_o, (1, proprio_size))

    # 4. IMIT branch (what wrapper computes for a_imit):
    norm_proprio_imit = running_statistics.normalize(raw_proprio, imit_norm.proprioception)
    z_imit, _ = frozen_prior.apply({"params": prior_params}, norm_proprio_imit)
    decoder_input = jnp.concatenate([z_imit, norm_proprio_imit], axis=-1)
    logits, _ = frozen_decoder.apply({"params": decoder_params}, decoder_input)
    parametric = distribution.NormalTanhDistribution(event_size=action_size)
    a_imit = parametric.mode(logits)

    # 5. TRAINABLE branch (what rollout would compute for a_taken at t=0).
    nets = make_dmpo_kl_anchor_networks(
        proprio_size=proprio_size,
        task_obs_size=task_obs_size,
        action_size=action_size,
        latent_size=latent_size,
        vision_shape=vision_shape,
        prior_layer_sizes=prior_layers,
        decoder_layer_sizes=decoder_layers,
        policy_head_layer_sizes=(32, 32),
        cfg=cfg,
        cnn_feature_size=8,
        cnn_channels=(2, 4, 8),
        mono_channels=1,
        shared_weights=True,
        warm_start_prior_params=prior_params,
        warm_start_decoder_params=decoder_params,
    )
    # Apply DMPO normalizer (which now carries imit proprio stats).
    from track_mjx.agent.observation_utils import normalize_dict_obs

    raw_obs = {
        "vision": jnp.zeros((1,) + vision_shape),
        "imitation_target": jnp.zeros((1, task_obs_size)),
        "proprioception": raw_proprio,
    }
    norm_obs = normalize_dict_obs(raw_obs, dmpo_norm)
    params = nets.policy.init(jax.random.PRNGKey(0), norm_obs)
    dist = nets.policy.apply(params, norm_obs)
    # At t=0, MPO's M-step has not run; the rollout takes the policy's mode
    # (after bind) for the deterministic check. (At rollout time the action
    # is sampled, but the *expected* anchor reward is computed against the
    # mode in deterministic eval; for the stochastic sample, σ ≈ 0.7 means
    # the *average* anchor reward over many samples is close-to-but-less-
    # than 1.0. We test the mode here for a tight assertion.)
    a_taken = bind(dist.mode())

    # 6. Compute the wrapper's anchor reward exactly as wrappers_kl_anchor.py:113-114.
    diff = a_taken[..., :action_size] - a_imit[..., :action_size]
    action_mse = jnp.mean(diff * diff)
    r_anchor = float(jnp.exp(-w_anchor * action_mse * action_size))

    assert r_anchor > 0.99, (
        f"Warm-start integration test failed: r_anchor={r_anchor:.4f} "
        f"(expected > 0.99). action_mse={float(action_mse):.6f}. "
        "Either the obs normalization, the σ parameterization, or the "
        "warm-start splice is broken end-to-end."
    )
