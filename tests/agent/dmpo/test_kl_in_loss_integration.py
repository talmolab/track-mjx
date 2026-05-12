"""End-to-end integration test for the kl-anchor KL-in-loss architecture.

Builds the full computation chain: frozen Prior + Decoder + non-trivial
imit normalizer + seeded DMPO normalizer + trainable kl-anchor policy
(with warm-started weights). Asserts that at step 0 the closed-form
KL(pi_theta || pi_imit) is essentially zero — equivalently,
exp(-w * KL) > 0.99 — confirming the warm-start invariant holds at the
distributional level (not just the mode-action level).
"""
from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from brax.training.acme import running_statistics

from scamper.agent.imitation.intention_network import Decoder as ScamperDecoder
from scamper.agent.mlp_prior.prior_networks import Prior as ScamperPrior

from track_mjx.agent.dmpo.config import DMPOConfig
from track_mjx.agent.dmpo.kl_anchor_utils import pretanh_gaussian_kl
from track_mjx.agent.dmpo.networks_kl_anchor import make_dmpo_kl_anchor_networks
from track_mjx.agent.dmpo.normalizer_seeding import seed_proprio_from_imit
from track_mjx.agent.observation_utils import (
    DictRunningStatisticsState,
    init_dict_normalizer,
    normalize_dict_obs,
)


@pytest.fixture
def cfg():
    return DMPOConfig(num_envs=4, unroll_length=4, batch_size=4, sequence_length=4)


def test_kl_in_loss_warm_start_invariant(cfg):
    """At step 0: KL(online_policy || frozen_anchor) ≈ 0 → exp(-w*KL) > 0.99."""
    rng = jax.random.PRNGKey(13)
    proprio_size = 32
    task_obs_size = 12
    action_size = 18
    vision_shape = (16, 16, 2)
    latent_size = 8
    prior_layers = (32, 32)
    decoder_layers = (32, 32)
    w_anchor = 0.5

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
            mean=proprio_mean, std=proprio_std,
            count=jnp.array(1_000_000.0, dtype=jnp.float32),
            summed_variance=proprio_std * proprio_std * 1_000_000.0,
            std_eps=1e-6, mode=running_statistics.NormalizationMode.WELFORD,
        ),
    )
    obs_template = {
        "vision": jnp.zeros(vision_shape, dtype=jnp.float32),
        "imitation_target": jnp.zeros((task_obs_size,), dtype=jnp.float32),
        "proprioception": jnp.zeros((proprio_size,), dtype=jnp.float32),
    }
    dmpo_norm = seed_proprio_from_imit(init_dict_normalizer(obs_template), imit_norm)

    raw_proprio = jax.random.normal(k_o, (1, proprio_size))

    # IMIT branch: pre-tanh (mu, log_std). The wrapper applies
    # log(softplus(raw)+1e-3) on log_std to match online_dist.stddev() units.
    norm_proprio_imit = running_statistics.normalize(raw_proprio, imit_norm.proprioception)
    z_imit, _ = frozen_prior.apply({"params": prior_params}, norm_proprio_imit)
    decoder_input = jnp.concatenate([z_imit, norm_proprio_imit], axis=-1)
    logits, _ = frozen_decoder.apply({"params": decoder_params}, decoder_input)
    mu_imit = logits[..., :action_size]
    raw_log_std_imit = logits[..., action_size:]
    # Apply the same softplus+1e-3 transform the wrapper applies before exposing
    # log_std_imit, so the units match log(online_dist.stddev()).
    log_std_imit = jnp.log(jax.nn.softplus(raw_log_std_imit) + 1e-3)

    # TRAINABLE branch.
    nets = make_dmpo_kl_anchor_networks(
        proprio_size=proprio_size, task_obs_size=task_obs_size,
        action_size=action_size, latent_size=latent_size,
        vision_shape=vision_shape,
        prior_layer_sizes=prior_layers, decoder_layer_sizes=decoder_layers,
        policy_head_layer_sizes=(32, 32),
        cfg=cfg, cnn_feature_size=8, cnn_channels=(2, 4, 8),
        mono_channels=1, shared_weights=True,
        warm_start_prior_params=prior_params,
        warm_start_decoder_params=decoder_params,
    )
    raw_obs = {
        "vision": jnp.zeros((1,) + vision_shape),
        "imitation_target": jnp.zeros((1, task_obs_size)),
        "proprioception": raw_proprio,
    }
    norm_obs = normalize_dict_obs(raw_obs, dmpo_norm)
    params = nets.policy.init(jax.random.PRNGKey(0), norm_obs)
    dist = nets.policy.apply(params, norm_obs)
    mu_theta = dist.mean()
    log_std_theta = jnp.log(dist.stddev())

    kl = pretanh_gaussian_kl(mu_theta, log_std_theta, mu_imit, log_std_imit)
    # kl is per-sample; aggregate via mean(exp(-w*kl)).
    r_anchor = float(jnp.mean(jnp.exp(-w_anchor * kl)))

    assert r_anchor > 0.99, (
        f"KL-in-loss warm-start failed: r_anchor={r_anchor:.4f}, "
        f"KL={float(jnp.mean(kl)):.6f}. The online policy distribution is "
        f"not reproducing the imit decoder distribution at step 0."
    )
