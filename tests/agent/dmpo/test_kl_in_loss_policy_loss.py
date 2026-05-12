"""Verify _policy_loss_fn applies the KL-anchor loss term when alpha > 0."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions

from track_mjx.agent.dmpo.config import DMPOConfig
from track_mjx.agent.dmpo.kl_anchor_utils import pretanh_gaussian_kl
from track_mjx.agent.dmpo.networks import DMPONetworks


def _make_tiny_networks(action_size=4, obs_size=6):
    import flax.linen as nn

    class _PolicyMod(nn.Module):
        action_size: int
        @nn.compact
        def __call__(self, obs):
            h = nn.Dense(8)(obs)
            mu = nn.Dense(self.action_size)(h)
            log_std = nn.Dense(self.action_size)(h)
            return tfd.MultivariateNormalDiag(loc=mu, scale_diag=jnp.exp(log_std) + 1e-3)

    class _CriticMod(nn.Module):
        num_atoms: int
        @nn.compact
        def __call__(self, obs, action):
            h = jnp.concatenate([obs, action], axis=-1)
            h = nn.Dense(8)(h)
            logits = nn.Dense(self.num_atoms)(h)
            return tfd.Categorical(logits=logits)

    return DMPONetworks(
        policy=_PolicyMod(action_size=action_size),
        critic=_CriticMod(num_atoms=51),
    )


def test_policy_loss_adds_anchor_term_when_alpha_positive():
    """_policy_loss_fn loss with kl_anchor_alpha=alpha must equal the loss
    with kl_anchor_alpha=0 plus -alpha * mean(exp(-w * KL)).
    """
    from track_mjx.agent.dmpo.learner import _policy_loss_fn

    cfg_alpha0 = DMPOConfig(
        num_envs=4, batch_size=4, sequence_length=4,
        kl_anchor_alpha=0.0, kl_anchor_w=0.5,
    )
    cfg_alpha1 = DMPOConfig(
        num_envs=4, batch_size=4, sequence_length=4,
        kl_anchor_alpha=1.5, kl_anchor_w=0.5,
    )
    nets = _make_tiny_networks(action_size=4, obs_size=6)

    rng = jax.random.PRNGKey(0)
    rng, k_init = jax.random.split(rng)
    policy_params = nets.policy.init(k_init, jnp.zeros((1, 6)))
    target_policy_params = jax.tree.map(lambda x: x + 0.0, policy_params)
    target_critic_params = nets.critic.init(rng, jnp.zeros((1, 6)), jnp.zeros((1, 4)))
    from track_mjx.agent.dmpo.losses import MPO
    dual = MPO(
        epsilon=cfg_alpha0.epsilon, epsilon_mean=cfg_alpha0.epsilon_mean,
        epsilon_stddev=cfg_alpha0.epsilon_stddev,
        init_log_temperature=cfg_alpha0.init_log_temperature,
        init_log_alpha_mean=cfg_alpha0.init_log_alpha_mean,
        init_log_alpha_stddev=cfg_alpha0.init_log_alpha_stddev,
        per_dim_constraining=True, action_penalization=True,
        epsilon_penalty=cfg_alpha0.epsilon_penalty,
    ).init_params(action_dim=4)

    obs_t0 = jnp.ones((cfg_alpha0.batch_size, 6))
    anchor_mu = jnp.full((cfg_alpha0.batch_size, 4), 0.2)
    anchor_log_std = jnp.full((cfg_alpha0.batch_size, 4), -0.5)
    rng, k_loss = jax.random.split(rng)

    loss0, _ = _policy_loss_fn(
        policy_params, dual, nets, obs_t0, target_policy_params,
        target_critic_params, cfg_alpha0, k_loss,
        anchor_mu_imit=anchor_mu, anchor_log_std_imit=anchor_log_std,
    )
    loss1, stats1 = _policy_loss_fn(
        policy_params, dual, nets, obs_t0, target_policy_params,
        target_critic_params, cfg_alpha1, k_loss,
        anchor_mu_imit=anchor_mu, anchor_log_std_imit=anchor_log_std,
    )

    # Manually compute the expected delta = -alpha * mean(exp(-w * KL)).
    online_dist = jax.vmap(nets.policy.apply, in_axes=(None, 0))(policy_params, obs_t0)
    mu_theta = online_dist.mean()
    log_std_theta = jnp.log(online_dist.stddev())
    kl = pretanh_gaussian_kl(mu_theta, log_std_theta, anchor_mu, anchor_log_std)
    expected_delta = -cfg_alpha1.kl_anchor_alpha * jnp.mean(
        jnp.exp(-cfg_alpha1.kl_anchor_w * kl)
    )

    np.testing.assert_allclose(
        float(loss1 - loss0), float(expected_delta), atol=1e-5,
        err_msg=f"loss1-loss0={loss1-loss0:.6f}, expected={expected_delta:.6f}",
    )
    # Stats must include the anchor metrics.
    assert "anchor_kl_mean" in stats1._asdict() or hasattr(stats1, "anchor_kl_mean"), \
        "MPOStats must expose anchor_kl_mean (add as new optional field)"


def test_policy_loss_alpha_zero_equals_pure_mpo():
    """With kl_anchor_alpha=0, the policy loss equals the pure MPO loss
    regardless of whether anchor_mu_imit is supplied.
    """
    from track_mjx.agent.dmpo.learner import _policy_loss_fn

    cfg = DMPOConfig(
        num_envs=4, batch_size=4, sequence_length=4,
        kl_anchor_alpha=0.0, kl_anchor_w=0.5,
    )
    nets = _make_tiny_networks(action_size=4, obs_size=6)

    rng = jax.random.PRNGKey(0)
    rng, k_init = jax.random.split(rng)
    policy_params = nets.policy.init(k_init, jnp.zeros((1, 6)))
    target_policy_params = jax.tree.map(lambda x: x + 0.0, policy_params)
    target_critic_params = nets.critic.init(rng, jnp.zeros((1, 6)), jnp.zeros((1, 4)))
    from track_mjx.agent.dmpo.losses import MPO
    dual = MPO(
        epsilon=cfg.epsilon, epsilon_mean=cfg.epsilon_mean,
        epsilon_stddev=cfg.epsilon_stddev,
        init_log_temperature=cfg.init_log_temperature,
        init_log_alpha_mean=cfg.init_log_alpha_mean,
        init_log_alpha_stddev=cfg.init_log_alpha_stddev,
        per_dim_constraining=True, action_penalization=True,
        epsilon_penalty=cfg.epsilon_penalty,
    ).init_params(action_dim=4)

    obs_t0 = jnp.ones((cfg.batch_size, 6))
    rng, k_loss = jax.random.split(rng)

    loss_no_anchor, _ = _policy_loss_fn(
        policy_params, dual, nets, obs_t0, target_policy_params,
        target_critic_params, cfg, k_loss,
        anchor_mu_imit=None, anchor_log_std_imit=None,
    )
    loss_with_anchor, _ = _policy_loss_fn(
        policy_params, dual, nets, obs_t0, target_policy_params,
        target_critic_params, cfg, k_loss,
        anchor_mu_imit=jnp.zeros((cfg.batch_size, 4)),
        anchor_log_std_imit=jnp.zeros((cfg.batch_size, 4)),
    )
    np.testing.assert_allclose(float(loss_no_anchor), float(loss_with_anchor), atol=1e-7)
