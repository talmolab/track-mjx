"""Verify the linear-decay schedule for the kl-anchor `w` parameter.

`_policy_loss_fn` should:
- Default to static `w == cfg.kl_anchor_w` when `kl_anchor_decay_sgd_steps == 0`.
- Linearly decay from `cfg.kl_anchor_w` (at step 0) to `cfg.kl_anchor_w_floor`
  (at step `cfg.kl_anchor_decay_sgd_steps`) and clamp at the floor afterwards.
- Use the schedule-current `w_now` in the anchor reward, and surface the
  current value as `stats.anchor_w_now`.
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions

from track_mjx.agent.dmpo.config import DMPOConfig
from track_mjx.agent.dmpo.kl_anchor_utils import pretanh_gaussian_kl
from track_mjx.agent.dmpo.losses import MPO
from track_mjx.agent.dmpo.networks import DMPONetworks


# ----------------------------------------------------------------------
# Scaffolding (mirrors test_kl_in_loss_policy_loss.py).
# ----------------------------------------------------------------------
def _make_tiny_networks(action_size=4, obs_size=6):
    class _PolicyMod(nn.Module):
        action_size: int

        @nn.compact
        def __call__(self, obs):
            h = nn.Dense(8)(obs)
            mu = nn.Dense(self.action_size)(h)
            log_std = nn.Dense(self.action_size)(h)
            return tfd.MultivariateNormalDiag(
                loc=mu, scale_diag=jnp.exp(log_std) + 1e-3
            )

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


def _build_inputs(cfg, action_size=4, obs_size=6, seed=0):
    """Build (nets, policy_params, target_policy_params, target_critic_params,
    dual_params, obs_t0, anchor_mu, anchor_log_std, key) for direct
    `_policy_loss_fn` invocation."""
    nets = _make_tiny_networks(action_size=action_size, obs_size=obs_size)
    rng = jax.random.PRNGKey(seed)
    rng, k_init = jax.random.split(rng)
    policy_params = nets.policy.init(k_init, jnp.zeros((1, obs_size)))
    target_policy_params = jax.tree.map(lambda x: x + 0.0, policy_params)
    target_critic_params = nets.critic.init(
        rng, jnp.zeros((1, obs_size)), jnp.zeros((1, action_size))
    )
    dual = MPO(
        epsilon=cfg.epsilon,
        epsilon_mean=cfg.epsilon_mean,
        epsilon_stddev=cfg.epsilon_stddev,
        init_log_temperature=cfg.init_log_temperature,
        init_log_alpha_mean=cfg.init_log_alpha_mean,
        init_log_alpha_stddev=cfg.init_log_alpha_stddev,
        per_dim_constraining=True,
        action_penalization=True,
        epsilon_penalty=cfg.epsilon_penalty,
    ).init_params(action_dim=action_size)
    obs_t0 = jnp.ones((cfg.batch_size, obs_size))
    anchor_mu = jnp.full((cfg.batch_size, action_size), 0.2)
    anchor_log_std = jnp.full((cfg.batch_size, action_size), -0.5)
    rng, k_loss = jax.random.split(rng)
    return (
        nets,
        policy_params,
        target_policy_params,
        target_critic_params,
        dual,
        obs_t0,
        anchor_mu,
        anchor_log_std,
        k_loss,
    )


def _call_policy_loss(cfg, step, *, nets_seed=0):
    """Run `_policy_loss_fn` with the given (cfg, step) and return
    (loss, stats).
    """
    from track_mjx.agent.dmpo.learner import _policy_loss_fn

    (
        nets,
        policy_params,
        target_policy_params,
        target_critic_params,
        dual,
        obs_t0,
        anchor_mu,
        anchor_log_std,
        k_loss,
    ) = _build_inputs(cfg, seed=nets_seed)
    return _policy_loss_fn(
        policy_params,
        dual,
        nets,
        obs_t0,
        target_policy_params,
        target_critic_params,
        cfg,
        k_loss,
        anchor_mu_imit=anchor_mu,
        anchor_log_std_imit=anchor_log_std,
        step=jnp.int32(step),
    )


# ----------------------------------------------------------------------
# Tests.
# ----------------------------------------------------------------------
def test_decay_disabled_default():
    """With `decay_sgd_steps == 0` (the default), `w_now` is constant at
    `cfg.kl_anchor_w` regardless of the SGD step.
    """
    cfg = DMPOConfig(
        num_envs=4,
        batch_size=4,
        sequence_length=4,
        kl_anchor_alpha=1.0,
        kl_anchor_w=0.5,
    )
    for step in (0, 1000, 10**9):
        _, stats = _call_policy_loss(cfg, step)
        np.testing.assert_allclose(
            float(stats.anchor_w_now), 0.5, atol=1e-7,
            err_msg=f"step={step}: w_now should be static 0.5",
        )


def test_decay_at_step_zero():
    """At `step == 0`, `w_now` equals the start value `cfg.kl_anchor_w`."""
    cfg = DMPOConfig(
        num_envs=4,
        batch_size=4,
        sequence_length=4,
        kl_anchor_alpha=1.0,
        kl_anchor_w=1.0,
        kl_anchor_w_floor=0.05,
        kl_anchor_decay_sgd_steps=1000,
    )
    _, stats = _call_policy_loss(cfg, 0)
    np.testing.assert_allclose(float(stats.anchor_w_now), 1.0, atol=1e-7)


def test_decay_at_step_end():
    """At `step == decay_sgd_steps`, `w_now` equals the floor."""
    cfg = DMPOConfig(
        num_envs=4,
        batch_size=4,
        sequence_length=4,
        kl_anchor_alpha=1.0,
        kl_anchor_w=1.0,
        kl_anchor_w_floor=0.05,
        kl_anchor_decay_sgd_steps=1000,
    )
    _, stats = _call_policy_loss(cfg, 1000)
    np.testing.assert_allclose(float(stats.anchor_w_now), 0.05, atol=1e-7)


def test_decay_at_midpoint():
    """At `step == decay_sgd_steps / 2`, `w_now` equals the linear midpoint."""
    cfg = DMPOConfig(
        num_envs=4,
        batch_size=4,
        sequence_length=4,
        kl_anchor_alpha=1.0,
        kl_anchor_w=1.0,
        kl_anchor_w_floor=0.05,
        kl_anchor_decay_sgd_steps=1000,
    )
    _, stats = _call_policy_loss(cfg, 500)
    # Linear midpoint of (1.0, 0.05): 1.0 + (0.05 - 1.0) * 0.5 = 0.525.
    np.testing.assert_allclose(float(stats.anchor_w_now), 0.525, atol=1e-6)
    # Guard against silent float64 promotion if someone removes the
    # `jnp.float32(...)` wrapper around `anchor_w_now`.
    assert stats.anchor_w_now.dtype == jnp.float32


def test_decay_clamped_past_end():
    """At `step >> decay_sgd_steps`, `w_now` is clamped at the floor (no
    overshoot below `kl_anchor_w_floor`).
    """
    cfg = DMPOConfig(
        num_envs=4,
        batch_size=4,
        sequence_length=4,
        kl_anchor_alpha=1.0,
        kl_anchor_w=1.0,
        kl_anchor_w_floor=0.05,
        kl_anchor_decay_sgd_steps=1000,
    )
    _, stats = _call_policy_loss(cfg, 10_000)
    np.testing.assert_allclose(float(stats.anchor_w_now), 0.05, atol=1e-7)


def test_anchor_reward_uses_w_now():
    """`anchor_reward_mean` must use the schedule-current `w_now`, not the
    static `cfg.kl_anchor_w`.

    We pick deterministic anchor inputs (anchor_mu = 0.2, anchor_log_std =
    -0.5, dist defaults from fresh init) and compare the reward at the
    midpoint of the decay against `mean(exp(-w_now * KL))` recomputed
    independently with `w_now == 0.525`.
    """
    cfg = DMPOConfig(
        num_envs=4,
        batch_size=4,
        sequence_length=4,
        kl_anchor_alpha=1.0,
        kl_anchor_w=1.0,
        kl_anchor_w_floor=0.05,
        kl_anchor_decay_sgd_steps=1000,
    )
    (
        nets,
        policy_params,
        target_policy_params,
        target_critic_params,
        dual,
        obs_t0,
        anchor_mu,
        anchor_log_std,
        k_loss,
    ) = _build_inputs(cfg, seed=0)

    from track_mjx.agent.dmpo.learner import _policy_loss_fn

    _, stats_mid = _policy_loss_fn(
        policy_params,
        dual,
        nets,
        obs_t0,
        target_policy_params,
        target_critic_params,
        cfg,
        k_loss,
        anchor_mu_imit=anchor_mu,
        anchor_log_std_imit=anchor_log_std,
        step=jnp.int32(500),
    )

    # Recompute the expected reward at w_now = 0.525.
    online_dist = jax.vmap(nets.policy.apply, in_axes=(None, 0))(
        policy_params, obs_t0
    )
    mu_theta = online_dist.mean()
    log_std_theta = jnp.log(online_dist.stddev())
    kl = pretanh_gaussian_kl(mu_theta, log_std_theta, anchor_mu, anchor_log_std)
    expected_reward = float(jnp.mean(jnp.exp(-0.525 * kl)))

    np.testing.assert_allclose(
        float(stats_mid.anchor_reward_mean),
        expected_reward,
        rtol=1e-5,
        atol=1e-6,
    )
    # Sanity: w_now reported in stats matches the value used.
    np.testing.assert_allclose(float(stats_mid.anchor_w_now), 0.525, atol=1e-6)


def test_anchor_w_now_falls_back_to_static_when_alpha_zero():
    """When `kl_anchor_alpha == 0.0`, the anchor branch is skipped entirely
    even if `kl_anchor_decay_sgd_steps > 0`. In that case `anchor_w_now`
    should default to the static `cfg.kl_anchor_w` (NOT a value computed
    from the decay schedule).
    """
    cfg = DMPOConfig(
        num_envs=4,
        batch_size=4,
        sequence_length=4,
        kl_anchor_alpha=0.0,
        kl_anchor_w=0.7,
        kl_anchor_w_floor=0.05,
        kl_anchor_decay_sgd_steps=1000,
    )
    _, stats = _call_policy_loss(cfg, 500)
    # The schedule midpoint would be 0.375 if the branch ran. Assert we get
    # the static start value 0.7 instead.
    np.testing.assert_allclose(float(stats.anchor_w_now), 0.7, atol=1e-7)
