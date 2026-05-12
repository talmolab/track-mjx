import dataclasses
import jax
import jax.numpy as jnp
import optax
from track_mjx.agent.dmpo.config import DMPOConfig
from track_mjx.agent.dmpo.learner import (
    TrainingState, init_training_state, make_optimizers, sgd_step,
)
from track_mjx.agent.dmpo.networks import make_dmpo_networks
from track_mjx.agent.dmpo.losses import MPOParams
from track_mjx.agent.dmpo.learner import compute_categorical_target


def test_make_optimizers_returns_three(dummy=None):
    cfg = DMPOConfig()
    pol_opt, crit_opt, dual_opt = make_optimizers(cfg)
    assert isinstance(pol_opt, optax.GradientTransformation)
    assert isinstance(crit_opt, optax.GradientTransformation)
    assert isinstance(dual_opt, optax.GradientTransformation)


def test_training_state_shapes(rng, env_spec):
    cfg = DMPOConfig()
    nets = make_dmpo_networks(env_spec["obs_size"], env_spec["action_size"], cfg)
    state = init_training_state(rng, nets, env_spec, cfg)

    assert isinstance(state, TrainingState)
    assert isinstance(state.dual_params, MPOParams)
    # Per-dim constraint => alpha shape matches action dim.
    assert state.dual_params.log_alpha_mean.shape == (env_spec["action_size"],)
    assert state.dual_params.log_alpha_stddev.shape == (env_spec["action_size"],)
    # Dual log-temperature is shape (1,) in vnl-ray and Acme. Confirm what we have.
    assert state.dual_params.log_temperature.shape == (1,)
    # Initial values match vnl-ray defaults (log-space, post softplus interpretation).
    assert jnp.isclose(state.dual_params.log_temperature.squeeze(), cfg.init_log_temperature)
    assert jnp.allclose(state.dual_params.log_alpha_mean, cfg.init_log_alpha_mean)
    assert jnp.allclose(state.dual_params.log_alpha_stddev, cfg.init_log_alpha_stddev)
    # Steps starts at 0, jnp.int32 scalar.
    assert state.steps.shape == ()
    assert state.steps.dtype == jnp.int32
    # Target params equal online params at init (no asymmetry yet).
    chex_equal_pol = jax.tree.map(
        lambda a, b: jnp.array_equal(a, b),
        state.policy_params, state.target_policy_params,
    )
    assert all(jax.tree_util.tree_leaves(chex_equal_pol))
    chex_equal_crit = jax.tree.map(
        lambda a, b: jnp.array_equal(a, b),
        state.critic_params, state.target_critic_params,
    )
    assert all(jax.tree_util.tree_leaves(chex_equal_crit))


def test_training_state_is_pytree(rng, env_spec):
    """TrainingState must be a JAX pytree so jit/grad/vmap work on it."""
    cfg = DMPOConfig()
    nets = make_dmpo_networks(env_spec["obs_size"], env_spec["action_size"], cfg)
    state = init_training_state(rng, nets, env_spec, cfg)
    leaves = jax.tree_util.tree_leaves(state)
    assert len(leaves) > 0
    # All leaves must be JAX-traceable: jnp arrays or Python numeric scalars.
    # Note: RunningStatisticsState.std_eps is a Python float — a valid JAX
    # pytree leaf that is handled correctly by jit/grad/vmap.
    assert all(
        isinstance(l, (jnp.ndarray, float, int)) for l in leaves
    )
    # Stronger check: the state must actually be jit-traceable.
    f = jax.jit(lambda s: s.steps)
    assert f(state).shape == ()


def test_compute_categorical_target_shapes(rng, env_spec):
    cfg = DMPOConfig()
    nets = make_dmpo_networks(env_spec["obs_size"], env_spec["action_size"], cfg)
    state = init_training_state(rng, nets, env_spec, cfg)

    B, T = 4, 5
    next_obs = jnp.zeros((B, T, env_spec["obs_size"]))
    next_action = jnp.zeros((B, T, env_spec["action_size"]))
    rewards = jnp.zeros((B, T))
    discounts = jnp.ones((B, T))  # raw not_done mask

    target_probs = compute_categorical_target(
        nets, state.target_critic_params,
        next_obs, next_action, rewards, discounts, cfg,
    )
    assert target_probs.shape == (B, T, cfg.num_atoms)
    # Each row must sum to 1 (probability distribution).
    assert jnp.allclose(target_probs.sum(axis=-1), 1.0, atol=1e-5)


def test_compute_categorical_target_zero_reward_zero_discount(rng, env_spec):
    """When r=0 and γ=0, target = δ(0) projected onto atoms (vmin..vmax with 0 in support).
    Verifies the projection numerically: probability mass concentrates near 0."""
    cfg = DMPOConfig()
    nets = make_dmpo_networks(env_spec["obs_size"], env_spec["action_size"], cfg)
    state = init_training_state(rng, nets, env_spec, cfg)

    B, T = 2, 1
    next_obs = jnp.zeros((B, T, env_spec["obs_size"]))
    next_action = jnp.zeros((B, T, env_spec["action_size"]))
    rewards = jnp.zeros((B, T))
    discounts = jnp.zeros((B, T))  # γ=0 zeroes out the bootstrapped distribution.

    target_probs = compute_categorical_target(
        nets, state.target_critic_params,
        next_obs, next_action, rewards, discounts, cfg,
    )
    atoms = jnp.linspace(cfg.vmin, cfg.vmax, cfg.num_atoms)
    expected_value = (target_probs * atoms[None, None, :]).sum(-1)
    # Mean of projected distribution should be ~0 (project δ(0) onto symmetric atom grid).
    assert jnp.allclose(expected_value, 0.0, atol=1e-3)


def test_sgd_step_runs_and_advances(rng, env_spec):
    cfg = DMPOConfig()
    nets = make_dmpo_networks(env_spec["obs_size"], env_spec["action_size"], cfg)
    state = init_training_state(rng, nets, env_spec, cfg)
    optimizers = make_optimizers(cfg)

    B, T = 4, 6
    obs_dim, act_dim = env_spec["obs_size"], env_spec["action_size"]
    batch = {
        "observation": jnp.zeros((B, T, obs_dim)),
        "action": jnp.zeros((B, T, act_dim)),
        "reward": jnp.zeros((B, T)),
        "discount": jnp.ones((B, T)),
        "next_observation": jnp.zeros((B, T, obs_dim)),
    }
    new_state, metrics = sgd_step(state, batch, nets, optimizers, cfg)

    assert int(new_state.steps) == int(state.steps) + 1
    assert "policy_loss" in metrics
    assert "critic_loss" in metrics
    assert jnp.isfinite(metrics["policy_loss"])
    assert jnp.isfinite(metrics["critic_loss"])
    # Dual variables are still valid (not negative log-temperature etc).
    # log_temperature has shape (1,); the clip in MPO floors at -18.
    assert float(new_state.dual_params.log_temperature.squeeze()) >= -18.0


def test_sgd_step_jittable(rng, env_spec):
    cfg = DMPOConfig()
    nets = make_dmpo_networks(env_spec["obs_size"], env_spec["action_size"], cfg)
    state = init_training_state(rng, nets, env_spec, cfg)
    optimizers = make_optimizers(cfg)
    jitted = jax.jit(lambda s, b: sgd_step(s, b, nets, optimizers, cfg))

    B, T = 4, 6
    obs_dim, act_dim = env_spec["obs_size"], env_spec["action_size"]
    batch = {
        "observation": jnp.zeros((B, T, obs_dim)),
        "action": jnp.zeros((B, T, act_dim)),
        "reward": jnp.zeros((B, T)),
        "discount": jnp.ones((B, T)),
        "next_observation": jnp.zeros((B, T, obs_dim)),
    }
    new_state, metrics = jitted(state, batch)
    assert jnp.isfinite(metrics["critic_loss"])


def test_sgd_step_target_update_schedule(rng, env_spec):
    """After cfg.target_critic_update_period steps, target_critic_params change.
    Before, they should remain equal to initial."""
    cfg = DMPOConfig()
    nets = make_dmpo_networks(env_spec["obs_size"], env_spec["action_size"], cfg)
    state = init_training_state(rng, nets, env_spec, cfg)
    optimizers = make_optimizers(cfg)

    B, T = 4, 6
    obs_dim, act_dim = env_spec["obs_size"], env_spec["action_size"]
    batch = {
        "observation": jax.random.normal(rng, (B, T, obs_dim)),
        "action": jax.random.normal(rng, (B, T, act_dim)),
        "reward": jax.random.normal(rng, (B, T)),
        "discount": jnp.ones((B, T)),
        "next_observation": jax.random.normal(rng, (B, T, obs_dim)),
    }
    initial_target_crit = state.target_critic_params
    # Run a few steps but fewer than the update period (107).
    for _ in range(5):
        state, _ = sgd_step(state, batch, nets, optimizers, cfg)
    # Target should still equal initial.
    eq = jax.tree.map(lambda a, b: jnp.array_equal(a, b),
                      state.target_critic_params, initial_target_crit)
    assert all(jax.tree_util.tree_leaves(eq))


def test_compute_categorical_target_applies_gamma(rng, env_spec):
    """Bellman target must scale next-state value by cfg.discount (γ).

    With reward=0 and not_done=1, the target equals γ * V(s') projected
    onto the atom grid. Two cfg.discount values produce expected values that
    scale linearly. If γ is silently 1.0 (the bug), both produce the same
    expected value.
    """
    cfg_high = dataclasses.replace(DMPOConfig(), discount=1.0)
    cfg_low = dataclasses.replace(DMPOConfig(), discount=0.5)
    nets = make_dmpo_networks(
        env_spec["obs_size"], env_spec["action_size"], cfg_high
    )
    state = init_training_state(rng, nets, env_spec, cfg_high)

    B, T = 2, 1
    k_obs, k_act = jax.random.split(rng)
    next_obs = jax.random.normal(k_obs, (B, T, env_spec["obs_size"]))
    next_action = jax.random.normal(k_act, (B, T, env_spec["action_size"]))
    rewards = jnp.zeros((B, T))
    discounts = jnp.ones((B, T))  # not_done mask = 1 (live transition)

    target_high = compute_categorical_target(
        nets, state.target_critic_params,
        next_obs, next_action, rewards, discounts, cfg_high,
    )
    target_low = compute_categorical_target(
        nets, state.target_critic_params,
        next_obs, next_action, rewards, discounts, cfg_low,
    )
    atoms = jnp.linspace(cfg_high.vmin, cfg_high.vmax, cfg_high.num_atoms)
    ev_high = (target_high * atoms[None, None, :]).sum(-1)
    ev_low = (target_low * atoms[None, None, :]).sum(-1)
    # Guard against the degenerate case where the critic emits ~zero values
    # — the ratio test below would be meaningless.
    assert jnp.all(jnp.abs(ev_high) > 0.1), (
        f"degenerate ev_high≈0; γ-ratio test would be meaningless: ev_high={ev_high}"
    )
    # γ_low/γ_high = 0.5 ⇒ ev_low ≈ 0.5 * ev_high. Allow projection slack.
    assert jnp.allclose(ev_low, 0.5 * ev_high, atol=1e-2), (
        f"γ not applied: ev_high={ev_high}, ev_low={ev_low}, "
        f"ratio={ev_low / ev_high}"
    )
