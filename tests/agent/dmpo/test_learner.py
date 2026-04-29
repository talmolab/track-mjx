import jax
import jax.numpy as jnp
import optax
from track_mjx.agent.dmpo.config import DMPOConfig
from track_mjx.agent.dmpo.learner import (
    TrainingState, init_training_state, make_optimizers,
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
    # All leaves should be jnp arrays (no python objects in the tree).
    assert all(isinstance(l, jnp.ndarray) for l in leaves)


def test_compute_categorical_target_shapes(rng, env_spec):
    cfg = DMPOConfig()
    nets = make_dmpo_networks(env_spec["obs_size"], env_spec["action_size"], cfg)
    state = init_training_state(rng, nets, env_spec, cfg)

    B, T = 4, 5
    next_obs = jnp.zeros((B, T, env_spec["obs_size"]))
    next_action = jnp.zeros((B, T, env_spec["action_size"]))
    rewards = jnp.zeros((B, T))
    discounts = 0.97 * jnp.ones((B, T))

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
