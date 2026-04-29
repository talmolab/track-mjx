import jax
import jax.numpy as jnp
import optax
from track_mjx.agent.dmpo.config import DMPOConfig
from track_mjx.agent.dmpo.learner import (
    TrainingState, init_training_state, make_optimizers,
)
from track_mjx.agent.dmpo.networks import make_dmpo_networks
from track_mjx.agent.dmpo.losses import MPOParams


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
