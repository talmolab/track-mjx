"""Sanity test for vendored MPO loss. Numerical parity vs vnl-ray comes in Task 7."""
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp
from track_mjx.agent.dmpo.losses import MPO, MPOParams, clip_mpo_params


tfd = tfp.distributions


def test_mpo_init_params():
    loss_fn = MPO(
        epsilon=0.1, epsilon_mean=0.0025, epsilon_stddev=1e-7,
        epsilon_penalty=0.1,
        init_log_temperature=10.0,
        init_log_alpha_mean=10.0,
        init_log_alpha_stddev=1000.0,
        per_dim_constraining=True,
        action_penalization=True,
    )
    params = loss_fn.init_params(action_dim=6, dtype=jnp.float32)
    assert isinstance(params, MPOParams)
    # Per-dim constraining → alphas have shape (action_dim,)
    assert params.log_alpha_mean.shape == (6,)
    assert params.log_alpha_stddev.shape == (6,)


def test_mpo_runs_on_dummy_inputs():
    loss_fn = MPO(
        epsilon=0.1, epsilon_mean=0.0025, epsilon_stddev=1e-7,
        epsilon_penalty=0.1,
        init_log_temperature=10.0,
        init_log_alpha_mean=10.0,
        init_log_alpha_stddev=1000.0,
        per_dim_constraining=True,
        action_penalization=True,
    )
    B, A, N = 4, 6, 20  # batch, action_dim, num samples
    online = tfd.MultivariateNormalDiag(loc=jnp.zeros((B, A)), scale_diag=jnp.ones((B, A)))
    target = tfd.MultivariateNormalDiag(loc=jnp.zeros((B, A)), scale_diag=jnp.ones((B, A)))
    params = loss_fn.init_params(action_dim=A, dtype=jnp.float32)
    sampled_actions = jax.random.normal(jax.random.PRNGKey(0), (N, B, A))
    q_values = jax.random.normal(jax.random.PRNGKey(1), (N, B))

    loss, stats = loss_fn(
        params=params,
        online_action_distribution=online,
        target_action_distribution=target,
        actions=sampled_actions,
        q_values=q_values,
    )
    assert jnp.isfinite(loss)
    # The full numerical-parity test against vnl-ray TF lives in Task 7.
